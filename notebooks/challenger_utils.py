"""Shared helpers for the tree-challenger experiments (Notebooks 16a and 16b).

Follows the Notebook 16 protocol:
- residual target: log1p(auction_volume) - log1p(median20_auction_volume)
- five-year rolling training windows, the preceding year for validation/early stopping
- 2022-2024 development years, 2025 confirmation year
- hyperparameters chosen on historical validation data only
- replacement gate: mean development log RMSE improves by >= 0.002, no development
  year worse by > 0.002, and the paired date-block log-MAE bootstrap 97.5% bound < 0

All candidates in a data tier share identical sampled rows and preprocessing, so
differences between candidates come from the change being tested, not from sampling.
"""

from pathlib import Path
import gc
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate blackrock_data.")
sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.modeling_utils import fit_preprocessor, apply_preprocessor, forecast_metrics  # noqa: E402

EDA = PROJECT_ROOT / "outputs" / "eda"
PANEL = EDA / "05_feature_engineering" / "enriched_daily_panel.parquet"
BENCHMARK_JSON = EDA / "09_global_finalists" / "frozen_global_benchmark.json"
BENCHMARK_PREDICTIONS = EDA / "09_global_finalists" / "predictions"
for required in [PANEL, BENCHMARK_JSON, BENCHMARK_PREDICTIONS]:
    if not required.exists():
        raise FileNotFoundError(f"{required} is missing; run Notebooks 05 and 09 first.")

with open(BENCHMARK_JSON) as handle:
    FROZEN = json.load(handle)
CHAMPION = FROZEN["champion"]
BASE_FEATURES = list(FROZEN["features"])
assert len(BASE_FEATURES) == 58

# New ratio features. Every input is already known by the end of t-1
# (all rolling auction windows in Notebook 05 end at 1 PRECEDING).
# Logged ratios with +1 smoothing; missing when either side is missing.
RATIO_SQL = {
    "ratio_lag1_med20": "ln((lag1_auction_volume + 1) / (median20_auction_volume + 1))",
    "ratio_avg5_med20": "ln((avg5_auction_volume + 1) / (median20_auction_volume + 1))",
    "ratio_lag1_med60": "ln((lag1_auction_volume + 1) / (median60_auction_volume + 1))",
}
RATIO_FEATURES = list(RATIO_SQL)
RATIO_SET = BASE_FEATURES + RATIO_FEATURES

BOOLEAN_ALL = {
    "auction_spike_lag1", "is_monday", "is_friday",
    "is_first_trading_day_month", "is_last_trading_day_month",
    "is_last_3_trading_days_month", "is_last_5_trading_days_month",
    "is_last_trading_day_quarter", "is_last_3_trading_days_quarter",
    "is_last_5_trading_days_quarter", "is_last_trading_day_year",
    "monthly_options_expiry", "quarterly_options_expiry",
    "after_long_market_gap", "before_long_market_gap", "is_early_close",
    "early_close_month_end_window", "expiry_month_end_window",
}
CATEGORICAL_ALL = {"month", "day_of_week"}
BOOLEAN = [f for f in RATIO_SET if f in BOOLEAN_ALL]
CATEGORICAL = [f for f in RATIO_SET if f in CATEGORICAL_ALL]
CONTINUOUS = [f for f in RATIO_SET if f not in BOOLEAN_ALL and f not in CATEGORICAL_ALL]

CONTEXT = [
    "date", "permno", "auction_volume", "log1p_auction_volume",
    "median20_auction_volume", "size_percentile", "prior_auction_observations",
    "quarterly_options_expiry", "monthly_options_expiry", "is_last_trading_day_month",
]

TEST_YEARS = [2022, 2023, 2024, 2025]
DEVELOPMENT_YEARS = [2022, 2023, 2024]
CONFIRMATION_YEAR = 2025
SEED = 20261018  # same base seed as Notebook 16

# Row budgets per data tier: (tune-train rows, validation rows, refit rows).
# "base" matches Notebook 16 exactly; "large" roughly triples the training data.
TIERS = {
    "base": {"tune": 250_000, "validation": 120_000, "refit": 350_000},
    "large": {"tune": 750_000, "validation": 150_000, "refit": 1_000_000},
}

# Smoke mode: tiny samples and two folds, only to check that the code runs end to end.
SMOKE = os.environ.get("CHALLENGER_SMOKE") == "1"
if SMOKE:
    TEST_YEARS = [2022, 2025]
    DEVELOPMENT_YEARS = [2022]
    TIERS = {
        "base": {"tune": 20_000, "validation": 10_000, "refit": 25_000},
        "large": {"tune": 30_000, "validation": 10_000, "refit": 40_000},
    }


def output_dir(name):
    """Results folder under outputs/eda; smoke runs go to a separate _smoke folder."""
    path = EDA / (f"{name}_smoke" if SMOKE else name)
    path.mkdir(parents=True, exist_ok=True)
    return path

con = duckdb.connect()
con.execute("PRAGMA threads=4")


def sql(query):
    return con.execute(query).df()


_needed = list(dict.fromkeys(CONTEXT + BASE_FEATURES + [
    "lag1_auction_volume", "avg5_auction_volume", "median60_auction_volume",
]))
_select = ", ".join([f'"{c}"' for c in _needed] + [f"{e} AS {n}" for n, e in RATIO_SQL.items()])


def load_period(start_year, end_year, sample_rows=None, seed=SEED):
    """Eligible rows with dates in [start_year-01-01, end_year-01-01)."""
    sample = f"USING SAMPLE {sample_rows} ROWS (reservoir, {seed})" if sample_rows else ""
    frame = sql(f"""
        SELECT * FROM (
            SELECT {_select} FROM read_parquet('{PANEL}')
            WHERE eligible_target
              AND date >= DATE '{start_year}-01-01' AND date < DATE '{end_year}-01-01'
        ) {sample}
    """)
    # Deterministic row order regardless of DuckDB's parallel scan order.
    return frame.sort_values(["date", "permno"], kind="mergesort").reset_index(drop=True)


def residual_target_and_base(train, score):
    fallback = float(np.median(train["auction_volume"]))
    train_base = train["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    score_base = score["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    target = train["log1p_auction_volume"].to_numpy(dtype=float) - np.log1p(train_base)
    return target, np.log1p(score_base)


def feature_mask(prep, features):
    allowed = set(features)
    return np.array([source in allowed for source in prep["output_sources"]])


def final_log_prediction(residual_prediction, base_log, train_log):
    lo, hi = np.quantile(train_log, [0.001, 0.999])
    return np.clip(base_log + residual_prediction, lo, hi)


def load_benchmark_predictions(year):
    path = BENCHMARK_PREDICTIONS / f"global_finalist_predictions_{year}.parquet"
    return sql(f'SELECT date, permno, "{CHAMPION}" AS frozen_prediction FROM read_parquet(\'{path}\')')


def attach_extra(frame, extra_frame):
    """Left-join date-level extra features (a DataFrame indexed by date) onto a panel frame."""
    if extra_frame is None:
        return frame
    extra_frame = extra_frame.copy()
    extra_frame.index = pd.to_datetime(extra_frame.index).astype("datetime64[ns]")
    left = frame.assign(_join_date=pd.to_datetime(frame["date"]).astype("datetime64[ns]"))
    merged = left.merge(extra_frame, left_on="_join_date", right_index=True, how="left")
    return merged.drop(columns="_join_date")


def prepare_tier(test_year, tier, extra=None):
    """Load and preprocess tuning, validation, refit and test data for one fold and tier.

    `extra(test_year, phase)` optionally returns date-level features (DataFrame indexed by date) for
    phase "tune" (used for the tuning and validation rows) or "refit" (used for refit and test rows).
    This lets fold-specific fitted features, such as an HMM fitted only on that phase's training
    window, enter without leaking later data. Extra columns are treated as continuous features.
    """
    rows = TIERS[tier]
    validation_year = test_year - 1
    tune = load_period(validation_year - 5, validation_year, rows["tune"], SEED + test_year)
    validation = load_period(validation_year, test_year, rows["validation"], SEED + 100 + test_year)
    refit = load_period(test_year - 5, test_year, rows["refit"], SEED + 200 + test_year)

    continuous = list(CONTINUOUS)
    if extra is not None:
        tune_extra, refit_extra = extra(test_year, "tune"), extra(test_year, "refit")
        continuous += [c for c in tune_extra.columns if c not in continuous]
        tune, validation = attach_extra(tune, tune_extra), attach_extra(validation, tune_extra)
        refit = attach_extra(refit, refit_extra)

    x_tune, prep_tune = fit_preprocessor(tune, continuous, BOOLEAN, CATEGORICAL)
    x_validation = apply_preprocessor(validation, prep_tune, continuous, BOOLEAN, CATEGORICAL)
    y_tune, validation_base = residual_target_and_base(tune, validation)
    y_validation_log = validation["log1p_auction_volume"].to_numpy(dtype=float)

    x_refit, prep = fit_preprocessor(refit, continuous, BOOLEAN, CATEGORICAL)
    y_refit, _ = residual_target_and_base(refit, refit)
    return {
        "tier": tier, "continuous": continuous,
        "x_tune": x_tune, "y_tune": y_tune, "prep_tune": prep_tune,
        "tune_log": tune["log1p_auction_volume"].to_numpy(dtype=float),
        "x_validation": x_validation, "y_validation_log": y_validation_log,
        "y_validation_residual": y_validation_log - validation_base,
        "validation_base": validation_base,
        "x_refit": x_refit, "y_refit": y_refit, "prep": prep,
        "refit_log": refit["log1p_auction_volume"].to_numpy(dtype=float),
        "refit_frame_for_base": refit[["auction_volume", "median20_auction_volume", "log1p_auction_volume"]],
        "rows": {"tune": len(tune), "validation": len(validation), "refit": len(refit)},
    }


def run_experiment(candidates, out_dir, tag, extra=None):
    """Run every candidate through every fold.

    `extra` is passed to prepare_tier (see there); test rows get the "refit" phase features.
    Each candidate is a dict with:
      name, family, tier, features,
      configs: list of hyperparameter dicts searched on validation,
      fit(x, y, x_val, y_val, config, seed) -> (model, best_iteration),
      refit(x, y, config, iterations, seed) -> model,
      importance(model) -> array of gains per input column (optional).
    """
    out_dir = Path(out_dir)
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, validation_rows, importance_rows = [], [], []
    started = time.time()

    for test_year in TEST_YEARS:
        test = load_period(test_year, test_year + 1)
        output = test[CONTEXT].copy()
        output = output.merge(load_benchmark_predictions(test_year), on=["date", "permno"],
                              how="left", validate="one_to_one")
        if output["frozen_prediction"].isna().any():
            raise AssertionError("Frozen benchmark predictions did not align with the test panel.")
        y_test_raw = test["auction_volume"].to_numpy(dtype=float)
        metric_rows.append({"test_year": test_year, "candidate": CHAMPION, "family": "frozen ridge",
                            **forecast_metrics(y_test_raw, output["frozen_prediction"])})

        for tier in dict.fromkeys(c["tier"] for c in candidates):
            data = prepare_tier(test_year, tier, extra)
            test_frame = attach_extra(test, extra(test_year, "refit")) if extra is not None else test
            x_test = apply_preprocessor(test_frame, data["prep"], data["continuous"], BOOLEAN, CATEGORICAL)
            _, test_base = residual_target_and_base(data["refit_frame_for_base"], test)
            print(f"[{time.time() - started:6.0f}s] {test_year} tier={tier} rows={data['rows']}", flush=True)

            for candidate in [c for c in candidates if c["tier"] == tier]:
                tune_mask = feature_mask(data["prep_tune"], candidate["features"])
                best = None
                for number, config in enumerate(candidate["configs"]):
                    model, iterations = candidate["fit"](
                        data["x_tune"][:, tune_mask], data["y_tune"],
                        data["x_validation"][:, tune_mask], data["y_validation_residual"],
                        config, SEED + test_year + number,
                    )
                    prediction = final_log_prediction(
                        model.predict(data["x_validation"][:, tune_mask]),
                        data["validation_base"], data["tune_log"],
                    )
                    rmse = float(np.sqrt(np.mean((data["y_validation_log"] - prediction) ** 2)))
                    row = {"test_year": test_year, "candidate": candidate["name"], "config_id": number,
                           "config": json.dumps(config, sort_keys=True), "best_iteration": iterations,
                           "validation_rmse_log": rmse}
                    validation_rows.append(row)
                    if best is None or rmse < best["validation_rmse_log"]:
                        best = {**row, "config_dict": config}
                    del model
                print(f"[{time.time() - started:6.0f}s]   {candidate['name']}: best config "
                      f"{best['config_id']} ({best['best_iteration']} iters, "
                      f"val rmse {best['validation_rmse_log']:.4f})", flush=True)

                mask = feature_mask(data["prep"], candidate["features"])
                model = candidate["refit"](data["x_refit"][:, mask], data["y_refit"], best["config_dict"],
                                           max(25, best["best_iteration"]), SEED + 300 + test_year)
                prediction_log = final_log_prediction(model.predict(x_test[:, mask]), test_base,
                                                      data["refit_log"])
                prediction_raw = np.maximum(np.expm1(prediction_log), 0)
                output[candidate["name"]] = prediction_raw
                metric_rows.append({"test_year": test_year, "candidate": candidate["name"],
                                    "family": candidate["family"], "selected_config": best["config_id"],
                                    "iterations": max(25, best["best_iteration"]),
                                    **forecast_metrics(y_test_raw, prediction_raw)})
                if candidate.get("importance"):
                    sources = np.array(data["prep"]["output_sources"])[mask]
                    gains = candidate["importance"](model)
                    fold = (pd.DataFrame({"source_feature": sources, "gain": gains})
                            .groupby("source_feature", as_index=False)["gain"].sum())
                    fold["gain_share"] = fold["gain"] / fold["gain"].sum()
                    fold["test_year"], fold["candidate"] = test_year, candidate["name"]
                    importance_rows.extend(fold.to_dict("records"))
                del model
                gc.collect()
            del data, x_test
            gc.collect()

        path = predictions_dir / f"{tag}_predictions_{test_year}.parquet"
        con.register("prediction_output", output)
        con.execute(f"COPY prediction_output TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.unregister("prediction_output")
        print(f"[{time.time() - started:6.0f}s] wrote {len(output):,} predictions for {test_year}", flush=True)
        del test, output
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    validation = pd.DataFrame(validation_rows)
    importance = pd.DataFrame(importance_rows)
    metrics.to_csv(out_dir / "candidate_metrics.csv", index=False)
    validation.to_csv(out_dir / "hyperparameter_validation.csv", index=False)
    if len(importance):
        importance.to_csv(out_dir / "feature_importance.csv", index=False)
    return metrics, validation, importance


def bootstrap_mean_interval(values, seed, draws=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def decision_table(metrics, prediction_glob, reference, candidates):
    """Apply the Notebook 16 replacement gate to each candidate against one reference column.

    `reference` is a candidate name; CHAMPION refers to the frozen_prediction column.
    """
    ref_column = "frozen_prediction" if reference == CHAMPION else reference
    rows = []
    for number, candidate in enumerate(candidates):
        daily = sql(f"""
            SELECT date, year(date) AS test_year,
                   avg(abs(ln(1 + "{candidate}") - log1p_auction_volume)
                     - abs(ln(1 + "{ref_column}") - log1p_auction_volume)) AS d
            FROM read_parquet('{prediction_glob}') GROUP BY 1, 2
        """)
        for period, years in {"development": DEVELOPMENT_YEARS, "confirmation": [CONFIRMATION_YEAR]}.items():
            mean, low, high = bootstrap_mean_interval(
                daily.loc[daily.test_year.isin(years), "d"], SEED + number)
            rows.append({"candidate": candidate, "period": period, "log_mae_diff": mean,
                         "ci_2_5": low, "ci_97_5": high})
    boot = pd.DataFrame(rows)

    annual = metrics.pivot_table(index="candidate", columns="test_year", values="rmse_log")
    dev = annual[DEVELOPMENT_YEARS]
    table = pd.DataFrame({
        "development_rmse_log": dev.mean(axis=1),
        "confirmation_rmse_log": annual[CONFIRMATION_YEAR],
        "development_wape": metrics[metrics.test_year.isin(DEVELOPMENT_YEARS)]
            .groupby("candidate").wape.mean(),
    }).loc[candidates + [reference]]
    table["rmse_change_vs_reference"] = table.development_rmse_log - table.loc[reference, "development_rmse_log"]
    table["worst_year_change"] = dev.sub(dev.loc[reference], axis=1).max(axis=1)
    table["confirmation_change"] = table.confirmation_rmse_log - table.loc[reference, "confirmation_rmse_log"]
    dev_boot = boot[boot.period == "development"].set_index("candidate")
    table["log_mae_diff"] = dev_boot.log_mae_diff
    table["log_mae_ci_97_5"] = dev_boot.ci_97_5
    table["practical_improvement"] = table.rmse_change_vs_reference <= -0.002
    table["stable_each_year"] = table.worst_year_change <= 0.002
    table["paired_significant"] = table.log_mae_ci_97_5 < 0
    table["passes_gate"] = table.practical_improvement & table.stable_each_year & table.paired_significant.fillna(False)
    flags = ["practical_improvement", "stable_each_year", "paired_significant", "passes_gate"]
    table[flags] = table[flags].astype(object)
    table.loc[reference, flags] = None  # the reference is not judged against itself
    table.insert(0, "reference", reference)
    return table.reset_index(names="candidate"), boot


def segment_table(prediction_glob, columns):
    """Development-year log RMSE by event type, size and target tail."""
    expressions = {
        "event": "CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry "
                 "THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",
        "size": "CASE WHEN size_percentile < 0.33 THEN 'small' WHEN size_percentile < 0.67 THEN 'mid' ELSE 'large' END",
        "actual_tail": "CASE WHEN pr >= 0.99 THEN 'top_1pct' ELSE 'lower_99pct' END",
    }
    parts = []
    for column in columns:
        for dimension, expression in expressions.items():
            parts.append(f"""SELECT '{column}' AS candidate, '{dimension}' AS dimension, {expression} AS segment,
                count(*) AS n, sqrt(avg(pow(ln(1 + "{column}") - log1p_auction_volume, 2))) AS rmse_log
                FROM ranked GROUP BY 1, 2, 3""")
    frame = sql(f"""
        WITH ranked AS (
            SELECT *, percent_rank() OVER (PARTITION BY year(date) ORDER BY auction_volume) AS pr
            FROM read_parquet('{prediction_glob}') WHERE year(date) IN (2022, 2023, 2024))
        {' UNION ALL '.join(parts)}""")
    return frame.pivot_table(index=["dimension", "segment"], columns="candidate", values="rmse_log")
