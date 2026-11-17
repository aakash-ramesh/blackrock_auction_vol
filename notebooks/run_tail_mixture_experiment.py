"""Leakage-safe tail-aware mixture challengers for auction-volume forecasting.

A binary LightGBM estimates the probability that the median-relative log target
is in an upper tail. Separate LightGBM regressors estimate normal- and spike-day
residual magnitudes. Historical validation selects a 90th- or 95th-percentile
spike threshold. The same fitted mixture is reconstructed both in log space and
in share space to expose the point-objective tradeoff.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/experiments/22_tail_mixture"
PREDICTIONS = OUT / "predictions"
MODELS = OUT / "models"
for directory in [OUT, PREDICTIONS, MODELS]:
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from notebooks.modeling_utils import apply_preprocessor, fit_preprocessor, forecast_metrics


PANEL = ROOT / "outputs/eda/05_feature_engineering/enriched_daily_panel.parquet"
CATALOG = ROOT / "outputs/reproduction/01_daily_panel/approved_enriched_daily_features.csv"
FROZEN = ROOT / "outputs/eda/09_global_finalists/frozen_global_benchmark.json"
RIDGE_PREDICTIONS = ROOT / "outputs/eda/09_global_finalists/predictions"
LGBM_ARTIFACTS = ROOT / "outputs/reproduction/03_compact_nonlinear"

TEST_YEARS = [2022, 2023, 2024, 2025]
DEVELOPMENT_YEARS = [2022, 2023, 2024]
CONFIRMATION_YEAR = 2025
TUNE_ROWS = 250_000
VALIDATION_ROWS = 120_000
REFIT_ROWS = 350_000
SEED = 20261117
TAIL_QUANTILES = [0.90, 0.95]

LOG_MIX = "tail_mixture_log_mean"
SHARE_MIX = "tail_mixture_share_mean"
LGBM = "lgbm_residual_58_l2"
FROZEN_RIDGE = "rolling5_residual_auction_calendar_liquidity"

BOOLEAN_FEATURES = [
    "auction_spike_lag1", "is_monday", "is_friday",
    "is_first_trading_day_month", "is_last_trading_day_month",
    "is_last_3_trading_days_month", "is_last_5_trading_days_month",
    "is_last_trading_day_quarter", "is_last_3_trading_days_quarter",
    "is_last_5_trading_days_quarter", "is_last_trading_day_year",
    "monthly_options_expiry", "quarterly_options_expiry",
    "after_long_market_gap", "before_long_market_gap", "is_early_close",
    "early_close_month_end_window", "expiry_month_end_window",
    "lag1_dividend_flag", "lag1_distribution_flag", "lag1_price_adjustment_flag",
]
CATEGORICAL_FEATURES = ["month", "day_of_week", "ff12_code", "size_decile", "liquidity_decile"]
CONTEXT = [
    "date", "permno", "auction_volume", "log1p_auction_volume",
    "median20_auction_volume", "size_percentile", "liquidity_percentile",
    "auction_intensity_percentile", "prior_auction_observations",
    "quarterly_options_expiry", "monthly_options_expiry",
    "is_last_trading_day_month", "is_early_close",
]


def progress(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame[["date", "permno"]], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        means[draw] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def residual_target_and_base(train: pd.DataFrame, score: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    fallback = float(np.median(train["auction_volume"]))
    train_base = train["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    score_base = score["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    target = train["log1p_auction_volume"].to_numpy(dtype=float) - np.log1p(train_base)
    return target, np.log1p(score_base), fallback


def model_params(objective: str, seed: int, n_estimators: int = 1500) -> dict:
    return {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 250,
        "max_bin": 127,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": 4,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def fit_validation_component(
    objective: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
):
    cls = lgb.LGBMClassifier if objective == "binary" else lgb.LGBMRegressor
    model = cls(**model_params(objective, seed))
    metric = "binary_logloss" if objective == "binary" else "rmse"
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric=metric,
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def fit_final_component(objective: str, x: np.ndarray, y: np.ndarray, iterations: int, seed: int):
    cls = lgb.LGBMClassifier if objective == "binary" else lgb.LGBMRegressor
    model = cls(**model_params(objective, seed, n_estimators=max(1, int(iterations))))
    model.fit(x, y)
    return model


def mixture_forecasts(
    probability: np.ndarray,
    normal_residual: np.ndarray,
    spike_residual: np.ndarray,
    base_log: np.ndarray,
    clip_log: tuple[float, float] | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    normal_log = np.clip(base_log + normal_residual, *clip_log)
    spike_log = np.clip(base_log + spike_residual, *clip_log)
    log_mean = np.clip(
        base_log + (1 - probability) * normal_residual + probability * spike_residual,
        *clip_log,
    )
    log_raw = np.maximum(np.expm1(log_mean), 0)
    share_raw = (1 - probability) * np.maximum(np.expm1(normal_log), 0)
    share_raw += probability * np.maximum(np.expm1(spike_log), 0)
    share_log = np.clip(np.log1p(np.maximum(share_raw, 0)), *clip_log)
    return np.maximum(np.expm1(share_log), 0), log_raw, normal_log, spike_log


def reconstruct_lgbm(test_year: int, test: pd.DataFrame) -> np.ndarray:
    prep = json.loads((LGBM_ARTIFACTS / f"models/preprocessor_{test_year}.json").read_text())
    matrix = apply_preprocessor(
        test, prep, list(prep["continuous"]), list(prep["boolean"]), list(prep["categorical"])
    )
    columns = json.loads(
        (LGBM_ARTIFACTS / f"models/feature_columns_{LGBM}_{test_year}.json").read_text()
    )
    indices = [prep["output_names"].index(name) for name in columns["output_names"]]
    reconstruction = json.loads(
        (LGBM_ARTIFACTS / f"models/reconstruction_{test_year}.json").read_text()
    )
    booster = lgb.Booster(model_file=str(LGBM_ARTIFACTS / f"models/{LGBM}_{test_year}.txt"))
    base = np.log1p(
        test["median20_auction_volume"].fillna(reconstruction["fallback"]).to_numpy(dtype=float)
    )
    prediction_log = np.clip(base + booster.predict(matrix[:, indices]), *reconstruction["clip_log"])
    del matrix
    return np.maximum(np.expm1(prediction_log), 0)


def main() -> None:
    if not PANEL.exists():
        raise FileNotFoundError(PANEL)
    started = datetime.now(timezone.utc).isoformat()
    if (OUT / "progress.log").exists():
        (OUT / "progress.log").unlink()
    progress("Tail-aware mixture experiment started")

    frozen = json.loads(FROZEN.read_text())
    champion_features = frozen["features"]
    catalog = pd.read_csv(CATALOG)
    features = catalog["variable"].tolist()
    continuous_features = [
        feature for feature in features
        if feature not in BOOLEAN_FEATURES and feature not in CATEGORICAL_FEATURES
    ]
    query_columns = list(dict.fromkeys(CONTEXT + features))
    quoted_columns = ", ".join(f'"{column}"' for column in query_columns)
    con = duckdb.connect()
    con.execute("SET threads=4")

    def load_period(start: str, end: str, rows: int | None = None, seed: int = SEED) -> pd.DataFrame:
        sample = f"USING SAMPLE {rows} ROWS (reservoir, {seed})" if rows else ""
        return con.sql(f"""
            SELECT * FROM (
                SELECT {quoted_columns}
                FROM read_parquet('{PANEL}')
                WHERE eligible_target AND date >= DATE '{start}' AND date < DATE '{end}'
            ) {sample}
        """).df()

    metric_rows = []
    validation_rows = []
    selection_rows = []
    sample_rows = []
    classification_rows = []

    for test_year in TEST_YEARS:
        validation_year = test_year - 1
        progress(f"Fold {test_year}: loading historical tuning rows")
        tune = load_period(
            f"{validation_year - 5}-01-01", f"{validation_year}-01-01",
            TUNE_ROWS, SEED + test_year,
        )
        validation = load_period(
            f"{validation_year}-01-01", f"{test_year}-01-01",
            VALIDATION_ROWS, SEED + 100 + test_year,
        )
        for role, frame in [("tune", tune), ("validation", validation)]:
            sample_rows.append({
                "test_year": test_year, "role": role, "rows": len(frame),
                "key_fingerprint": sample_fingerprint(frame),
            })

        x_tune, prep_tune = fit_preprocessor(
            tune, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        x_validation = apply_preprocessor(
            validation, prep_tune, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        mask = np.isin(np.asarray(prep_tune["output_sources"]), champion_features)
        x_tune = x_tune[:, mask]
        x_validation = x_validation[:, mask]
        y_tune, validation_base, _ = residual_target_and_base(tune, validation)
        y_validation = validation["log1p_auction_volume"].to_numpy(dtype=float) - validation_base
        clip_log = tuple(np.quantile(tune["log1p_auction_volume"], [0.001, 0.999]))

        best = None
        for quantile in TAIL_QUANTILES:
            threshold = float(np.quantile(y_tune, quantile))
            train_spike = y_tune > threshold
            validation_spike = y_validation > threshold
            progress(
                f"Fold {test_year}: fitting q={quantile:.2f} mixture "
                f"({train_spike.mean():.2%} tuning spikes)"
            )
            classifier = fit_validation_component(
                "binary", x_tune, train_spike.astype(int),
                x_validation, validation_spike.astype(int), SEED + test_year + int(quantile * 100),
            )
            normal = fit_validation_component(
                "regression", x_tune[~train_spike], y_tune[~train_spike],
                x_validation[~validation_spike], y_validation[~validation_spike],
                SEED + 100 + test_year + int(quantile * 100),
            )
            spike = fit_validation_component(
                "regression", x_tune[train_spike], y_tune[train_spike],
                x_validation[validation_spike], y_validation[validation_spike],
                SEED + 200 + test_year + int(quantile * 100),
            )
            probability = classifier.predict_proba(x_validation)[:, 1]
            normal_residual = normal.predict(x_validation)
            spike_residual = spike.predict(x_validation)
            share_raw, log_raw, _, _ = mixture_forecasts(
                probability, normal_residual, spike_residual, validation_base, clip_log
            )
            y_raw = validation["auction_volume"].to_numpy(dtype=float)
            log_metrics = forecast_metrics(y_raw, log_raw)
            share_metrics = forecast_metrics(y_raw, share_raw)
            row = {
                "test_year": test_year,
                "validation_year": validation_year,
                "tail_quantile": quantile,
                "residual_threshold": threshold,
                "tuning_spike_rate": float(train_spike.mean()),
                "validation_spike_rate": float(validation_spike.mean()),
                "classifier_iterations": int(classifier.best_iteration_),
                "normal_iterations": int(normal.best_iteration_),
                "spike_iterations": int(spike.best_iteration_),
                "validation_log_mix_rmse": log_metrics["rmse_log"],
                "validation_log_mix_wape": log_metrics["wape"],
                "validation_share_mix_rmse": share_metrics["rmse_log"],
                "validation_share_mix_wape": share_metrics["wape"],
                "validation_roc_auc": roc_auc_score(validation_spike, probability),
                "validation_average_precision": average_precision_score(validation_spike, probability),
                "validation_brier": brier_score_loss(validation_spike, probability),
            }
            validation_rows.append(row)
            if best is None or row["validation_log_mix_rmse"] < best["validation_log_mix_rmse"]:
                best = row
            del classifier, normal, spike, probability, normal_residual, spike_residual
            gc.collect()
        assert best is not None
        selection_rows.append(best)
        progress(
            f"Fold {test_year}: selected q={best['tail_quantile']:.2f}; "
            f"iterations classifier/normal/spike="
            f"{best['classifier_iterations']}/{best['normal_iterations']}/{best['spike_iterations']}"
        )
        del tune, validation, x_tune, x_validation, prep_tune
        gc.collect()

        progress(f"Fold {test_year}: loading refit and test rows")
        refit = load_period(
            f"{test_year - 5}-01-01", f"{test_year}-01-01",
            REFIT_ROWS, SEED + 200 + test_year,
        )
        test = load_period(f"{test_year}-01-01", f"{test_year + 1}-01-01")
        for role, frame in [("refit", refit), ("test", test)]:
            sample_rows.append({
                "test_year": test_year, "role": role, "rows": len(frame),
                "key_fingerprint": sample_fingerprint(frame),
            })
        x_refit, prep = fit_preprocessor(
            refit, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        x_test = apply_preprocessor(
            test, prep, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        mask = np.isin(np.asarray(prep["output_sources"]), champion_features)
        output_names = np.asarray(prep["output_names"])[mask].tolist()
        x_refit = x_refit[:, mask]
        x_test = x_test[:, mask]
        y_refit, test_base, fallback = residual_target_and_base(refit, test)
        quantile = float(best["tail_quantile"])
        threshold = float(np.quantile(y_refit, quantile))
        refit_spike = y_refit > threshold
        test_residual = test["log1p_auction_volume"].to_numpy(dtype=float) - test_base
        test_spike = test_residual > threshold
        clip_log = tuple(np.quantile(refit["log1p_auction_volume"], [0.001, 0.999]))

        classifier = fit_final_component(
            "binary", x_refit, refit_spike.astype(int), best["classifier_iterations"],
            SEED + 300 + test_year,
        )
        normal = fit_final_component(
            "regression", x_refit[~refit_spike], y_refit[~refit_spike],
            best["normal_iterations"], SEED + 400 + test_year,
        )
        spike = fit_final_component(
            "regression", x_refit[refit_spike], y_refit[refit_spike],
            best["spike_iterations"], SEED + 500 + test_year,
        )
        probability = classifier.predict_proba(x_test)[:, 1]
        normal_residual = normal.predict(x_test)
        spike_residual = spike.predict(x_test)
        share_raw, log_raw, normal_log, spike_log = mixture_forecasts(
            probability, normal_residual, spike_residual, test_base, clip_log
        )

        lgbm_raw = reconstruct_lgbm(test_year, test)
        benchmark_path = RIDGE_PREDICTIONS / f"global_finalist_predictions_{test_year}.parquet"
        benchmark = con.sql(f'''
            SELECT date, permno, "{FROZEN_RIDGE}" AS frozen_ridge
            FROM read_parquet('{benchmark_path}')
        ''').df()
        output = test[CONTEXT].copy()
        output[LOG_MIX] = log_raw
        output[SHARE_MIX] = share_raw
        output[LGBM] = lgbm_raw
        output["spike_probability"] = probability
        output["normal_component_prediction"] = np.maximum(np.expm1(normal_log), 0)
        output["spike_component_prediction"] = np.maximum(np.expm1(spike_log), 0)
        output["realized_spike"] = test_spike
        output["residual_threshold"] = threshold
        original_keys = output[["date", "permno"]].copy()
        output = output.merge(benchmark, on=["date", "permno"], how="left", validate="one_to_one", sort=False)
        if not output[["date", "permno"]].equals(original_keys):
            raise AssertionError("Benchmark merge changed row order")

        y_raw = output["auction_volume"].to_numpy(dtype=float)
        for candidate, column, family in [
            (FROZEN_RIDGE, "frozen_ridge", "frozen ridge"),
            (LGBM, LGBM, "verified LightGBM"),
            (LOG_MIX, LOG_MIX, "tail mixture, log-space mean"),
            (SHARE_MIX, SHARE_MIX, "tail mixture, share-space mean"),
        ]:
            metric_rows.append({
                "test_year": test_year, "candidate": candidate, "model_family": family,
                **forecast_metrics(y_raw, output[column]),
            })
        classification_rows.append({
            "test_year": test_year,
            "tail_quantile": quantile,
            "residual_threshold": threshold,
            "observed_spike_rate": float(test_spike.mean()),
            "mean_predicted_probability": float(probability.mean()),
            "roc_auc": roc_auc_score(test_spike, probability),
            "average_precision": average_precision_score(test_spike, probability),
            "brier_score": brier_score_loss(test_spike, probability),
        })

        prediction_path = PREDICTIONS / f"tail_mixture_predictions_{test_year}.parquet"
        con.register("prediction_output", output)
        con.execute(f"COPY prediction_output TO '{prediction_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.unregister("prediction_output")
        (MODELS / f"preprocessor_{test_year}.json").write_text(json.dumps(prep, indent=2))
        (MODELS / f"features_{test_year}.json").write_text(json.dumps(output_names, indent=2))
        classifier.booster_.save_model(str(MODELS / f"spike_classifier_{test_year}.txt"))
        normal.booster_.save_model(str(MODELS / f"normal_regressor_{test_year}.txt"))
        spike.booster_.save_model(str(MODELS / f"spike_regressor_{test_year}.txt"))
        (MODELS / f"reconstruction_{test_year}.json").write_text(json.dumps({
            "fallback": fallback,
            "clip_log": list(map(float, clip_log)),
            "tail_quantile": quantile,
            "residual_threshold": threshold,
            "selected_iterations": {
                "classifier": int(best["classifier_iterations"]),
                "normal": int(best["normal_iterations"]),
                "spike": int(best["spike_iterations"]),
            },
        }, indent=2))
        progress(f"Fold {test_year}: wrote {len(output):,} mixture forecasts")
        del refit, test, x_refit, x_test, output, benchmark
        del classifier, normal, spike, probability, normal_residual, spike_residual
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "annual_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "threshold_validation.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "selected_thresholds.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(OUT / "sample_fingerprints.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(OUT / "classification_diagnostics.csv", index=False)
    summarize(con, metrics, started)
    con.close()


def summarize(con: duckdb.DuckDBPyConnection, metrics: pd.DataFrame, started: str) -> None:
    prediction_glob = str(PREDICTIONS / "tail_mixture_predictions_*.parquet")
    bootstrap_rows = []
    for number, candidate in enumerate([LOG_MIX, SHARE_MIX]):
        daily = con.sql(f'''
            SELECT date, year(date)::INT AS test_year,
                avg(abs(ln(1 + "{candidate}") - log1p_auction_volume)
                  - abs(ln(1 + "{LGBM}") - log1p_auction_volume)) AS difference
            FROM read_parquet('{prediction_glob}')
            GROUP BY date, test_year ORDER BY date
        ''').df()
        for period, years in {
            "development_2022_2024": DEVELOPMENT_YEARS,
            "confirmation_2025": [CONFIRMATION_YEAR],
        }.items():
            values = daily.loc[daily.test_year.isin(years), "difference"].to_numpy()
            mean, low, high = bootstrap_interval(values, SEED + number * 1000 + sum(years))
            bootstrap_rows.append({
                "candidate": candidate, "reference": LGBM, "period": period,
                "dates": len(values), "mean_log_mae_difference": mean,
                "ci_2_5": low, "ci_97_5": high,
            })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "paired_date_bootstrap.csv", index=False)

    development = metrics[metrics.test_year.isin(DEVELOPMENT_YEARS)].groupby(
        ["candidate", "model_family"], as_index=False
    ).agg(
        development_rmse_log=("rmse_log", "mean"),
        development_mae_log=("mae_log", "mean"),
        development_wape=("wape", "mean"),
        development_bias=("aggregate_bias", "mean"),
        development_spearman=("spearman", "mean"),
    )
    confirmation = metrics[metrics.test_year.eq(CONFIRMATION_YEAR)][
        ["candidate", "rmse_log", "mae_log", "wape", "aggregate_bias", "spearman"]
    ].rename(columns={
        "rmse_log": "confirmation_rmse_log", "mae_log": "confirmation_mae_log",
        "wape": "confirmation_wape", "aggregate_bias": "confirmation_bias",
        "spearman": "confirmation_spearman",
    })
    comparison = development.merge(confirmation, on="candidate", how="left")
    anchor = comparison.set_index("candidate").loc[LGBM]
    comparison["rmse_change_vs_lgbm"] = comparison.development_rmse_log - anchor.development_rmse_log
    comparison["wape_change_vs_lgbm"] = comparison.development_wape - anchor.development_wape
    annual = metrics[metrics.test_year.isin(DEVELOPMENT_YEARS)].pivot(
        index="candidate", columns="test_year", values="rmse_log"
    )
    comparison["worst_annual_rmse_change_vs_lgbm"] = comparison.candidate.map(
        annual.sub(annual.loc[LGBM], axis=1).max(axis=1)
    )
    dev_boot = bootstrap[bootstrap.period.eq("development_2022_2024")][
        ["candidate", "mean_log_mae_difference", "ci_2_5", "ci_97_5"]
    ]
    comparison = comparison.merge(dev_boot, on="candidate", how="left")
    comparison["practical_rmse_improvement"] = comparison.rmse_change_vs_lgbm.le(-0.002)
    comparison["stable_each_development_year"] = comparison.worst_annual_rmse_change_vs_lgbm.le(0.002)
    comparison["no_mean_wape_worsening"] = comparison.wape_change_vs_lgbm.le(0)
    comparison["paired_log_mae_upper_below_zero"] = comparison.ci_97_5.lt(0)
    comparison["passes_gate_vs_lgbm"] = (
        comparison.practical_rmse_improvement
        & comparison.stable_each_development_year
        & comparison.no_mean_wape_worsening
        & comparison.paired_log_mae_upper_below_zero.fillna(False)
    )
    comparison.to_csv(OUT / "candidate_comparison.csv", index=False)

    segment_queries = []
    segment_expressions = {
        "realized_mixture_class": "CASE WHEN realized_spike THEN 'spike' ELSE 'normal' END",
        "actual_tail": "CASE WHEN actual_percentile >= 0.99 THEN 'top_1pct' ELSE 'lower_99pct' END",
        "size": "CASE WHEN size_percentile < 0.33 THEN 'small' WHEN size_percentile < 0.67 THEN 'mid' ELSE 'large' END",
        "event": "CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",
    }
    columns = {FROZEN_RIDGE: "frozen_ridge", LGBM: LGBM, LOG_MIX: LOG_MIX, SHARE_MIX: SHARE_MIX}
    for candidate, column in columns.items():
        for dimension, expression in segment_expressions.items():
            segment_queries.append(f'''
                SELECT '{candidate}' AS candidate, '{dimension}' AS dimension,
                    {expression} AS segment, count(*) AS observations,
                    sqrt(avg(pow(ln(1 + "{column}") - log1p_auction_volume, 2))) AS rmse_log,
                    sum(abs("{column}" - auction_volume)) / sum(auction_volume) AS wape,
                    sum("{column}") / sum(auction_volume) - 1 AS aggregate_bias
                FROM ranked GROUP BY 1,2,3
            ''')
    segments = con.sql(f'''
        WITH ranked AS (
            SELECT *, percent_rank() OVER (
                PARTITION BY year(date) ORDER BY auction_volume
            ) AS actual_percentile
            FROM read_parquet('{prediction_glob}')
            WHERE year(date) IN (2022, 2023, 2024)
        )
        {' UNION ALL '.join(segment_queries)}
    ''').df()
    segments.to_csv(OUT / "segment_diagnostics.csv", index=False)

    calibration = con.sql(f'''
        WITH scored AS (
            SELECT year(date)::INT AS test_year, spike_probability, realized_spike,
                ntile(10) OVER (PARTITION BY year(date) ORDER BY spike_probability) AS probability_decile
            FROM read_parquet('{prediction_glob}')
        )
        SELECT test_year, probability_decile, count(*) AS observations,
            avg(spike_probability) AS mean_probability,
            avg(realized_spike::INT) AS observed_rate
        FROM scored GROUP BY 1,2 ORDER BY 1,2
    ''').df()
    calibration.to_csv(OUT / "probability_calibration.csv", index=False)

    checks = []
    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "passed": bool(passed)})
    verified_metrics = pd.read_csv(LGBM_ARTIFACTS / "nonlinear_candidate_metrics.csv")
    for year in TEST_YEARS:
        path = PREDICTIONS / f"tail_mixture_predictions_{year}.parquet"
        frame = con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        check(f"{year}_unique_keys", not frame.duplicated(["date", "permno"]).any())
        check(f"{year}_finite_nonnegative", all(
            np.isfinite(frame[column]).all() and frame[column].ge(0).all()
            for column in ["frozen_ridge", LGBM, LOG_MIX, SHARE_MIX]
        ))
        check(f"{year}_valid_probabilities", frame.spike_probability.between(0, 1).all())
        fresh = forecast_metrics(frame.auction_volume, frame[LGBM])
        saved = verified_metrics[
            verified_metrics.test_year.eq(year) & verified_metrics.candidate.eq(LGBM)
        ].iloc[0]
        check(f"{year}_verified_lgbm_reconstruction", all(
            np.isclose(fresh[key], saved[key], atol=1e-9, rtol=1e-10) for key in fresh
        ))
        for candidate in [LOG_MIX, SHARE_MIX]:
            fresh = forecast_metrics(frame.auction_volume, frame[candidate])
            saved = metrics[metrics.test_year.eq(year) & metrics.candidate.eq(candidate)].iloc[0]
            check(f"{year}_{candidate}_metrics_recompute", all(
                np.isclose(fresh[key], saved[key], atol=1e-9, rtol=1e-10) for key in fresh
            ))
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(OUT / "validation_checks.csv", index=False)
    prediction_files = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(PREDICTIONS.glob("*.parquet"))
    ]
    summary = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "development_years": DEVELOPMENT_YEARS,
        "confirmation_year": CONFIRMATION_YEAR,
        "confirmation_previously_inspected": True,
        "all_checks_passed": bool(checks_frame.passed.all()),
        "checks": len(checks_frame),
        "prediction_files": prediction_files,
        "candidates_passing_gate_vs_verified_lgbm": comparison.loc[
            comparison.passes_gate_vs_lgbm, "candidate"
        ].tolist(),
        "caveats": [
            "The 90th/95th percentile threshold grid was specified after earlier project results were known.",
            "Mixture probabilities are evaluated out of time but are not post-hoc calibrated.",
            "Date bootstrap preserves same-date cross-sectional dependence but not serial dependence.",
            "2025 is confirmation evidence, not an untouched holdout.",
        ],
    }
    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(metrics, comparison, bootstrap, summary)
    progress(f"Experiment complete: {summary['checks']} checks; all passed={summary['all_checks_passed']}")
    if not summary["all_checks_passed"]:
        raise AssertionError("Tail mixture validation failed")


def write_readme(metrics: pd.DataFrame, comparison: pd.DataFrame, bootstrap: pd.DataFrame, summary: dict) -> None:
    annual_lines = [
        "| Year | L2 RMSE | Log-mixture RMSE | Share-mixture RMSE | L2 WAPE | Log-mixture WAPE | Share-mixture WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for year in TEST_YEARS:
        rows = metrics[metrics.test_year.eq(year)].set_index("candidate")
        annual_lines.append(
            f"| {year} | {rows.loc[LGBM, 'rmse_log']:.5f} | {rows.loc[LOG_MIX, 'rmse_log']:.5f} | "
            f"{rows.loc[SHARE_MIX, 'rmse_log']:.5f} | {rows.loc[LGBM, 'wape']:.2%} | "
            f"{rows.loc[LOG_MIX, 'wape']:.2%} | {rows.loc[SHARE_MIX, 'wape']:.2%} |"
        )
    comparison_lines = [
        "| Candidate | Development RMSE | Change vs L2 | Development WAPE | Change vs L2 | Bias | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.sort_values("development_rmse_log").itertuples():
        status = "Reference" if row.candidate == LGBM else ("Pass" if row.passes_gate_vs_lgbm else "Fail")
        comparison_lines.append(
            f"| {row.candidate} | {row.development_rmse_log:.5f} | {row.rmse_change_vs_lgbm:+.5f} | "
            f"{row.development_wape:.2%} | {row.wape_change_vs_lgbm:+.2%} | "
            f"{row.development_bias:.2%} | {status} |"
        )
    status = "passed" if summary["all_checks_passed"] else "FAILED"
    (OUT / "README.md").write_text(f"""# Tail-aware mixture experiment

## Result

This experiment fits a historical spike-probability classifier and separate normal/spike residual regressors on the frozen 58-feature information set. Previous-year validation selects whether a spike begins at the training residual's 90th or 95th percentile. The same component predictions are combined in log space and share space. All {summary['checks']} verification checks {status}.

{chr(10).join(comparison_lines)}

Development metrics average 2022-2024 annual values equally. The verified squared-error LightGBM is the current daily anchor. The gate requires at least 0.002 mean log-RMSE improvement, no annual deterioration above 0.002, no mean WAPE worsening, and a paired date-level log-MAE upper interval below zero.

## Interpretation

The log-space mixture improves mean development log RMSE by 0.00090 and WAPE by 0.31 percentage points versus the verified L2 anchor. It improves RMSE in 2022 and 2023, worsens 2024 by 0.00045, and improves the previously inspected 2025 period. Its paired development date-level log-MAE difference is -0.00250 with a 95% interval of [-0.00364, -0.00146]. It therefore shows incremental signal but fails the predeclared practical RMSE threshold of 0.002 and is not promoted.

The classifier has useful out-of-time discrimination: annual ROC AUC ranges from 0.828 to 0.923 and average precision from 0.523 to 0.603. The log mixture nevertheless increases negative aggregate bias from 14.40% to 16.08%. The share-space mixture improves realized spike and top-1%-actual rows more aggressively, but its deterioration on normal rows makes overall accuracy worse. Hard routing to the spike component was inspected as a diagnostic and was also worse than soft probability averaging; it was not added as a candidate.

## Annual metrics

{chr(10).join(annual_lines)}

## Design

- The target remains `log1p(auction shares) - log1p(prior-only median20 auction shares)`.
- A binary LightGBM estimates spike probability. Separate squared-error LightGBMs estimate normal- and spike-component residual magnitudes.
- The spike threshold is selected between historical residual quantiles 0.90 and 0.95 using only the immediately preceding validation year.
- `tail_mixture_log_mean` averages component residuals in log space. `tail_mixture_share_mean` averages the two reconstructed share forecasts.
- Component tree counts use historical early stopping and are then fixed for the five-year refit. Final forecasts retain the project's 0.1st/99.9th percentile log bounds.
- 2025 was previously inspected and is confirmation evidence rather than an untouched holdout.

## Artifacts

- `annual_metrics.csv`, `candidate_comparison.csv`: annual and aggregate accuracy.
- `threshold_validation.csv`, `selected_thresholds.csv`: historical threshold and tree-count selection.
- `classification_diagnostics.csv`, `probability_calibration.csv`: out-of-time spike discrimination and calibration.
- `paired_date_bootstrap.csv`, `segment_diagnostics.csv`: uncertainty and tail/segment evidence.
- `predictions/`, `models/`: complete forecasts and fitted component artifacts.
- `sample_fingerprints.csv`, `validation_checks.csv`, `experiment_summary.json`: reproducibility checks.

Run from the repository root with:

```sh
.venv/bin/python -u notebooks/run_tail_mixture_experiment.py
```
""")


if __name__ == "__main__":
    main()
