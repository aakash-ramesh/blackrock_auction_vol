"""Leakage-safe spline residual challengers for closing-auction volume.

Candidates
----------
1. Pooled penalized cubic splines on the frozen benchmark's 58 source features.
2. The identical spline forecast plus a shrunk stock-specific mean residual.

The experiment preserves the existing five-year rolling folds, previous-year
validation, median20-relative log target, historical sampling sizes, and final
log-forecast clipping. 2022-2024 are development years and 2025 is previously
inspected confirmation evidence.
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
OUT = ROOT / "outputs/experiments/21_spline_residual"
PREDICTIONS = OUT / "predictions"
MODELS = OUT / "models"
for directory in [OUT, PREDICTIONS, MODELS]:
    directory.mkdir(parents=True, exist_ok=True)
for env, directory in [("MPLCONFIGDIR", ".matplotlib")]:
    os.environ.setdefault(env, str(OUT / directory))

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from notebooks.modeling_utils import (
    apply_preprocessor,
    fit_preprocessor,
    forecast_metrics,
)


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
SEED = 20261116

# Small, prespecified grids. Ridge alpha below is normalized by sample size so
# these values are comparable to the project's existing normalized penalties.
KNOT_GRID = [4, 6, 8]
RIDGE_STRENGTH_GRID = [1e-5, 1e-4, 1e-3]
STOCK_SHRINKAGE_GRID = [25.0, 100.0, 250.0]

POOLED = "spline_residual_58"
HIERARCHICAL = "spline_residual_58_stock_effect"
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
    keys = frame[["date", "permno"]].copy()
    values = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def bootstrap_interval(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        means[draw] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def residual_target_and_base(train: pd.DataFrame, score: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fallback = float(np.median(train["auction_volume"]))
    train_base = train["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    score_base = score["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    target = train["log1p_auction_volume"].to_numpy(dtype=float) - np.log1p(train_base)
    return target, np.log1p(score_base)


def compact_layout(prep: dict, champion_features: list[str], continuous_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    allowed = set(champion_features)
    continuous = set(continuous_features) & allowed
    sources = np.asarray(prep["output_sources"])
    names = np.asarray(prep["output_names"])
    compact = np.flatnonzero(np.isin(sources, list(allowed)))
    continuous_main = np.asarray([
        idx for idx in compact if sources[idx] in continuous and names[idx] == sources[idx]
    ], dtype=int)
    passthrough = np.asarray([idx for idx in compact if idx not in set(continuous_main)], dtype=int)
    return continuous_main, passthrough


def fit_design(
    matrix: np.ndarray,
    prep: dict,
    champion_features: list[str],
    continuous_features: list[str],
    n_knots: int,
) -> tuple[sparse.csr_matrix, dict]:
    continuous_idx, passthrough_idx = compact_layout(prep, champion_features, continuous_features)
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        knots="quantile",
        extrapolation="constant",
        include_bias=False,
        sparse_output=True,
    )
    spline_matrix = spline.fit_transform(matrix[:, continuous_idx])
    scaler = StandardScaler(with_mean=False)
    spline_matrix = scaler.fit_transform(spline_matrix)
    design = sparse.hstack(
        [spline_matrix, sparse.csr_matrix(matrix[:, passthrough_idx])],
        format="csr",
        dtype=np.float64,
    )
    bundle = {
        "spline": spline,
        "spline_scaler": scaler,
        "continuous_idx": continuous_idx,
        "passthrough_idx": passthrough_idx,
        "design_columns": int(design.shape[1]),
    }
    return design, bundle


def apply_design(matrix: np.ndarray, bundle: dict) -> sparse.csr_matrix:
    spline_matrix = bundle["spline"].transform(matrix[:, bundle["continuous_idx"]])
    spline_matrix = bundle["spline_scaler"].transform(spline_matrix)
    return sparse.hstack(
        [spline_matrix, sparse.csr_matrix(matrix[:, bundle["passthrough_idx"]])],
        format="csr",
        dtype=np.float64,
    )


def fit_spline_ridge(design: sparse.csr_matrix, target: np.ndarray, strength: float) -> Ridge:
    model = Ridge(
        alpha=float(strength * len(target)),
        fit_intercept=True,
        solver="lsqr",
        tol=1e-6,
        max_iter=5000,
    )
    model.fit(design, target)
    return model


def stock_effect_table(
    permno: pd.Series,
    target: np.ndarray,
    pooled_prediction: np.ndarray,
) -> pd.DataFrame:
    residual = target - pooled_prediction
    frame = pd.DataFrame({"permno": permno.to_numpy(), "residual": residual})
    return frame.groupby("permno", as_index=False).agg(
        stock_residual_mean=("residual", "mean"),
        stock_residual_observations=("residual", "size"),
    )


def apply_stock_effect(frame: pd.DataFrame, effects: pd.DataFrame, shrinkage: float) -> np.ndarray:
    aligned = frame[["permno"]].merge(effects, on="permno", how="left", sort=False)
    mean = aligned["stock_residual_mean"].fillna(0).to_numpy(dtype=float)
    count = aligned["stock_residual_observations"].fillna(0).to_numpy(dtype=float)
    return mean * count / (count + shrinkage)


def reconstruct_lgbm(test_year: int, test: pd.DataFrame, x_test: np.ndarray) -> np.ndarray:
    prep = json.loads((LGBM_ARTIFACTS / f"models/preprocessor_{test_year}.json").read_text())
    # x_test belongs to the spline refit preprocessor, so rebuild the verified
    # LightGBM matrix with its own saved fold preprocessor.
    verified_x = apply_preprocessor(
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
    prediction_log = np.clip(base + booster.predict(verified_x[:, indices]), *reconstruction["clip_log"])
    del verified_x
    return np.maximum(np.expm1(prediction_log), 0)


def main() -> None:
    if not PANEL.exists():
        raise FileNotFoundError(PANEL)
    started = datetime.now(timezone.utc).isoformat()
    if (OUT / "progress.log").exists():
        (OUT / "progress.log").unlink()
    progress("Spline residual experiment started")

    frozen = json.loads(FROZEN.read_text())
    champion_features = frozen["features"]
    catalog = pd.read_csv(CATALOG)
    features = catalog["variable"].tolist()
    continuous_features = [
        feature for feature in features
        if feature not in BOOLEAN_FEATURES and feature not in CATEGORICAL_FEATURES
    ]
    assert len(champion_features) == 58
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

    checkpoint_paths = {
        "metrics": OUT / "_checkpoint_metrics.csv",
        "validation": OUT / "_checkpoint_validation.csv",
        "samples": OUT / "_checkpoint_samples.csv",
        "selection": OUT / "_checkpoint_selection.csv",
    }
    resume = "--resume" in sys.argv
    if resume and checkpoint_paths["metrics"].exists():
        metric_rows = pd.read_csv(checkpoint_paths["metrics"]).to_dict("records")
        validation_rows = pd.read_csv(checkpoint_paths["validation"]).to_dict("records")
        sample_rows = pd.read_csv(checkpoint_paths["samples"]).to_dict("records")
        selection_rows = pd.read_csv(checkpoint_paths["selection"]).to_dict("records")
        completed_years = set(pd.DataFrame(metric_rows).test_year.astype(int))
        progress(f"Resuming after completed folds {sorted(completed_years)}")
    else:
        metric_rows: list[dict] = []
        validation_rows: list[dict] = []
        sample_rows: list[dict] = []
        selection_rows: list[dict] = []
        completed_years = set()
        for path in checkpoint_paths.values():
            if path.exists():
                path.unlink()

    for test_year in TEST_YEARS:
        if test_year in completed_years:
            continue
        validation_year = test_year - 1
        progress(f"Fold {test_year}: loading tuning and validation rows")
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
                "test_year": test_year,
                "role": role,
                "rows": len(frame),
                "key_fingerprint": sample_fingerprint(frame),
            })

        x_tune, prep_tune = fit_preprocessor(
            tune, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        x_validation = apply_preprocessor(
            validation, prep_tune, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        y_tune, validation_base = residual_target_and_base(tune, validation)
        y_validation_log = validation["log1p_auction_volume"].to_numpy(dtype=float)

        best: dict | None = None
        best_validation_model = None
        best_validation_bundle = None
        best_tune_design = None
        best_validation_design = None
        for n_knots in KNOT_GRID:
            progress(f"Fold {test_year}: fitting {n_knots}-knot spline design")
            z_tune, bundle = fit_design(
                x_tune, prep_tune, champion_features, continuous_features, n_knots
            )
            z_validation = apply_design(x_validation, bundle)
            for strength in RIDGE_STRENGTH_GRID:
                model = fit_spline_ridge(z_tune, y_tune, strength)
                prediction_log = validation_base + model.predict(z_validation)
                lo, hi = np.quantile(tune["log1p_auction_volume"], [0.001, 0.999])
                prediction_log = np.clip(prediction_log, lo, hi)
                rmse = float(np.sqrt(np.mean((y_validation_log - prediction_log) ** 2)))
                row = {
                    "test_year": test_year,
                    "validation_year": validation_year,
                    "candidate": POOLED,
                    "n_knots": n_knots,
                    "ridge_strength": strength,
                    "stock_shrinkage": np.nan,
                    "design_columns": z_tune.shape[1],
                    "validation_rmse_log": rmse,
                }
                validation_rows.append(row)
                if best is None or rmse < best["validation_rmse_log"]:
                    best = row
                    best_validation_model = model
                    best_validation_bundle = bundle
                    best_tune_design = z_tune
                    best_validation_design = z_validation
            if best is None or best["n_knots"] != n_knots:
                del z_tune, z_validation
            gc.collect()

        assert best is not None
        pooled_tune = best_validation_model.predict(best_tune_design)
        pooled_validation = best_validation_model.predict(best_validation_design)
        tune_effects = stock_effect_table(tune["permno"], y_tune, pooled_tune)
        best_stock: dict | None = None
        for shrinkage in STOCK_SHRINKAGE_GRID:
            effect = apply_stock_effect(validation, tune_effects, shrinkage)
            prediction_log = validation_base + pooled_validation + effect
            lo, hi = np.quantile(tune["log1p_auction_volume"], [0.001, 0.999])
            prediction_log = np.clip(prediction_log, lo, hi)
            rmse = float(np.sqrt(np.mean((y_validation_log - prediction_log) ** 2)))
            row = {
                "test_year": test_year,
                "validation_year": validation_year,
                "candidate": HIERARCHICAL,
                "n_knots": best["n_knots"],
                "ridge_strength": best["ridge_strength"],
                "stock_shrinkage": shrinkage,
                "design_columns": best["design_columns"],
                "validation_rmse_log": rmse,
            }
            validation_rows.append(row)
            if best_stock is None or rmse < best_stock["validation_rmse_log"]:
                best_stock = row
        assert best_stock is not None
        selection_rows.extend([best, best_stock])
        progress(
            f"Fold {test_year}: selected knots={best['n_knots']}, strength={best['ridge_strength']}, "
            f"stock shrinkage={best_stock['stock_shrinkage']}"
        )

        del tune, validation, x_tune, x_validation, best_validation_model
        del best_validation_bundle, best_tune_design, best_validation_design, tune_effects
        del model, pooled_tune, pooled_validation
        gc.collect()

        progress(f"Fold {test_year}: loading refit and test rows")
        refit = load_period(
            f"{test_year - 5}-01-01", f"{test_year}-01-01",
            REFIT_ROWS, SEED + 200 + test_year,
        )
        test = load_period(f"{test_year}-01-01", f"{test_year + 1}-01-01")
        for role, frame in [("refit", refit), ("test", test)]:
            sample_rows.append({
                "test_year": test_year,
                "role": role,
                "rows": len(frame),
                "key_fingerprint": sample_fingerprint(frame),
            })

        x_refit, prep = fit_preprocessor(
            refit, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        x_test = apply_preprocessor(
            test, prep, continuous_features, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        z_refit, bundle = fit_design(
            x_refit, prep, champion_features, continuous_features, int(best["n_knots"])
        )
        z_test = apply_design(x_test, bundle)
        y_refit, test_base = residual_target_and_base(refit, test)
        model = fit_spline_ridge(z_refit, y_refit, float(best["ridge_strength"]))
        pooled_refit = model.predict(z_refit)
        pooled_test = model.predict(z_test)
        effects = stock_effect_table(refit["permno"], y_refit, pooled_refit)
        stock_test = apply_stock_effect(test, effects, float(best_stock["stock_shrinkage"]))
        lo, hi = np.quantile(refit["log1p_auction_volume"], [0.001, 0.999])
        pooled_log = np.clip(test_base + pooled_test, lo, hi)
        hierarchical_log = np.clip(test_base + pooled_test + stock_test, lo, hi)
        pooled_raw = np.maximum(np.expm1(pooled_log), 0)
        hierarchical_raw = np.maximum(np.expm1(hierarchical_log), 0)

        progress(f"Fold {test_year}: reconstructing verified anchors")
        lgbm_raw = reconstruct_lgbm(test_year, test, x_test)
        benchmark_path = RIDGE_PREDICTIONS / f"global_finalist_predictions_{test_year}.parquet"
        benchmark = con.sql(f'''
            SELECT date, permno, "{FROZEN_RIDGE}" AS frozen_ridge
            FROM read_parquet('{benchmark_path}')
        ''').df()

        output = test[CONTEXT].copy()
        output[POOLED] = pooled_raw
        output[HIERARCHICAL] = hierarchical_raw
        output[LGBM] = lgbm_raw
        original_keys = output[["date", "permno"]].copy()
        output = output.merge(benchmark, on=["date", "permno"], how="left", validate="one_to_one", sort=False)
        if not output[["date", "permno"]].equals(original_keys):
            raise AssertionError("Benchmark merge changed row order")
        if output["frozen_ridge"].isna().any():
            raise AssertionError("Frozen ridge did not align")

        y_raw = output["auction_volume"].to_numpy(dtype=float)
        for candidate, column, family in [
            (FROZEN_RIDGE, "frozen_ridge", "frozen ridge"),
            (LGBM, LGBM, "verified LightGBM"),
            (POOLED, POOLED, "penalized spline"),
            (HIERARCHICAL, HIERARCHICAL, "penalized spline plus shrunk stock effect"),
        ]:
            metric_rows.append({
                "test_year": test_year,
                "candidate": candidate,
                "model_family": family,
                **forecast_metrics(y_raw, output[column]),
            })

        prediction_path = PREDICTIONS / f"spline_predictions_{test_year}.parquet"
        con.register("prediction_output", output)
        con.execute(f"COPY prediction_output TO '{prediction_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.unregister("prediction_output")
        prep_path = MODELS / f"preprocessor_{test_year}.json"
        prep_path.write_text(json.dumps(prep, indent=2))
        joblib.dump(
            {"model": model, "design": bundle},
            MODELS / f"spline_model_{test_year}.joblib",
            compress=3,
        )
        effects.to_csv(MODELS / f"stock_effects_{test_year}.csv", index=False)
        (MODELS / f"reconstruction_{test_year}.json").write_text(json.dumps({
            "fallback": float(np.median(refit["auction_volume"])),
            "clip_log": [float(lo), float(hi)],
            "pooled_selection": best,
            "stock_selection": best_stock,
        }, indent=2))
        progress(f"Fold {test_year}: wrote {len(output):,} forecasts")
        pd.DataFrame(metric_rows).to_csv(checkpoint_paths["metrics"], index=False)
        pd.DataFrame(validation_rows).to_csv(checkpoint_paths["validation"], index=False)
        pd.DataFrame(sample_rows).to_csv(checkpoint_paths["samples"], index=False)
        pd.DataFrame(selection_rows).to_csv(checkpoint_paths["selection"], index=False)

        del refit, test, x_refit, x_test, z_refit, z_test, output, benchmark
        del model, bundle, effects, pooled_refit, pooled_test, stock_test
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    validation_results = pd.DataFrame(validation_rows)
    selections = pd.DataFrame(selection_rows)
    sample_fingerprints = pd.DataFrame(sample_rows)
    metrics.to_csv(OUT / "annual_metrics.csv", index=False)
    validation_results.to_csv(OUT / "hyperparameter_validation.csv", index=False)
    selections.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    sample_fingerprints.to_csv(OUT / "sample_fingerprints.csv", index=False)

    summarize(con, metrics, started)
    con.close()


def summarize(con: duckdb.DuckDBPyConnection, metrics: pd.DataFrame, started: str) -> None:
    prediction_glob = str(PREDICTIONS / "spline_predictions_*.parquet")
    candidates = [POOLED, HIERARCHICAL]
    references = [LGBM, POOLED]
    bootstrap_rows = []
    for candidate_number, candidate in enumerate(candidates):
        for reference in references:
            if candidate == POOLED and reference == POOLED:
                continue
            reference_column = reference
            daily = con.sql(f'''
                SELECT date, year(date)::INT AS test_year,
                    avg(abs(ln(1 + "{candidate}") - log1p_auction_volume)
                      - abs(ln(1 + "{reference_column}") - log1p_auction_volume)) AS difference
                FROM read_parquet('{prediction_glob}')
                GROUP BY date, test_year ORDER BY date
            ''').df()
            for period, years in {
                "development_2022_2024": DEVELOPMENT_YEARS,
                "confirmation_2025": [CONFIRMATION_YEAR],
            }.items():
                values = daily.loc[daily.test_year.isin(years), "difference"].to_numpy()
                mean, low, high = bootstrap_interval(
                    values, SEED + 1000 * candidate_number + len(reference) + sum(years)
                )
                bootstrap_rows.append({
                    "candidate": candidate,
                    "reference": reference,
                    "period": period,
                    "dates": len(values),
                    "mean_log_mae_difference": mean,
                    "ci_2_5": low,
                    "ci_97_5": high,
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
        "rmse_log": "confirmation_rmse_log",
        "mae_log": "confirmation_mae_log",
        "wape": "confirmation_wape",
        "aggregate_bias": "confirmation_bias",
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
    dev_boot = bootstrap[
        bootstrap.period.eq("development_2022_2024") & bootstrap.reference.eq(LGBM)
    ][["candidate", "mean_log_mae_difference", "ci_2_5", "ci_97_5"]]
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
        "actual_tail": "CASE WHEN actual_percentile >= 0.99 THEN 'top_1pct' ELSE 'lower_99pct' END",
        "size": "CASE WHEN size_percentile < 0.33 THEN 'small' WHEN size_percentile < 0.67 THEN 'mid' ELSE 'large' END",
        "history": "CASE WHEN prior_auction_observations < 5 THEN 'cold_0_4' WHEN prior_auction_observations < 20 THEN 'developing_5_19' ELSE 'seasoned_20_plus' END",
        "event": "CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",
    }
    columns = {
        FROZEN_RIDGE: "frozen_ridge", LGBM: LGBM,
        POOLED: POOLED, HIERARCHICAL: HIERARCHICAL,
    }
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

    checks = []
    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "passed": bool(passed)})

    verified_metrics = pd.read_csv(LGBM_ARTIFACTS / "nonlinear_candidate_metrics.csv")
    for year in TEST_YEARS:
        path = PREDICTIONS / f"spline_predictions_{year}.parquet"
        frame = con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        check(f"{year}_unique_keys", not frame.duplicated(["date", "permno"]).any())
        check(f"{year}_finite_nonnegative", all(
            np.isfinite(frame[column]).all() and frame[column].ge(0).all()
            for column in ["frozen_ridge", LGBM, POOLED, HIERARCHICAL]
        ))
        fresh = forecast_metrics(frame.auction_volume, frame[LGBM])
        saved = verified_metrics[
            verified_metrics.test_year.eq(year) & verified_metrics.candidate.eq(LGBM)
        ].iloc[0]
        check(f"{year}_verified_lgbm_reconstruction", all(
            np.isclose(fresh[key], saved[key], atol=1e-9, rtol=1e-10) for key in fresh
        ))
        for candidate in [POOLED, HIERARCHICAL]:
            fresh = forecast_metrics(frame.auction_volume, frame[candidate])
            saved = metrics[
                metrics.test_year.eq(year) & metrics.candidate.eq(candidate)
            ].iloc[0]
            check(f"{year}_{candidate}_metrics_recompute", all(
                np.isclose(fresh[key], saved[key], atol=1e-9, rtol=1e-10) for key in fresh
            ))
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(OUT / "validation_checks.csv", index=False)

    prediction_files = []
    for path in sorted(PREDICTIONS.glob("*.parquet")):
        prediction_files.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
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
            "Hyperparameter grids were specified for this experiment after earlier project results were known.",
            "Date bootstrap preserves same-date cross-sectional dependence but not serial dependence.",
            "The stock effect uses sampled five-year fitting rows and falls back to zero for unseen stocks.",
            "2025 is confirmation evidence, not an untouched holdout.",
        ],
    }
    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(metrics, comparison, bootstrap, summary)
    progress(
        f"Experiment complete: {summary['checks']} checks; all passed={summary['all_checks_passed']}"
    )
    if not summary["all_checks_passed"]:
        raise AssertionError("Spline experiment validation failed")


def write_readme(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    summary: dict,
) -> None:
    annual_lines = [
        "| Year | Verified L2 RMSE | Pooled spline RMSE | Spline + stock RMSE | Verified L2 WAPE | Pooled spline WAPE | Spline + stock WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for year in TEST_YEARS:
        year_metrics = metrics[metrics.test_year.eq(year)].set_index("candidate")
        annual_lines.append(
            f"| {year} | {year_metrics.loc[LGBM, 'rmse_log']:.5f} | "
            f"{year_metrics.loc[POOLED, 'rmse_log']:.5f} | {year_metrics.loc[HIERARCHICAL, 'rmse_log']:.5f} | "
            f"{year_metrics.loc[LGBM, 'wape']:.2%} | {year_metrics.loc[POOLED, 'wape']:.2%} | "
            f"{year_metrics.loc[HIERARCHICAL, 'wape']:.2%} |"
        )
    comparison_lines = [
        "| Candidate | Development RMSE | Change vs L2 | Development WAPE | Change vs L2 | Bias | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.sort_values("development_rmse_log").itertuples():
        gate_status = "Reference" if row.candidate == LGBM else ("Pass" if row.passes_gate_vs_lgbm else "Fail")
        comparison_lines.append(
            f"| {row.candidate} | {row.development_rmse_log:.5f} | {row.rmse_change_vs_lgbm:+.5f} | "
            f"{row.development_wape:.2%} | {row.wape_change_vs_lgbm:+.2%} | "
            f"{row.development_bias:.2%} | {gate_status} |"
        )
    stock_boot = bootstrap[
        bootstrap.candidate.eq(HIERARCHICAL)
        & bootstrap.reference.eq(POOLED)
        & bootstrap.period.eq("development_2022_2024")
    ].iloc[0]
    status = "passed" if summary["all_checks_passed"] else "FAILED"
    (OUT / "README.md").write_text(f"""# Spline residual challengers

## Result

This bounded experiment compares a pooled penalized cubic-spline residual model with the identical model plus a shrunk stock-specific residual effect. It uses the frozen 58-feature input set, five-year rolling history, previous-year validation, median20-relative log target, historical sample sizes, and the existing forecast clipping rule. All {summary['checks']} verification checks {status}.

{chr(10).join(comparison_lines)}

Development metrics average 2022-2024 annual values equally. The verified squared-error LightGBM is the current daily anchor. The gate requires at least 0.002 mean log-RMSE improvement, no annual deterioration above 0.002, no mean WAPE worsening, and a paired date-level log-MAE upper interval below zero.

## Annual metrics

{chr(10).join(annual_lines)}

## Does the stock effect help the spline?

For development dates, the hierarchical candidate minus pooled-spline mean date-level log-MAE difference is {stock_boot.mean_log_mae_difference:+.6f}, with a 95% whole-date bootstrap interval of [{stock_boot.ci_2_5:+.6f}, {stock_boot.ci_97_5:+.6f}]. Negative favors the stock effect. This interval preserves same-date cross-sectional dependence but not serial dependence across dates.

## Design

- Continuous inputs use cubic B-splines with quantile-spaced knots. Binary variables, missing indicators, and encoded categories pass through linearly.
- Knot counts 4, 6, and 8 and normalized ridge strengths 1e-5, 1e-4, and 1e-3 are selected on the immediately preceding validation year.
- The hierarchical candidate uses the selected pooled spline unchanged. A stock's fitting-sample mean residual is multiplied by `n / (n + k)`, with `k` selected from 25, 100, and 250 on the same historical validation year. Unseen stocks receive zero correction.
- The final model refits on the same 350,000-row five-year sample size used by the nonlinear challenger experiment and scores every eligible row.
- 2025 was previously inspected and is confirmation evidence rather than an untouched holdout.

## Artifacts

- `annual_metrics.csv`, `candidate_comparison.csv`: annual and aggregate comparisons.
- `hyperparameter_validation.csv`, `selected_hyperparameters.csv`: complete bounded search and fold selections.
- `paired_date_bootstrap.csv`, `segment_diagnostics.csv`: uncertainty and segment evidence.
- `predictions/`: complete row-level forecasts for both candidates and reconstructed verified anchors.
- `models/`: fitted spline/ridge models, preprocessing, reconstruction settings, and stock effects.
- `sample_fingerprints.csv`, `validation_checks.csv`, `experiment_summary.json`: chronology and reproducibility checks.

Run from the repository root with:

```sh
.venv/bin/python -u notebooks/run_spline_residual_experiment.py
```
""")


if __name__ == "__main__":
    main()
