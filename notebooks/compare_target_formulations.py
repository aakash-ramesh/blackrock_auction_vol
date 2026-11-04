"""Controlled six-model target experiment; run with the project's .venv Python.

On macOS, set DYLD_LIBRARY_PATH to sklearn/.dylibs as in the preceding reproduction.
No original research artifact is overwritten. Samples are reused from step 03.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/experiments/17_target_formulations"
PREVIOUS = ROOT / "outputs/reproduction/03_compact_nonlinear"
PANEL = ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from notebooks.modeling_utils import fit_preprocessor, apply_preprocessor, fit_ridge, predict_ridge, forecast_metrics

SEED = 20261018
TARGETS = ["log_level", "lag1_log_growth", "median20_log_residual"]
FAMILIES = ["ridge", "lgbm"]
RIDGE_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LEAF_GRID = [15, 31, 63]
YEARS = [2022, 2023, 2024, 2025]
CONTEXT = ["date", "permno", "auction_volume", "log1p_auction_volume", "median20_auction_volume",
           "lag1_auction_volume", "size_percentile", "liquidity_percentile", "prior_auction_observations",
           "quarterly_options_expiry", "monthly_options_expiry", "is_last_trading_day_month", "is_early_close"]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def progress(message):
    line = datetime.now(timezone.utc).isoformat() + " " + message
    print(line, flush=True)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(line + "\n")


def offset(frame, target, fallback):
    if target == "log_level":
        return np.zeros(len(frame), dtype=float)
    name = "lag1_auction_volume" if target == "lag1_log_growth" else "median20_auction_volume"
    return np.log1p(frame[name].astype(float).fillna(fallback).to_numpy(dtype=float))


def parameters(year, leaves, iterations, tuning):
    return {"objective": "regression", "n_estimators": iterations, "learning_rate": 0.03,
            "num_leaves": leaves, "min_child_samples": 500, "max_bin": 127,
            "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 5.0,
            "random_state": SEED + year + leaves if tuning else SEED + 300 + year,
            "n_jobs": 4, "deterministic": True, "force_col_wise": True, "verbosity": -1}


def prepare():
    for directory in [OUT, OUT / "predictions", OUT / "models"]:
        directory.mkdir(parents=True, exist_ok=True)
    previous_summary = json.loads((PREVIOUS / "reproduction_summary.json").read_text())
    assert previous_summary["all_checks_passed"]
    inventory = pd.read_csv(PREVIOUS / "training_sample_fingerprints.csv")
    for r in inventory.itertuples():
        assert sha(ROOT / r.path) == r.sha256, f"Sample changed: {r.path}"
    for r in previous_summary["predictions"]:
        assert sha(ROOT / r["path"]) == r["sha256"]
    original_inputs = pd.read_csv(ROOT / "outputs/reproduction/02_benchmark_reconciliation/input_fingerprints.csv")
    for r in original_inputs.itertuples():
        if r.path.endswith("enriched_daily_panel.parquet") or r.path.endswith("modeling_utils.py"):
            assert sha(ROOT / r.path) == r.sha256
    plan = {"targets": {
        "log_level": "log1p(A_t)",
        "lag1_log_growth": "log1p(A_t) - log1p(A_previous_stock_row); missing reference uses historical training median",
        "median20_log_residual": "log1p(A_t) - log1p(median of available auctions in prior 20 stock rows); missing reference uses historical training median"},
        "families": FAMILIES, "features": "Same 58 sources and shared preprocessing as step 03",
        "samples": "Exact saved tuning, validation and refit rows in their saved order from step 03",
        "development_years": [2022, 2023, 2024], "confirmation_year": 2025,
        "confirmation_previously_inspected": True, "ridge_penalties": RIDGE_GRID,
        "lgbm_leaves": LEAF_GRID, "lgbm_max_trees": 2000, "lgbm_early_stop_patience": 100,
        "primary_metric": "equal-year mean development log RMSE",
        "additional_metrics": ["WAPE", "raw MAE", "log MAE", "Spearman", "aggregate bias", "segment and tail errors"],
        "target_change_gate": "vs own-family median target: development RMSE improvement >=0.002, no annual worsening >0.002, paired development date-bootstrap log-MAE upper CI <0",
        "forecast_clipping": "common training log-target quantiles 0.001 and 0.999",
        "excluded_changes": ["new features", "Huber search", "blends", "calibration", "anomaly-history policy", "new clipping policy"],
        "median_lgbm_reuse": "Validated prior squared-error model, validation grid and test forecasts; other five combinations fitted on identical historical rows",
        "script_sha256": sha(Path(__file__)), "created_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "experiment_plan.json").write_text(json.dumps(plan, indent=2))
    inventory.to_csv(OUT / "shared_sample_fingerprints.csv", index=False)
    (OUT / "environment.json").write_text(json.dumps({n: importlib.metadata.version(n) for n in
        ["numpy", "pandas", "duckdb", "lightgbm", "scikit-learn"]}, indent=2))
    return plan


def run():
    plan = prepare()
    con = duckdb.connect()
    con.execute("SET threads=4")
    validation_rows, metric_rows, records = [], [], []
    selected_features = json.loads((ROOT / "outputs/eda/09_global_finalists/frozen_global_benchmark.json").read_text())["features"]
    historical_validation = pd.read_csv(PREVIOUS / "nonlinear_hyperparameter_validation.csv")
    progress("Plan frozen; beginning six-combination experiment with shared samples")
    for year in YEARS:
        progress(f"Fold {year}: loading identical saved historical samples")
        read = lambda name: con.sql(f"SELECT * FROM read_parquet('{PREVIOUS / 'samples' / (str(year) + '_' + name + '.parquet')}')").df()
        train, validation, refit = read("tune"), read("validation"), read("refit")
        test = con.sql(f"SELECT * FROM read_parquet('{PANEL}') WHERE eligible_target AND year={year} ORDER BY date,permno").df()
        prep_final = json.loads((PREVIOUS / f"models/preprocessor_{year}.json").read_text())
        types = [list(prep_final[k]) for k in ["continuous", "boolean", "categorical"]]
        xt, prep_tune = fit_preprocessor(train, *types)
        xv = apply_preprocessor(validation, prep_tune, *types)
        xr = apply_preprocessor(refit, prep_final, *types)
        xe = apply_preprocessor(test, prep_final, *types)
        mask_tune = np.array([f in set(selected_features) for f in prep_tune["output_sources"]])
        mask_final = np.array([f in set(selected_features) for f in prep_final["output_sources"]])
        xt, xv, xr, xe = xt[:, mask_tune], xv[:, mask_tune], xr[:, mask_final], xe[:, mask_final]
        (OUT / f"models/preprocessor_{year}.json").write_text(json.dumps(prep_final, indent=2))
        (OUT / f"models/tuning_preprocessor_{year}.json").write_text(json.dumps(prep_tune, indent=2))
        (OUT / f"models/feature_names_{year}.json").write_text(json.dumps(np.array(prep_final["output_names"])[mask_final].tolist()))
        fallback_tune = float(np.median(train.auction_volume))
        fallback_final = float(np.median(refit.auction_volume))
        clip_tune = np.quantile(train.log1p_auction_volume, [0.001, 0.999])
        clip_final = np.quantile(refit.log1p_auction_volume, [0.001, 0.999])
        old = con.sql(f"SELECT * FROM read_parquet('{PREVIOUS / 'predictions' / f'nonlinear_challenger_predictions_{year}.parquet'}')").df()
        output = test[CONTEXT].merge(old[["date", "permno", "frozen_prediction", "lgbm_residual_58_l2"]], on=["date", "permno"], validate="one_to_one", sort=False)
        assert output[["date", "permno"]].equals(test[["date", "permno"]])
        output = output.rename(columns={"frozen_prediction": "historical_frozen_ridge", "lgbm_residual_58_l2": "lgbm__median20_log_residual"})
        selected = {}
        for target in TARGETS:
            zt = train.log1p_auction_volume.to_numpy() - offset(train, target, fallback_tune)
            base_validation = offset(validation, target, fallback_tune)
            zv = validation.log1p_auction_volume.to_numpy() - base_validation
            for family in FAMILIES:
                candidate = family + "__" + target
                progress(f"Fold {year}: tuning {candidate}")
                choices = []
                if family == "lgbm" and target == "median20_log_residual":
                    cached = historical_validation[(historical_validation.test_year == year) & historical_validation.candidate.eq("lgbm_residual_58_l2")]
                    for r in cached.itertuples():
                        choices.append({"test_year": year, "validation_year": year-1, "candidate": candidate,
                                        "parameter": int(r.num_leaves), "iterations": int(r.best_iteration),
                                        "validation_rmse_log": float(r.validation_rmse_log), "reused": True})
                else:
                    for parameter in RIDGE_GRID if family == "ridge" else LEAF_GRID:
                        if family == "ridge":
                            coef = fit_ridge(xt, zt, parameter)
                            component = predict_ridge(xv, coef)
                            iterations = 0
                        else:
                            progress(f"Fold {year}: {candidate}, leaves={parameter}")
                            model = lgb.LGBMRegressor(**parameters(year, parameter, 2000, True))
                            model.fit(xt, zt, eval_set=[(xv, zv)], eval_metric="rmse",
                                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
                            component = model.predict(xv)
                            iterations = int(model.best_iteration_)
                        pred_log = np.clip(base_validation + component, *clip_tune)
                        choices.append({"test_year": year, "validation_year": year-1, "candidate": candidate,
                                        "parameter": parameter, "iterations": iterations,
                                        "validation_rmse_log": float(np.sqrt(np.mean((validation.log1p_auction_volume.to_numpy()-pred_log)**2))), "reused": False})
                validation_rows.extend(choices)
                best = min(choices, key=lambda r: r["validation_rmse_log"])
                selected[candidate] = best
                progress(f"Fold {year}: selected {candidate}: parameter={best['parameter']}, iterations={best['iterations']}")
        for target in TARGETS:
            z = refit.log1p_auction_volume.to_numpy() - offset(refit, target, fallback_final)
            base = offset(test, target, fallback_final)
            for family in FAMILIES:
                candidate = family + "__" + target
                best = selected[candidate]
                if candidate == "lgbm__median20_log_residual":
                    progress(f"Fold {year}: reusing verified median-relative LightGBM forecasts")
                else:
                    progress(f"Fold {year}: final fit {candidate}")
                    if family == "ridge":
                        coef = fit_ridge(xr, z, best["parameter"])
                        np.save(OUT / f"models/{candidate}_{year}.npy", coef)
                        component = predict_ridge(xe, coef)
                    else:
                        model = lgb.LGBMRegressor(**parameters(year, int(best["parameter"]), max(25,best["iterations"]), False))
                        model.fit(xr, z)
                        model.booster_.save_model(str(OUT / f"models/{candidate}_{year}.txt"))
                        component = model.predict(xe)
                    output[candidate] = np.maximum(np.expm1(np.clip(base + component, *clip_final)), 0)
                metric_rows.append({"test_year": year, "candidate": candidate, "family": family,
                                    "target": target, **forecast_metrics(output.auction_volume, output[candidate])})
                records.append({"test_year": year, "candidate": candidate,
                                "missing_tuning_reference": int(train["lag1_auction_volume" if target == "lag1_log_growth" else "median20_auction_volume"].isna().sum()) if target != "log_level" else 0,
                                "missing_test_reference": int(test["lag1_auction_volume" if target == "lag1_log_growth" else "median20_auction_volume"].isna().sum()) if target != "log_level" else 0})
        metric_rows.append({"test_year": year, "candidate": "historical_frozen_ridge", "family": "historical_reference", "target": "median20_log_residual", **forecast_metrics(output.auction_volume, output.historical_frozen_ridge)})
        (OUT / f"models/reconstruction_{year}.json").write_text(json.dumps({"fallback": fallback_final, "clip_log": clip_final.tolist(), "selected": selected}, indent=2))
        path = OUT / f"predictions/target_predictions_{year}.parquet"
        con.register("prediction_output", output)
        con.execute(f"COPY prediction_output TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.unregister("prediction_output")
        pd.DataFrame(metric_rows).to_csv(OUT / "annual_metrics.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(OUT / "hyperparameter_validation.csv", index=False)
        pd.DataFrame(records).to_csv(OUT / "reference_missingness.csv", index=False)
        progress(f"Fold {year}: saved {len(output):,} common evaluation rows")
        del train, validation, refit, test, xt, xv, xr, xe, output, old
        import gc
        gc.collect()
    con.close()
    summarize()


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat
    con = duckdb.connect()
    con.execute("SET threads=4")
    annual = pd.read_csv(OUT / "annual_metrics.csv")
    comparisons = [f + "__" + t for f in FAMILIES for t in TARGETS]
    checks, prediction_files = [], []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
    for year in YEARS:
        path = OUT / f"predictions/target_predictions_{year}.parquet"
        frame = con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        old_path = PREVIOUS / f"predictions/nonlinear_challenger_predictions_{year}.parquet"
        check(f"{year}_unique_keys", not frame.duplicated(["date", "permno"]).any())
        check(f"{year}_all_dates_in_year", frame.date.dt.year.eq(year).all())
        mismatch = con.sql(f'''SELECT count(*) FROM read_parquet('{path}') n FULL JOIN read_parquet('{old_path}') o USING(date,permno)
            WHERE n.permno IS NULL OR o.permno IS NULL OR n.auction_volume IS DISTINCT FROM o.auction_volume
              OR n.historical_frozen_ridge IS DISTINCT FROM o.frozen_prediction
              OR n.lgbm__median20_log_residual IS DISTINCT FROM o.lgbm_residual_58_l2''').fetchone()[0]
        check(f"{year}_identical_labels_reference_and_cached_lgbm", mismatch == 0)
        for candidate in comparisons + ["historical_frozen_ridge"]:
            check(f"{year}_{candidate}_complete_finite_nonnegative", np.isfinite(frame[candidate]).all() and frame[candidate].ge(0).all())
            metrics = forecast_metrics(frame.auction_volume, frame[candidate])
            recorded = annual[(annual.test_year == year) & annual.candidate.eq(candidate)].iloc[0]
            check(f"{year}_{candidate}_recomputed_metrics", all(np.isclose(v, recorded[k], atol=1e-9, rtol=1e-10) for k,v in metrics.items()))
        # Independent reconstruction from saved model coefficients / boosters.
        sample = con.sql(f"SELECT * FROM read_parquet('{PANEL}') WHERE eligible_target AND year={year} ORDER BY hash(permno,date) LIMIT 200").df()
        prep = json.loads((OUT / f"models/preprocessor_{year}.json").read_text())
        reconstruction = json.loads((OUT / f"models/reconstruction_{year}.json").read_text())
        matrix = apply_preprocessor(sample, prep, *[list(prep[k]) for k in ["continuous", "boolean", "categorical"]])
        names = json.loads((OUT / f"models/feature_names_{year}.json").read_text())
        indices = [prep["output_names"].index(n) for n in names]
        matrix = matrix[:, indices]
        for candidate in comparisons:
            family, target = candidate.split("__")
            if family == "ridge":
                component = predict_ridge(matrix, np.load(OUT / f"models/{candidate}_{year}.npy"))
            else:
                model_path = (PREVIOUS / f"models/lgbm_residual_58_l2_{year}.txt") if target == "median20_log_residual" else OUT / f"models/{candidate}_{year}.txt"
                component = lgb.Booster(model_file=str(model_path)).predict(matrix)
            pred = np.maximum(np.expm1(np.clip(offset(sample, target, reconstruction["fallback"])+component, *reconstruction["clip_log"])),0)
            aligned = sample[["date", "permno"]].merge(frame[["date", "permno", candidate]], on=["date", "permno"], validate="one_to_one")
            check(f"{year}_{candidate}_model_reconstruction", np.allclose(pred, aligned[candidate], rtol=1e-10, atol=1e-8))
        prediction_files.append({"path": str(path.relative_to(ROOT)), "rows": len(frame), "sha256": sha(path)})
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv", index=False)
    assert all(c["passed"] for c in checks), "Verification failed; inspect validation_checks.csv"

    glob = str(OUT / "predictions/target_predictions_*.parquet")
    bootstrap_rows = []
    for index, candidate in enumerate(comparisons):
        family, target = candidate.split("__")
        reference = family + "__median20_log_residual"
        daily = con.sql(f'''SELECT date, year(date) test_year,
            avg(abs(ln(1+"{candidate}")-log1p_auction_volume)-abs(ln(1+"{reference}")-log1p_auction_volume)) delta
            FROM read_parquet('{glob}') GROUP BY date,year(date) ORDER BY date''').df()
        for label, years in [("development", [2022,2023,2024]), ("confirmation", [2025])]:
            values = daily.loc[daily.test_year.isin(years), "delta"].to_numpy()
            rng = np.random.default_rng(SEED + index)
            draws = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
            low, high = np.quantile(draws,[0.025,0.975])
            bootstrap_rows.append({"candidate": candidate, "reference": reference, "period": label,
                                   "dates": len(values), "mean_difference": values.mean(), "ci_2_5": low, "ci_97_5": high})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "paired_target_bootstrap.csv", index=False)
    development = annual[annual.test_year.isin([2022,2023,2024])].groupby(["candidate", "family", "target"],as_index=False).agg(
        development_rmse_log=("rmse_log","mean"),development_wape=("wape","mean"),development_bias=("aggregate_bias","mean"),
        development_log_mae=("mae_log","mean"))
    confirmation = annual[annual.test_year.eq(2025)][["candidate","rmse_log","wape","aggregate_bias"]].rename(
        columns={"rmse_log":"confirmation_rmse_log","wape":"confirmation_wape","aggregate_bias":"confirmation_bias"})
    decision = development.merge(confirmation,on="candidate",validate="one_to_one").merge(
        bootstrap[bootstrap.period.eq("development")][["candidate","mean_difference","ci_2_5","ci_97_5"]],on="candidate",how="left",validate="one_to_one")
    annual_pivot = annual.pivot(index="candidate",columns="test_year",values="rmse_log")
    for index, row in decision.iterrows():
        if row.family == "historical_reference":
            continue
        ref = row.family + "__median20_log_residual"
        base = development.set_index("candidate").loc[ref,"development_rmse_log"]
        change = float(row.development_rmse_log-base)
        stable = bool((annual_pivot.loc[row.candidate,[2022,2023,2024]]-annual_pivot.loc[ref,[2022,2023,2024]]).max()<=0.002)
        decision.loc[index,"rmse_change_vs_same_family_median"] = change
        decision.loc[index,"stable_development_years"] = stable
        decision.loc[index,"target_change_passes_gate"] = bool(change<=-0.002 and stable and row.ci_97_5<0)
    decision.to_csv(OUT / "target_comparison_summary.csv",index=False)
    segment_rows = []
    segments = {"size":"CASE WHEN size_percentile<0.33 THEN 'small' WHEN size_percentile<0.67 THEN 'mid' ELSE 'large' END",
                "history":"CASE WHEN prior_auction_observations<5 THEN 'cold_0_4' WHEN prior_auction_observations<20 THEN 'developing_5_19' ELSE 'seasoned' END",
                "tail":"CASE WHEN actual_percentile>=0.99 THEN 'top_1pct' ELSE 'lower_99pct' END"}
    for candidate in comparisons:
        for dimension, expression in segments.items():
            part = con.sql(f'''WITH ranked AS (SELECT *,percent_rank() OVER(PARTITION BY year(date) ORDER BY auction_volume) actual_percentile
                FROM read_parquet('{glob}') WHERE year(date) IN(2022,2023,2024))
                SELECT '{candidate}' candidate,'{dimension}' dimension,{expression} segment,count(*) observations,
                sqrt(avg(pow(ln(1+"{candidate}")-log1p_auction_volume,2))) rmse_log,
                sum(abs("{candidate}"-auction_volume))/sum(auction_volume) wape,
                sum("{candidate}")/sum(auction_volume)-1 aggregate_bias FROM ranked GROUP BY 1,2,3''').df()
            segment_rows.append(part)
    pd.concat(segment_rows).to_csv(OUT / "segment_diagnostics.csv",index=False)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    labels = {"log_level":"Direct log level","lag1_log_growth":"One-day log growth","median20_log_residual":"Median-relative"}
    for family, color in [("ridge","#5477a0"),("lgbm","#cb8b32")]:
        d = decision[decision.family.eq(family)].set_index("target").loc[TARGETS]
        x = np.arange(3)+(0.18 if family=="lgbm" else -0.18)
        axes[0].bar(x,d.development_rmse_log,width=.36,label=family,color=color)
        axes[1].bar(x,100*d.development_wape,width=.36,label=family,color=color)
    for ax,title in zip(axes,["Development log RMSE","Development WAPE (%)"]):
        ax.set_xticks(range(3),[labels[t] for t in TARGETS],rotation=12)
        ax.set_title(title+"; lower is better")
        ax.legend()
    fig.suptitle("Same 58 inputs, historical samples and annual folds (2022-2024)")
    fig.tight_layout()
    fig.savefig(OUT / "target_comparison.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    manifest = {"all_checks_passed":True,"checks":len(checks),"completed_utc":datetime.now(timezone.utc).isoformat(),
                "prediction_files":prediction_files,"total_rows":sum(f["rows"] for f in prediction_files),
                "eligible_target_changes":decision.loc[decision.target_change_passes_gate.eq(True),"candidate"].tolist(),
                "median_lightgbm_exactly_reused":True,"plan_sha256":sha(OUT / "experiment_plan.json")}
    (OUT / "experiment_summary.json").write_text(json.dumps(manifest,indent=2))
    lines = ["| Model | Target | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in decision.sort_values(["family","development_rmse_log"]).itertuples():
        lines.append(f"| {r.family} | {r.target} | {r.development_rmse_log:.5f} | {r.development_wape:.2%} | {r.development_bias:.2%} | {r.confirmation_rmse_log:.5f} | {r.confirmation_wape:.2%} |")
    eligible = manifest["eligible_target_changes"]
    interpretation = "No alternative target passes the prespecified target-change gate." if not eligible else "Alternatives passing the prespecified target-change gate: " + ", ".join(eligible) + "."
    report = f'''# Controlled target-formulation comparison

{interpretation}

All {len(checks)} forecast verification checks passed across {manifest['total_rows']:,} common stock-days. The experiment fits or reuses exactly the six planned model/target combinations. Historical notebooks and results remain unchanged.

{chr(10).join(lines)}

Development statistics are equal-year averages for 2022-2024, not pooled metrics. The 2025 period was previously inspected and is confirmation evidence, not an untouched holdout. Selection uses development results only. The historical frozen ridge is an external reference; the controlled ridge refits use the shared step-03 samples, so they need not reproduce the earlier ridge coefficients or metrics.

## Controls and definitions

- The three estimation targets are direct log1p auction shares, log1p change from the previous stock row's auction, and log1p departure from the prior 20-row auction median.
- All forecasts reconstruct closing-auction shares and use the same training-derived final log-forecast bounds.
- Missing growth/median references use that fitting window's global median auction shares. No eligible row is removed for missing history. Reference missingness is recorded separately.
- Features are the same 58 sources. Shared preprocessing uses the prior 99-feature preparation and the same feature mask. No feature, calibration, anomaly-history, weighting, or clipping-policy changes were introduced.
- Exact tuning, validation and final-fitting samples from step 03 are reused for all targets. Their fingerprints are in `shared_sample_fingerprints.csv`.
- Each ridge target tests seven penalties. Each new LightGBM target tests 15, 31 and 63 leaves with at most 2,000 trees, the same seeds, and historical validation/early stopping. The verified median-relative LightGBM model and complete original validation grid are reused.
- Test labels are not used for tuning or fallback construction. Fold settings are saved under `models/`.

## Decision rule and limits

A target change is compared with the median-relative target within the same model family. It must improve mean development log RMSE by at least 0.002, avoid annual development deterioration above 0.002, and have a paired development log-MAE bootstrap upper interval below zero. Exact differences and intervals appear in `target_comparison_summary.csv` and `paired_target_bootstrap.csv`.

The bootstrap resamples whole dates, preserving same-date dependence but not serial dependence between dates. Intervals are not adjusted for multiple comparisons. Log RMSE, WAPE and aggregate bias answer different questions; a numerical winner need not improve every metric or difficult stock group. Top-actual-volume groups are ex-post diagnostics, not model inputs.

## Verification and outputs

Saved forecasts were checked for unique keys, completeness, nonnegative finite values, exact historical-reference alignment, and exact reuse of the previous median-relative LightGBM forecasts. Metrics were recomputed from written prediction files. Every model was independently reloaded to reconstruct 200 deterministic rows per year using the saved preprocessing and inverse transformation.

- `annual_metrics.csv`: every candidate/year, including the external historical ridge.
- `target_comparison_summary.csv`: development/confirmation results and target-change decisions.
- `hyperparameter_validation.csv`: every validation trial, with reused trials labeled.
- `predictions/`: full row-level forecasts for 2022-2025.
- `models/`: coefficients, boosters, preprocessors, clipping/fallback settings; cached median LightGBM models remain in the verified step-03 folder.
- `segment_diagnostics.csv`: pooled development size, history, and tail results.
- `reference_missingness.csv`: missing references by target, model and year.
- `validation_checks.csv`, `experiment_summary.json`, `experiment_plan.json`: checks, fingerprints and the plan frozen before fitting.

## Reproduce

Use `.venv/bin/python notebooks/compare_target_formulations.py` with the macOS OpenMP library environment used in step 03. To rebuild only summaries/checks from existing forecasts, append `--summarize`. This reuses existing source data and samples; it does not change prior experiments.
'''
    (OUT / "README.md").write_text(report)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# 17 — Controlled target-formulation comparison\n\nResults and reproduction notebook. The experiment was executed locally through `compare_target_formulations.py`; the cells below inspect the saved results. The six combinations use common historical samples and 58 input features."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'blackrock_data').exists())\nout = root / 'outputs/experiments/17_target_formulations'"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'target_comparison_summary.csv'))\ndisplay(Image(filename=str(out / 'target_comparison.png')))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'annual_metrics.csv'))\ndisplay(pd.read_csv(out / 'paired_target_bootstrap.csv'))"),
        nbformat.v4.new_code_cell("checks = pd.read_csv(out / 'validation_checks.csv')\nassert checks.passed.all()\ndisplay(checks)"),
        nbformat.v4.new_markdown_cell("To rerun model fitting, execute `.venv/bin/python notebooks/compare_target_formulations.py` from the repository root with the step-03 OpenMP runtime environment. See the saved experiment plan and README for target definitions, fallbacks and limitations.")])
    notebook.metadata.kernelspec = {"display_name":"Auction local","name":"auction-local","language":"python"}
    nbformat.write(notebook,ROOT / "notebooks/17_target_formulation_comparison.ipynb")
    con.close()
    progress("All forecast checks passed; target comparison and notebook saved")
    print(decision.to_string(index=False),flush=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        run()
