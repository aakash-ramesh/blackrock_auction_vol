"""Execute an isolated compact-only copy of Notebook 16 and verify its forecasts.

macOS runtime:
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/sklearn/.dylibs" \
  .venv/bin/python -u notebooks/reproduce_compact_nonlinear.py
"""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/reproduction/03_compact_nonlinear"
OUT.mkdir(parents=True, exist_ok=True)
for env, directory in [("MPLCONFIGDIR", ".matplotlib"), ("IPYTHONDIR", ".ipython"),
                       ("JUPYTER_RUNTIME_DIR", ".jupyter_runtime")]:
    os.environ.setdefault(env, str(OUT / directory))

import duckdb
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    started = datetime.now(timezone.utc).isoformat()
    verified = json.loads((ROOT / "outputs/reproduction/02_benchmark_reconciliation/verified_benchmark_manifest.json").read_text())
    assert verified["all_integrity_checks_passed"]
    fingerprints = pd.read_csv(ROOT / "outputs/reproduction/02_benchmark_reconciliation/input_fingerprints.csv")
    for row in fingerprints.itertuples():
        if row.path.endswith(".parquet") or row.path.endswith("modeling_utils.py"):
            assert digest(ROOT / row.path) == row.sha256, f"Verified input changed: {row.path}"
    source = ROOT / "notebooks/16_nonlinear_model_challengers.ipynb"
    nb = nbformat.read(source, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def replace_once(old, new):
        found = 0
        for cell in nb.cells:
            if cell.cell_type == "code":
                found += cell.source.count(old)
                cell.source = cell.source.replace(old, new)
        assert found == 1, (old[:100], found)

    replace_once('PANEL = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering" / "enriched_daily_panel.parquet"',
                 'PANEL = PROJECT_ROOT / "outputs" / "reproduction" / "01_daily_panel" / "enriched_daily_panel.parquet"')
    replace_once('OUT = PROJECT_ROOT / "outputs" / "eda" / "16_nonlinear_challengers"',
                 'OUT = PROJECT_ROOT / "outputs" / "reproduction" / "03_compact_nonlinear"')
    replace_once('    "lgbm_residual_99_l2": {"features": FEATURES, "objective": "regression"},\n', '')
    replace_once('print(f"Feature sets: compact={len(CHAMPION_FEATURES)}, full={len(FEATURES)}")', '''print(f"Feature sets: compact={len(CHAMPION_FEATURES)}, full={len(FEATURES)}")
SAMPLES = OUT / "samples"
MODELS = OUT / "models"
SAMPLES.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
def progress(message):
    from datetime import datetime, timezone
    line = datetime.now(timezone.utc).isoformat() + " " + message
    print(line, flush=True)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(line + "\\n")
def save_sample(frame, label):
    path = SAMPLES / (label + ".parquet")
    con.register("sample_snapshot", frame)
    con.execute(f"COPY sample_snapshot TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("sample_snapshot")
    return path
progress("Local compact nonlinear reproduction started")''')
    replace_once('    print(f"Running nonlinear challengers for {test_year}...")',
                 '    progress(f"Starting fold {test_year}")')
    replace_once('    x_tune, prep_tune = fit_preprocessor(', '''    save_sample(tune_train, f"{test_year}_tune")
    save_sample(validation, f"{test_year}_validation")
    progress(f"Fold {test_year}: loaded historical tuning and validation rows")
    x_tune, prep_tune = fit_preprocessor(''')
    replace_once('        for leaves in LEAF_GRID:\n            model = fit_validation_model(', '''        for leaves in LEAF_GRID:
            progress(f"Fold {test_year}: tuning {candidate}, leaves={leaves}")
            model = fit_validation_model(''')
    replace_once('        selected[candidate] = best', '''        selected[candidate] = best
        progress(f"Fold {test_year}: selected {candidate}, leaves={best['num_leaves']}, iterations={best['best_iteration']}")''')
    replace_once('    x_refit, prep = fit_preprocessor(', '''    save_sample(refit, f"{test_year}_refit")
    progress(f"Fold {test_year}: refitting on {len(refit):,} rows; scoring {len(test):,} rows")
    x_refit, prep = fit_preprocessor(''')
    replace_once('    y_refit_residual, test_base_log = residual_target_and_base(refit, test)', '''    (MODELS / f"preprocessor_{test_year}.json").write_text(json.dumps(prep, indent=2))
    y_refit_residual, test_base_log = residual_target_and_base(refit, test)
    (MODELS / f"reconstruction_{test_year}.json").write_text(json.dumps({
        "fallback": float(np.median(refit["auction_volume"])),
        "clip_log": np.quantile(refit["log1p_auction_volume"], [0.001, 0.999]).tolist(),
        "selected": selected,
    }, indent=2))''')
    replace_once('    output = output.merge(benchmark, on=["date", "permno"], how="left", validate="one_to_one")', '''    original_keys = output[["date", "permno"]].copy()
    output = output.merge(benchmark, on=["date", "permno"], how="left", validate="one_to_one", sort=False)
    assert output[["date", "permno"]].equals(original_keys), "Merge changed row ordering"
    assert np.array_equal(output["auction_volume"].to_numpy(), y_test_raw)''')
    replace_once('        model.fit(x_refit[:, mask], y_refit_residual)', '''        progress(f"Fold {test_year}: final fit {candidate}, iterations={iterations}")
        model.fit(x_refit[:, mask], y_refit_residual)
        model.booster_.save_model(str(MODELS / f"{candidate}_{test_year}.txt"))
        (MODELS / f"feature_columns_{candidate}_{test_year}.json").write_text(json.dumps({
            "output_names": np.array(prep["output_names"])[mask].tolist(),
            "source_features": spec["features"],
        }, indent=2))''')
    replace_once('    print(f"  wrote {len(output):,} predictions to {path.name}")', '''    pd.DataFrame(metric_rows).to_csv(OUT / "nonlinear_candidate_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "nonlinear_hyperparameter_validation.csv", index=False)
    progress(f"Fold {test_year}: saved {len(output):,} complete forecasts")''')

    nb.cells.insert(0, nbformat.v4.new_markdown_cell('''# Local compact nonlinear reproduction

Isolated copy of Notebook 16: unchanged 58-feature L2 and Huber specifications and their fixed 50/50 log blends. The 99-feature candidate is omitted. Uses the verified rebuilt panel and frozen historical ridge forecasts. Adds sample, model, preprocessing and row-alignment records. Historical research outputs remain unchanged.

2022-2024 are development years; 2025 is previously inspected confirmation evidence. This is a reproduction on the current data snapshot, not a new holdout or a guarantee of historical bitwise identity. Historical training samples were not retained.
'''))
    nb.metadata["reproduction"] = {"source_sha256": digest(source), "started_utc": started,
        "changes": ["redirect panel/output paths", "omit 99-feature candidate", "save samples/models/preprocessing", "assert merge ordering", "progress/checkpoint records"]}
    nb.metadata.kernelspec = {"name": "auction-local", "display_name": "Auction local", "language": "python"}
    executed = OUT / "16_compact_nonlinear_rebuilt.ipynb"
    nbformat.write(nb, executed)
    (OUT / "requirements.lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    environment = {name: importlib.metadata.version(name) for name in
                   ["lightgbm", "duckdb", "numpy", "pandas", "scikit-learn", "nbclient"]}
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2))
    client = NotebookClient(nb, timeout=None, kernel_name="auction-local", resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, executed)
    audit(verified, environment, started, source)


def audit(verified, environment, started, source):
    from notebooks.modeling_utils import forecast_metrics
    con = duckdb.connect()
    con.execute("SET threads=4")
    metrics = pd.read_csv(OUT / "nonlinear_candidate_metrics.csv")
    reference = pd.read_csv(ROOT / "outputs/reproduction/02_benchmark_reconciliation/recomputed_annual_metrics.csv")
    champion = verified["champion"]
    candidates = [c for c in metrics.candidate.unique() if c != champion]
    checks, files = [], []
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
    panel = ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
    for year in range(2022, 2026):
        path = OUT / f"predictions/nonlinear_challenger_predictions_{year}.parquet"
        df = con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        check(f"{year}_unique_keys", not df.duplicated(["date", "permno"]).any())
        original = ROOT / f"outputs/eda/09_global_finalists/predictions/global_finalist_predictions_{year}.parquet"
        mismatches = con.sql(f'''SELECT count(*) FROM read_parquet('{path}') p
            FULL JOIN read_parquet('{original}') r USING(date,permno)
            WHERE p.permno IS NULL OR r.permno IS NULL
               OR p.auction_volume IS DISTINCT FROM r.auction_volume
               OR p.frozen_prediction IS DISTINCT FROM r."{champion}"''').fetchone()[0]
        check(f"{year}_ridge_and_labels_exactly_match_verified_files", mismatches == 0)
        for candidate in [champion] + candidates:
            column = "frozen_prediction" if candidate == champion else candidate
            check(f"{year}_{candidate}_finite_nonnegative", np.isfinite(df[column]).all() and df[column].ge(0).all())
            fresh = forecast_metrics(df.auction_volume, df[column])
            saved = metrics[(metrics.test_year == year) & (metrics.candidate == candidate)].iloc[0]
            check(f"{year}_{candidate}_metrics_recompute", all(np.isclose(fresh[k], saved[k], atol=1e-9, rtol=1e-10) for k in fresh))
            if candidate == champion:
                old = reference[(reference.test_year == year) & (reference.candidate == champion)].iloc[0]
                check(f"{year}_ridge_matches_step02", all(np.isclose(fresh[k], old[k], atol=1e-9, rtol=1e-10) for k in fresh))
        # Reconstruct a deterministic slice from saved models and preprocessing.
        context = pd.read_csv(ROOT / "outputs/eda/05_feature_engineering/approved_enriched_daily_features.csv")
        all_features = context.variable.tolist()
        sample = con.sql(f"SELECT * FROM read_parquet('{panel}') WHERE eligible_target AND year={year} ORDER BY hash(permno,date) LIMIT 200").df()
        prep = json.loads((OUT / f"models/preprocessor_{year}.json").read_text())
        reconstruction = json.loads((OUT / f"models/reconstruction_{year}.json").read_text())
        from notebooks.modeling_utils import apply_preprocessor
        continuous = list(prep["continuous"])
        boolean = list(prep["boolean"])
        categorical = list(prep["categorical"])
        x = apply_preprocessor(sample, prep, continuous, boolean, categorical)
        import lightgbm as lgb
        base = np.log1p(sample.median20_auction_volume.fillna(reconstruction["fallback"]).to_numpy(dtype=float))
        for candidate in ["lgbm_residual_58_l2", "lgbm_residual_58_huber"]:
            columns = json.loads((OUT / f"models/feature_columns_{candidate}_{year}.json").read_text())
            indices = [prep["output_names"].index(c) for c in columns["output_names"]]
            booster = lgb.Booster(model_file=str(OUT / f"models/{candidate}_{year}.txt"))
            pred = np.maximum(np.expm1(np.clip(base + booster.predict(x[:, indices]), *reconstruction["clip_log"])), 0)
            aligned = sample[["date", "permno"]].merge(df[["date", "permno", candidate]], on=["date", "permno"], validate="one_to_one")
            check(f"{year}_{candidate}_saved_model_reconstructs_predictions", np.allclose(pred, aligned[candidate], rtol=1e-10, atol=1e-8))
        files.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(df)})
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv", index=False)
    pd.DataFrame([{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in sorted((OUT / "samples").glob("*.parquet"))]).to_csv(OUT / "training_sample_fingerprints.csv", index=False)
    decision = pd.read_csv(OUT / "nonlinear_candidate_selection.csv")
    old = pd.read_csv(ROOT / "outputs/eda/16_nonlinear_challengers/nonlinear_candidate_selection.csv")
    old[old.candidate.isin(decision.candidate)].merge(decision, on="candidate", suffixes=("_historical", "_rebuilt")).to_csv(OUT / "historical_summary_comparison.csv", index=False)
    summary = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "source_notebook_sha256": digest(source), "environment": environment,
               "all_checks_passed": all(c["passed"] for c in checks), "checks": len(checks),
               "predictions": files, "total_prediction_rows": sum(f["rows"] for f in files),
               "eligible_under_original_gate": decision.loc[decision.eligible_replacement, "candidate"].tolist(),
               "caveats": ["2025 previously inspected", "single-date bootstrap does not preserve serial dependence across dates",
                           "original gate has no multiplicity correction and allows annual RMSE worsening up to 0.002",
                           "historical samples unavailable; identical settings need not yield historical bitwise identity"]}
    (OUT / "reproduction_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, decision, metrics)
    con.close()
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["all_checks_passed"]:
        raise AssertionError("Nonlinear forecast verification failed")


def write_report(summary, decision, metrics):
    table = ["| Candidate | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE | Original gate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in decision.sort_values("development_rmse_log").itertuples():
        table.append(f"| {r.candidate} | {r.development_rmse_log:.5f} | {r.development_wape:.2%} | {r.development_bias:.2%} | {r.confirmation_rmse_log:.5f} | {r.confirmation_wape:.2%} | {'Pass' if r.eligible_replacement else 'Reference'} |")
    validation = pd.read_csv(OUT / "nonlinear_hyperparameter_validation.csv")
    selected = validation.sort_values("validation_rmse_log").groupby(["test_year", "candidate"], as_index=False).first()
    selected.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    segments = pd.read_csv(OUT / "nonlinear_segment_diagnostics.csv")
    segment_table = ["| Segment | Ridge log RMSE | L2 log RMSE | Ridge WAPE | L2 WAPE | Ridge bias | L2 bias |",
                     "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for segment in ["small", "cold_0_4", "top_1pct", "quarterly_expiry", "month_end"]:
        ridge_segment = segments[(segments.candidate == "rolling5_residual_auction_calendar_liquidity") & (segments.segment == segment)].iloc[0]
        tree_segment = segments[(segments.candidate == "lgbm_residual_58_l2") & (segments.segment == segment)].iloc[0]
        segment_table.append(f"| {segment} | {ridge_segment.rmse_log:.4f} | {tree_segment.rmse_log:.4f} | {ridge_segment.wape:.2%} | {tree_segment.wape:.2%} | {ridge_segment.aggregate_bias:.2%} | {tree_segment.aggregate_bias:.2%} |")
    bootstrap = pd.read_csv(OUT / "nonlinear_date_block_bootstrap.csv")
    intervals = ["| Candidate | Mean date-level log-MAE difference vs ridge | 95% interval |",
                 "| --- | ---: | --- |"]
    for r in bootstrap[bootstrap.period.eq("development_2022_2024")].itertuples():
        intervals.append(f"| {r.candidate} | {r.mean_difference:.6f} | [{r.ci_2_5:.6f}, {r.ci_97_5:.6f}] |")
    champion = "rolling5_residual_auction_calendar_liquidity"
    baseline = metrics[metrics.candidate.eq(champion)].set_index("test_year")
    annual = ["| Year | Ridge log RMSE | L2 log RMSE | Huber log RMSE | Ridge WAPE | L2 WAPE | Huber WAPE |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for year in range(2022, 2026):
        l2 = metrics[(metrics.test_year == year) & metrics.candidate.eq("lgbm_residual_58_l2")].iloc[0]
        huber = metrics[(metrics.test_year == year) & metrics.candidate.eq("lgbm_residual_58_huber")].iloc[0]
        annual.append(f"| {year} | {baseline.loc[year,'rmse_log']:.5f} | {l2.rmse_log:.5f} | {huber.rmse_log:.5f} | {baseline.loc[year,'wape']:.2%} | {l2.wape:.2%} | {huber.wape:.2%} |")
    status = "passed" if summary["all_checks_passed"] else "FAILED; review validation_checks.csv before using results"
    (OUT / "README.md").write_text(f'''# Step 3: compact nonlinear reproduction

## Scope and result

The original Notebook 16's 58-feature squared-error (L2) and Huber LightGBM models and fixed 50/50 log blends were rerun against the verified saved ridge forecasts. All {summary['checks']} forecast verification checks {status}. The run saved {summary['total_prediction_rows']:,} stock-day rows across four years, with a prediction from every candidate on every row. Historical source data, notebooks, and `outputs/eda/` results were preserved.

{chr(10).join(table)}

Development metrics average annual 2022-2024 values with equal year weight. The 2025 results are confirmation evidence already inspected during research, not a fresh holdout. Bias is forecast share total divided by actual share total minus one; averaging annual bias differs from pooling all rows.

## Annual comparisons

{chr(10).join(annual)}

## Experiment settings preserved

- Estimation target: log1p(auction shares) minus log1p(prior-only 20-row auction median).
- Inputs: the original 58 auction-history, calendar, and liquidity source features; the original shared preprocessing still fits all 99 approved features before applying the 58-feature mask.
- Missing baseline: historical training-sample median auction volume.
- Validation: previous calendar year; tuning data use the preceding five years. Final refit uses the five years before the evaluation year.
- Samples: 250,000 tuning rows, 120,000 validation rows, 350,000 final-fitting rows. Original seed formulas and sampling method preserved; realized samples are saved.
- Tree sizes: 15, 31, or 63 leaves. Learning rate 0.03; maximum 2,000 trees; early-stopping patience 100; remaining regularization and subsampling unchanged.
- Final forecasts clipped to training log-target 0.1st/99.9th percentiles. No calibration or anomaly-history correction added.
- Blends: equal weights in log1p forecast space, without test-period fitting.

Some selected fits are near the original tree-count limit. No larger search was introduced in this reproduction.

## Remaining calibration and segment tradeoffs

{chr(10).join(segment_table)}

These are pooled 2022-2024 segment diagnostics, unlike the equal-year averages above. The L2 model improves log RMSE and WAPE for small stocks, the top actual-volume tail, and the displayed event groups. However, top-tail and expiry bias worsen, and cold-start log RMSE worsens even though its WAPE improves. Only 758 development rows have fewer than five prior observed auctions. Actual top-tail membership is an ex-post diagnostic, not a usable forecast input. Better average accuracy does not establish better calibration in every group.

## Paired uncertainty and original replacement rule

{chr(10).join(intervals)}

Negative differences favor the challenger. The original bootstrap averages paired errors within dates and resamples dates 2,000 times. This preserves within-date dependence but not serial dependence across dates. Intervals are unadjusted for multiple candidate comparisons.

The original gate requires mean development log RMSE improvement of at least 0.002, no annual development deterioration larger than 0.002, and an upper paired log-MAE interval below zero. Passing this gate is evidence under that research rule, not an automatic production promotion. The historical ridge benchmark remains intact.

All four challengers pass the development gate. In 2025, the pure L2 model's paired log-MAE interval crosses zero despite improved log RMSE and WAPE; its log-MAE confirmation result should not be described as statistically established by this test. The other three candidates' 2025 intervals are below zero under the original bootstrap.

## Verification and reproducibility

All saved prediction keys, target values, and ridge forecasts are checked against the verified historical files. Predictions must be finite and nonnegative. Metrics are recalculated from written Parquet files. Saved tree models plus preprocessing and reconstruction settings reproduce a deterministic 200-row slice in every year for each fitted model.

The original Notebook 16 training samples and prediction files were absent. This run uses its settings and the verified current input snapshot; historical bitwise identity cannot be established. Differences from the old metrics are recorded in `historical_summary_comparison.csv`. The exact cause of the old ridge-reference discrepancy remains unknown; the new comparison repairs the evidence by using verified row-level forecasts.

## Artifacts

- `16_compact_nonlinear_rebuilt.ipynb`: executed isolated notebook.
- `predictions/`: complete 2022-2025 forecast files, including actuals and unchanged ridge forecasts.
- `models/`: eight fitted tree models, fold preprocessors, reconstruction parameters, and feature-column order.
- `samples/`: actual historical tuning, validation, and refit samples for each fold.
- `nonlinear_candidate_metrics.csv` and `nonlinear_candidate_selection.csv`: annual results and development comparison.
- `nonlinear_date_block_bootstrap.csv`: original paired uncertainty calculation.
- `nonlinear_segment_diagnostics.csv`: development tail, size, history, and calendar diagnostics.
- `validation_checks.csv`, `training_sample_fingerprints.csv`, and `reproduction_summary.json`: forecast checks and fingerprints.
- `environment.json`, `requirements.lock.txt`, and `progress.log`: environment and execution record.

## Next step

Use these verified forecasts to assess the accuracy/calibration tradeoff, then conduct the prespecified direct-log versus one-day-log-growth versus median-relative target comparison. Any anomaly-history or clipping-policy change should be separately labeled so that it is not confused with a target-formulation improvement.
''')


if __name__ == "__main__":
    if "--audit-only" in sys.argv:
        verified = json.loads((ROOT / "outputs/reproduction/02_benchmark_reconciliation/verified_benchmark_manifest.json").read_text())
        environment = json.loads((OUT / "environment.json").read_text())
        rebuilt = nbformat.read(OUT / "16_compact_nonlinear_rebuilt.ipynb", as_version=4)
        audit(verified, environment, rebuilt.metadata["reproduction"]["started_utc"], ROOT / "notebooks/16_nonlinear_model_challengers.ipynb")
    else:
        main()
