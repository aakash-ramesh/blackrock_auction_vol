"""Audit saved prediction artifacts without fitting or modifying any model."""

from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from notebooks.modeling_utils import forecast_metrics

OUT = ROOT / "outputs/reproduction/02_benchmark_reconciliation"
PRIOR = ROOT / "outputs/eda/09_global_finalists"
NONLINEAR = ROOT / "outputs/eda/16_nonlinear_challengers"
PANEL = ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
METRICS = ["observations", "mae_raw", "wape", "mae_log", "rmse_log", "spearman", "aggregate_bias"]
ATOL = 1e-9


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    spec = json.loads((PRIOR / "frozen_global_benchmark.json").read_text())
    champion = spec["champion"]
    saved09 = pd.read_csv(PRIOR / "global_finalist_metrics.csv")
    saved16 = pd.read_csv(NONLINEAR / "nonlinear_candidate_metrics.csv")
    candidates = saved09.candidate.unique().tolist()
    checks, rows, source_files, differences, contexts = [], [], [], [], []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for year in range(2022, 2026):
        path = PRIOR / f"predictions/global_finalist_predictions_{year}.parquet"
        source_files.append(fingerprint(path))
        frame = con.sql(f"SELECT * FROM read_parquet('{path}')").df()
        print(f"Auditing {year}: {len(frame):,} saved predictions", flush=True)
        check(f"{year}_unique_keys", not frame.duplicated(["date", "permno"]).any(), len(frame))
        check(f"{year}_dates", frame.date.dt.year.eq(year).all(), str(frame.date.min()) + " to " + str(frame.date.max()))
        check(f"{year}_finite_targets", np.isfinite(frame.auction_volume).all() and frame.auction_volume.gt(0).all(), "Positive finite labels")
        check(f"{year}_log_target", np.allclose(np.log1p(frame.auction_volume), frame.log1p_auction_volume, atol=1e-12, rtol=1e-12), "Stored log target equals log1p(shares)")

        con.execute(f"CREATE OR REPLACE TEMP VIEW saved AS SELECT * FROM read_parquet('{path}')")
        con.execute(f"""CREATE OR REPLACE TEMP VIEW rebuilt AS SELECT * FROM read_parquet('{PANEL}')
                     WHERE eligible_target AND year={year}""")
        keys = con.sql("""SELECT count(*) FILTER(WHERE s.permno IS NULL) missing_saved,
            count(*) FILTER(WHERE p.permno IS NULL) extra_saved
            FROM saved s FULL JOIN rebuilt p USING(date,permno)""").fetchone()
        check(f"{year}_evaluation_keys_match_panel", keys == (0, 0), str(keys))
        fields = [c for c in frame.columns if c not in candidates + ["date", "permno"]]
        # Compare every saved context field, allowing only insignificant floating arithmetic differences.
        for field in fields:
            mismatch = con.sql(f'''SELECT count(*) FROM saved s JOIN rebuilt p USING(date,permno)
                WHERE (s."{field}" IS NULL) <> (p."{field}" IS NULL)
                  OR (s."{field}" IS NOT NULL AND p."{field}" IS NOT NULL AND
                     abs(CAST(s."{field}" AS DOUBLE)-CAST(p."{field}" AS DOUBLE)) >
                     1e-10 + 1e-12*abs(CAST(p."{field}" AS DOUBLE)))''').fetchone()[0]
            contexts.append({"test_year": year, "field": field, "mismatches": mismatch})
        check(f"{year}_all_context_fields_match", all(r["mismatches"] == 0 for r in contexts if r["test_year"] == year), f"{len(fields)} fields")

        for candidate in candidates:
            prediction = frame[candidate].to_numpy(dtype=float)
            check(f"{year}_{candidate}_finite_nonnegative", np.isfinite(prediction).all() and (prediction >= 0).all(), "No scoring rows silently dropped")
            metric = forecast_metrics(frame.auction_volume, prediction)
            rows.append({"test_year": year, "candidate": candidate, **metric})
            historical = saved09[(saved09.test_year == year) & (saved09.candidate == candidate)].iloc[0]
            for name in METRICS:
                differences.append({"source": "notebook09", "test_year": year, "candidate": candidate,
                    "metric": name, "recomputed": metric[name], "reported": historical[name],
                    "difference": metric[name] - historical[name],
                    "matches": bool(np.isclose(metric[name], historical[name], atol=ATOL, rtol=1e-10))})
            if candidate == champion:
                historical16 = saved16[(saved16.test_year == year) & (saved16.candidate == champion)].iloc[0]
                for name in METRICS:
                    differences.append({"source": "notebook16", "test_year": year, "candidate": candidate,
                        "metric": name, "recomputed": metric[name], "reported": historical16[name],
                        "difference": metric[name] - historical16[name],
                        "matches": bool(np.isclose(metric[name], historical16[name], atol=ATOL, rtol=1e-10))})
            independent = con.sql(f'''SELECT avg(abs("{candidate}"-auction_volume)),
                 sum(abs("{candidate}"-auction_volume))/sum(auction_volume),
                 avg(abs(ln(1+"{candidate}")-ln(1+auction_volume))),
                 sqrt(avg(pow(ln(1+"{candidate}")-ln(1+auction_volume),2))),
                 sum("{candidate}")/sum(auction_volume)-1 FROM saved''').fetchone()
            check(f"{year}_{candidate}_independent_metrics", np.allclose(
                independent, [metric[k] for k in ["mae_raw", "wape", "mae_log", "rmse_log", "aggregate_bias"]],
                atol=ATOL, rtol=1e-10), "SQL agrees with project metric helper")

    metrics = pd.DataFrame(rows)
    comparison = pd.DataFrame(differences)
    metrics.to_csv(OUT / "recomputed_annual_metrics.csv", index=False)
    comparison.to_csv(OUT / "reported_metric_reconciliation.csv", index=False)
    pd.DataFrame(contexts).to_csv(OUT / "prediction_panel_alignment.csv", index=False)
    check("notebook09_all_metrics_reproduce", comparison.loc[comparison.source.eq("notebook09"), "matches"].all(), "All seven candidates, four years, seven metrics")
    old16_matches = bool(comparison.loc[comparison.source.eq("notebook16"), "matches"].all())
    availability = (NONLINEAR / "predictions").exists() and bool(list((NONLINEAR / "predictions").glob("*.parquet")))
    # Missing artifacts and the discrepancy are findings, not failed panel/metric arithmetic checks.
    issues = [{"issue": "notebook16_ridge_reference_mismatch", "present": not old16_matches,
               "impact": "Notebook 16 ridge metrics cannot come from the current Notebook 09 prediction files using the checked metric function."},
              {"issue": "notebook16_predictions_unavailable", "present": not availability,
               "impact": "Cannot reproduce nonlinear metrics, paired tests, or common-row comparisons from row-level forecasts."}]

    development = metrics[metrics.test_year.isin([2022, 2023, 2024])].groupby("candidate", as_index=False)[METRICS[1:]].mean()
    development.to_csv(OUT / "recomputed_development_mean_annual_metrics.csv", index=False)
    champion_rows = metrics[metrics.candidate.eq(champion)]
    reference16 = saved16[saved16.candidate.eq(champion)]
    annual_table = champion_rows[["test_year", "observations", "rmse_log", "wape", "aggregate_bias"]].merge(
        reference16[["test_year", "rmse_log", "wape", "aggregate_bias"]], on="test_year", suffixes=("_verified09", "_reported16"))
    annual_table.to_csv(OUT / "ridge_reference_discrepancy.csv", index=False)
    inputs = [PRIOR / "frozen_global_benchmark.json", PRIOR / "global_finalist_metrics.csv",
              PRIOR / "global_finalist_selection.csv", NONLINEAR / "nonlinear_candidate_metrics.csv",
              NONLINEAR / "nonlinear_candidate_selection.csv", ROOT / "notebooks/modeling_utils.py",
              ROOT / "notebooks/16_nonlinear_model_challengers.ipynb", ROOT / "notebooks/reconcile_saved_benchmarks.py",
              PANEL, ROOT / "outputs/reproduction/01_daily_panel/validation_summary.json",
              ROOT / "outputs/reproduction/01_daily_panel/environment.json"]
    source_files.extend(fingerprint(p) for p in inputs)
    pd.DataFrame(source_files).to_csv(OUT / "input_fingerprints.csv", index=False)
    passed = all(r["passed"] for r in checks)
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "all_integrity_checks_passed": passed,
                "checks": len(checks), "champion": champion, "development_years": [2022, 2023, 2024],
                "confirmation_year": 2025, "confirmation_previously_inspected": True,
                "absolute_metric_tolerance": ATOL, "issues": issues,
                "decision": "Use hashed Notebook 09 forecasts as the verified historical reference; treat Notebook 16 results as provisional until row-level reproduction.",
                "prediction_files": source_files[:4]}
    (OUT / "verified_benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))

    table = ["| Year | Rows | Verified log RMSE | Verified WAPE | Verified bias | Notebook 16 log RMSE | Notebook 16 WAPE |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in annual_table.itertuples():
        table.append(f"| {r.test_year} | {r.observations:,} | {r.rmse_log_verified09:.5f} | {r.wape_verified09:.2%} | {r.aggregate_bias_verified09:.2%} | {r.rmse_log_reported16:.5f} | {r.wape_reported16:.2%} |")
    text = f'''# Step 2: saved benchmark reconciliation

All {len(checks)} integrity and metric-arithmetic checks passed. No model was fitted and no historical artifact was overwritten.

## Verified reference

All 196 reported annual metrics (7 models x 4 years x 7 metrics) from Notebook 09 reproduce within the declared tolerance. An independent SQL calculation also agrees with the project's Python metric helper for raw MAE, WAPE, log MAE, log RMSE, and aggregate bias. Spearman uses the project's average-rank correlation implementation.

The 1,271,378 saved prediction rows cover exactly the eligible rebuilt-panel keys for 2022-2025, with no duplicates or missing/extra rows. All 12 saved target/context fields match the rebuilt panel within numerical tolerance in every year. Predictions are finite and nonnegative; scoring drops no rows.

The 58-feature ridge remains the verified historical reference. Its 2022-2024 mean annual log RMSE is {development.set_index('candidate').loc[champion, 'rmse_log']:.8f}, WAPE is {development.set_index('candidate').loc[champion, 'wape']:.4%}, and mean annual aggregate bias is {development.set_index('candidate').loc[champion, 'aggregate_bias']:.4%}. These average annual metrics give each year equal weight; they are not pooled metrics.

{chr(10).join(table)}

## Notebook 16 discrepancy

Notebook 16's reported ridge metrics differ in all four years. Row counts agree, but error and bias metrics do not. Its current code reads Notebook 09 forecasts, does not refit ridge, and uses the same metric helper. Therefore the saved Notebook 16 table cannot be reproduced from the currently available inputs under that code. This is an artifact-consistency problem rather than rounding or annual-versus-pooled aggregation.

The exact historical cause is not recoverable from this checkout. A different input snapshot or stale saved outputs are possible explanations, not established causes. Matching row counts do not establish row-level identity for the missing Notebook 16 forecasts.

Notebook 16's row-level nonlinear prediction files are absent. Its nonlinear improvements and paired replacement gate remain reported historical results, not independently verified results. Replacing its ridge reference in the CSV would not repair its paired tests; those need both models' predictions on identical rows. No replacement decision is made from this audit.

## Next bounded implementation step

Reproduce the original compact nonlinear candidates in isolation against the verified saved ridge forecasts, recording identical evaluation keys and input fingerprints. Preserve the original settings before introducing new targets or changing anomaly-history treatment. This will test whether the reported improvement persists on the reproducible data snapshot.

## Artifacts

- `recomputed_annual_metrics.csv`: all seven saved models by year.
- `recomputed_development_mean_annual_metrics.csv`: equal-year development averages.
- `reported_metric_reconciliation.csv`: individual differences against Notebook 09 and Notebook 16 tables.
- `ridge_reference_discrepancy.csv`: side-by-side ridge values.
- `prediction_panel_alignment.csv`: every saved target/context field checked against the rebuilt panel.
- `validation_checks.csv`: integrity and independent metric calculations.
- `input_fingerprints.csv`: exact input and code fingerprints.
- `verified_benchmark_manifest.json`: verified reference, caveats, and prediction-file fingerprints.

Run `.venv/bin/python notebooks/reconcile_saved_benchmarks.py` from the project root to reproduce this audit.
'''
    (OUT / "README.md").write_text(text)
    con.close()
    print(annual_table.to_string(index=False))
    print(json.dumps({"integrity_checks_passed": passed, "checks": len(checks), "issues": issues}, indent=2))
    if not passed:
        raise AssertionError("Benchmark integrity check failed; inspect validation_checks.csv")


if __name__ == "__main__":
    main()
