# Step 1: local daily-panel reconstruction

Completed November 2, 2026 (America/New_York).

The original Notebook 05 was executed locally with only its output directory changed. Historical notebooks, source data, and `outputs/eda/` research artifacts were preserved. No model was fitted in this step.

## Result

All 23 additional validation checks passed. The original notebook's integrity assertions also passed, and all eight code cells completed without errors.

| Measure | Rebuilt result |
| --- | ---: |
| Panel rows | 3,594,404 |
| Securities | 2,149 |
| Trading dates | 2,766 |
| Eligible auction targets | 3,499,401 |
| Approved daily source features | 99 |
| Duplicate panel keys | 0 |
| Eligible targets on excluded anomaly date | 0 |
| Nonfinite nonmissing numeric feature values on eligible rows | 0 |

Annual eligible counts match the historical research in all 11 years. The complete feature catalog and the null counts for every approved feature match the saved Notebook 05 artifacts. Panel target values and observed-target flags match the source auction records, including missing values. CRSP and auction source keys are unique.

An independent pandas reconstruction checked the prior 20-row auction median, previous-row auction volume, previous-row daily trading volume, and cumulative prior observed-auction count. All four matched across 28,333 panel rows from 16 deterministically selected stocks. This sample includes the stocks' available pre-2015 CRSP history when constructing lags.

## Files

- `05_feature_engineering_rebuilt.ipynb`: executed notebook and its outputs.
- `enriched_daily_panel.parquet`: rebuilt panel (1,237,207,283 bytes).
- `validation_checks.csv`: all 23 checks and outcomes.
- `validation_summary.json`: machine-readable status and panel fingerprint.
- `annual_reconciliation.csv`: rebuilt annual sample counts.
- `missingness_reconciliation.csv`: comparison with the saved feature null counts.
- `independent_history_checks.csv`: independent history-feature checks.
- `source_inventory.csv`: sizes and SHA-256 fingerprints for all 31 required input files.
- `environment.json` and `requirements.lock.txt`: local software versions.
- Remaining CSV and PNG files: ordinary Notebook 05 outputs, redirected here.

## Data treatment preserved

Missing auction labels remain missing. The supervised eligibility rule excludes the documented August 12, 2019 anomaly. Daily histories end before the forecast date; the 20-row auction median ignores missing values inside the preceding 20 stock rows, rather than collecting the last 20 nonmissing auctions. Missing covariates remain missing in the panel: imputation, transformations, and predictor clipping belong inside historical model-fitting folds.

The panel's `sample_split` field is a legacy descriptive split. Subsequent modeling must use the explicit annual chronological folds, not treat this field as the current experimental protocol.

## Limits and follow-up

These checks establish reconstruction and selected feature-timing consistency. They are not a complete independent validation of all 99 feature formulas, upstream point-in-time availability, or source entitlements. The absent historical panel prevents an all-cell comparison with the original panel.

The original feature logic excludes the anomaly date from supervised labels but does not remove its observed source values before constructing subsequent auction-history features. This behavior was preserved for reproduction; whether to mask those historical values is a separate, explicit data-treatment decision before new target experiments.

Next: recompute metrics from the saved Notebook 09 forecasts and reconcile their ridge reference with Notebook 16. No growth-rate, calibration, or nonlinear refitting experiments have been run in this step.

## Reproduce

From the repository root, using its local virtual environment:

```sh
.venv/bin/python -m ipykernel install --prefix .venv --name auction-local
.venv/bin/python -u notebooks/reproduce_daily_panel.py
```

To rerun only the checks on an already rebuilt panel:

```sh
.venv/bin/python -u notebooks/reproduce_daily_panel.py --audit-only
```
