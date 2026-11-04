# Step 2: saved benchmark reconciliation

All 81 integrity and metric-arithmetic checks passed. No model was fitted and no historical artifact was overwritten.

## Verified reference

All 196 reported annual metrics (7 models x 4 years x 7 metrics) from Notebook 09 reproduce within the declared tolerance. An independent SQL calculation also agrees with the project's Python metric helper for raw MAE, WAPE, log MAE, log RMSE, and aggregate bias. Spearman uses the project's average-rank correlation implementation.

The 1,271,378 saved prediction rows cover exactly the eligible rebuilt-panel keys for 2022-2025, with no duplicates or missing/extra rows. All 12 saved target/context fields match the rebuilt panel within numerical tolerance in every year. Predictions are finite and nonnegative; scoring drops no rows.

The 58-feature ridge remains the verified historical reference. Its 2022-2024 mean annual log RMSE is 0.48764270, WAPE is 36.8007%, and mean annual aggregate bias is -14.3778%. These average annual metrics give each year equal weight; they are not pooled metrics.

| Year | Rows | Verified log RMSE | Verified WAPE | Verified bias | Notebook 16 log RMSE | Notebook 16 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 328,634 | 0.53351 | 38.11% | -14.44% | 0.53450 | 38.00% |
| 2023 | 318,679 | 0.48366 | 36.97% | -14.16% | 0.48379 | 36.97% |
| 2024 | 316,212 | 0.44576 | 35.32% | -14.53% | 0.44544 | 35.23% |
| 2025 | 307,853 | 0.44512 | 34.56% | -9.62% | 0.44661 | 34.45% |

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
