# Step 3: compact nonlinear reproduction

## Scope and result

The original Notebook 16's 58-feature squared-error (L2) and Huber LightGBM models and fixed 50/50 log blends were rerun against the verified saved ridge forecasts. All 60 forecast verification checks passed. The run saved 1,271,378 stock-day rows across four years, with a prediction from every candidate on every row. Historical source data, notebooks, and `outputs/eda/` results were preserved.

| Candidate | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE | Original gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lgbm_residual_58_l2 | 0.46941 | 35.45% | -14.40% | 0.43065 | 32.94% | Pass |
| blend50_residual_58_l2 | 0.46963 | 35.50% | -14.84% | 0.43040 | 33.11% | Pass |
| blend50_residual_58_huber | 0.47139 | 35.45% | -15.43% | 0.43087 | 33.08% | Pass |
| lgbm_residual_58_huber | 0.47208 | 35.29% | -15.69% | 0.43209 | 32.94% | Pass |
| rolling5_residual_auction_calendar_liquidity | 0.48764 | 36.80% | -14.38% | 0.44512 | 34.56% | Reference |

Development metrics average annual 2022-2024 values with equal year weight. The 2025 results are confirmation evidence already inspected during research, not a fresh holdout. Bias is forecast share total divided by actual share total minus one; averaging annual bias differs from pooling all rows.

## Annual comparisons

| Year | Ridge log RMSE | L2 log RMSE | Huber log RMSE | Ridge WAPE | L2 WAPE | Huber WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.53351 | 0.51010 | 0.51469 | 38.11% | 36.68% | 36.53% |
| 2023 | 0.48366 | 0.46093 | 0.46313 | 36.97% | 35.20% | 35.03% |
| 2024 | 0.44576 | 0.43720 | 0.43842 | 35.32% | 34.47% | 34.33% |
| 2025 | 0.44512 | 0.43065 | 0.43209 | 34.56% | 32.94% | 32.94% |

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

| Segment | Ridge log RMSE | L2 log RMSE | Ridge WAPE | L2 WAPE | Ridge bias | L2 bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 0.6055 | 0.5800 | 46.37% | 44.69% | -27.69% | -23.70% |
| cold_0_4 | 1.4172 | 1.5097 | 92.19% | 85.80% | -55.93% | -65.74% |
| top_1pct | 1.3219 | 1.1463 | 55.13% | 52.71% | -43.25% | -46.14% |
| quarterly_expiry | 0.8078 | 0.7679 | 54.85% | 52.83% | -27.59% | -33.11% |
| month_end | 0.6862 | 0.6559 | 50.24% | 47.03% | -27.05% | -28.32% |

These are pooled 2022-2024 segment diagnostics, unlike the equal-year averages above. The L2 model improves log RMSE and WAPE for small stocks, the top actual-volume tail, and the displayed event groups. However, top-tail and expiry bias worsen, and cold-start log RMSE worsens even though its WAPE improves. Only 758 development rows have fewer than five prior observed auctions. Actual top-tail membership is an ex-post diagnostic, not a usable forecast input. Better average accuracy does not establish better calibration in every group.

## Paired uncertainty and original replacement rule

| Candidate | Mean date-level log-MAE difference vs ridge | 95% interval |
| --- | ---: | --- |
| lgbm_residual_58_l2 | -0.006703 | [-0.011754, -0.002398] |
| blend50_residual_58_l2 | -0.009535 | [-0.012796, -0.006842] |
| lgbm_residual_58_huber | -0.008204 | [-0.013217, -0.004254] |
| blend50_residual_58_huber | -0.010131 | [-0.013217, -0.007623] |

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
