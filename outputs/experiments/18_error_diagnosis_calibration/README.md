# Error diagnosis and historical calibration

No correction passes the prespecified accuracy/calibration gate; retain the uncalibrated forecast as the main reference.

All 87 checks passed for 1,271,378 evaluation stock-days. The verified 2022-2025 base forecasts and models were preserved. A prerequisite 2021 model was tuned on 2020, fitted through 2020, and used only to obtain earlier out-of-time calibration evidence.

| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| uncalibrated | 0.46941 | 35.45% | -14.40% | 0.43065 | 32.94% | Reference |
| global_share_ratio | 0.49292 | 39.07% | 1.16% | 0.46192 | 37.48% | Fail |
| global_log_shift | 0.46921 | 35.53% | -13.37% | 0.43143 | 33.15% | Fail |
| global_smearing | 0.55687 | 47.63% | 16.60% | 0.45994 | 37.22% | Fail |
| shrunk_decile_ratio | 0.50474 | 39.08% | 1.16% | 0.45893 | 37.62% | Fail |

Development figures average annual 2022-2024 metrics, giving years equal weight. The previously inspected 2025 confirmation period is reported separately and does not determine the adoption gate.

## Residual findings

- The largest 1% of absolute-share-error rows contribute 35.65% of total development absolute share error. These are ex-post error groups, not usable predictors.
- The top 1% of actual auction volumes contribute 31.82% of absolute share error.
- Fewer-than-20-history rows contribute 0.29% of absolute share error. Large relative cold-start errors should not be mistaken for the main source of universe-wide share error.
- The upper forecast bound removes shares equal to 4.22% of the observed net development volume shortfall. This is a mechanical bound-effect diagnostic, not a claim about causal economic error drivers or a tested change in clipping policy.
- `segment_diagnostics.csv` separates stock size, history, forecast-size deciles, ex-post actual tails and disjoint calendar groups. Event precedence is early close, quarterly expiry, monthly expiry, month-end, regular.
- `largest_share_errors.csv` and `largest_log_errors.csv` contain different failure rankings with known calendar and lagged-market context. Those annotations do not establish why an error occurred.
- `stock_error_summary.csv` and `date_error_summary.csv` distinguish persistent stock-level bias from common-date misses. The stock table retains sparse stocks with their counts; no stock-specific correction was fitted.

## Calibration design

For evaluation year Y, all parameters are fitted on the full Y-1 out-of-time forecast sample. No correction uses the labels from the year it evaluates. Each historical forecast's base model was fitted before that forecast year. The common model specification was selected retrospectively in earlier research, so this chronology is not a claim of untouched specification validation.

Four fixed corrections were tested: global share-total scaling; a shift by mean historical log residual; mean exponential-residual smearing; and ten historical forecast-size bins with ratio factors shrunk toward the global factor using n/(n+10000). Historical forecast decile thresholds are retained when scoring later dates. No calibration parameters, bounds, shrinkage strengths, or extra candidates were tuned on evaluation results.

The underlying base-model clipping remains unchanged. Corrections apply to its saved forecasts without a second clipping step; otherwise the intended correction would be partly reversed. The unclipped model is reconstructed only to diagnose the mechanical effect of existing bounds, not introduced as an alternative forecast.

The gate requires lower mean absolute annual aggregate bias, no mean annual WAPE worsening, no more than 0.002 deterioration in mean log RMSE, no annual WAPE deterioration above 0.002, and an upper paired development daily-WAPE interval below zero. Bias alone cannot establish a better forecast. The date-level bootstrap preserves cross-sectional dependence but not serial dependence, and does not adjust for multiple comparisons. Daily bootstrap weighting differs from the equal-year headline aggregation.

## Interpretation

Near-zero average log residual and negative raw-share volume bias can coexist: predicting a typical log volume and predicting arithmetic expected shares are different objectives in a skewed distribution. Correcting total-volume bias may increase absolute or log errors. This experiment measures that tradeoff; it does not assign the shortfall entirely to retransformation, structural growth, or clipping.

## Outputs

- `annual_metrics.csv`, `calibration_comparison.csv`, `paired_bootstrap.csv`: annual results, adoption criteria and uncertainty.
- `parameters/`, `calibration_parameter_summary.csv`: prior-year fitting windows and exact correction factors.
- `predictions/`: full actuals, fixed base forecasts, four corrections, and diagnostic context.
- `historical_2021/`: prerequisite out-of-time model, samples, predictions and provenance.
- `segment_diagnostics.csv`, `error_concentration.csv`, `clipping_diagnostics.csv`: error structure.
- `largest_share_errors.csv`, `largest_log_errors.csv`, `stock_error_summary.csv`, `date_error_summary.csv`: detailed audit tables.
- `validation_checks.csv`, `input_fingerprints.csv`, `experiment_plan.json`, `experiment_summary.json`: chronology, exact-input and forecast checks.

Run `.venv/bin/python notebooks/diagnose_and_calibrate.py` with the step-03 OpenMP environment. Append `--summarize` to rebuild diagnostics from existing forecasts without model fitting.
