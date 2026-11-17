# Spline residual challengers

## Result

This bounded experiment compares a pooled penalized cubic-spline residual model with the identical model plus a shrunk stock-specific residual effect. It uses the frozen 58-feature input set, five-year rolling history, previous-year validation, median20-relative log target, historical sample sizes, and the existing forecast clipping rule. All 20 verification checks passed.

| Candidate | Development RMSE | Change vs L2 | Development WAPE | Change vs L2 | Bias | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lgbm_residual_58_l2 | 0.46941 | +0.00000 | 35.45% | +0.00% | -14.40% | Reference |
| spline_residual_58_stock_effect | 0.48526 | +0.01584 | 36.33% | +0.87% | -16.09% | Fail |
| spline_residual_58 | 0.48535 | +0.01593 | 36.36% | +0.91% | -16.24% | Fail |
| rolling5_residual_auction_calendar_liquidity | 0.48791 | +0.01850 | 36.73% | +1.28% | -14.59% | Fail |

Development metrics average 2022-2024 annual values equally. The verified squared-error LightGBM is the current daily anchor. The gate requires at least 0.002 mean log-RMSE improvement, no annual deterioration above 0.002, no mean WAPE worsening, and a paired date-level log-MAE upper interval below zero.

## Annual metrics

| Year | Verified L2 RMSE | Pooled spline RMSE | Spline + stock RMSE | Verified L2 WAPE | Pooled spline WAPE | Spline + stock WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.51010 | 0.53119 | 0.53114 | 36.68% | 37.62% | 37.60% |
| 2023 | 0.46093 | 0.48204 | 0.48184 | 35.20% | 36.52% | 36.48% |
| 2024 | 0.43720 | 0.44281 | 0.44279 | 34.47% | 34.94% | 34.90% |
| 2025 | 0.43065 | 0.44550 | 0.44547 | 32.94% | 34.26% | 34.25% |

## Does the stock effect help the spline?

For development dates, the hierarchical candidate minus pooled-spline mean date-level log-MAE difference is -0.000009, with a 95% whole-date bootstrap interval of [-0.000067, +0.000045]. Negative favors the stock effect. This interval preserves same-date cross-sectional dependence but not serial dependence across dates.

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
