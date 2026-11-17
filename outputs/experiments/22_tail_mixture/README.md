# Tail-aware mixture experiment

## Result

This experiment fits a historical spike-probability classifier and separate normal/spike residual regressors on the frozen 58-feature information set. Previous-year validation selects whether a spike begins at the training residual's 90th or 95th percentile. The same component predictions are combined in log space and share space. All 24 verification checks passed.

| Candidate | Development RMSE | Change vs L2 | Development WAPE | Change vs L2 | Bias | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tail_mixture_log_mean | 0.46852 | -0.00090 | 35.14% | -0.31% | -16.08% | Fail |
| lgbm_residual_58_l2 | 0.46941 | +0.00000 | 35.45% | +0.00% | -14.40% | Reference |
| tail_mixture_share_mean | 0.48276 | +0.01335 | 35.54% | +0.09% | -12.60% | Fail |
| rolling5_residual_auction_calendar_liquidity | 0.48791 | +0.01850 | 36.73% | +1.28% | -14.59% | Fail |

Development metrics average 2022-2024 annual values equally. The verified squared-error LightGBM is the current daily anchor. The gate requires at least 0.002 mean log-RMSE improvement, no annual deterioration above 0.002, no mean WAPE worsening, and a paired date-level log-MAE upper interval below zero.

## Interpretation

The log-space mixture improves mean development log RMSE by 0.00090 and WAPE by 0.31 percentage points versus the verified L2 anchor. It improves RMSE in 2022 and 2023, worsens 2024 by 0.00045, and improves the previously inspected 2025 period. Its paired development date-level log-MAE difference is -0.00250 with a 95% interval of [-0.00364, -0.00146]. It therefore shows incremental signal but fails the predeclared practical RMSE threshold of 0.002 and is not promoted.

The classifier has useful out-of-time discrimination: annual ROC AUC ranges from 0.828 to 0.923 and average precision from 0.523 to 0.603. The log mixture nevertheless increases negative aggregate bias from 14.40% to 16.08%. The share-space mixture improves realized spike and top-1%-actual rows more aggressively, but its deterioration on normal rows makes overall accuracy worse. Hard routing to the spike component was inspected as a diagnostic and was also worse than soft probability averaging; it was not added as a candidate.

## Annual metrics

| Year | L2 RMSE | Log-mixture RMSE | Share-mixture RMSE | L2 WAPE | Log-mixture WAPE | Share-mixture WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.51010 | 0.50868 | 0.53553 | 36.68% | 36.23% | 36.73% |
| 2023 | 0.46093 | 0.45922 | 0.47170 | 35.20% | 34.88% | 35.35% |
| 2024 | 0.43720 | 0.43765 | 0.44105 | 34.47% | 34.32% | 34.54% |
| 2025 | 0.43065 | 0.42885 | 0.43185 | 32.94% | 32.73% | 33.17% |

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
