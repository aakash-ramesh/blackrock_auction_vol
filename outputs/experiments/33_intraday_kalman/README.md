# Sequential checkpoint and Kalman exploration

[Master notebook 11](../../../master_research/11_sequential_intraday_and_kalman_filters.ipynb) explains and executes the [research implementation](../../../notebooks/explore_intraday_kalman.py).

## Decision

Retain the incumbent tuned daily LightGBM plus annual 15:30 ridge. No tested sequential candidate passes the frozen development gate against both the matched latest-only ridge and the incumbent.

| Method | Development log RMSE | Development stock WAPE | 2025 log RMSE |
| --- | ---: | ---: | ---: |
| Incumbent daily + 15:30 ridge | 0.454705 | 33.9832% | 0.404183 |
| Matched latest-only ridge | 0.454767 | 33.9624% | 0.405170 |
| Multi-checkpoint ridge | 0.454808 | 33.9968% | 0.405436 |
| Ridge on checkpoint signals | 0.454762 | 33.9753% | 0.405646 |
| Correlated-error Kalman | 0.455299 | 34.0173% | 0.405699 |
| Innovation-clipped Kalman | 0.455969 | 34.0793% | 0.406803 |
| Independent-error Kalman | 0.475699 | 35.8844% | 0.425068 |
| Variance-inflated Kalman | 0.563178 | 44.0215% | 0.504872 |

Development averages annual 2022–2024 scores equally. Previously inspected 2025 is retrospective confirmation. All 1,260,168 eligible regular-session stock-days are retained across the four evaluation years.

## Protocol

The daily anchors are saved out-of-time tuned LightGBM forecasts. Components use the daily sources and information available at 10:30, noon, 14:00, 15:00 and 15:30, with prior-only historical normalization. Final auction labels do not enter the observation stream.

For evaluation year Y, select component ridge penalties on Q2 of Y−1 after fitting on earlier 2021-onward history. Refit components through June Y−1 and freeze them. Q3 supplies out-of-time measurement calibration; Q4 selects filter settings and signal-stack penalties. After selection, refit measurement statistics and stack coefficients on Q3+Q4, keeping the exact component means frozen. Every layer precedes Y.

This intentionally sacrifices the latest six months of component fitting relative to the incumbent, but keeps measurement calibration genuinely out of time and matched to the exact components evaluated. Latest-only and multi-checkpoint controls share the frozen window. Their close scores do not make their protocols identical to the incumbent's.

The latent state is the fixed terminal log residual relative to daily. Correlated filtering uses causal Cholesky whitening of calibrated measurement errors, with diagonal shrinkage. Independent-error, process-variance-inflation and innovation-clipping variants are bounded sensitivities. No state is carried between stock-days, no future-data smoothing is used, and actual auction outcomes only enter historical fitting or evaluation scoring.

Latest-only ridge falls back to daily when the latest checkpoint is missing. Sequential candidates retain earlier available observations. All-observations-missing prefixes return daily exactly. Complete-checkpoint sensitivity tables expose that difference.

## Uncertainty and verification

Independent-error Kalman's nominal 95% log bands cover about 88.3% of development outcomes at 15:30. Correlated Kalman covers about 95.9%, but its nominal 80% bands cover about 88.5%. Near-nominal coverage at one level therefore does not validate a Gaussian liquidity distribution for execution control.

Paired intervals recompute annual log RMSE, WAPE, signed bias and absolute annual bias in 5,000 circular 20-session block draws, then average years equally. They condition on fitted forecasts and omit refitting and repeated research-selection uncertainty. The gate requires at least 0.002 development log-RMSE improvement, no annual deterioration above 0.002, no mean WAPE worsening and a favorable direct paired log-RMSE interval against both references.

Checks include known scalar-filter arithmetic, equality to a batch Gaussian update with correlated/missing observations, ridge-system agreement with the original helper, forecast reconstruction from saved parameters, unique/common labels, chronological fitting boundaries, exact missing fallback, positive finite state variances and unchanged prefix estimates when future observation values and availability are perturbed. Notebook 11 independently reconstructs three prior-only history features for 16 stocks at all five times and reconciles noon/15:30 availability with the earlier audits.

## Artifacts

- `plan.json`: fixed design, grids and code/raw-panel/checkpoint/anchor fingerprints.
- `panels/`: thin prior-only checkpoint panels and their wide matched input panel.
- `samples/manifest_YEAR.csv`: ordered sample fingerprints, counts and chronological boundaries.
- `models/`: component preprocessing, coefficients, measurement calibration, selected filter settings and signal-stack coefficients.
- `component_validation_YEAR.csv` and `meta_validation_YEAR.csv`: all historical validation trials.
- `predictions/forecasts_YEAR.parquet`: complete prefix forecasts, measurements, missing flags and filter variances.
- `annual_metrics.csv`, `summary.csv`, `paired_intervals.csv`, `selection_gate.csv`: point scores, direct metric intervals and adoption decision.
- `uncertainty_diagnostics.csv`, `availability_sensitivity.csv`, `annual_segment_metrics.csv`: calibration, missingness and subgroup diagnostics.
- `decision.json`, `notebook_verification.json`, `notebook_environment.json`, `fitted_source_manifest.csv`, `independent_history_checks.csv`: decisions, environment and verification records.

From the project root, with the project model dependencies available, run `python -m notebooks.explore_intraday_kalman`. Existing complete forecasts are reconstructed and verified rather than refitted. A changed pinned plan requires an explicit new run/rebuild. Execution-control ideation remains deferred.
