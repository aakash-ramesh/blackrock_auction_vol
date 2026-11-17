# Full-feature ridge with one-day log growth

For stock-day auction shares `A_t`, the target is `log1p(A_t) - log1p(B_t)`, where `B_t` is the preceding stock row's auction shares. An unavailable or nonpositive reference uses the fitting sample's median auction shares. Ridge estimates the growth component, and `expm1(log1p(B_t) + predicted_growth)` recovers shares before applying the common training-based forecast bounds.

The fit solves the penalized normal equations in NumPy with an unpenalized intercept and uniform stock-day weights. All four fresh fits use the exact historical tuning, validation, and refit rows from Notebook 21. Each fold fits preprocessing using its training rows and picks the ridge penalty by validation log RMSE after reconstruction. The 99 approved daily source features include the compact 58 plus 41 from returns and price, market state, sector state, security and corporate actions, and size and history. The encoded column count can vary by fold.

## Development results (2022–2024 equal-year means)

| Ridge specification | Log RMSE | WAPE | MAE shares | RMSE shares | Share R² | Bias |
|---|---:|---:|---:|---:|---:|---:|
| 58 features · 20-day median log residual | 0.48811 | 36.74% | 82,272 | 657,160 | 0.350 | -14.50% |
| 58 features · one-day log growth | 0.49098 | 37.43% | 83,831 | 659,832 | 0.344 | -14.80% |
| 99 features · 20-day median log residual | 0.48783 | 36.66% | 82,074 | 656,927 | 0.350 | -14.48% |
| 99 features · one-day log growth | 0.49049 | 37.40% | 83,746 | 659,286 | 0.345 | -14.28% |
| Frozen Notebook 09 ridge | 0.48764 | 36.80% | 82,399 | 658,440 | 0.347 | -14.38% |

The 99-feature growth model minus the matched current 58-feature fit was +0.00238 log RMSE and +0.66 percentage points of WAPE. Paired calendar-month resampling within years gave approximate 95% intervals of [+0.00164, +0.00307] for log RMSE and [+0.41, +0.89] points for WAPE. The bootstrap conditions on the fitted forecasts.

## 2025 confirmation

| Ridge specification | Log RMSE | WAPE | MAE shares | RMSE shares | Share R² | Bias |
|---|---:|---:|---:|---:|---:|---:|
| 58 features · 20-day median log residual | 0.44665 | 34.50% | 109,470 | 657,649 | 0.450 | -9.73% |
| 58 features · one-day log growth | 0.44860 | 35.00% | 111,046 | 664,597 | 0.438 | -10.50% |
| 99 features · 20-day median log residual | 0.44956 | 34.14% | 108,330 | 657,873 | 0.449 | -13.02% |
| 99 features · one-day log growth | 0.45146 | 34.71% | 110,129 | 664,865 | 0.437 | -13.75% |
| Frozen Notebook 09 ridge | 0.44512 | 34.56% | 109,663 | 658,547 | 0.448 | -9.62% |

The project has previously inspected 2025. The frozen Notebook 09 forecast is included as a historical reference on identical stock-days. The fresh 58-feature median-residual fit is the control for isolating changes to the target and features.

All 76 verification checks passed across 1,271,378 evaluation stock-days per specification. Checks cover historical sample identity, fold chronology, matching compact-model metrics, reconstruction from saved coefficients, and alignment with the frozen benchmark.

Run `.venv/bin/python -u notebooks/compare_full_ridge_growth.py` with Notebook 21's panel and sample files available. The [annual metrics](annual_metrics.csv), [validation scores](validation_grid.csv), saved coefficients and share forecasts are in this directory.
