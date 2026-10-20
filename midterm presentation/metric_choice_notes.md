# Error metrics for the target and ridge comparisons

These are diagnostic additions to the existing saved forecasts. No model is fitted and the provisional ridge reference is unchanged. All comparisons use the same 963,525 stock-days over development years 2022–2024, with 328,634, 318,679, and 316,212 observations respectively. The displayed values average each metric over the three years, giving years equal weight. They are not pooled-sample metrics.

## Definitions and rationale

- Log RMSE: square root of mean squared differences between `log(1 + prediction)` and `log(1 + actual shares)`. Its log scale compares stock scales more directly than raw-share squared error; squaring still penalizes large log misses. It was the existing development selection criterion, not a claim that it is universally the best execution objective.
- MAPE (%): `100 × mean(abs(prediction − actual) / actual)`.
- MSPE: `mean(((prediction − actual) / actual)^2)`, displayed as an unscaled squared ratio. Multiplying by 10,000 would express squared percentage-point units, which are not used in the tables.
- WAPE: `sum(abs(prediction − actual)) / sum(actual)`, retained as a complementary measure of share-volume error. Unlike MAPE, it weights absolute percentage errors in proportion to actual shares.
- All evaluated actuals are positive (minimum one share). MAPE/MSPE would be undefined at zero; there are no such rows to drop here. Missing targets remain missing upstream.
- MAPE and MSPE divide each stock-day error by that day’s actual shares, making small denominators influential. Metric-dependent rankings should be reported rather than used to declare a final model.

## Mean annual metrics, 2022–2024

| Saved specification | Log RMSE | MAPE (%) | MSPE (ratio squared) | WAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| rolling5_log_full | 0.49609 | 37.49 | 2.773 | 37.17 |
| rolling5_residual_full | 0.48710 | 34.68 | 1.483 | 36.69 |
| rolling5_residual_recent_full | 0.48677 | 34.82 | 1.171 | 36.73 |
| rolling5_residual_auction_calendar_liquidity | 0.48764 | 34.59 | 1.325 | 36.80 |
| rolling5_residual_auction_calendar | 0.49059 | 35.26 | 1.278 | 36.90 |

## Interpretation

The controlled direct-log versus residual comparison holds the five-year window, 99 inputs, and equal observation weights fixed. All three displayed measures improve with the residual formulation: log RMSE 0.49609→0.48710; MAPE 37.49→34.68%; MSPE 2.773→1.483. These are saved forecast comparisons, not new fits.

Among the three specifications on the ridge slide, the 58-input reference has the lowest MAPE (34.59%). The recently weighted 99-input model has the lowest log RMSE (0.48677) and MSPE (1.171). The 42-input model has lower MSPE than the 58-input model but worse log RMSE and MAPE. Thus the reference does not dominate every metric.

## Low-denominator sensitivity

There are 2,377 actual auctions of at most 100 shares, 0.2467% of development stock-days. In the pooled concentration diagnostic, these observations contribute 75.2% of squared relative errors for the direct-log model and 42.2% for the 58-input ridge reference. This concentration uses pooled sums, unlike the equal-year averages in the comparison tables. It illustrates sensitivity to small auctions without implying that they should be removed or that relative errors are unimportant.

## Reproduction

Run `build_metric_comparison.py`. It reads saved 09 forecast files, reproduces saved log RMSE and WAPE by year, and exports annual metrics, summaries, denominator sensitivity, and definitions under `assets/`. It also checks CRSP counts for the common 2015–2025 source-table window. No earlier research artifact is overwritten.
