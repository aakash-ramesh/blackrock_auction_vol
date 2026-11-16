# Frozen daily versus 15 30 evaluation findings

The frozen 15:30 ridge improves historical log RMSE, WAPE and absolute annual aggregate bias versus the frozen daily LightGBM anchor. Direct metric-matched date-block intervals remain favorable in development and previously inspected 2025 confirmation. **This is retrospective evidence conditional on the fitted forecasts, not a fresh holdout assessment or production certification. No new period was scored.**

## Primary 20 trading date block results

All differences below are intraday minus daily. Negative is favorable for error measures and absolute bias. Signed bias is a directional quantity: positive indicates an upward shift, not automatically an improvement. Here both forecasts underpredict, so the upward shift accompanies reduced absolute annual bias. WAPE and bias differences and intervals are expressed in percentage points.

| Period | Metric | Daily | Intraday | Difference | 95 percent interval for difference |
| --- | --- | ---: | ---: | ---: | --- |
| development | Log RMSE | 0.46949 | 0.45995 | -0.00954 | [-0.01135, -0.00781] |
| development | WAPE pp | 35.46758 | 34.59194 | -0.87563 | [-1.03511, -0.71696] |
| development | Signed bias pp | -14.46838 | -11.80007 | +2.66831 | [+2.08103, +3.28041] |
| development | Absolute annual bias pp | 14.46838 | 11.80007 | -2.66831 | [-3.28041, -2.08103] |
| confirmation | Log RMSE | 0.42901 | 0.40916 | -0.01986 | [-0.02287, -0.01609] |
| confirmation | WAPE pp | 32.82976 | 31.78146 | -1.04831 | [-1.47897, -0.64252] |
| confirmation | Signed bias pp | -9.00034 | -6.91064 | +2.08970 | [+1.06261, +3.12116] |
| confirmation | Absolute annual bias pp | 9.00034 | 6.91064 | -2.08970 | [-3.12116, -1.06261] |

Development statistics are equal-year averages of annual metrics for 2022–2024. Confirmation is 2025 alone. They are not pooled metrics or averages of date-level percentage errors. For absolute annual bias, take the absolute bias within each year and then average, rather than taking the absolute value of the mean signed bias.

## What changed statistically

Earlier intraday intervals addressed log MAE, while the headline metric was log RMSE. This module recomputes log RMSE itself in every bootstrap replicate. WAPE and bias also recompute share-total ratios using each replicate's actual-volume denominator. Stock-days within each date stay together, and the same dates and weights are used for both forecasts and all metrics. Resampling runs separately within years before equal-year aggregation.

There are 5000 circular moving-block replicates at each length. The prespecified primary length is 20 trading dates; 5 and 60 are sensitivity analyses. Overall log-RMSE and WAPE intervals favor intraday at all three lengths in both periods. No favorable block length was selected after scoring.

Circular wrapping makes date inclusion uniform but introduces artificial adjacency between each year's end and start. Inference relies on within-year stability and limited dependence; it does not guarantee coverage under structural changes. These conditional intervals omit parameter refitting, tuning and repeated research-selection uncertainty. They must not be presented as a new independent test after the many earlier experiments. Secondary and subgroup intervals are pointwise, unadjusted for multiplicity. No p-values, noninferiority or production promotion are declared.

## Subgroup reporting

Annual counts, distinct dates, actual share totals and metrics are retained for predefined size thirds, liquidity thirds, history depth, disjoint calendar events, and month-end crossed with liquidity. Explicit missing groups are included, even when empty. Period summaries require all constituent years; no year is silently dropped. Actual-volume tails are excluded from forecast-time subgroup definitions.

Intervals are suppressed if any year has fewer than 30 subgroup dates, fewer than 500 observations, or an invalid share denominator. The full trading calendar is resampled for eligible subgroups, including dates with zero subgroup observations. An invalid replicate suppresses the interval rather than being discarded. Expiry and month-end groups can have thousands of stocks but too few dates for defensible subgroup intervals; their point estimates and sample counts remain visible. Read metric_intervals.csv status/reason fields and annual_subgroup_metrics.csv before interpreting a subgroup.

All subgroup intervals are exploratory, not evidence for a new routing or specialist rule. The main frozen contrast is full intraday ridge versus daily; shrinkage and feature variants are not reopened here.

## Unused period availability

The local source audit found **no ready forward period beyond December 31 2025**. All six core date-bearing components—auction records, CRSP inputs, 15:30 aggregates, market context, sector context and calendar—end on that date. The 2026 FOMC filename does not supply auction-model coverage. The audit inspected source schemas, dates, counts and filenames, without reading forward auction volumes or generating/scoring forward forecasts. Local presence alone cannot prove non-use elsewhere; an attestation and provenance record are required. See unused_period_audit.md and the exact per-file date queries in forward_source_file_inventory.csv.

## Next action before any new score

Acquire compatible post-2025 source data or begin prospective collection. Before examining outcomes, freeze a fixed evaluation window and duration, the annual training and inner historical tuning procedures, eligibility and checkpoint-delivery rules, the primary metric and an independently justified WAPE acceptance/noninferiority margin if needed. Fit every layer only on earlier information and retain out-of-time inputs for stacked models. Record prior-use attestation and lock forecast snapshots. Score once at the fixed end without optional stopping. A chronological historical reanalysis cannot make already inspected years untouched again.

## Reproduction and artifacts

From the repository root, run:

```sh
python notebooks/evaluate_frozen_intraday.py
python notebooks/audit_unused_intraday_period.py
python notebooks/summarize_frozen_intraday_evaluation.py
```

The existing local project environment is sufficient. No model fitting or new external dependency is needed. Large historical row-level forecasts under experiment 19 must be present or regenerated using that experiment's documented pipeline. The reader notebook displays saved results.

- evaluation_protocol.md defines estimands, bootstrap, subgroup rules and forward scoring restrictions.
- annual_subgroup_metrics.csv and period_metric_differences.csv retain counts and matched point estimates.
- metric_intervals.csv contains direct metric intervals, primary/sensitivity labels and sparse-group suppressions.
- daily_sufficient_statistics.csv and bootstrap_weight_fingerprints.csv preserve the computation and resampling provenance.
- unused_period_audit.json, unused_period_audit.md and forward_source_coverage.csv record local availability and the unresolved provenance requirement.
- validation_checks.csv and input_fingerprints.csv retain verification evidence.

All **94 checks passed** across **1,260,168 stock-days**. Checks include unequal-denominator synthetic formulas, root and year weighting, paired resampling arithmetic and determinism, source hashes, complete/unique keys, prediction validity, exact fallback, annual reconciliation, exhaustive date-level partitions, finite bootstrap metrics and the unchanged protocol fingerprint. The source audit inventoried **57 local Parquet files**; no forward period was scored.
