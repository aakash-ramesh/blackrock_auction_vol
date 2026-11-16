# Frozen daily versus 15 30 evaluation protocol

Protocol date November 13 2026

## Status and scope

This is a retrospective evaluation of the frozen experiment 19 forecasts, not a new holdout test and not a model search. Daily LightGBM is the baseline and the full 15:30 ridge correction is the single primary candidate. No shrinkage, feature subset, specialist or alternative checkpoint can enter the primary contrast. All model fitting, preprocessing and selection stay unchanged. Model specification and source artifacts are fingerprinted.

2022 through 2024 are development years already examined repeatedly. 2025 is previously inspected confirmation, never an untouched test. Results will not be used to declare a new statistical discovery or select another configuration. Input hashes must match the retained forecast manifest. The module must not silently drop invalid predictions.

## Universe and estimands

Score the identical eligible regular-session date and PERMNO pairs for both forecasts. Missing or invalid checkpoints retain the exact daily forecast. Early closes are excluded because 15:30 is after those auctions. Use the same exclusions and target eligibility as experiment 19.

Primary estimand is the equal-year mean of annual log1p RMSE, candidate minus baseline. Compute each annual RMSE from all stock-days in that year, then average years equally. Negative is better. Secondary estimands are equal-year annual WAPE difference and equal-year signed aggregate bias difference. WAPE is sum absolute share error divided by sum actual shares, not the mean of stock-day or daily percentage errors. Signed bias is forecast share sum divided by actual sum minus one. A positive bias difference is not inherently an improvement. Also report the difference in equal-year absolute annual bias as a diagnostic.

For 2025, these reduce to that year's statistics. Never pool across years while labeling a result equal-year. All metric intervals must recompute the same nonlinear statistics as their point estimates, including RMSE square roots, ratio denominators and absolute bias.

## Paired dependence-aware uncertainty

Use 5000 circular moving-block bootstrap replicates, resampling trading dates within each calendar year. A sampled date carries every stock and both forecasts together. Primary block length is 20 trading dates. Lengths 5 and 60 are prespecified sensitivity analyses, not opportunities to select a favorable interval. Each replicate samples enough consecutive circular date blocks to preserve the original number of dates per year, then recomputes each annual statistic and averages years equally.

Circular wrapping gives dates equal marginal inclusion but introduces an artificial adjacency between the last and first date of a year. Intervals assume sufficient within-year local stability and weak serial dependence; market regime change can violate those assumptions. Sensitivity across block lengths does not prove validity. These percentile 95 percent intervals condition on the fitted historical forecasts. They do not capture refitting, hyperparameter selection, repeated research decisions, arbitrary stock-universe sampling or unseen regime uncertainty. Bootstrap draws use a fixed seed and identical date weights across both forecasts, metrics and subgroup analyses.

Daily sufficient statistics may be used to make computation exact and efficient: stock-day count, squared log errors for each forecast, absolute share errors, actual share total and predicted share totals. All metric calculations must reconcile against row-level forecasts and the saved annual summaries.

## Predefined subgroup reporting

Use lagged size and liquidity percentile thirds with cutoffs one third and two thirds, plus explicit missing groups. History groups are fewer than 5 prior auctions, 5 through 19, and at least 20, plus missing. Disjoint events have precedence quarterly expiry, monthly expiry, month-end, regular. Additionally report disjoint month-end crossed with liquidity. No actual-volume tail is used as a forecast-time subgroup.

Report annual counts, distinct dates, actual share totals, metrics and paired differences for every observed group. Period subgroup summaries require representation in every constituent year. Pointwise subgroup intervals use the same full calendar date resampling, including dates with zero observations in a group. Suppress subgroup intervals if any year has fewer than 30 observed subgroup dates, fewer than 500 stock-days, or a nonpositive share total. Empty bootstrap subgroup denominators invalidate an interval rather than being silently discarded. Sparse counts remain visible.

Subgroup and secondary metric intervals are descriptive, unadjusted for multiplicity. They cannot support separate significance or specialist-model claims. Only the overall log-RMSE contrast is designated primary; the designation now does not undo prior selection. No p-value-based promotion will follow from retrospective subgroup results.

## Decision limits

Report estimates and intervals without changing the existing reference or declaring production readiness. The historical 0.002 log-RMSE practical threshold can be shown as context, not relabeled as an economically justified threshold. No WAPE noninferiority margin has been agreed. Noninferiority cannot be claimed until a margin is justified independently by the intended use and frozen before a new evaluation. Preserve any WAPE versus bias tradeoffs.

## Audit unused forward data before scoring

After building the historical evaluation, inspect local source inventories, date ranges and coverage counts only. Do not read forward auction volume values, generate forward predictions, fit against forward outcomes or score a new period. The earliest potential forward dates must be after December 31 2025, because all earlier years have appeared in training, development, descriptive work or confirmation.

Audit the presence of auction labels, CRSP daily history, 15:30 trading aggregates, market and sector inputs and calendar/identifier references. A future FOMC or calendar file alone does not establish forecast readiness. A source's maximum date does not prove continuous coverage, valid labels or point-in-time availability. Local presence cannot prove that outcomes were never inspected elsewhere. Require a provenance and prior-use attestation before calling any new period untouched.

Before a future test, freeze the evaluation start/end and minimum duration, annual refit schedule, training-only preprocessing and hyperparameter rules, checkpoint delivery cutoff, eligibility/fallback policy, primary metric and operational acceptance criteria. Reproduce the daily anchor and correction using earlier data only; every calibration layer must use earlier out-of-time forecasts. Log and lock forecasts before outcomes where feasible. Score once at the fixed end without optional stopping. If future data are absent, report the block explicitly and propose acquisition or prospective collection; do not manufacture a fresh holdout from inspected years.
