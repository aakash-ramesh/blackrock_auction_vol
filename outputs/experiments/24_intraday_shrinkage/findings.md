# Results of historical intraday correction shrinkage

**Retain the full 15:30 ridge update under the predefined adoption rule.** Shrinkage improves development proportional accuracy, especially in low-liquidity stocks, but introduces tradeoffs in WAPE and bias and does not improve 2025 log accuracy. This is useful evidence of time-varying correction quality, not a reason to discard the intraday signal.

## Main results

All development numbers are equal-year means for 2022–2024. The 2025 year was already inspected earlier and is separate confirmation, not a pristine holdout. No strength or variant is chosen from 2025 labels.

| Variant | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full intraday ridge | 0.459954 | 34.5919% | -11.800% | 0.409157 | 31.7815% |
| Global shrinkage | 0.458352 | 34.5433% | -12.558% | 0.409777 | 31.6436% |
| Month-end only | 0.459929 | 34.5861% | -11.820% | 0.409792 | 31.7788% |
| Low liquidity only | 0.456747 | 34.5981% | -11.877% | 0.410782 | 31.7977% |
| Month-end and liquidity | 0.456957 | 34.6031% | -11.885% | 0.410856 | 31.8032% |

## Why no variant was adopted

The global variant improves development log RMSE by 0.00160, below the 0.002 practical threshold, and its development log-MAE interval crosses zero. The month-end-only variant has negligible development improvement and its interval also crosses zero.

Low-liquidity shrinkage improves log RMSE in each development year, by 0.00321 on average, and its 20-date paired log-MAE interval is favorable. However, development WAPE worsens by only about **0.0061 percentage points**. The combined variant similarly worsens WAPE by about **0.0112 percentage points**. These are very small differences, but the rule frozen before fitting required no WAPE deterioration, so both fail. We did not relax that rule after seeing the results. The log-accuracy gains remain meaningful research findings rather than being dismissed as worthless.

The decision to retain the full model was saved before this run scored 2025 and fingerprinted to confirm it remained unchanged. In 2025 every shrinkage variant has worse log RMSE than full ridge, although global shrinkage improves WAPE from 31.7815% to 31.6436%. Low-liquidity and combined variants also have worse paired daily log MAE than full ridge in 2025 under the exploratory block intervals. This illustrates objective and temporal instability rather than supporting one permanent shrinkage factor.

## What happens to weak segments

In development, low-liquidity-only shrinkage improves that group's log RMSE from 0.52852 to 0.51930, beating the daily anchor's 0.52603. However, group WAPE increases from 36.54% to 36.68% and bias deepens from -11.59% to -13.69%. In 2025, that group's log RMSE instead worsens from 0.45695 to 0.46140. The development weakness is reduced on the proportional metric, but the remedy does not consistently generalize or improve all objectives.

Month-end-only shrinkage barely improves development month-end results: log RMSE moves from 0.66227 to 0.66186 and WAPE from 47.57% to 47.51%, still above the daily anchor's 0.66095 and 47.18%. In 2025 it worsens month-end log RMSE from 0.48433 to 0.49625. Month-end scarcity and changing effects remain unresolved. Segment comparisons are exploratory point estimates, not new specialist-selection tests.

## How it was done

The formula is log1p(daily forecast) plus strength times the difference between the bounded full ridge log forecast and the daily log forecast. Strength zero exactly reproduces daily; strength one exactly reproduces the full update. There is no extra clipping, so intermediate forecasts remain between the two endpoints. Missing checkpoints retain the exact daily forecast. Early-close days remain excluded.

The global strength uses historical validation only: 0.9 for 2022, 0.7 for 2023, 0.7 for 2024, and 0.8 for 2025. The grid is 0.0 to 1.0 in steps of 0.1. The objective is validation mean squared log error; exact ties prefer less change. Targeted variants keep full ridge outside the prespecified target. Low liquidity means lagged percentile below one third. Month-end uses the same disjoint definition as the earlier audit, excluding monthly and quarterly expiries.

The combined variant partitions month-end-only, low-liquidity-only and their intersection. A target cell with fewer than eight validation dates or 500 rows uses the global strength. In the initial fold, Q4 2021 supplies only three month-end dates, so those month-end strengths fall back to global. No coefficients are fitted from those three dates. The 2025 month-end validation sample has eleven dates, so even qualifying cells remain small. selected_strengths.csv records every cell, sample size, fallback and validation window.

For each test year, the original historical tuning ridge is reconstructed and predicts the saved validation rows. Its selected penalty is unchanged and its original validation loss is verified. That historical validation sample selects both the original penalty and the new strength: this is joint historical tuning, not an independent nested calibration split. All fitting precedes the evaluation year; final daily and full-ridge forecasts remain unchanged. The experiment addresses bounded shrinkage only, not amplification, different nonlinear transformations, new ridge fits or new data.

## Decision and uncertainty

Four variants and their boundaries, sparse-cell rules, objective and adoption gate were fixed in experiment_plan.json before fitting. The gate requires at least 0.002 mean development log-RMSE improvement versus full ridge, no annual deterioration above 0.002, no mean WAPE worsening, and a paired development log-MAE upper interval below zero; daily-reference guardrails are also applied. No variant passes.

Uncertainty uses 2,000 moving blocks of twenty consecutive trading dates, sampled within each year, with equal-year mean daily log-MAE differences. These intervals concern daily log MAE, not headline RMSE or WAPE, and are unadjusted for multiple comparisons. Hypotheses were motivated by inspected development diagnostics, and 2025 was previously examined, so this remains retrospective research despite the chronological fitting and frozen development decision.

## Artifacts and next research question

The folder retains the frozen plan and development decision, strengths and validation grid, annual and segment metrics, full row-level forecasts, input fingerprints, verification records and paired daily errors. The reader notebook displays saved results without refitting.

The next question is whether the desired operational objective permits a small WAPE tradeoff for proportional accuracy, and whether correction strength can be estimated more stably over time using only earlier out-of-time intraday forecasts. Do not deploy the low-liquidity rule or tune additional subgroup boundaries from these results. Retain full ridge as the research reference until an explicitly designed follow-up is evaluated.

All **152 checks passed** across **1,260,168 stock-days**. Checks cover historical inputs and chronology, original validation reproduction, exact reference metrics, alpha endpoints and grid, endpoint bounds, exact non-target behavior, checkpoint fallback, saved-parameter reconstruction, written forecasts, partition counts and the unchanged development decision.
