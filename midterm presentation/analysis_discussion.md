# Selected descriptive, analysis, and closing sections

The descriptive section uses three slides, planned for about two minutes:

1. Auction-size distribution and median auction volume across size deciles. Takeaway: account for stock scale and use share-scale as well as log-scale error measures.
2. Median auction share over time and calendar-event contrasts. Takeaway: evaluate forward in time and test known calendar information. The event contrasts are pooled and overlap.
3. A summary of eligible labels, annual target coverage, target skew, stock-size differences, and time/calendar variation.

Coverage appears as a summary statistic rather than a separate plot. Intraday checkpoint, sector, market-return, and field-missingness graphs remain in the full descriptive report. The main deck contains no local file citations or notebook references.

## Analysis section added

Ten main slides, planned for approximately eight and a quarter minutes:

1. Individual relationships with the target in shares. Show the mean and median auction across deciles of the trailing 20-day auction median and prior-day trading volume. Notes cover the broader variable families and distinguish scale, within-stock variation, and incremental information.
2. Prior-close market capitalization and relative quoted spread. Distinguish pooled stock scale from changes within a stock.
3. Prior-day signed and absolute returns. Include weak/non-monotonic curves rather than only strong associations.
4. Prior-day market and sector trading volume. Use date/sector-date group outcomes to avoid repeating common context as independent stock-level evidence.
5. One-family-at-a-time removal from the original controlled ridge audit. Recompute the summary from 2022–2024 annual results. Auction history and calendar dominate; liquidity is a small, metric-dependent increment. This audit has a different history/target specification from the final residual model.
6. Exact log1p residual around the trailing 20-day median, inverse transformation to shares, and a controlled direct-level versus residual comparison.
7. Specification comparison within ridge and the 0.002 practical tolerance supporting a 58-input ridge reference for controlled comparisons. Include the architecture challenge in narration and explain why the smaller auction/calendar model does not qualify.
8. Recent-median rule versus ridge by year, with one WAPE bar chart and a plain-language error example. Label 2025 confirmation separately. Headline mean annual development WAPE: 43.47% versus 36.80%.
9. Full-history actual and predicted 20-day rolling medians for the same JPM and CATO examples. Show 2015 actuals only, predictions beginning in 2016, and separate log scales. This visual smoothing differs from the model's prior-only median offset. Allocate 30 seconds and disclose that smoothing can conceal daily spikes.
10. Unsmoothed daily actual versus predicted volume for JPM in 2021 and CATO in 2017. Existing year selection uses the year nearest each stock's median annual log RMSE. Mark the largest absolute log errors and connect missed spikes to remaining challenges. Label different years/axis ranges and retrospective specification selection. Allocate 45 seconds, shortening narration for the individual-relationship, model-selection, and annual-performance slides by 15 seconds each to limit the analysis budget to eight and a quarter minutes including the additional individual-covariate slides.

One unnumbered appendix slide provides training, historical validation, and forward-evaluation log RMSE/WAPE. The forward-year axis does not imply identical date windows for the three series. Lower forward error alone cannot establish absence of overfitting. This slide is backup material outside the narration budget.

The extensive individual-variable review informs the economic narrative without placing every graph in the main deck. Its full atlas remains available for preparation. The new diagnostic work is retrospective and does not rewrite the historical model-selection sequence.

## Challenges and roadmap added

Two closing slides take approximately two and a half minutes:

- Slide 22: large-auction underprediction and missed spikes, weaker results for small stocks and limited histories, and the information boundary of the prior-close daily forecast. Preserve the distinction between observed forecasting errors and unverified economic causes.
- Slide 23: diagnose errors and refine how existing auction-history features adapt, test nonlinear models alongside ridge, test lagged VIX/rates and scheduled CPI/NFP indicators, and develop intraday updates from prepared checkpoints. Compare extensions with the frozen benchmark chronologically and seek additional periods unused in selection.

These incorporate the user's handwritten B/A&B items. The A-only rebalance, FOMC implied-probability, and pension ideas are outside this selected roadmap. New daily context and intraday models are future tests, not completed results. No assumption is made that the additional data have been sourced or improve performance. The 2025 year remains confirmation evidence already inspected during research.

The main deck now has twenty-three slides plus one backtest appendix slide and one literature reference slide. Roadmap narration is shortened from two minutes to 90 seconds. Following removal of the 40-second data-assembly slide, numbered speaker-note durations total 19 minutes 10 seconds, with 50 seconds for transitions and recording buffer.

Evidence remains limited to the first 13 notebooks. Stock-specific options-market analysis is outside the midterm scope.


## Review edits

Removed the standalone data-assembly slide and moved its essential details into optional notes under data sources. Removed the data-documentation reference slide and all D1/D2 citations from audience materials, retaining the literature references. Revised slide titles throughout to describe the topic or analysis. Findings and implications stay in the slide body. Current section ranges: background/data 1–8, descriptive statistics 9–11, analysis 12–21, challenges/roadmap 22–23, backtest appendix 24, literature references 25.


## Further review: scope, ridge status, and backtest interpretation

Generalize the target notation to any candidate predictors available at the forecast origin. Keep the empirical information boundary and distinguish proposed inputs from completed tests. Present ridge as one model built so far, with no final-model claim. The within-ridge specification comparison does not select a winner across all model classes.

The annual main slide compares two forecasting methods on common forward rows. Simplify it to WAPE only, define the recent-median rule, and explain absolute error per 100 actual shares. The appendix compares one ridge specification across training, historical validation, and later-year evaluation, using different samples/periods and retaining two metrics. It is a separate fitting diagnostic and cannot rule out overfitting from line ordering.

Acknowledge the large stock-example WAPEs openly. Add the same-row median-rule values (JPM 37.9% versus ridge 30.8%, CATO 56.6% versus ridge 46.4%). Improvement relative to the simple rule does not establish adequate absolute accuracy or execution value. Keep the timing and slide count unchanged.


## Further review: purpose, weights, and nonlinear work

Make the descriptive role of slides 9–10 explicit: distribution/scale and time/calendar patterns motivate the modeling and evaluation choices. Clarify on slide 18 that weights refer to training rows, with equal row influence or a two-year recency half-life. Keep the within-ridge comparison caveat because feature counts also differ. Slide 19 states that the median-rule forecast is the prior-only 20-day median itself. Add a distinct roadmap row for nonlinear models, such as boosted trees, to test thresholds and interactions on the same inputs and folds before testing new data. The subsequent covariate extension adds three slides and reallocates narration to stay within the 20-minute budget.


## Additional covariate slides

Add slides 13–15 immediately after the original two-variable relationship slide. Keep both strong and weak results, six prior-close inputs in total, and the target in shares. Interpret market/sector plots as summaries of group-average auction shares. These curves use existing 2015–2025 diagnostic results, extend economic interpretation, and do not alter the original model-selection chronology. The new slides take 35 seconds each. Condense descriptive narration to 40/40/30 seconds. Total narration becomes 19:10 plus a 50-second recording buffer, while quantitative model results remain unchanged.


## Recording-script conversion

The current `speaker_notes.md` replaces the earlier narration/Q&A outline with a continuous, conversational script for slides 1–23 and separate complete appendix scripts. It retains essential interpretation caveats in spoken form without reading every technical footnote. The earlier technical notes remain in `sources/speaker_reference_notes.md`. Use the current script's approximately 2,400 words and 130–135-word-per-minute speaking estimate for rehearsal; the historical per-slide timings above describe earlier drafts. No presentation slide or numerical result changes in this update.


## Current revision: economic framework and error diagnostics

Retain the 23-slide main sequence. Slide 4 provides economic hypotheses for recent activity, scale/liquidity, common conditions, and known calendar inputs while retaining the general predictor vector and availability rule. Slides 12–16 distinguish individual associations from incremental family information within the earlier controlled ridge audit. Slide 19 addresses forecast accuracy. Smoothing on slide 20 motivates the unsmoothed misses on slide 21; no economic cause is assigned to example spikes.

Replace the text-heavy challenges slide with size/history error panels on a common 2022–2024 sample. WAPE pools stock-day errors within each group; counts and the 0.33% share of under-20-history observations prevent overstating rare-group evidence. The new chart is a reaggregation of saved in-scope forecasts, checked against the original full-period diagnostic. The script describes the distinct aggregation from the mean annual headline. The prior-close information boundary leads into intraday updates in the roadmap narration. New stock-specific options and index-rebalancing work remains deferred until after the midterm.

The complete read-aloud script is approximately 2,450 spoken words, about 18–19 minutes at 130–135 words per minute before pauses. Earlier per-slide timing notes are historical. The cover includes Aakash Ramesh, Arjo Bhattacharya, Alisa Priyadarsini, and Ivan Sysoev.


## Current revision: calibration, group gains, and baseline-adjusted support

Add one main slide (20) showing mean-share calibration and predicted/actual ratios by forecast decile on 2022–2024 saved ridge predictions. The aggregate shortfall is 14.4%; all decile means are underpredicted. Explain the arithmetic-mean calibration question separately from log-space training, without assigning a unique cause. Any future correction must use earlier fitting data and forward evaluation. The roadmap now explicitly includes calibration tests.

Upgrade the existing group-error slide (now 23) to compare ridge with the exact saved median-rule definition on the same rows. Use the saved historical training fallbacks and verify annual baseline metrics against original results. Ridge improves all displayed groups, but short-history absolute errors remain large and the 0–4 baseline WAPE exceeds 100%. Keep counts and the 0.33% under-20 share visible.

Keep the new baseline-adjusted covariate study in appendix page 26 and a separate report. It examines log departures from the recent median for volume, cap, spread, and absolute return using development-year pairs, excluding missing stock medians. Economic interpretation remains descriptive. No extra individual-variable slide enters the main recording. The main deck is 24 slides with three appendix pages; the complete main narration is about 2,480 words, preserving the 20-minute target with rehearsal.


## Descriptive summary removed after review

Remove the former main slide 11 because it repeats the preceding descriptive graphs and later modeling decisions. Preserve 3.50 million eligible labels and 93.0%–99.0% annual observed-target coverage in main slide 7 and its script. Add a spoken transition after the time/calendar graph. The first covariate slide is now 11; subsequent pages shift back by one. The main recording covers 23 slides, with three appendix pages (24–26). No chart data or model results change.
