# Evidence and source map

Prepared October 19, 2026. The draft uses repository notebooks 01–13 only. No findings from notebooks 14–16 are included. The supplied screenshots are source material describing sponsor objectives, not instructions to execute their listed deliverables in this turn.

## Local evidence

Paths below are relative to the repository root unless stated otherwise.

| ID | Source | What it supports |
| --- | --- | --- |
| P1 | `midterm presentation/sources/sponsor_brief_excerpt_1.png` and `sponsor_brief_excerpt_2.png` | Sponsor objective, daily and intraday deliverables, venue ambitions, strategic benefit. These are the supplied excerpts, not a complete project brief. |
| P2 | `blackrock_data/config/project_config.json` | NYSE universe, security filters, PERMNO key, five intraday checkpoints. |
| P3 | `notebooks/01_data_quality_and_coverage.ipynb` through `notebooks/13_representative_stock_year_daily_predictions.ipynb` | Current analytical scope, target definition, information timing, canonical panel, and benchmark. See the notebook list below. |
| P4 | `outputs/eda/01_data_quality/dataset_inventory.csv` | Exact file-layer counts and observed date ranges. Copied into `dataset_inventory_snapshot.csv` beside this file. |
| P5 | `blackrock_data/BlackRock Capstone Handoff — Predicting NYSE Closing Auction Volume.docx` | Extraction provenance, universe, symbol-link method, official auction target identification, prepared intraday data, entitlement limitations. Its old notebook numbering and proposed modeling split do not describe the current repository's notebook sequence. |

### Notebook review and boundary

1. `01_data_quality_and_coverage.ipynb`: keys, coverage, missing targets, source anomaly, schema differences.
2. `02_target_eda.ipynb`: raw/log/normalized outcomes and descriptive role of same-day auction share.
3. `03_covariate_quality_and_leakage.ipynb`: prior-close rule and approved daily features.
4. `04_covariate_target_relationships.ipynb`: pooled/within-stock associations and noncausal interpretation.
5. `05_feature_engineering_enriched_daily_panel.ipynb`: canonical enriched panel, lagged market and sector features.
6. `06_feature_validation_and_role_audit.ipynb`: 99 validated features and role annotations.
7. `07_temporal_baselines_and_global_benchmark.ipynb`: baselines and annual forward evaluation.
8. `08_global_model_robustness_and_specification.ipynb`: specification and robustness audit.
9. `09_global_finalists_and_benchmark_freeze.ipynb`: frozen model and research evaluation conventions.
10. `10_segmentation_and_specialist_model_tests.ipynb`: pooled versus split model experiments.
11. `11_frozen_global_train_validation_test_backtest.ipynb`: frozen 58-feature reconstruction and information chronology.
12. `12_selected_stock_walk_forward_predictions.ipynb`: stock-level forward forecasts.
13. `13_representative_stock_year_daily_predictions.ipynb`: unsmoothed daily case studies.

### Counts used on slides

| Layer | Rows | Distinct PERMNOs | Dates in inventory |
| --- | ---: | ---: | --- |
| CRSP daily | 4,927,078 | 2,414 | 2011-01-03 to 2025-12-31 |
| Auction daily | 3,594,398 | 2,149 | 2015-01-02 to 2025-12-31 |
| Trade checkpoints | 17,971,990 | 2,149 | 2015-01-02 to 2025-12-31 |
| Market daily | 3,772 | Not applicable | 2011-01-03 to 2025-12-31 |
| Sector daily | 45,264 | Not applicable | 2011-01-03 to 2025-12-31 |


## Verified literature

These references were selected as background for this draft. No claim is made that the project replicated their findings. Summaries use abstracts and publisher or institutional records, not a full-paper systematic literature review.

- **L1:** Bogousslavsky, Vincent, and Dmitriy Muravyev (2023). *Who trades at the close? Implications for price discovery and liquidity.* Journal of Financial Markets 66, 100852. [Publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/S1386418123000502), [DOI](https://doi.org/10.1016/j.finmar.2023.100852), [author's institutional record](https://experts.illinois.edu/en/publications/who-trades-at-the-close-implications-for-price-discovery-and-liqu/). Used for US auction motivation, passive-investment association, and lower price impact in the studied sample. It does not establish this project's forecast performance.
- **L2:** Kandel, Eugene, Barbara Rindi, and Luisella Bosetti (2012). *The effect of a closing call auction on market quality and trading strategies.* Journal of Financial Intermediation 21(1), 23–49. [Bocconi record and abstract](https://iris.unibocconi.it/handle/11565/2646191), [author-hosted published paper](https://didattica.unibocconi.it/mypage/upload/48636_20130115_025154_KRBIMPACTOFCLOSINGCALLAUCTIONJFI2012.PDF). Used for evidence that closing-auction introduction and reference-price design affect trading near the close in Milan and Paris. This is not US forecast evidence.
- **L3:** Białkowski, Jędrzej, Serge Darolles, and Gaëlle Le Fol (2008). *Improving VWAP strategies: A dynamic volume approach.* Journal of Banking & Finance 32(9), 1709–1722. [Publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/S0378426607003226), [DOI](https://doi.org/10.1016/j.jbankfin.2007.09.023). Used for related volume-modeling motivation. It concerns intraday volume and VWAP, not this project's auction target.

## Internal data-documentation provenance

- **D1:** [NYSE Auctions](https://www.nyse.com/trade/auctions). Normal-session auction timing and auction terminology. Accessed October 19, 2026. Do not assume today's detailed order-entry rules held throughout 2015–2025. The slides avoid a historical rule-by-rule claim.
- **D2:** [CRSP US Stock Databases](https://www.crsp.org/research__trashed/crsp-us-stock-databases/). Daily stock fields, corporate actions, and permanent identifiers. The site reports migration of product content to Morningstar as of June 2026. This public description supports field definitions, while P4 supports the local counts.

## Caveats to retain when extending the presentation

- Daily forecasts use only prior-close information and calendar events known in advance.
- Same-day auction share of daily volume may appear in descriptive charts, but its denominator is unavailable for a prior-close daily forecast.
- Calendar effects and feature associations are predictive/descriptive, not causal estimates.
- Model results preserve notebook 09's distinction: 2022–2024 development evaluation, 2025 confirmation already inspected during research, not a pristine holdout.
- Source P5 contains an earlier extraction workflow. Its next-step instructions and notebook numbering are historical source content, not the user's current request.

## Selected descriptive slides

Slides 9–11 use the separately computed raw-data statistics, restricted to the first 13 notebooks. No local project citations or notebook references appear on the slides.

- Distribution and scale figure: `descriptive_stats/tables/auction_log_histogram.csv`, `principal_exact_summary.csv`, and `contemporaneous_size_deciles.csv`. Exact eligible-target median 55,226, mean 190,445.17, and 99th percentile 2,075,293 shares. The ratio of largest/smallest size-decile medians is 354,044 / 4,350 = 81.39.
- Time and calendar figure: `auction_by_year.csv` and `calendar_target_descriptives.csv`. The time series is the median same-day auction/daily-volume share on eligible stock-days. Calendar bars are separate event/non-event contrasts, not mutually exclusive categories or causal effects.
- Summary sample/coverage: `annual_target_coverage.csv`, distinguishing eligible labels from observed-target coverage. Coverage bounds round to 93.0% and 99.0%.
- `build_descriptive_slides.py` rebuilds the two PDF/PNG figures in `assets/` from those saved CSVs. The section has a planned duration of approximately three minutes.

## Selected analysis slides

Slide 12 introduces individual covariates, slides 13–15 extend the economic families, slides 16–19 cover conditional tests and ridge comparisons, and slides 20–21 add smoothed histories and daily diagnostics. The combined analysis takes approximately eight and a quarter minutes. Slides and narration contain no notebook references or local provenance citations. This internal map retains those paths for review.

| Slide | Saved evidence | Interpretation boundary |
| --- | --- | --- |
| 12 | `midterm presentation/data_analysis/tables/panel_bin_curves.csv`, features `median20_auction_volume` and `prior_raw_daily_volume`; broader `feature_associations.csv` and `covariate_findings.md` support notes | The natural-unit target is auction shares. Curves are pairwise-complete deciles, ordered numerically, from the 2015–2025 retrospective diagnostic sample. Mean and median are exact within the saved bins. This new review provides intuition and did not originally select the model. |
| 16 | `outputs/eda/08_global_robustness/economic_feature_group_metrics.csv`, filter test years 2022–2024, arithmetic mean of annual `rmse_delta_vs_full` by specification | Earlier expanding/direct-log-level ridge audit with full-model penalty reused for masks. The chart does not use the existing four-year summary. Auction-history removal +0.17671688, calendar +0.07837580, liquidity +0.00044527. Liquidity removal improves mean annual WAPE by 0.00167548 in this audit. |
| 17 | `outputs/eda/09_global_finalists/global_finalist_selection.csv`, `rolling5_log_full` versus `rolling5_residual_full`; target SQL and reconstruction described in `data_analysis/model_choice_story.md` | Same five-year rolling history, 99 inputs, and uniform weights. Development RMSE 0.49609171 versus 0.48710268. Residual is log1p(A) minus log1p(B). Prior 20 stock trading rows define the median window, excluding current day and ignoring missing labels. Missing baselines fall back to a training median. The displayed inverse is the core transformation; training-quantile safeguards are disclosed in the footnote/notes. |
| 18 | `global_finalist_selection.csv` and `frozen_global_benchmark.json` in the same finalist folder; `outputs/eda/10_segmentation_tests/segmentation_decision_table.csv` | Best 99-input weighted residual 0.48676537, selected 58-input uniform residual 0.48764270, 42-input auction/calendar residual 0.49058764. Tolerance 0.002. Selected source-family counts 19/23/16, before encoding. Size-split gain 0.00116232 is below the practical threshold. Its uncertainty interval concerns paired log MAE, not log RMSE. |
| 19 | `midterm presentation/data_analysis/tables/historical_baseline_comparison.csv`; `outputs/eda/09_global_finalists/champion_segment_and_tail_diagnostics.csv` supports tail discussion | Same eligible counts within each year. Headline development results are equal-weight averages of annual metrics. WAPE 0.43467781 versus 0.36800686, log RMSE 0.62113569 versus 0.48764270. 2025 is separately labeled as previously inspected confirmation evidence. Largest-1% bias is a pooled 2022–2025 diagnostic, not a development-only statistic. |

`build_analysis_slides.py` rebuilds three PDF/PNG figures from these saved tables without fitting models. Compact inputs are copied to `assets/analysis_*_inputs.csv`, with development-family metrics in `analysis_family_development.csv` and exact displayed metric provenance in `analysis_slide_metrics.json`. The presentation's two numerical comparison tables remain native editable LaTeX.

## Challenges and roadmap slides

Slides 22–23 conclude the main deck, adding approximately two and a half minutes. Slide 22 draws on the tail/segment diagnostics and daily stock studies already mapped for the completed analysis. Its information-boundary statement follows the prior-close forecast definition. The findings do not establish the economic causes of missed spikes or isolate coverage as the cause of segment errors.

The user's supplied handwritten ideas are saved as `midterm presentation/sources/roadmap_ideas.png`. Only items marked B or A&B inform the selected additions: VIX/rates, economic-release calendar (CPI/NFP), intraday model, and stock auction history. The image is source material for discussion, not an instruction to implement all written ideas. A-only rebalance, FOMC implied-probability, and pension proposals are not added to the roadmap.

Slide 23 presents proposed work rather than completed empirical claims. VIX/rates and macro-release indicators need data sourcing, timestamp checks, and incremental-value testing. Stock auction history already enters the frozen model, so the proposal refines its responsiveness and fallback behavior. Intraday checkpoints exist, but an intraday model and its predictive gains remain future work. All new evaluations should preserve information timing and distinguish unexamined periods from 2025 confirmation evidence already inspected during research. No research outputs or source data were modified by this presentation update.

## Backtest exhibits added

Main slide 21 uses `outputs/eda/12_selected_stock_predictions/selected_stock_daily_walk_forward_predictions.parquet` and the existing `outputs/eda/13_representative_stock_year_daily/representative_stock_years.csv`. It selects JPM (PERMNO 47896, 2021, 252 observations) and CATO (PERMNO 11267, 2017, 251 observations) from the six existing continuously observed examples. The original selection computes each stock's median annual log RMSE across 2016–2025 and chooses the closest year, with year as tie-breaker. The two stock examples are editorial choices, not a representative sample or a controlled size comparison. Source profiles were selected using recent cross-sectional characteristics, so the chart labels only ticker and year rather than implying a historical size classification.

`build_backtest_slides.py` verifies point-level WAPE and log RMSE against the saved selection metadata. JPM WAPE 0.3082149183 and log RMSE 0.3576979966. CATO WAPE 0.4636206893 and log RMSE 0.5511624152. Crosses mark five largest absolute log1p errors per example. Actual/predicted share volumes are unsmoothed and plotted on logarithmic axes with independent ranges. The chart discloses retrospective specification selection and different example years. Plot rows and metadata are saved as `assets/backtest_daily_examples_inputs.csv` and `backtest_example_selection.csv`.

The unnumbered backtest appendix (PDF page 24) uses `outputs/eda/11_frozen_global_backtest/frozen_global_train_validation_test_metrics.csv`, copied to `assets/backtest_split_metrics_inputs.csv`. It shows log RMSE and WAPE only. Each x value identifies the forward year. Training uses the in-sample final historical refit (350k sampled rows); validation is a prior-year sample (120k rows) forecast by an earlier tuning model; forward results use all eligible forecast-year rows. Validation-model training excludes its validation year, while the subsequent final training window can include it. Source implementation: `notebooks/11_frozen_global_train_validation_test_backtest.ipynb`. These different fitted stages, periods, and samples preclude a simple overfitting conclusion from line ordering. Preserve the 2025 confirmation disclosure.

The backtest chart pairs are reproducible from saved research outputs, with no new fitting or changes to notebooks. The main analysis narration takes eight and a quarter minutes. The individual-relationship, finalist, and annual-performance narration remains concise, and the roadmap narration is shortened to 90 seconds. The main narration now totals 19 minutes 10 seconds plus a 50-second buffer after removal of the data-assembly slide.


## Rolling-median history slide

Main slide 20 shows JPM and CATO from `outputs/eda/12_selected_stock_predictions/selected_stock_daily_walk_forward_predictions.parquet`, matching the stock identities in the adjacent daily-error slide. For each stock and each series separately, calculate a trailing 20-observation median with `min_periods=5`, including the current plotted date. This preserves the display-smoothing convention in `notebooks/12_selected_stock_walk_forward_predictions.ipynb`. It is distinct from the model's prior-only median offset and does not change targets, predictions, or model metrics. The chart deliberately omits faint daily lines because the following slide displays unsmoothed errors.

`assets/backtest_rolling_median_inputs.csv` retains raw actuals, raw predictions, and both rolling curves for all 5,530 stock-days. Actuals cover 2015–2025, with no predictions in 2015. Mark that year as warm-up and start the forecast-median curve after five non-missing predictions in 2016. Disclose retrospective specification selection and independent log-axis ranges. The 2016 historical reconstruction uses the documented modal-penalty fallback before nested historical tuning is feasible. Annual refits use earlier data, with daily lagged features updating during each forecast year. The first 13 notebooks remain the completed-results boundary.


## Review changes and source retention

Removed the data-documentation reference slide and D1/D2 citations from slides and speaker notes at the user's request. The public documentation links above remain internal provenance only. Removed the former data-assembly/sample slide, retaining essential construction details as optional notes beneath the auction/stock-history data-source slide. Reviewed all main titles and changed analytical titles to subject descriptions, retaining findings in the body. The final deck now contains 23 main slides, one backtest appendix slide, and one literature reference slide (25 PDF pages). The numbered narration totals 19 minutes 10 seconds, with 50 seconds available for transitions and recording buffer.


## Review clarification: model status and WAPE comparisons

The cover now reads “Forecasting Closing Auction Volume,” with subtitle “Data Analysis and Preliminary Modeling.” The forecast equation uses a general predictor vector available at the prior-close origin. This changes presentation notation, not feature data or model fitting. Ridge is one modeling approach built so far. The existing 58-input result supplies a reference within ridge trials, without claiming a final project model. Historical artifact names containing finalist/champion/selected remain source identifiers, not current project-level conclusions.

Main slide 19 now shows only annual WAPE, as paired bars comparing the recent-median rule and ridge on the same eligible rows. Values and headline means are unchanged. The earlier two-metric comparison has been simplified, and an error-per-100-shares explanation added. Appendix page 24 separately shows the ridge specification across training, historical validation, and forward-evaluation stages, retaining log RMSE/WAPE and disclosing different periods and samples.

For slide 21, `build_backtest_slides.py` additionally reads `median20_auction_volume` from the saved daily parquet for the same JPM 2021 and CATO 2017 stock-days. All 252 JPM and 251 CATO rows have this baseline observed, so no fallback or row dropping occurs. WAPE is computed from sums of absolute share errors and actual shares. JPM median-rule WAPE 0.3787498777 (rounded 37.9%) versus ridge 0.3082149183 (30.8%). CATO median-rule WAPE 0.5660981995 (56.6%) versus ridge 0.4636206893 (46.4%). Exact computed values are saved in `assets/backtest_example_baseline_comparison.csv`. Substantial absolute error remains despite improvement over this rule; no execution-use acceptance criterion has been established.


## Clarifications: observation weighting and nonlinear roadmap

Slide 18's weighting definition is verified against `training_weights` in `notebooks/09_global_finalists_and_benchmark_freeze.ipynb`: uniform returns no explicit weight array (equal weights), while recent-data weighting is `0.5 ** (age_years / 2.0)`, with age measured from January 1 of the prediction year and days divided by 365.25. These weights affect the fitting objective, not the predictor coefficients or WAPE formula. Displayed feature sets also differ, so the row comparison cannot isolate a weighting effect.

Slide 19's simple baseline forecasts the prior-only stock median itself, with historical training-median fallback if unavailable. Its median window excludes the current forecast day, unlike the rolling display smoother. Slides 9–10 now state their descriptive purposes explicitly. Slide 23 adds proposed nonlinear-model tests (e.g., boosted trees) for thresholds/interactions. No nonlinear fitting, gain, or new data acquisition is claimed. Compare model classes on identical eligible inputs and chronological folds, using training-only tuning, before adding new information. All existing numerical results remain unchanged.


## Expanded individual-covariate slides (13–15)

`build_covariate_slides.py` reads existing `data_analysis/tables/panel_bin_curves.csv` for `prior_raw_market_cap_usd`, `lag1_quoted_spread`, `prior_raw_return_total`, and `lag1_abs_return`. It reads `native_bin_curves.csv` for `prior_raw_market_total_volume` and `prior_raw_sector_total_volume`. Rows are sorted by numeric bin order. Each variable has ten pairwise-complete input groups with saved quantile boundaries and ties preserved. No raw source, research output, or fitted forecast is changed.

Slide 13 uses stock-day outcomes and prior-close market cap/spread. Market-cap raw Pearson correlation is 0.288742 pooled versus 0.021558 within stock; spread −0.050078 versus −0.000511. Separate panel ranges are disclosed. Slide 14 uses signed and absolute prior-day returns, with the same zero-based target range. Raw pooled Pearson correlations are 0.001156 and −0.003108; the absolute-return within-stock correlation is 0.019023. Curves do not establish causal responses or rule out conditional/nonlinear relationships.

Slide 15 uses 2,765 dates and 33,180 sector-dates. Its outcome is mean eligible auction shares per stock in the corresponding date/sector-date. Navy and teal summarize the mean and median of these group-average outcomes within input deciles, not the distribution of individual-stock auction shares. Native dates or sector-days receive equal weight. Market context shows an increasing pattern; sector context is non-monotonic and mixes sectors/composition/time. Correlations at different aggregation levels should not be compared as a common experiment.

All six inputs precede the auction day and contain no auction-history components. The prior-day total trading volume already shown on slide 12 is also distinct from auction history. New exhibits retain natural share outcomes, use the retrospective 2015–2025 sample, and supplement economic interpretation rather than select/refit a model. Exact curves and six association rows are copied to `assets/covariate_extension_plot_inputs.csv` and `covariate_extension_associations.csv`. New narration is 35 seconds per slide; descriptive narration is 40/40/30 seconds, for total 19:10 plus 50 seconds of buffer.


## Recording-script conversion

The speaker notes are now a continuous read-aloud script, including natural slide transitions and no Q&A branches. Slides 1–23 form the main narration, with separate scripts for the two appendix slides. Approximate main-script length is 2,400 words, supporting roughly 18–19 minutes of speech at 130–135 words per minute before pauses. Earlier detailed technical notes were retained as `sources/speaker_reference_notes.md`, distinct from the spoken recording script. Interpretation and empirical values follow the evidence already mapped above. The presentation PDF, chart assets, raw data, and fitted results remain unchanged.


## Group-error exhibit added for the revised narrative

`midterm presentation/build_group_error_slide.py` reads saved predictions from `outputs/eda/09_global_finalists/predictions/global_finalist_predictions_2022.parquet` through `_2024.parquet`, using `rolling5_residual_auction_calendar_liquidity` (58-input ridge reference). It reproduces the source 2022–2025 size/history entries in `champion_segment_and_tail_diagnostics.csv` as a cross-check, then reports development-only 2022–2024 results. Definitions follow `notebooks/build_notebook_09.py`; underlying daily size ranks and prior-target counts follow `build_notebook_05.py`. There is no model refit or use of post-13 research.

The diagnostic pools errors across 963,525 stock-days: size WAPEs small 46.2139%, mid 39.7105%, large 34.3544%; history WAPEs 0–4 prior auctions 92.1939% (n=758), 5–19 51.6708% (n=2,466), 20+ 36.6264% (n=960,301). Under-20 cases comprise 0.3346% of observations. Counts are stock-days, not unique stocks. Stocks may change size groups and enter multiple history groups over time; group differences are observational. This pooled metric differs from the mean annual headline used on the annual benchmark slide. Compact data and metadata are saved in `assets/group_error_development_inputs.csv` and `assets/group_error_metadata.json`.


## Calibration and same-row baseline comparisons

Current slide map: 1–19 unchanged order, calibration 20, smoothed/daily examples 21–22, group comparison 23, roadmap 24; appendix diagnostic 25, baseline-adjusted covariates 26, literature 27. The main script has approximately 2,480 spoken words.

`build_group_error_slide.py` reconstructs `median20_fallback` using the saved `median20_auction_volume` in the 09 predictions and annual `training_global_median_raw_target` from `outputs/eda/07_temporal_baselines/preprocessing_manifest.json`. The earlier point-level baseline predictions are unavailable locally, but these saved inputs recover annual WAPEs and observation counts to numerical tolerance. No new fallback is estimated. This preserves all 963,525 development stock-days, including 616 fallback cases. The comparison therefore matches the simple rule used in the annual main slide.

Size-group baseline/ridge WAPEs: small 53.2174/46.2139%, mid 47.0868/39.7105%, large 40.6947/34.3544%. History-group values: 0–4 prior auctions 142.5067/92.1939%, 5–19 67.5732/51.6708%, 20+ 43.2212/36.6264%. Group definitions and the original ridge-only checks are unchanged. WAPE above 100% is allowed by its definition. Saved CSVs include `assets/group_error_baseline_fallbacks.csv`, `group_error_development_inputs.csv`, and `forecast_diagnostic_annual_checks.csv`.

Calibration uses the 58-input ridge's saved 2022–2024 share predictions. Pooled predicted/actual totals are 185,058,667,712 / 216,150,063,209 shares, yielding −14.384171% aggregate bias. Ten pooled equal-count prediction groups show lower mean forecast than mean actual in each group. Deciles use prediction ranks, with date and PERMNO tie-breaks, rather than realized-volume ranks. Data and metrics are `assets/calibration_development_inputs.csv` and `calibration_development_metrics.json`. No recalibration is fitted on evaluation outcomes.

## Baseline-adjusted covariate support

`data_analysis/build_baseline_adjusted_analysis.py` joins saved 09 forecast keys to the isolated `data_analysis/cache/analysis_panel.parquet` derived from the in-scope canonical panel. Key uniqueness, exact actual-target equality, and 963,525 matched rows are checked. Missing stock median cases (616) are excluded only from this descriptive study. The target is the existing log1p deviation; predictors use prior-close values. Quantile groups keep equal input values together.

Same-pair rank correlations with raw shares versus log deviation: daily volume 0.857092/0.075029 (962,909 pairs), market cap 0.633994/0.011704 (962,909), spread −0.447779/−0.014653 (962,802), absolute return −0.052598/0.067043 (962,850). These distinguish scale associations from baseline departures without controlling other covariates or proving incremental forecasting gains. Complete curves, summaries, metadata, two chart variants, and a report are saved under the presentation folder. This support analysis adds no model fit or post-13 input.
