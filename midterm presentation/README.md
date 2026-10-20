# Midterm presentation draft

This folder is self-contained for the current presentation draft. Existing notebooks, reports, and outputs were not modified. Evidence is limited to repository notebooks 01–13.

## Files

- `midterm_draft.pdf`: presentation to review, twenty-three main slides plus one backtest appendix slide and one literature reference slide.
- `midterm_draft.tex`: editable 16:9 Beamer source.
- `speaker_notes.md`: complete conversational recording script for slides 1–23, with separate appendix narration.
- `analysis_discussion.md`: record of the selected descriptive, analysis, and closing sections.
- `sources/speaker_reference_notes.md`: earlier technical notes retained for preparation, separate from the recording script.
- `sources/source_map.md`: evidence paths, verified literature links, local-source distinctions, and limitations.
- `sources/dataset_inventory_snapshot.csv`: small snapshot supporting the displayed dataset counts.
- `sources/sponsor_brief_excerpt_1.png` and `sponsor_brief_excerpt_2.png`: copies of the supplied excerpts.
- `sources/roadmap_ideas.png`: supplied handwritten ideas, with B and A&B items used for the roadmap.
- `build.sh`: PDF rebuild helper using Tectonic, with latexmk fallback.

## Rebuild

From this folder, run:

```bash
./build.sh
```

Alternatively open `midterm_draft.tex` in a LaTeX editor or upload it to Overleaf. Upload the `assets/` folder alongside the source because the deck includes eleven chart PDFs. It uses standard Beamer packages and no bibliography tool. The first Tectonic build may need internet access to retrieve its TeX bundle.

## Scope

The current draft covers project understanding, problem definition, background literature, data sources, a compact descriptive section, ten analysis slides, and two closing slides on challenges and the roadmap. The descriptive section has two graph slides followed by one summary slide, planned for about two minutes. Allow about six and a half minutes for the opening, eight and a quarter minutes for analysis, and two and a half minutes for challenges and the roadmap. The recording script is approximately 2,400 words, about 18–19 minutes at 130–135 words per minute, leaving room for pauses and slide changes within 20 minutes. Appendix material is outside that narration budget.

The deck uses a neutral academic style rather than official BlackRock branding. Replace the project-level author line with the group's names before recording. The date is October 20, 2026, based on the supplied deadline and current project context.

This folder does not automatically prevent Git conflicts, but it isolates presentation edits from the research files. Its local `.gitignore` excludes LaTeX build intermediates. No commit or push has been made.

## Descriptive statistics

The `descriptive_stats/` subfolder contains the separately prepared raw-data analysis for the first 13 notebooks. A small selection of its results now appears in slides 9–11. The full report remains separate for reference. Options-market data are outside this scope. Known monthly and quarterly expiry calendar flags remain part of the daily analysis.

## Descriptive chart assets

`build_descriptive_slides.py` rebuilds the two paired chart figures in `assets/` from the saved descriptive CSV tables. It requires the Python packages listed in `descriptive_stats/requirements.txt`. Run it before `./build.sh` when the descriptive inputs change. Each figure is saved as PDF for the slides and PNG for preview.

## Analysis slides and reproducibility

Slides 12–21 connect individual-variable relationships to economic-family removal tests, the median log-residual formulation, specification comparisons within ridge, annual baseline comparisons, rolling-median histories, and daily stock examples. The complete 193-specification individual-variable review remains in `data_analysis/covariate_atlas.html`; the main slides show representative curves across four individual-relationship slides, while the notes discuss the full set of economic families. They do not imply that every individual graph appears in the deck or that this retrospective review originally selected the model.

`build_analysis_slides.py` rebuilds three chart PDF/PNG pairs and saves compact chart-input CSVs and exact metrics in `assets/`. It reads the saved covariate tables, annual family-audit outputs, finalist-selection results, and historical-baseline comparison. It does not refit models. Run it from the repository with a Python environment containing pandas, NumPy, and Matplotlib, then run `./build.sh` from this folder. Existing external research outputs are required to regenerate the analysis assets, but the saved assets make the LaTeX deck itself self-contained.

Model-selection comparisons and the family-removal chart use mean annual metrics from 2022–2024. Annual performance plots separately label 2025 as confirmation evidence already inspected during research. Natural-unit relationship plots use the retrospective 2015–2025 descriptive sample. The earlier `data_analysis/` reports and hash record describe the diagnostic stage before these selected slides were added; their statements about leaving the presentation unchanged refer to that stage.

`build_backtest_slides.py` builds three additional PDF/PNG chart pairs from saved outputs, using pandas, NumPy, Matplotlib, and DuckDB. It performs no model fitting. Main slide 20 shows 20-trading-observation rolling medians of actual and predicted volume for JPM and CATO over 2015–2025, with an actuals-only 2015 warm-up. The median curves require five non-missing values and include the current plotted observation, matching the original display smoother. Main slide 21 uses unsmoothed daily JPM (2021) and CATO (2017) examples with example-year selection and point-level metrics verified against saved results. The main annual-error slide compares the ridge benchmark with a recent-median rule using WAPE only. The appendix addresses a separate question: how errors vary across training, historical validation, and forward evaluation for one ridge specification, retaining log RMSE and WAPE. Compact data inputs and selection metadata are saved as `assets/backtest_*.csv`. Different periods/samples prevent interpreting lower forward error as proof against overfitting. Stock examples use a retrospectively selected specification, different years, and separate axis ranges.

## Challenges and roadmap

Slides 22–23 link large-auction underprediction, weaker performance for small stocks or limited histories, and the prior-close information boundary to future work. The roadmap includes diagnosing errors and refining existing auction-history features, comparing nonlinear models with ridge, testing lagged VIX/rates and scheduled CPI/NFP indicators, and developing updates using the prepared intraday checkpoints. These are planned extensions, with data availability and incremental predictive gains still to be established. Auction history is already part of the current model. No new datasets or models were added for these slides, and the completed-results boundary remains the first 13 notebooks.


## Presentation edits after review

Removed the standalone data-assembly/sample slide because its key definitions already appear in the target and source discussion. Essential construction details remain in optional speaker notes. Removed the data-documentation reference slide and its D1/D2 citations from slides and notes. Public documentation remains recorded in the internal source map for provenance. Literature references remain in the deck. All slide titles now name the subject or analysis, with conclusions retained in the body.


## Model status and metric clarification

The cover heading is “Forecasting Closing Auction Volume,” with subtitle “Data Analysis and Preliminary Modeling.” Neither includes an exchange name. NYSE remains the actual completed empirical scope in the project/data discussion. The forecast notation uses a general predictor vector with an explicit forecast origin. Availability restricts eligibility, rather than a fixed list of economic families.

Ridge is one modeling approach built so far. The 58-input specification is a reproducible reference within these ridge trials, not the project's final model. Historical output filenames and fields containing “finalist,” “champion,” or “selected” remain unchanged for reproducibility; those labels do not establish final project status. Hold the reference fixed during comparisons while continuing model development.

The daily-example slide now supplies a same-row median-rule comparison: JPM 2021 WAPE 37.8750% versus ridge 30.8215%, and CATO 2017 WAPE 56.6098% versus ridge 46.3621%. These are meaningful reductions relative to the simple rule, with substantial remaining absolute error. They do not establish suitability for execution decisions. The verified comparison is saved in `assets/backtest_example_baseline_comparison.csv`. WAPE is total absolute share error divided by total actual shares, not the mean daily percentage error or an accuracy percentage.


## Clarifications on descriptive slides, training weights, and the baseline

Slides 9–10 describe the sample rather than evaluate forecasts: target skew/stock scale, followed by time/calendar variation in participation. Their purpose statements and notes now make the motivation explicit. The ridge comparison uses observation weights in the fitting objective. Equal weights give every training stock-day equal weight. Recent-data weights have a two-year half-life, distinct from model coefficients or evaluation metrics. The recent-median rule simply predicts the prior-only trailing 20-day auction median, with the historical training fallback if needed.

The roadmap now explicitly includes nonlinear model comparisons (for example, boosted trees) to test thresholds and interactions. These are planned tests, with no model fitting or new performance claim. Compare on the same information set and chronological folds before adding new data families.


## Expanded individual-covariate analysis

Three new slides (13–15) extend the existing auction-history/daily-volume comparison. They cover prior-day market capitalization and relative quoted spread, signed and absolute stock returns, and market/sector total-volume measures. Each shows mean and median outcomes in shares across input deciles. Stock-specific inputs use paired stock-days. Market and sector context use equal-weight dates and sector-days with mean auction shares per eligible stock as the group outcome. These are retrospective associations, with no causal or incremental-model-value claim.

`build_covariate_slides.py` creates three additional PDF/PNG pairs from saved diagnostic tables, plus `assets/covariate_extension_plot_inputs.csv` and `covariate_extension_associations.csv`. It does not refit models. The complete recording script now integrates all main slides with natural transitions. Its word count supports approximately 18–19 minutes of speech at 130–135 words per minute, plus pauses within the 20-minute budget. Existing model metrics and research outputs remain unchanged.


## Read-aloud recording script

`speaker_notes.md` is now a complete first-person group narration. Read only the paragraphs, using headings to navigate slides. It includes chart-reading explanations, the meaning of training weights and WAPE, the median-rule baseline, ridge's provisional status, uncertainty in the associations, confirmation-year limits, and transitions into the challenges and roadmap. There are no “If asked” branches or placeholders in the spoken script. All numerical model results remain unchanged.

Slides 24–25 have complete appendix scripts after a separate divider and are outside the main recording. The previous detailed notes are retained in `sources/speaker_reference_notes.md` for technical preparation. Timing is now based on script length and speaking pace rather than the earlier slide-by-slide time labels. The presentation PDF itself is unchanged in this notes-only update.
