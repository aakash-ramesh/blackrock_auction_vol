# Analysis leading to model choice, and individual covariate diagnostics

Start with [the model-choice story](model_choice_story.md), then [the new economic findings](covariate_findings.md). The [searchable covariate atlas](covariate_atlas.html) contains an individual graph and results for every specification.

The presentation is unchanged. Everything in this folder is for discussion before choosing analysis slides. Evidence is restricted to the first 13 notebooks. Stock-specific options-market data and later modeling work are excluded.

## The core story

Recent auction history estimates a stock's normal auction scale. Known calendar effects add substantial forecasting information, while liquidity contributes a smaller increment. The chosen specification predicts a log1p residual around a trailing auction median using one global, regularized model. Its 58 source features are a smaller practical tie with 99-feature alternatives. Tested interactions, hard splits and specialists have not justified replacing it under the stated criteria.

This conclusion is narrower than saying that ridge is the best possible model, that every retained input is individually necessary, or that omitted variables have no economic relevance.

## What is historical and what is new

- `model_choice_story.md` reviews the original analysis and saved results, with a complete 13-notebook evidence map and the actual selection tradeoffs. It does not refit or reselect models.
- `covariate_findings.md` reports the new retrospective natural-unit associations and economic distinctions.
- `covariate_atlas.html` offers 193 entries: 99 existing daily features, 31 contemporaneous economic source variables, 31 prior-close counterparts, two auction-outcome diagnostics, and six intraday measurements at each of five checkpoints. All 58 selected model features are covered.
- `tables/selected_feature_dictionary.csv` maps all retained features to their timing, family, economic hypothesis, new isolated associations, original feature-role audit, and historical conditional-coefficient context.
- `tables/raw_covariate_comparison.csv` compares current and prior-close measurements for each of the 31 source covariates. Calendar and industry effects are covered in the daily feature set.
- `tables/` also contains bin means/medians, annual and period associations, missingness contrasts, family ablation, model comparisons, selection decisions, and review manifests.
- `figures/` holds individual plots and four broader discussion figures. These are analysis artifacts, not additions to the deck.

## Natural target and interpretation

The primary outcome is official closing-auction volume in shares, with no log transformation for Pearson correlations, OLS slopes, within-stock share-level associations, or binned outcomes. Each curve reports raw-share means and medians. Secondary ranks are explicitly labeled. Contemporary inputs are kept separate from information available for a prior-close daily forecast.

The analysis uses finite pairs without imputation or winsorization. Stock-specific primary associations use stock-dates. Shared calendar/market inputs use dates and the mean auction shares per eligible stock; sector inputs use sector-dates and that group's mean shares. Stock-day-pooled results are separately recorded. Within-stock demeaning removes security levels, not common dates or other confounding.

Stock-date Spearman and within-stock rank measures use a deterministic approximately 7% hash sample; date and sector-date context ranks use all native observations. Exact rank sample counts are in the tables. Full-population numeric bins keep ties together. Constant variables cannot identify an effect. Low-cardinality inputs use their actual observed values. Highly concentrated inputs separate their common value from the remaining-value groups so that rare dividends or adjustments do not disappear into a single bin.

These are one-variable-at-a-time associations, not identified causal effects or incremental forward feature tests. No IID p-values are reported. The new full-period diagnostics are not a new holdout and must not be described as the historical basis for selecting the existing model. The 2025 year was inspected during the original research and is confirmation evidence only.

## Reproduce

Use Python 3.12 with the packages in `../descriptive_stats/requirements.txt`. From this folder:

```bash
python build_panels.py
python analyze_covariates.py
python build_story.py
python render_covariates.py
python verify_analysis.py
```

`build_panels.py` extracts and evaluates only the `panel_sql` assignment from the existing feature-engineering notebook. It does not execute the notebook or import a builder that writes research files. The exact extracted SQL is saved in `canonical_panel_query.sql`.

The panel rebuild and source-level joins write a roughly 2.9 GB `cache/`, excluded by this folder's `.gitignore`. The cache prevents repeated feature reconstruction and is not needed to read the reports or plots. Temporary query spills use `/tmp/blackrock_covariate_spill`. Existing raw sources, notebooks, model outputs, and presentation files are not modified. If source inputs or feature SQL change, remove the disposable cache before rebuilding it. `render_covariates.py --refresh-figures` forces regeneration of existing individual plots.

The saved notebooks and historical metrics are reviewed rather than retrained. `notebook_review_manifest.csv` records their hashes, execution counts and saved error outputs. `panel_build_metadata.json` records the copied SQL's hash, sample count and auction-share total. `analysis_metadata.json` records analysis definitions. The presentation's hashes are recorded for an unchanged-file check.
