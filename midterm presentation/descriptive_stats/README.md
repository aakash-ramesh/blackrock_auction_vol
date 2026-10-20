# Raw-data descriptive statistics

Scope: data used through repository notebooks 01–13. The presentation and its speaker notes use audience-facing terms rather than notebook references. The full report remains separate from the slides. Selected results now appear in a compact three-slide descriptive section.

## Review

- `report.html`: self-contained report with tables, eight embedded charts, a topic filter, and links to the full CSV tables. Open it in a browser.
- `report.md`: the same narrative and selected tables, suitable for GitHub review.
- `figures/`: eight standalone PNG charts for inspection. The presentation uses a compact selection, rebuilt as paired figures by `../build_descriptive_slides.py`.
- `tables/`: full-precision source inventory, per-field completeness, numeric and categorical summaries, yearly statistics, sample eligibility, and quality checks.
- `run_metadata.json`: software versions, scope, and run timestamp.
- `build_descriptive_stats.py` and `render_report.py`: reproducible analysis and reporting code.

## Reproduce

From this directory, using Python 3.12 or later:

```bash
python -m venv /tmp/blackrock_midterm_stats_env
/tmp/blackrock_midterm_stats_env/bin/python -m pip install -r requirements.txt
/tmp/blackrock_midterm_stats_env/bin/python build_descriptive_stats.py
```

The script locates the repository relative to itself. It reads source Parquet files and writes only this directory's report, charts, tables, and metadata. Temporary query spill files use `/tmp/blackrock_midterm_duckdb`. The supplied market/sector extracts are aggregated context tables, not exchange tick data.

## Populations and statistics

- Full raw extracts remain intact. Missing targets are never treated as zero. No imputation or clipping is performed.
- Stock data use stock-days. Market/calendar data use trading dates. Sector data use sector-days. Intraday data use stock-day-checkpoints. Industry history uses effective-date intervals, followed by a separate match to eligible stock-days.
- Source inventory and all-field profiles cover the full available local history. Additional numeric summaries restrict CRSP, market, and sector data to 2015–2025.
- Eligible target statistics require an identified, nonmissing auction target and exclude the source-anomaly date, August 12, 2019.
- `numeric_summary.csv` profiles every numeric measure except identifiers and category codes. Quantiles with `_approx` suffixes use DuckDB approximate quantiles over all observations. Means, standard deviations, counts, minima, and maxima use all finite observations.
- `principal_exact_summary.csv`, selected annual auction summaries, and checkpoint medians use exact quantiles.
- `field_profile.csv` provides a row for every field, including identifiers, text, and dates. `schema_absent_rows` distinguishes missing columns in annual files from null values inside files where the column exists. Distinct counts in this all-field profile are approximate and labeled accordingly.
- `categorical_frequencies_top30.csv` reports up to 30 observed levels for category fields and booleans. `<NULL>` differs from `<EMPTY STRING>`. It omits identifier-frequency lists. `flag_rates_by_year.csv` gives complete boolean counts by year, including event-date frequencies.
- Numeric identifiers such as PERMNO and FF12 codes are not treated as continuous economic variables. Units are indicated in the numeric tables. All stored ratios and returns retain their native decimal scale.
- A size-decile comparison uses contemporaneous market capitalization for description only. Daily forecasting features in the research pipeline use lagged information.
- Source-row counts reconcile against the saved research inventory. Universe, observed-label, and eligible-label counts reconcile against the saved sample-eligibility table. An additional direct panel check runs when the enriched daily Parquet panel is present. Those existing outputs are reconciliation controls, not input observations for the raw-data statistics.

## Scope distinctions

Eight tables are read directly in the first 13 notebooks: CRSP daily, auction daily, intraday trade checkpoints, market daily, sector daily, event calendar, industry history, and the source-anomaly control. Three additional upstream reference layers are profiled separately: security master, basic trading calendar, and the TAQ–CRSP symbol bridge.

Intraday checkpoints were audited but are not predictors in the daily models. The industry reference contains many securities outside the NYSE study universe, so its raw record count is not a model-universe security count. Do not add row counts from different layers to claim a combined sample size.

Options-market data and FOMC-specific datasets are excluded. Known monthly and quarterly options-expiry flags remain included as calendar information. Model predictions, engineered features, and model-performance statistics are outside this raw-data report.
