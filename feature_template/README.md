# Closing-auction volume: feature construction template (E0–E5)

One notebook that takes the project's raw data, cleans it, and builds the model-ready feature table behind the E0–E5 specifications from the midterm presentation.

## How to run

1. Install the packages: `pip install pandas numpy duckdb pyarrow python-dateutil jupyter`
2. Open `01_feature_pipeline_template.ipynb`.
3. In Section 0, set `DATA_ROOT` to the `blackrock_data` folder of the project repo. If the notebook sits next to (or inside) a `blackrock_auction_vol` folder, it finds the data on its own.
4. Run all cells (about 2–3 minutes, roughly 6 GB of memory).

## What it reads

| Input | Location | Source |
|---|---|---|
| CRSP daily stock file | `blackrock_data/stock_daily/crsp_daily/` | WRDS `crsp.dsf_v2` + `crsp.stksecurityinfohist`; NYSE-primary US common stocks |
| Closing-auction prints | `blackrock_data/stock_daily/auction_daily/` | WRDS TAQ; exchange `N`, correction `00`, sale condition containing `6` |
| Options stock-day aggregates (optional) | `blackrock_data/stock_daily/options_daily/` | WRDS OptionMetrics, built by `11_options_daily.ipynb` |
| FOMC meetings 2015–2026 | `reference_inputs/fomc_meetings.csv` | federalreserve.gov FOMC calendar pages, checked 2026-11-17 |
| Index event dates | `reference_inputs/index_event_dates.csv` | MSCI, FTSE Russell, S&P schedules (from the team's index-rebalance dataset) |
| S&P 500 changes and membership | `reference_inputs/sp500_constituent_changes.csv`, `sp500_membership_spells.csv` | Team's index-rebalance dataset |

## What it writes (`outputs/`)

| File | Content |
|---|---|
| `model_features/year=YYYY/model_features_YYYY.parquet` | One row per stock-day, 2015–2025: target, eligibility flags, 40 features |
| `feature_dictionary.csv` | Every feature: group, exact formula, raw source, when it is known |
| `calendar_features.csv` | Date-level calendar and event flags |
| `qc_report.csv` | Every data-quality and look-ahead check with its result |
| `live_features_YYYYMMDD.parquet` | Feature row for the next trading day after the data ends |
| `sanity_model_ladder.csv` | E0→E5 expanding-window OLS results |

Key columns: `date`, `permno`, `ticker`, `auction_volume` (target in shares), `y_log_ratio` (target used by the model), `model_ready` (use these rows for fitting).

## Feature groups

| Group | Features | Count |
|---|---|---|
| E0 | 20-row median auction volume (the reference), its log, number of prints behind it | 3 |
| E1 | S&P quarterly rebalance, Russell reconstitution, MSCI review, any index event; S&P 500 membership, pending and trade-day flags for announced additions and deletions | 11 |
| E2 | Monthly options expiration excluding quarterly | 1 |
| E2+ | Stock-level options exposure: share of open interest expiring today and within 7 days, total open interest, put/call ratio, lagged option volume, implied vol, gamma size | 8 |
| E3 | Month-, quarter-, year-end; first trading day of month; early close | 5 |
| E4 | Scheduled FOMC statement day | 1 |
| E5 | Stock volatility, abnormal volume and size; market volatility and volume; high-volatility and high-volume regimes; event interactions | 11 |

## How we know it is right

- **40 checks** run inside the notebook (keys, value ranges, calendar, reference lists).
- **Look-ahead test**: features for a forecast day are rebuilt after deleting that day's data; none change.
- **Cross-checks against the project repo**: the rule-based calendar matches all 3,772 trading days and the repo's event calendar; the recomputed market series matches the repo's `market_daily`.
- **Reproduces the midterm slide**: with at least 10 prior auction prints, the test sample is 1,882,373 stock-days (slide: 1.88M) and the out-of-sample log RMSE is:

| Spec | This notebook | Midterm slide |
|---|---|---|
| E0 | 0.6163 | 0.6164 |
| E1 | 0.5404 | 0.540 |
| E2 | 0.5400 | 0.539 |
| E3 | 0.5307 | 0.529 |
| E4 | 0.5303 | 0.528 |
| E5 | 0.5142 | 0.5250 |

E5 here also includes stock-level volatility, volume and size, so its gain is larger than on the slide.

## Known limits

- **MSCI and S&P 500 lists come from the team's index-rebalance dataset.** The dates were not re-checked one by one against the providers' websites. S&P quarterly and Russell dates are also derived by rule and match the list.
- **S&P 500 changes with a missing or estimated announcement date are not used** (52 of 358), so those events have no pending or trade-day flag.
- **MSCI reviews after November 2025 are not in the list.** Add them before using `msci_review` on later dates; the notebook warns about this.
- **The options layer ends 2025-08-29** (latest OptionMetrics data on WRDS), so E2+ features are missing after that date.
- **Open-interest features need the OCC morning file in live use.** Historical OptionMetrics rows report open interest as of the previous close.
- **Not included from the index-rebalance dataset**: ETF and index-fund ownership (quarterly holdings; release timing needs documenting first) and announcement-window flags for MSCI and Russell (the dates are in `index_event_dates.csv` if wanted).
- **E5 is our reading of the slide description**; thresholds are documented in the notebook and easy to change.

## Extending for live use

1. Append the next day's CRSP-equivalent row and auction print to the raw tables.
2. Add new FOMC, MSCI and S&P 500 announcements to `reference_inputs/` as they are published.
3. Re-run. Section 13 shows the feature row a forecast for the next trading day would use.
