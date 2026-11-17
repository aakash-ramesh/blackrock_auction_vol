# Closing-auction volume: feature template (E0–E5)

Takes the project's raw data, cleans it, and builds the model-ready features for the E0–E5 specifications from the midterm presentation.

## How to run

1. `pip install pandas numpy duckdb python-dateutil jupyter`
2. Open `01_feature_pipeline_template.ipynb`.
3. In the first code cell, set `DATA_ROOT` to the repo's `blackrock_data` folder. (Inside the repo it finds the folder on its own.)
4. Run all cells. Takes about a minute.

## Files

| File | What it is |
|---|---|
| `01_feature_pipeline_template.ipynb` | **The template.** Load → clean → build features → save. |
| `02_checks_optional.ipynb` | Optional. Checks the template: data quality, no look-ahead, and a comparison with the midterm results. Not needed to build the features. |
| `reference_inputs/` | Small date lists that can't be computed by rule: FOMC meetings, MSCI review dates, S&P 500 changes and membership. |

## What it reads

| Data | Folder | Source |
|---|---|---|
| CRSP daily | `blackrock_data/stock_daily/crsp_daily/` | WRDS CRSP (NYSE-listed US common stocks) |
| Closing-auction prints | `blackrock_data/stock_daily/auction_daily/` | WRDS TAQ |
| Options (optional) | `blackrock_data/stock_daily/options_daily/` | WRDS OptionMetrics |

## What it produces (`outputs/`)

- `model_features/year=YYYY/…parquet`: one row per stock per day, 2015–2025, with the target (`auction_volume`, `y_log_ratio`) and 40 features. Use rows where `model_ready` is true.
- `feature_dictionary.csv`: the formula, source and timing of every feature.

## The features

| Step | What it adds | Count |
|---|---|---|
| E0 | Each stock's typical auction volume: median of its last 20 auctions | 3 |
| E1 | Index rebalance days (S&P quarterly, Russell, MSCI) and S&P 500 adds/deletes | 11 |
| E2 | Monthly options expiration days | 1 |
| E2+ | How much of each stock's options open interest is expiring (optional) | 8 |
| E3 | Month-end, quarter-end, year-end, first day of month, early closes | 5 |
| E4 | FOMC announcement days | 1 |
| E5 | Recent volatility and volume, high-volatility periods, and event interactions | 11 |

Every feature uses only information available before the trading day it predicts.

## Result check

With these features, the same simple model as the midterm gives almost the same accuracy (log RMSE, 2020–2025, 1.88M stock-days):

| Spec | This template | Midterm slide |
|---|---|---|
| E0 | 0.6163 | 0.6164 |
| E1 | 0.5404 | 0.540 |
| E2 | 0.5400 | 0.539 |
| E3 | 0.5307 | 0.529 |
| E4 | 0.5303 | 0.528 |
| E5 | 0.5142 | 0.5250 |

E5 is built from the slide's description and includes a few more stock-level inputs, so it improves slightly more.

## Known limits

- MSCI and S&P 500 dates come from the team's index-rebalance dataset. S&P 500 changes with a missing or estimated announcement date (52 of 358) are left out.
- MSCI dates after November 2025 need to be added before using the template on later dates.
- The options data ends 2025-08-29.
