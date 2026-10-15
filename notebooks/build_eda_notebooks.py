"""Build the first two capstone EDA notebooks.

Run from the repository root. The generated notebooks find the project root
automatically, so they can also be opened and rerun from the notebooks folder.
"""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


COMMON_SETUP = r'''
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (11, 5.5), "figure.dpi": 120})

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate blackrock_data. Run the notebook inside the Capstone project.")

DATA = PROJECT_ROOT / "blackrock_data"
OUTPUT = PROJECT_ROOT / "outputs" / "eda"
con = duckdb.connect()

AUCTION = str(DATA / "stock_daily/auction_daily/year=*/auction_daily_*.parquet")
CRSP = str(DATA / "stock_daily/crsp_daily/year=*/crsp_daily_*.parquet")
TRADE = str(DATA / "stock_intraday/trade_checkpoints/year=*/trade_checkpoints_*.parquet")
EVENTS = str(DATA / "events/event_calendar.parquet")
INDUSTRY = str(DATA / "reference/industry_history.parquet")
ANOMALIES = str(DATA / "quality/auction_source_anomalies.parquet")

def sql(query, params=None):
    return con.execute(query, params or []).df()

print(f"Project root: {PROJECT_ROOT}")
'''


def notebook_01():
    cells = [
        md('''
# 01 — Data quality and coverage

This notebook establishes the reliable analytical sample before modeling.

**Unit of observation**

- Daily data: `date × permno`
- Intraday data: `date × permno × checkpoint`

**Target rule**

- A target is observed only when `target_found = TRUE` and `auction_volume` is non-null.
- Missing auction records are not converted to zero.
- Dates explicitly marked `exclude_from_model` are ineligible.
- Intraday checkpoints after an early close are not treated as missing data failures.
'''),
        code(COMMON_SETUP + '\nOUT = OUTPUT / "01_data_quality"\nOUT.mkdir(parents=True, exist_ok=True)'),
        md('## Dataset inventory'),
        code(r'''
inventory = sql(f"""
WITH crsp AS (
  SELECT count(*) AS rows, count(DISTINCT date) AS dates,
         count(DISTINCT permno) AS permnos, min(date) AS min_date, max(date) AS max_date
  FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
), auction AS (
  SELECT count(*) AS rows, count(DISTINCT date) AS dates,
         count(DISTINCT permno) AS permnos, min(date) AS min_date, max(date) AS max_date
  FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
), trades AS (
  SELECT count(*) AS rows, count(DISTINCT date) AS dates,
         count(DISTINCT permno) AS permnos, min(date) AS min_date, max(date) AS max_date
  FROM read_parquet('{TRADE}', union_by_name=true, hive_partitioning=true)
), market AS (
  SELECT count(*) AS rows, count(DISTINCT date) AS dates,
         NULL::BIGINT AS permnos, min(date) AS min_date, max(date) AS max_date
  FROM read_parquet('{str(DATA / "market/market_daily.parquet")}')
), sector AS (
  SELECT count(*) AS rows, count(DISTINCT date) AS dates,
         NULL::BIGINT AS permnos, min(date) AS min_date, max(date) AS max_date
  FROM read_parquet('{str(DATA / "sector/sector_daily.parquet")}')
)
SELECT 'CRSP daily' AS dataset, * FROM crsp
UNION ALL SELECT 'Auction daily', * FROM auction
UNION ALL SELECT 'Trade checkpoints', * FROM trades
UNION ALL SELECT 'Market daily', * FROM market
UNION ALL SELECT 'Sector daily', * FROM sector
""")
display(inventory)
inventory.to_csv(OUT / "dataset_inventory.csv", index=False)
'''),
        md('## Annual target coverage'),
        code(r'''
annual = sql(f"""
WITH c AS (
  SELECT date, permno
  FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
  WHERE date >= DATE '2015-01-01'
), a AS (
  SELECT date, permno, auction_volume, auction_price, target_found
  FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
)
SELECT year(c.date) AS year,
       count(*) AS stock_days,
       count(a.permno) AS auction_rows,
       count(a.auction_volume) AS observed_targets,
       count(*) - count(a.auction_volume) AS missing_targets,
       count(a.auction_volume)::DOUBLE / count(*) AS target_coverage,
       count(DISTINCT c.date) AS trading_days,
       count(DISTINCT c.permno) AS securities
FROM c LEFT JOIN a USING (date, permno)
GROUP BY 1 ORDER BY 1
""")
display(annual)
annual.to_csv(OUT / "annual_target_coverage.csv", index=False)

fig, ax = plt.subplots()
ax.plot(annual["year"], 100 * annual["target_coverage"], marker="o", linewidth=2)
ax.axhline(95, color="#b91c1c", linestyle="--", linewidth=1, label="95%")
ax.set(title="Observed closing-auction targets by year", ylabel="Coverage (%)", xlabel="Year", ylim=(90, 100))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "annual_target_coverage.png", bbox_inches="tight")
plt.show()
'''),
        md('## Daily coverage and source anomalies'),
        code(r'''
daily = sql(f"""
WITH c AS (
  SELECT date, permno
  FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
  WHERE date >= DATE '2015-01-01'
), a AS (
  SELECT date, permno, auction_volume
  FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
), an AS (
  SELECT date, exclude_from_model, reason
  FROM read_parquet('{ANOMALIES}')
)
SELECT c.date, count(*) AS stocks, count(a.auction_volume) AS observed,
       count(a.auction_volume)::DOUBLE / count(*) AS coverage,
       coalesce(bool_or(an.exclude_from_model), false) AS exclude_from_model,
       any_value(an.reason) AS reason
FROM c LEFT JOIN a USING (date, permno)
LEFT JOIN an USING (date)
GROUP BY c.date ORDER BY c.date
""")
display(daily.nsmallest(12, "coverage"))
daily.to_csv(OUT / "daily_target_coverage.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily["date"], 100 * daily["coverage"], color="#1d4ed8", linewidth=0.8)
flagged = daily[daily["exclude_from_model"]]
ax.scatter(flagged["date"], 100 * flagged["coverage"], color="#b91c1c", s=45, zorder=4, label="Excluded source anomaly")
ax.axhline(95, color="#6b7280", linestyle="--", linewidth=1, label="95%")
ax.set(title="Daily closing-auction target coverage", ylabel="Coverage (%)", xlabel="")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "daily_target_coverage.png", bbox_inches="tight")
plt.show()
'''),
        md('## Integrity checks'),
        code(r'''
checks = sql(f"""
WITH a AS (
  SELECT * FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
), c AS (
  SELECT * FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
), t AS (
  SELECT * FROM read_parquet('{TRADE}', union_by_name=true, hive_partitioning=true)
)
SELECT 'auction duplicate date-permno groups' AS check_name,
       (SELECT count(*) FROM (SELECT date, permno FROM a GROUP BY 1,2 HAVING count(*) > 1)) AS issue_count
UNION ALL SELECT 'CRSP duplicate date-permno groups',
       (SELECT count(*) FROM (SELECT date, permno FROM c GROUP BY 1,2 HAVING count(*) > 1))
UNION ALL SELECT 'trade duplicate date-permno-checkpoint groups',
       (SELECT count(*) FROM (SELECT date, permno, checkpoint FROM t GROUP BY 1,2,3 HAVING count(*) > 1))
UNION ALL SELECT 'negative auction volumes', (SELECT count(*) FROM a WHERE auction_volume < 0)
UNION ALL SELECT 'nonpositive observed auction prices', (SELECT count(*) FROM a WHERE auction_volume IS NOT NULL AND auction_price <= 0)
UNION ALL SELECT 'auction share above 100%', (SELECT count(*) FROM a WHERE auction_share_daily_volume > 1)
UNION ALL SELECT 'target flag/value disagreements',
       (SELECT count(*) FROM a WHERE coalesce(target_found, false) <> (auction_volume IS NOT NULL))
""")
display(checks)
checks.to_csv(OUT / "integrity_checks.csv", index=False)
'''),
        md('## Missing targets by market-cap decile'),
        code(r'''
size_coverage = sql(f"""
WITH c AS (
  SELECT date, permno, market_cap_usd,
         ntile(10) OVER (PARTITION BY date ORDER BY market_cap_usd) AS size_decile
  FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
  WHERE date >= DATE '2015-01-01'
), a AS (
  SELECT date, permno, auction_volume
  FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
)
SELECT size_decile, count(*) AS stock_days, count(a.auction_volume) AS observed_targets,
       count(a.auction_volume)::DOUBLE / count(*) AS target_coverage
FROM c LEFT JOIN a USING (date, permno)
GROUP BY 1 ORDER BY 1
""")
display(size_coverage)
size_coverage.to_csv(OUT / "target_coverage_by_size_decile.csv", index=False)

fig, ax = plt.subplots()
sns.barplot(data=size_coverage, x="size_decile", y=size_coverage["target_coverage"] * 100, color="#2563eb", ax=ax)
ax.set(title="Target coverage by daily market-cap decile", xlabel="Market-cap decile (1 = smallest)", ylabel="Coverage (%)", ylim=(90, 100))
fig.tight_layout()
fig.savefig(OUT / "target_coverage_by_size_decile.png", bbox_inches="tight")
plt.show()
'''),
        md('## Intraday checkpoint coverage'),
        code(r'''
checkpoint = sql(f"""
SELECT checkpoint, count(*) AS rows,
       avg(checkpoint_data_found::INTEGER) AS data_found_rate,
       count(*) FILTER (WHERE checkpoint_after_market_close) AS after_close_rows,
       count(*) FILTER (WHERE cum_volume IS NULL) AS missing_cum_volume,
       count(*) FILTER (WHERE vwap IS NULL) AS missing_vwap
FROM read_parquet('{TRADE}', union_by_name=true, hive_partitioning=true)
GROUP BY 1 ORDER BY strptime(checkpoint, '%H:%M')
""")
display(checkpoint)
checkpoint.to_csv(OUT / "intraday_checkpoint_coverage.csv", index=False)

fig, ax = plt.subplots()
sns.barplot(data=checkpoint, x="checkpoint", y=checkpoint["data_found_rate"] * 100, color="#0f766e", ax=ax)
ax.set(title="Intraday trade data found by checkpoint", xlabel="Checkpoint", ylabel="Data found (%)", ylim=(98, 100))
fig.tight_layout()
fig.savefig(OUT / "intraday_checkpoint_coverage.png", bbox_inches="tight")
plt.show()
'''),
        md('## Auction schema drift'),
        code(r'''
files = sorted((DATA / "stock_daily/auction_daily").glob("year=*/*.parquet"))
schemas = {}
for file in files:
    year = int(file.parent.name.split("=")[1])
    schemas[year] = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file}')").fetchall()]

all_columns = sorted(set().union(*map(set, schemas.values())))
schema_matrix = pd.DataFrame({year: [col in columns for col in all_columns] for year, columns in schemas.items()}, index=all_columns)
schema_summary = pd.DataFrame({
    "year": sorted(schemas),
    "column_count": [len(schemas[y]) for y in sorted(schemas)],
    "schema_family": ["TAQ-oriented" if len(schemas[y]) < 40 else "CRSP-enriched" for y in sorted(schemas)],
})
display(schema_summary)
display(schema_matrix.loc[schema_matrix.nunique(axis=1) > 1].astype(int))
schema_summary.to_csv(OUT / "auction_schema_summary.csv", index=False)
schema_matrix.astype(int).to_csv(OUT / "auction_schema_matrix.csv")
'''),
        md('## Modeling sample definition'),
        code(r'''
splits = sql(f"""
WITH c AS (
  SELECT date, permno
  FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
  WHERE date >= DATE '2015-01-01'
), a AS (
  SELECT date, permno, auction_volume, target_found
  FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
), an AS (
  SELECT date, exclude_from_model FROM read_parquet('{ANOMALIES}')
)
SELECT CASE WHEN c.date <= DATE '2021-12-31' THEN 'Train: 2015–2021'
            WHEN c.date <= DATE '2023-12-31' THEN 'Validation: 2022–2023'
            ELSE 'Test: 2024–2025' END AS sample,
       count(*) AS stock_days,
       count(a.auction_volume) AS observed_targets,
       count(*) FILTER (WHERE a.target_found AND a.auction_volume IS NOT NULL
                         AND NOT coalesce(an.exclude_from_model, false)) AS eligible_targets,
       count(*) FILTER (WHERE a.target_found AND a.auction_volume IS NOT NULL
                         AND NOT coalesce(an.exclude_from_model, false))::DOUBLE / count(*) AS eligible_rate
FROM c LEFT JOIN a USING (date, permno)
LEFT JOIN an USING (date)
GROUP BY 1 ORDER BY min(c.date)
""")
display(splits)
splits.to_csv(OUT / "modeling_sample_by_split.csv", index=False)
'''),
        md('''
## Decisions from this notebook

1. Use 2015–2025 for target modeling; retain 2011–2014 only as daily-history context.
2. Exclude documented source-anomaly dates, currently 2019-08-12.
3. Keep missing targets as missing, never as zero.
4. Use `date × permno` and `date × permno × checkpoint` as required unique keys.
5. Mask checkpoint observations marked after the market close.
6. Build a canonical auction staging view before feature engineering because the yearly raw schemas differ.
7. Report results by year and size group because target coverage is not constant over time or across the cross-section.
'''),
    ]
    return cells


def notebook_02():
    cells = [
        md('''
# 02 — Closing-auction target EDA

This notebook studies the distribution and stability of the closing-auction target after applying the quality rules from Notebook 01.

**Targets examined**

- Raw closing-auction volume
- `log1p(auction_volume)`, the recommended modeling scale
- Auction volume divided by trailing 20-day average daily volume, a scale-normalized diagnostic target

`auction_share_daily_volume` is retained only for descriptive analysis because same-day final daily volume is unavailable before the close.
'''),
        code(COMMON_SETUP + '\nOUT = OUTPUT / "02_target_eda"\nOUT.mkdir(parents=True, exist_ok=True)'),
        md('## Create the eligible target panel'),
        code(r'''
con.execute(f"""
CREATE OR REPLACE TEMP VIEW eligible_targets AS
WITH c AS (
  SELECT *,
         avg(daily_volume) OVER (
           PARTITION BY permno ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
         ) AS avg20_daily_volume
  FROM read_parquet('{CRSP}', union_by_name=true, hive_partitioning=true)
  WHERE date >= DATE '2015-01-01'
), a AS (
  SELECT date, permno, auction_volume, auction_price, auction_share_daily_volume, target_found
  FROM read_parquet('{AUCTION}', union_by_name=true, hive_partitioning=true)
), e AS (
  SELECT * FROM read_parquet('{EVENTS}')
), an AS (
  SELECT date, exclude_from_model FROM read_parquet('{ANOMALIES}')
)
SELECT c.date, c.permno, c.ticker, c.market_cap_usd, c.daily_volume,
       c.return_total, c.avg20_daily_volume,
       a.auction_volume, a.auction_price, a.auction_share_daily_volume,
       ln(1 + a.auction_volume) AS log1p_auction_volume,
       a.auction_volume / nullif(c.avg20_daily_volume, 0) AS auction_to_adv20,
       e.day_of_week, e.is_monday, e.is_friday,
       e.is_last_trading_day_month, e.is_last_trading_day_quarter,
       e.monthly_options_expiry, e.quarterly_options_expiry,
       year(c.date) AS year, month(c.date) AS month
FROM c JOIN a USING (date, permno)
LEFT JOIN e USING (date)
LEFT JOIN an USING (date)
WHERE a.target_found AND a.auction_volume IS NOT NULL
  AND a.auction_volume >= 0
  AND NOT coalesce(an.exclude_from_model, false)
""")

panel_summary = sql("""
SELECT count(*) AS eligible_rows, count(DISTINCT date) AS trading_days,
       count(DISTINCT permno) AS securities, min(date) AS first_date, max(date) AS last_date
FROM eligible_targets
""")
display(panel_summary)
'''),
        md('## Distribution summary'),
        code(r'''
distribution = sql("""
SELECT count(*) AS n,
       min(auction_volume) AS min,
       quantile_cont(auction_volume, 0.01) AS p01,
       quantile_cont(auction_volume, 0.10) AS p10,
       quantile_cont(auction_volume, 0.25) AS p25,
       quantile_cont(auction_volume, 0.50) AS median,
       quantile_cont(auction_volume, 0.75) AS p75,
       quantile_cont(auction_volume, 0.90) AS p90,
       quantile_cont(auction_volume, 0.99) AS p99,
       max(auction_volume) AS max,
       avg(auction_volume) AS mean,
       stddev_samp(auction_volume) AS std_dev,
       skewness(auction_volume) AS skewness
FROM eligible_targets
""")
display(distribution.T.rename(columns={0: "value"}))
distribution.to_csv(OUT / "target_distribution_summary.csv", index=False)
'''),
        md('## Raw and transformed distributions'),
        code(r'''
sample = sql("""
SELECT auction_volume, log1p_auction_volume, auction_to_adv20, auction_share_daily_volume
FROM eligible_targets USING SAMPLE 400000 ROWS (reservoir, 20261011)
""")
raw_cap = sample["auction_volume"].quantile(0.995)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(sample.loc[sample["auction_volume"] <= raw_cap, "auction_volume"], bins=80, ax=axes[0], color="#2563eb")
axes[0].set(title="Raw auction volume, capped at sample 99.5th percentile", xlabel="Shares")
sns.histplot(sample["log1p_auction_volume"], bins=80, ax=axes[1], color="#0f766e")
axes[1].set(title="log1p closing-auction volume", xlabel="log1p(shares)")
fig.tight_layout()
fig.savefig(OUT / "target_raw_and_log_distribution.png", bbox_inches="tight")
plt.show()
'''),
        md('## Target distribution by year'),
        code(r'''
by_year = sql("""
SELECT year, count(*) AS observations,
       avg(auction_volume) AS mean_volume,
       median(auction_volume) AS median_volume,
       quantile_cont(auction_volume, 0.90) AS p90_volume,
       quantile_cont(auction_volume, 0.99) AS p99_volume,
       avg(log1p_auction_volume) AS mean_log_volume,
       median(auction_to_adv20) AS median_auction_to_adv20,
       median(auction_share_daily_volume) AS median_auction_share
FROM eligible_targets GROUP BY 1 ORDER BY 1
""")
display(by_year)
by_year.to_csv(OUT / "target_summary_by_year.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(by_year["year"], by_year["median_volume"], marker="o", color="#2563eb")
axes[0].set(title="Median auction volume by year", xlabel="Year", ylabel="Shares")
axes[1].plot(by_year["year"], 100 * by_year["median_auction_share"], marker="o", color="#0f766e")
axes[1].set(title="Median auction share of daily volume", xlabel="Year", ylabel="Percent")
fig.tight_layout()
fig.savefig(OUT / "target_by_year.png", bbox_inches="tight")
plt.show()
'''),
        md('## Market-wide time series'),
        code(r'''
daily_target = sql("""
SELECT date, count(*) AS observed_stocks,
       sum(auction_volume) AS total_auction_volume,
       median(auction_volume) AS median_stock_auction_volume,
       median(auction_share_daily_volume) AS median_auction_share,
       avg(log1p_auction_volume) AS mean_log_volume
FROM eligible_targets GROUP BY 1 ORDER BY 1
""")
daily_target["rolling_20d_median"] = daily_target["median_stock_auction_volume"].rolling(20, min_periods=10).median()
daily_target.to_csv(OUT / "daily_target_summary.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
axes[0].plot(daily_target["date"], daily_target["total_auction_volume"] / 1e9, linewidth=0.7, color="#2563eb")
axes[0].set(title="Aggregate observed NYSE closing-auction volume", ylabel="Billions of shares")
axes[1].plot(daily_target["date"], daily_target["rolling_20d_median"], linewidth=1.1, color="#0f766e")
axes[1].set(title="20-day rolling median of stock-level auction volume", ylabel="Shares", xlabel="")
fig.tight_layout()
fig.savefig(OUT / "daily_target_time_series.png", bbox_inches="tight")
plt.show()
'''),
        md('## Calendar effects'),
        code(r'''
calendar_effects = sql("""
SELECT effect, category, observations, median_volume, mean_log_volume, median_auction_share
FROM (
  SELECT 'Day of week' AS effect,
         CASE day_of_week WHEN 0 THEN 'Monday' WHEN 1 THEN 'Tuesday' WHEN 2 THEN 'Wednesday'
              WHEN 3 THEN 'Thursday' WHEN 4 THEN 'Friday' ELSE cast(day_of_week AS VARCHAR) END AS category,
         count(*) AS observations, median(auction_volume) AS median_volume,
         avg(log1p_auction_volume) AS mean_log_volume,
         median(auction_share_daily_volume) AS median_auction_share
  FROM eligible_targets GROUP BY day_of_week
  UNION ALL
  SELECT 'Last trading day of month', cast(is_last_trading_day_month AS VARCHAR), count(*),
         median(auction_volume), avg(log1p_auction_volume), median(auction_share_daily_volume)
  FROM eligible_targets GROUP BY is_last_trading_day_month
  UNION ALL
  SELECT 'Last trading day of quarter', cast(is_last_trading_day_quarter AS VARCHAR), count(*),
         median(auction_volume), avg(log1p_auction_volume), median(auction_share_daily_volume)
  FROM eligible_targets GROUP BY is_last_trading_day_quarter
  UNION ALL
  SELECT 'Monthly options expiry', cast(monthly_options_expiry AS VARCHAR), count(*),
         median(auction_volume), avg(log1p_auction_volume), median(auction_share_daily_volume)
  FROM eligible_targets GROUP BY monthly_options_expiry
  UNION ALL
  SELECT 'Quarterly options expiry', cast(quarterly_options_expiry AS VARCHAR), count(*),
         median(auction_volume), avg(log1p_auction_volume), median(auction_share_daily_volume)
  FROM eligible_targets GROUP BY quarterly_options_expiry
)
ORDER BY effect, category
""")
display(calendar_effects)
calendar_effects.to_csv(OUT / "calendar_effects.csv", index=False)

binary_effects = calendar_effects[(calendar_effects["effect"] != "Day of week") & (calendar_effects["category"] == "true")].copy()
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=binary_effects, y="effect", x="median_auction_share", color="#7c3aed", ax=ax)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
ax.set(title="Median auction share on selected calendar events", xlabel="Median share of daily volume", ylabel="")
fig.tight_layout()
fig.savefig(OUT / "calendar_event_auction_share.png", bbox_inches="tight")
plt.show()
'''),
        md('## Cross-sectional size relationship'),
        code(r'''
size_deciles = sql("""
WITH ranked AS (
  SELECT *, ntile(10) OVER (PARTITION BY date ORDER BY market_cap_usd) AS size_decile
  FROM eligible_targets WHERE market_cap_usd IS NOT NULL
)
SELECT size_decile, count(*) AS observations,
       median(market_cap_usd) AS median_market_cap,
       median(auction_volume) AS median_auction_volume,
       avg(log1p_auction_volume) AS mean_log_volume,
       median(auction_to_adv20) AS median_auction_to_adv20,
       median(auction_share_daily_volume) AS median_auction_share
FROM ranked GROUP BY 1 ORDER BY 1
""")
display(size_deciles)
size_deciles.to_csv(OUT / "target_by_market_cap_decile.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=size_deciles, x="size_decile", y="median_auction_volume", color="#2563eb", ax=axes[0])
axes[0].set(title="Median auction volume by market-cap decile", xlabel="Market-cap decile", ylabel="Shares")
sns.barplot(data=size_deciles, x="size_decile", y="median_auction_to_adv20", color="#0f766e", ax=axes[1])
axes[1].set(title="Median auction volume / trailing ADV", xlabel="Market-cap decile", ylabel="Ratio")
fig.tight_layout()
fig.savefig(OUT / "target_by_market_cap_decile.png", bbox_inches="tight")
plt.show()
'''),
        md('## Industry relationship'),
        code(r'''
industry = sql(f"""
WITH mapped AS (
  SELECT t.*, i.ff12_code, i.ff12_name,
         row_number() OVER (
           PARTITION BY t.date, t.permno
           ORDER BY i.secinfostartdt DESC, i.secinfoenddt DESC
         ) AS mapping_rank
  FROM eligible_targets t
  LEFT JOIN read_parquet('{INDUSTRY}') i
    ON t.permno = i.permno
   AND t.date BETWEEN i.secinfostartdt AND i.secinfoenddt
)
SELECT coalesce(ff12_name, 'Unmapped') AS industry,
       count(*) AS observations, count(DISTINCT permno) AS securities,
       median(auction_volume) AS median_auction_volume,
       avg(log1p_auction_volume) AS mean_log_volume,
       median(auction_to_adv20) AS median_auction_to_adv20,
       median(auction_share_daily_volume) AS median_auction_share
FROM mapped WHERE mapping_rank = 1
GROUP BY 1 ORDER BY median_auction_volume DESC
""")
display(industry)
industry.to_csv(OUT / "target_by_industry.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(data=industry, y="industry", x="median_auction_volume", color="#2563eb", ax=ax)
ax.set(title="Median auction volume by Fama–French 12 industry", xlabel="Shares", ylabel="")
fig.tight_layout()
fig.savefig(OUT / "target_by_industry.png", bbox_inches="tight")
plt.show()
'''),
        md('## Persistence of the target'),
        code(r'''
persistence = sql("""
WITH z AS (
  SELECT date, permno, log1p_auction_volume,
         lag(log1p_auction_volume, 1) OVER (PARTITION BY permno ORDER BY date) AS lag1,
         lag(log1p_auction_volume, 5) OVER (PARTITION BY permno ORDER BY date) AS lag5
  FROM eligible_targets
)
SELECT corr(log1p_auction_volume, lag1) AS pooled_lag1_correlation,
       corr(log1p_auction_volume, lag5) AS pooled_lag5_correlation,
       count(*) FILTER (WHERE lag1 IS NOT NULL) AS lag1_pairs,
       count(*) FILTER (WHERE lag5 IS NOT NULL) AS lag5_pairs
FROM z
""")
display(persistence)
persistence.to_csv(OUT / "target_persistence.csv", index=False)

stock_persistence = sql("""
WITH z AS (
  SELECT date, permno, log1p_auction_volume,
         lag(log1p_auction_volume, 1) OVER (PARTITION BY permno ORDER BY date) AS lag1
  FROM eligible_targets
), by_stock AS (
  SELECT permno, count(lag1) AS pairs, corr(log1p_auction_volume, lag1) AS lag1_correlation
  FROM z GROUP BY permno
)
SELECT * FROM by_stock WHERE pairs >= 100 AND lag1_correlation IS NOT NULL
""")
stock_persistence.to_csv(OUT / "stock_level_target_persistence.csv", index=False)

fig, ax = plt.subplots()
sns.histplot(stock_persistence["lag1_correlation"], bins=60, color="#0f766e", ax=ax)
ax.axvline(stock_persistence["lag1_correlation"].median(), color="#b91c1c", linestyle="--", label="Median stock")
ax.set(title="Stock-level lag-1 correlation of log auction volume", xlabel="Correlation", ylabel="Stocks")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "stock_level_target_persistence.png", bbox_inches="tight")
plt.show()
'''),
        md('## Compact findings table'),
        code(r'''
headline = pd.DataFrame({
    "finding": [
        "Eligible observations",
        "Median auction volume",
        "99th percentile auction volume",
        "Maximum auction volume",
        "Pooled lag-1 correlation on log scale",
        "Median stock-level lag-1 correlation",
    ],
    "value": [
        int(panel_summary.loc[0, "eligible_rows"]),
        float(distribution.loc[0, "median"]),
        float(distribution.loc[0, "p99"]),
        float(distribution.loc[0, "max"]),
        float(persistence.loc[0, "pooled_lag1_correlation"]),
        float(stock_persistence["lag1_correlation"].median()),
    ],
})
display(headline)
headline.to_csv(OUT / "headline_target_findings.csv", index=False)
'''),
        md('''
## Decisions from this notebook

1. Use `log1p(auction_volume)` as the primary estimation target and translate predictions back to shares for reporting.
2. Keep raw volume and volume-to-trailing-ADV as economic interpretation measures.
3. Always include stock-size controls or stock effects: much of the cross-sectional target difference is scale-related.
4. Preserve calendar flags because closing liquidity changes materially around month-end, quarter-end, and expiry dates.
5. Include lagged auction-volume features because the target is persistent, but measure their value against simple lag and rolling-median baselines.
6. Continue reporting results by year, size decile, industry, and event day; the target distribution is not stationary.
'''),
    ]
    return cells


def write_notebook(path, cells):
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(nb, path)


if __name__ == "__main__":
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "01_data_quality_and_coverage.ipynb", notebook_01())
    write_notebook(NOTEBOOK_DIR / "02_target_eda.ipynb", notebook_02())
    print("Built 01_data_quality_and_coverage.ipynb and 02_target_eda.ipynb")
