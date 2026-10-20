
WITH crsp_row AS (
    SELECT date, permno, ticker, close, open, high, low, bid, ask,
           daily_volume, daily_price_volume, return_total, market_cap_usd,
           shares_outstanding_thousands, ordinary_dividend, nonordinary_dividend,
           daily_price_factor, distribution_flag,
           security_type, security_subtype, share_type, us_incorporated, issuer_type,
           daily_volume * 1.0 / nullif(shares_outstanding_thousands * 1000.0, 0) AS turnover,
           abs(return_total) * 1000000.0 / nullif(daily_price_volume, 0) AS amihud,
           CASE WHEN ask >= bid AND ask + bid > 0
                THEN (ask - bid) / ((ask + bid) / 2.0) END AS quoted_spread,
           (high - low) / nullif(close, 0) AS hl_range,
           open / nullif(lag(close) OVER (PARTITION BY permno ORDER BY date), 0) - 1 AS overnight_gap
    FROM read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/stock_daily/crsp_daily/year=*/*.parquet', union_by_name=true, hive_partitioning=true)
), auction AS (
    SELECT date, permno, auction_volume, auction_price,
           auction_share_daily_volume, target_found
    FROM read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/stock_daily/auction_daily/year=*/*.parquet', union_by_name=true, hive_partitioning=true)
), base AS (
    SELECT c.*, a.auction_volume, a.auction_price, a.auction_share_daily_volume,
           coalesce(a.target_found, false) AS target_found
    FROM crsp_row c LEFT JOIN auction a USING (date, permno)
), stock_history AS (
    SELECT *,
           lag(auction_volume, 1) OVER w AS lag1_auction_volume,
           lag(auction_volume, 2) OVER w AS lag2_auction_volume,
           lag(auction_volume, 5) OVER w AS lag5_auction_volume,
           avg(auction_volume) OVER w5 AS avg5_auction_volume,
           avg(auction_volume) OVER w20 AS avg20_auction_volume,
           avg(auction_volume) OVER w60 AS avg60_auction_volume,
           median(auction_volume) OVER w20 AS median20_auction_volume,
           median(auction_volume) OVER w60 AS median60_auction_volume,
           stddev_samp(auction_volume) OVER w20 AS std20_auction_volume,
           stddev_samp(auction_volume) OVER w60 AS std60_auction_volume,
           lag(auction_share_daily_volume, 1) OVER w AS lag1_auction_share,
           avg(auction_share_daily_volume) OVER w20 AS avg20_auction_share,
           stddev_samp(auction_share_daily_volume) OVER w20 AS std20_auction_share,
           date_diff('day', max(CASE WHEN auction_volume IS NOT NULL THEN date END) OVER wall, date)
             AS days_since_last_auction,
           count(auction_volume) OVER wall AS prior_auction_observations,
           count(*) OVER wall AS prior_stock_observations,
           date_diff('day', min(date) OVER (PARTITION BY permno), date) AS data_age_days,
           lag(daily_volume, 1) OVER w AS lag1_daily_volume,
           avg(daily_volume) OVER w5 AS avg5_daily_volume,
           avg(daily_volume) OVER w20 AS avg20_daily_volume,
           avg(daily_volume) OVER w60 AS avg60_daily_volume,
           stddev_samp(daily_volume) OVER w20 AS std20_daily_volume,
           stddev_samp(daily_volume) OVER w60 AS std60_daily_volume,
           lag(daily_price_volume, 1) OVER w AS lag1_dollar_volume,
           avg(daily_price_volume) OVER w20 AS avg20_dollar_volume,
           lag(turnover, 1) OVER w AS lag1_turnover,
           avg(turnover) OVER w20 AS avg20_turnover,
           lag(amihud, 1) OVER w AS lag1_amihud,
           avg(amihud) OVER w20 AS avg20_amihud,
           lag(quoted_spread, 1) OVER w AS lag1_quoted_spread,
           lag(hl_range, 1) OVER w AS lag1_hl_range,
           lag(return_total, 1) OVER w AS lag1_return,
           lag(abs(return_total), 1) OVER w AS lag1_abs_return,
           exp(sum(CASE WHEN return_total > -1 THEN ln(1 + return_total) END) OVER w5) - 1 AS momentum5,
           exp(sum(CASE WHEN return_total > -1 THEN ln(1 + return_total) END) OVER w20) - 1 AS momentum20,
           exp(sum(CASE WHEN return_total > -1 THEN ln(1 + return_total) END) OVER w60) - 1 AS momentum60,
           stddev_samp(return_total) OVER w5 AS volatility5,
           stddev_samp(return_total) OVER w20 AS volatility20,
           stddev_samp(return_total) OVER w60 AS volatility60,
           max(abs(return_total)) OVER w20 AS max_abs_return20,
           lag(close, 1) OVER w AS lag1_price,
           lag(overnight_gap, 1) OVER w AS lag1_overnight_gap,
           lag(market_cap_usd, 1) OVER w AS lag1_market_cap,
           lag(shares_outstanding_thousands * 1000.0, 1) OVER w AS lag1_shares_outstanding,
           lag((ordinary_dividend IS NOT NULL AND ordinary_dividend != 0)
               OR (nonordinary_dividend IS NOT NULL AND nonordinary_dividend != 0), 1) OVER w AS lag1_dividend_flag,
           lag(coalesce(distribution_flag, 'NO') != 'NO', 1) OVER w AS lag1_distribution_flag,
           lag(daily_price_factor IS NOT NULL AND abs(daily_price_factor - 1.0) > 1e-12, 1) OVER w
             AS lag1_price_adjustment_flag,
           lag(security_type, 1) OVER w AS lag1_security_type,
           lag(security_subtype, 1) OVER w AS lag1_security_subtype,
           lag(share_type, 1) OVER w AS lag1_share_type,
           lag(issuer_type, 1) OVER w AS lag1_issuer_type,
           lag(us_incorporated, 1) OVER w AS lag1_us_incorporated
    FROM base
    WINDOW
        w AS (PARTITION BY permno ORDER BY date),
        wall AS (PARTITION BY permno ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        w5 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
        w20 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        w60 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING)
), stock_features AS (
    SELECT *,
           std20_auction_volume / nullif(avg20_auction_volume, 0) AS auction_cv20,
           avg5_auction_volume / nullif(avg20_auction_volume, 0) AS auction_short_long_ratio,
           lag1_auction_volume / nullif(avg20_daily_volume, 0) AS lag1_auction_to_adv20,
           lag1_auction_volume > 2 * median20_auction_volume AS auction_spike_lag1,
           lag1_daily_volume / nullif(avg20_daily_volume, 0) AS abnormal_volume_lag1,
           avg5_daily_volume / nullif(avg20_daily_volume, 0) AS volume_short_long_ratio
    FROM stock_history
), market_history AS (
    SELECT date,
           lag(market_return_vw, 1) OVER w AS lag1_market_return,
           stddev_samp(market_return_vw) OVER w20 AS market_volatility20,
           lag(market_total_volume, 1) OVER w AS lag1_market_volume,
           lag(market_total_dollar_volume, 1) OVER w AS lag1_market_dollar_volume,
           lag(market_total_volume, 1) OVER w / nullif(avg(market_total_volume) OVER w20, 0)
             AS market_abnormal_volume_lag1,
           lag(market_return_dispersion, 1) OVER w AS lag1_market_dispersion,
           lag(market_breadth, 1) OVER w AS lag1_market_breadth,
           lag(market_median_hl_range, 1) OVER w AS lag1_market_hl_range
    FROM read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/market/market_daily.parquet')
    WINDOW w AS (ORDER BY date), w20 AS (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
), sector_history AS (
    SELECT date, ff12_code,
           lag(sector_return_vw, 1) OVER w AS lag1_sector_return,
           stddev_samp(sector_return_vw) OVER w20 AS sector_volatility20,
           lag(sector_total_volume, 1) OVER w AS lag1_sector_volume,
           lag(sector_total_dollar_volume, 1) OVER w AS lag1_sector_dollar_volume,
           lag(sector_return_dispersion, 1) OVER w AS lag1_sector_dispersion,
           lag(sector_breadth, 1) OVER w AS lag1_sector_breadth
    FROM read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/sector/sector_daily.parquet')
    WINDOW
        w AS (PARTITION BY ff12_code ORDER BY date),
        w20 AS (PARTITION BY ff12_code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
), candidate AS (
    SELECT s.*, m.* EXCLUDE(date),
           e.* EXCLUDE(date, year, month, quarter, day),
           i.ff12_code, i.ff12_name, sh.* EXCLUDE(date, ff12_code),
           coalesce(an.exclude_from_model, false) AS exclude_source_anomaly,
           row_number() OVER (
               PARTITION BY s.date, s.permno
               ORDER BY i.secinfostartdt DESC NULLS LAST, i.secinfoenddt DESC NULLS LAST
           ) AS industry_rank
    FROM stock_features s
    LEFT JOIN market_history m USING (date)
    LEFT JOIN read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/events/event_calendar.parquet') e USING (date)
    LEFT JOIN read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/reference/industry_history.parquet') i
      ON s.permno = i.permno
     AND s.date BETWEEN i.secinfostartdt AND i.secinfoenddt
    LEFT JOIN sector_history sh
      ON s.date = sh.date AND i.ff12_code = sh.ff12_code
    LEFT JOIN read_parquet('/Users/aramesh/Documents/GitHub/blackrock_auction_vol/blackrock_data/quality/auction_source_anomalies.parquet') an USING (date)
    WHERE s.date >= DATE '2015-01-01'
), deduplicated AS (
    SELECT * EXCLUDE(industry_rank) FROM candidate WHERE industry_rank = 1
), ranked AS (
    SELECT *,
           percent_rank() OVER (PARTITION BY date ORDER BY lag1_market_cap) AS size_percentile,
           percent_rank() OVER (PARTITION BY date ORDER BY avg20_daily_volume) AS liquidity_percentile,
           percent_rank() OVER (PARTITION BY date ORDER BY avg20_turnover) AS turnover_percentile,
           percent_rank() OVER (PARTITION BY date ORDER BY lag1_auction_to_adv20) AS auction_intensity_percentile,
           ntile(10) OVER (PARTITION BY date ORDER BY lag1_market_cap) AS size_decile,
           ntile(10) OVER (PARTITION BY date ORDER BY avg20_daily_volume) AS liquidity_decile
    FROM deduplicated
)
SELECT
    date, permno, ticker, year(date) AS year, month(date) AS month,
    CASE WHEN date <= DATE '2021-12-31' THEN 'train'
         WHEN date <= DATE '2023-12-31' THEN 'validation'
         ELSE 'test' END AS sample_split,
    target_found AS observed_target, exclude_source_anomaly,
    target_found AND auction_volume IS NOT NULL AND NOT exclude_source_anomaly AS eligible_target,
    auction_volume,
    CASE WHEN auction_volume IS NOT NULL THEN ln(1 + auction_volume) END AS log1p_auction_volume,
    auction_price, auction_share_daily_volume,
    daily_volume AS current_daily_volume, return_total AS current_return, close AS current_close,
    lag1_auction_volume, lag2_auction_volume, lag5_auction_volume,
    avg5_auction_volume, avg20_auction_volume, avg60_auction_volume,
    median20_auction_volume, median60_auction_volume,
    std20_auction_volume, std60_auction_volume, auction_cv20,
    auction_short_long_ratio, lag1_auction_share, avg20_auction_share,
    std20_auction_share, lag1_auction_to_adv20, auction_spike_lag1,
    days_since_last_auction, prior_auction_observations,
    lag1_daily_volume, avg5_daily_volume, avg20_daily_volume, avg60_daily_volume,
    std20_daily_volume, std60_daily_volume, abnormal_volume_lag1,
    volume_short_long_ratio, lag1_dollar_volume, avg20_dollar_volume,
    lag1_turnover, avg20_turnover, lag1_amihud, avg20_amihud,
    lag1_quoted_spread, lag1_hl_range,
    lag1_return, lag1_abs_return, momentum5, momentum20, momentum60,
    volatility5, volatility20, volatility60, max_abs_return20,
    lag1_price, lag1_overnight_gap,
    lag1_market_cap, lag1_shares_outstanding, prior_stock_observations,
    data_age_days, size_percentile, liquidity_percentile, turnover_percentile,
    auction_intensity_percentile, size_decile, liquidity_decile,
    lag1_market_return, market_volatility20, lag1_market_volume,
    lag1_market_dollar_volume, market_abnormal_volume_lag1,
    lag1_market_dispersion, lag1_market_breadth, lag1_market_hl_range,
    lag1_sector_return, sector_volatility20, lag1_sector_volume,
    lag1_sector_dollar_volume, lag1_sector_dispersion, lag1_sector_breadth,
    lag1_return - lag1_sector_return AS stock_minus_sector_return,
    lag1_daily_volume / nullif(lag1_sector_volume, 0) AS stock_share_sector_volume,
    is_monday, is_friday, day_of_week,
    trading_day_of_month, trading_days_to_month_end,
    is_first_trading_day_month, is_last_trading_day_month,
    is_last_3_trading_days_month, is_last_5_trading_days_month,
    trading_days_to_quarter_end, is_last_trading_day_quarter,
    is_last_3_trading_days_quarter, is_last_5_trading_days_quarter,
    trading_days_to_year_end, is_last_trading_day_year,
    monthly_options_expiry, quarterly_options_expiry,
    after_long_market_gap, before_long_market_gap, is_early_close,
    early_close_month_end_window, expiry_month_end_window,
    ff12_code, ff12_name, lag1_dividend_flag, lag1_distribution_flag,
    lag1_price_adjustment_flag, lag1_security_type, lag1_security_subtype,
    lag1_share_type, lag1_issuer_type, lag1_us_incorporated
FROM ranked
