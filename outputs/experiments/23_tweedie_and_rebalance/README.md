# Daily NYSE auction shares with Tweedie loss and scheduled event features

## Results

Each number below weights 2022–2024 equally.

| Model | Log RMSE | Stock-level WAPE | Daily-total WAPE | Share bias |
|---|---:|---:|---:|---:|
| Published LightGBM | 0.46941 | 35.45% | 17.25% | −14.40% |
| Matched log control | 0.47109 | 35.47% | 17.79% | −15.25% |
| Log plus events | 0.46795 | 35.25% | 17.59% | −15.34% |
| Tweedie | 0.52620 | 37.79% | 14.21% | −4.74% |
| Tweedie plus events | 0.51770 | 37.58% | 13.45% | −4.74% |

The log model with events improved stock-level WAPE and log RMSE in all four years compared with the matched log control. Its development WAPE improvement was 0.22 percentage points, with a paired 95% interval of 0.06 to 0.43 points. The block interval for its change in aggregate bias crosses zero. Its development Russell-close daily-total WAPE fell from 57.01% to 40.35%, across three sessions.

The combined Tweedie model improved daily-total WAPE versus its matched log-plus-events control by 4.14 points, with a paired 95% interval of 3.27 to 4.94 points. Stock-level WAPE rose by 2.32 points. The original prior-year daily-total multiplier benchmark was 14.34% WAPE in development. The combined Tweedie model reached 13.45% WAPE, with yearly values of 11.92%, 11.85%, and 16.59%. The multiplier had lower error in 2024. On the previously inspected 2025 period, Tweedie plus events recorded 10.63% daily-total WAPE and +0.04% share bias, versus 12.25% and −8.90% for published LightGBM.

All 1,271,378 evaluation stock-days matched the historical evaluation keys and labels. The full annual table includes ordinary month-end, quarter-end, Russell, and realized top-1% auction slices. The `regular` group contains dates outside the three event categories. [Annual results](annual_metrics.csv) · [Development results](development_metrics.csv) · [Paired intervals](paired_development_intervals.csv)

## Design

The experiment crosses the existing median-relative squared-error log target and raw-share Tweedie loss with the compact 58 daily features and the same features plus four event predictors. All four candidates use the same stock-days and tree settings within each fold. The saved 58-feature squared-error LightGBM serves as a historical reference. Comparisons that isolate the new loss or event features use the freshly fitted controls.

For test year Y, tuning fits on Y−6 through Y−2, selects using Y−1, and refits on Y−5 through Y−1. Uniform seeded reservoir samples cap these sets at 180,000, 100,000 and 300,000 stock-days. Evaluation includes every eligible stock-day in Y. The saved original training samples were unavailable in this checkout. Fresh samples came from the local daily panel cache, and all test keys and labels matched the frozen ridge reference. The saved LightGBM reference reproduced its published annual log RMSE, WAPE and share bias on these rows. Development averages weight 2022–2024 equally. The project previously inspected the 2025 confirmation period.

The Tweedie target is auction shares divided by a constant fitted on the training sample. A log-link offset starts predictions at each stock's trailing 20-observation median plus one share, with a training-only median fallback. This preserves the historic scale anchor. The model is fitted using powers 1.3 and 1.7. Early stopping uses historical validation Tweedie loss at the candidate power. The selection across powers uses one common validation loss at power 1.5 after bounds. The log candidates select iterations by validation log RMSE. Each refit applies the same annual lower and upper share bounds to all four candidates, using the 0.1st and 99.9th percentiles of refit labels. The common search has 31 leaves per tree and a cap of 1,000 rounds. The log control reached that cap in two folds. The run used Python 3.13.13, LightGBM 4.6.0, DuckDB 1.5.6, pandas 3.1.1, and NumPy 2.4.6.

## Prior-close event features

- Russell US annual reconstitution close flag.
- Each stock's average log auction lift over up to six strictly earlier events of the same type. Lift is the difference between log1p auction shares and log1p trailing 20-row median on those historical dates, bounded to [−3, 3] before averaging.
- The stock's count of observations contributing to that event history.
- The existing lagged stock size percentile, active on these event dates.

The event types are Russell close, other quarter-end, and other month-end, in that precedence order. Stock history for a type uses only earlier dated observations. Each reported Russell segment contains one close per year. Event flags are available in advance, and stock size and previous auction outcomes are known by the prior close. This experiment uses historical auction response and stock size as proxies for stock exposure. Point-in-time index additions, deletions and weight changes were unavailable across the full training span. Published Russell dates come from the historical June scheduling rule and annual announcements. [Historical rule](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/research/four-decades-russell-reconstitution.pdf) · [2017](https://research.ftserussell.com/products/index-notices/home/getmethodology/?id=2062382) · [2019](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2019/ftse-russell-announces-2019-russell-us-indexes-reconstitution-schedule) · [2022](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2022/ftse-russell-announces-2022-russell-us-indexes-reconstitution-schedule) · [2023](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2023/ftse-russell-announces-2023-russell-us-indexes-reconstitution-schedule) · [2024](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2024/russell-reconstitution-2024-schedule) · [2025](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2025/ftse-russell-announces-2025-us-indexes-reconstitution-schedule)

## Reproduce

From the repository root with LightGBM 4.6.0 and its OpenMP runtime available in the Python environment:

```sh
.venv/bin/python -u notebooks/compare_daily_tweedie_and_rebalance.py
```

The script supports individual `--year` runs and a final `--summarize` run. The summary includes paired 95% intervals from 4,000 circular moving-block draws of 20 trading sessions within each development year. The intervals condition on the fitted forecasts and use common resampled sessions across models. The top 1% segment uses realized volume for evaluation. The annual fit manifests record sample fingerprints and selections.
