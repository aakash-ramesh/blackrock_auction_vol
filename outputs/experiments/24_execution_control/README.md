# Close-targeted execution prototype

Reproduce from the repository root with `.venv/bin/python notebooks/simulate_execution_control.py`. The replay uses the project's 2024–2025 auction and trade-checkpoint Parquet files and requires DuckDB, NumPy, pandas, and Matplotlib.

## Replay design

The parent order arrives at noon ET. The policy schedules continuous shares for noon–14:00, observes the first window's volume and its simulated fill at 14:00, and schedules continuous shares for 14:00–15:30. It commits the remaining inventory to the closing auction at 15:30. The continuous-only baseline leaves its residual unfilled. The fixed split requests 25% at noon and catches up to 50% continuous execution at 14:00 where capacity permits.

The trailing-history optimizer uses 60 completed eligible sessions for the stock. The checkpoint version selects the 36 of those sessions closest in log cumulative noon volume and groups them by noon–14:00 volume into thirds. At 14:00, it selects the group using the just-completed window. Each historical scenario keeps both execution-window volumes and the auction volume from the same day. The dynamic program uses a 21-point inventory and action grid, expected scenario costs at 14:00, and linear interpolation for the continuation value at noon.

Cohort selection uses 2024 data. Among securities with at least 220 usable sessions, 24 stocks are selected at evenly spaced ranks of median window volume. Usable replay stock-days have positive matched official auction prices and volumes and valid checkpoint increments on regular NYSE sessions. Parent order sizes are 1% and 5% of the trailing median of noon–15:30 continuous volume plus auction volume. Both order sizes and all policies use the same 5,925 stock-days on 247 dates in 2025.

## Cost and fill assumptions

All costs are in basis points per parent-order notional. Continuous fills are capped at 10% of each window's observed volume and auction fills at 5% of auction volume in the base case. The stress cases change the auction cap to 2.5% and 10% and re-optimize. These caps are scenario assumptions about executable shares. The public tape has no simulated order's queue position, fills, or market impact.

The model charges 2 bps of continuous spread per filled share. Temporary impact per filled share is 60 bps multiplied by continuous participation or 100 bps multiplied by auction participation. A 100-bp charge per unfilled share prices completion risk. The close-tracking risk weight is 0.001 per bp. The variance-rate proxy is the trailing variance of log price changes between the two interval VWAPs in bps, divided by the 1.75-hour spacing between interval midpoints. Tracking exposure is the fraction already executed, and accrues through the close under the zero-drift benchmark model. The modeled friction-and-penalty column excludes the risk term. The empirical timing RMS is computed from the observed VWAP and auction-price differences weighted by simulated fills.

The replay proxies each execution window with aggregate volume and VWAP, treats historical prices and liquidity as exogenous, and assumes uniform fills within each window for the risk integral. The reported objective is conditional on those assumptions. The 2025 replay excludes incomplete rows, and the sample is a fixed liquidity-stratified cohort.

## Output files

- [Summary](summary.csv) has costs, allocations, completion, and close timing risk for each policy, order size, and auction cap.
- [Paired comparisons](paired_differences.csv) report the checkpoint policy minus trailing history. The intervals resample whole dates and preserve within-date cross-sectional dependence. They do not account for serial dependence or cost model uncertainty.
- [Stock-day replay](stock_day_replay.csv.gz) contains each order, request, proxy fill, and cost component. Fill and order allocation columns are parent-order fractions.
- [Cohort](cohort.csv) lists the 24 stocks and their 2024 selection statistics.
- [Grid check](grid_sensitivity.csv) reports the change with 41 grid points for six stocks. Mean objective changes are below 0.007 bps in absolute value for both optimized policies at the base auction cap.
- [Results chart](execution_results.png) shows the mean objective and its components at the base cap.
