# Frozen intraday forecast robustness audit

No models were refitted. Annual and monthly results and partitions by prior size, liquidity, history, disjoint calendar events and actual-volume tails are in the accompanying CSVs. Actual-tail membership is an ex-post diagnostic, never a usable prediction input. Early closes remain excluded.

## Main results

```text
              method       period  years  mean_annual_rmse_change  mean_annual_wape_change  months  months_rmse_improved  months_wape_improved
        ridge_update  development      3                -0.009537                -0.008756      36                    32                    31
        ridge_update confirmation      1                -0.019857                -0.010483      12                    12                    11
lgbm_intraday_update  development      3                -0.007345                -0.007021      36                    30                    29
lgbm_intraday_update confirmation      1                -0.013420                -0.007080      12                    12                    11
```

## Serial dependence sensitivity

```text
              method       period  block_trading_dates  mean_log_mae_change  ci_lower  ci_upper
        ridge_update  development                    5            -0.010020 -0.011870 -0.008637
        ridge_update  development                   20            -0.010020 -0.012011 -0.008560
        ridge_update confirmation                    5            -0.017825 -0.019720 -0.015220
        ridge_update confirmation                   20            -0.017825 -0.020562 -0.014475
lgbm_intraday_update  development                    5            -0.007307 -0.009429 -0.005812
lgbm_intraday_update  development                   20            -0.007307 -0.009743 -0.006153
lgbm_intraday_update confirmation                    5            -0.009011 -0.010580 -0.007313
lgbm_intraday_update confirmation                   20            -0.009011 -0.010908 -0.006799
```

Changes are update minus daily anchor; negative error changes favor the update. Block intervals concern equal-year mean daily log MAE, not log RMSE or aggregate WAPE. Five- and twenty-trading-date blocks are sampled within years. These are retrospective diagnostics, unadjusted for multiple comparisons, and not pristine holdout validation.

A matching daily-only ridge control remains necessary to isolate the contribution of current checkpoint features to ridge gains. This audit makes no promotion decision, changes no model, and has not implemented other checkpoints or new event data.

## Interpretation

Ridge improves annual log RMSE in every evaluated year, 32 of 36 development months, and all 12 confirmation months. Both 5-date and 20-date bootstrap intervals favor both update families. However, equal-year development segment means show worse ridge log RMSE for the low-liquidity and large-stock groups. Disjoint month-end days worsen in both log RMSE and WAPE; monthly expiry WAPE is slightly worse. The top actual-volume 1 percent still has roughly 43 percent negative aggregate bias. These are exploratory segment findings, not thresholds chosen to route predictions. See the annual segment counts before interpreting rare groups.

All 19 metric/partition checks passed; all four source forecast fingerprints verified.
