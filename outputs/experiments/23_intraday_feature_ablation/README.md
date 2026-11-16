# Intraday ridge feature group ablation

Four disjoint current-session groups are added individually to the daily-only control and removed individually from the full intraday ridge. Historical checkpoint information is retained throughout. Each candidate is refitted and tunes its own penalty historically. Source features carry their missingness flags together.

## Results

```text
           candidate  rmse_log     wape  aggregate_bias       period
           add_price  0.474695 0.357021       -0.130356  development
 add_recent_activity  0.463362 0.347248       -0.129573  development
          add_trades  0.462628 0.348187       -0.116432  development
          add_volume  0.461665 0.346535       -0.127438  development
          drop_price  0.460058 0.345900       -0.117825  development
drop_recent_activity  0.460882 0.346866       -0.118925  development
         drop_trades  0.460400 0.345539       -0.126215  development
         drop_volume  0.460601 0.346259       -0.117991  development
         fixed_daily  0.469491 0.354676       -0.144684  development
 ridge_daily_control  0.474878 0.356962       -0.129938  development
        ridge_update  0.459954 0.345919       -0.118001  development
           add_price  0.426436 0.330452       -0.078781 confirmation
 add_recent_activity  0.411851 0.319387       -0.075220 confirmation
          add_trades  0.413208 0.320141       -0.070203 confirmation
          add_volume  0.411763 0.319035       -0.074136 confirmation
          drop_price  0.409249 0.317750       -0.069038 confirmation
drop_recent_activity  0.410568 0.318684       -0.070733 confirmation
         drop_trades  0.410082 0.318035       -0.072661 confirmation
         drop_volume  0.409697 0.317939       -0.069382 confirmation
         fixed_daily  0.429014 0.328298       -0.090003 confirmation
 ridge_daily_control  0.426726 0.330393       -0.078790 confirmation
        ridge_update  0.409157 0.317815       -0.069106 confirmation
```

The group definitions and frozen design are in experiment_plan.json. Segment diagnostics include disjoint calendar events, liquidity thirds, and month-end crossed with liquidity. Headline and development segment metrics are equal-year averages; annual segment counts are retained separately.

Add-one changes are compared with daily-only ridge; drop-one changes with full intraday ridge. These are conditional predictive comparisons, not a causal or additive allocation of importance. Correlated groups may substitute for one another. Twenty-date block intervals concern daily log MAE, not headline log RMSE; unadjusted multiple comparisons are exploratory. 2025 is previously inspected confirmation and is not used to choose groups. No specialized routing or production promotion is performed.

All 179 checks passed. Existing references were preserved and reconstructed, historical sample and forecast hashes verified, preprocessing and bounds matched, optimized ridge fitting compared to the original helper, and saved forecasts checked for exact keys, nonnegativity, fallback, model reconstruction and exhaustive segment counts.
