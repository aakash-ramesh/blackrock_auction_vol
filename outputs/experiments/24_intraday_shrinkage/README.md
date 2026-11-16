# Historical intraday correction shrinkage

The development-only decision retains **ridge_update**. Model strength is chosen on prior historical validation rows, never evaluation-year labels. The development decision was saved and fingerprinted before scoring 2025, and verified unchanged afterward. The specification itself is motivated by previously inspected diagnostics and is retrospective research.

```text
                 candidate  rmse_log     wape  aggregate_bias       period
               fixed_daily  0.469491 0.354676       -0.144684  development
             global_shrink  0.458352 0.345433       -0.125583  development
      low_liquidity_shrink  0.456747 0.345981       -0.118774  development
month_end_liquidity_shrink  0.456957 0.346031       -0.118845  development
          month_end_shrink  0.459929 0.345861       -0.118201  development
              ridge_update  0.459954 0.345919       -0.118001  development
               fixed_daily  0.429014 0.328298       -0.090003 confirmation
             global_shrink  0.409777 0.316436       -0.074786 confirmation
      low_liquidity_shrink  0.410782 0.317977       -0.069521 confirmation
month_end_liquidity_shrink  0.410856 0.318032       -0.072724 confirmation
          month_end_shrink  0.409792 0.317788       -0.072193 confirmation
              ridge_update  0.409157 0.317815       -0.069106 confirmation
```

## Design and interpretation

Strength 0 retains the daily anchor exactly; 1 retains the full bounded intraday ridge exactly. Intermediate strengths interpolate log1p share forecasts without re-clipping, preserving endpoints. Each year uses saved historical tuning and validation samples to reconstruct the original validation ridge forecast and then fit strength; the same validation year selected the original ridge penalty, so this is joint historical tuning rather than an independent nested calibration split. Full evaluation-year ridge and anchor forecasts are unchanged.

Global, disjoint month-end, low-liquidity and combined segment variants were fixed before fitting. Non-targeted rows retain full ridge. Target cells with fewer than 8 dates or 500 rows use the global strength. Missing liquidity is treated as not low and separately reported. Read selected_strengths.csv for sparse-cell fallbacks and counts.

Development means weight years equally. Gates, all annual metrics, and separate 2025 confirmation results are saved. A lower subgroup error alone cannot promote a candidate. Confidence intervals concern paired daily log MAE using 2000 within-year moving 20-date blocks, not headline log RMSE or WAPE; multiple comparisons are unadjusted. 2025 was already inspected in earlier research. Early closes remain excluded; missing checkpoints retain daily forecasts exactly.

All 152 checks passed, including input hashes, original validation forecast reproduction, chronology, exact alpha endpoints, forecast bounds between endpoints, fallback, saved-parameter reconstruction, written forecasts, exhaustive group counts and the frozen development decision.
