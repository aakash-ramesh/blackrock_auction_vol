# Quarterly walk-forward audit of the full 15:30 ridge

The quarterly procedure improves log RMSE versus the daily anchor in all 16 quarters. It has a small aggregate point-estimate advantage over the original annual intraday ridge, but their development paired log-RMSE and WAPE intervals include zero. Do not declare quarterly refitting superior or automatically replace the annual reference.

```text
      candidate  rmse_log     wape  aggregate_bias       period
   annual_ridge  0.459954 0.345919       -0.118001  development
    fixed_daily  0.469491 0.354676       -0.144684  development
quarterly_ridge  0.458885 0.345259       -0.117596  development
   annual_ridge  0.409157 0.317815       -0.069106 confirmation
    fixed_daily  0.429014 0.328298       -0.090003 confirmation
quarterly_ridge  0.408919 0.316324       -0.070642 confirmation
```

## Direct 20-date block intervals

```text
      period    reference               metric  block_dates  difference  ci_lower  ci_upper
 development  fixed_daily             rmse_log           20   -0.010606 -0.012352 -0.008860
 development  fixed_daily                 wape           20   -0.009416 -0.011049 -0.007703
 development  fixed_daily          signed_bias           20    0.027088  0.019516  0.034813
 development  fixed_daily absolute_annual_bias           20   -0.027088 -0.034813 -0.019516
 development annual_ridge             rmse_log           20   -0.001069 -0.002129  0.000052
 development annual_ridge                 wape           20   -0.000660 -0.001776  0.000445
 development annual_ridge          signed_bias           20    0.000405 -0.004511  0.005465
 development annual_ridge absolute_annual_bias           20   -0.000405 -0.005465  0.004511
confirmation  fixed_daily             rmse_log           20   -0.020095 -0.023269 -0.016091
confirmation  fixed_daily                 wape           20   -0.011973 -0.015885 -0.008105
confirmation  fixed_daily          signed_bias           20    0.019362  0.008380  0.031042
confirmation  fixed_daily absolute_annual_bias           20   -0.019362 -0.031042 -0.008380
confirmation annual_ridge             rmse_log           20   -0.000238 -0.000923  0.000441
confirmation annual_ridge                 wape           20   -0.001490 -0.002867 -0.000326
confirmation annual_ridge          signed_bias           20   -0.001535 -0.004043  0.001014
confirmation annual_ridge absolute_annual_bias           20    0.001535 -0.001014  0.004043
```

Each of 16 quarters uses the preceding complete quarter to choose a penalty, earlier 2021-onward history for tuning fits, and all available earlier history for final refitting. No shrinkage calibration stage is used. The daily LightGBM anchor remains annually fitted. New samples, fitting windows and bounds accompany the quarterly schedule, so the annual comparison does not isolate frequency alone. Actual sample keys and all fitted objects are saved.

Development years are 2022-2024; previously inspected 2025 is separate confirmation. This is retrospective evaluation of a fixed procedure, not a fresh holdout. Block intervals recompute annual RMSE/WAPE/bias and then equal-year means. They condition on fitted forecasts and exclude refitting and prior research-selection uncertainty. Quarter counts are descriptive, not independent significance tests. No new period or other checkpoint is scored.

All 178 checks passed across 1,260,168 rows. The report covering all intraday work is in reports/intraday_progress_2026_11_16.
