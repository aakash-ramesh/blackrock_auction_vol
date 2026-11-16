# Matched daily-only ridge control

The frozen full 15:30 ridge and daily anchor remain unchanged. The new control uses the same samples, residual target, preprocessing, penalty grid, clipping and missing-checkpoint fallback but excludes all ten current checkpoint inputs. Prior checkpoint history remains in both models. Each control chooses its own penalty using historical validation.

```text
          candidate  rmse_log     wape  aggregate_bias       period
        fixed_daily  0.469491 0.354676       -0.144684  development
ridge_daily_control  0.474878 0.356962       -0.129938  development
       ridge_update  0.459954 0.345919       -0.118001  development
        fixed_daily  0.429014 0.328298       -0.090003 confirmation
ridge_daily_control  0.426726 0.330393       -0.078790 confirmation
       ridge_update  0.409157 0.317815       -0.069106 confirmation
```

Paired differences below are full intraday ridge minus reference; negative favors the intraday update. Confidence intervals concern equal-year mean daily log MAE, not log RMSE or WAPE.

```text
          reference       period  block_dates  mean_log_mae_delta  ci_lower  ci_upper
        fixed_daily  development            5           -0.010020 -0.011870 -0.008637
        fixed_daily  development           20           -0.010020 -0.012011 -0.008560
        fixed_daily confirmation            5           -0.017825 -0.019720 -0.015220
        fixed_daily confirmation           20           -0.017825 -0.020562 -0.014475
ridge_daily_control  development            5           -0.012902 -0.013981 -0.012017
ridge_daily_control  development           20           -0.012902 -0.014109 -0.011993
ridge_daily_control confirmation            5           -0.016024 -0.017169 -0.014688
ridge_daily_control confirmation           20           -0.016024 -0.017722 -0.014431
```

This retrospective study uses previously inspected 2025 confirmation results. The shared requirement for a valid checkpoint in historical training and daily fallback on missing checkpoints is retained for a matched comparison, rather than an independently optimized daily-only forecast. Blocks preserve some serial dependence; intervals do not adjust for multiple comparisons. No model is promoted automatically.

All 60 checks passed, including historical sample hashes, original full-ridge validation selection and coefficient reconstruction, original forecasts, final preprocessing and bounds, exact keys and fallback, and saved control coefficients.
