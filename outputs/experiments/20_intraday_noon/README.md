# First 12:00 intraday forecast update

Noon provides a modest early improvement, while 15:30 remains the stronger update. Ridge has the best aggregate point estimates at both times: development log RMSE is 0.464589 at noon versus 0.459954 at 15:30 and 0.469491 for the daily anchor. Noon retains about 51% of the 15:30 ridge log-RMSE improvement. Noon LightGBM fails the full prespecified gate against the daily anchor (mean improvement below 0.002 and one annual deterioration above 0.002), despite beating the learned daily-only control. Paired comparisons favor 15:30 for both model families in development and 2025. Keep the daily anchor and 15:30 ridge as the leading research candidates; noon ridge is an optional earlier update requiring robustness assessment.

The 12:00 LightGBM update does not pass every prespecified incremental-information criterion; retain the references and inspect the comparison.

All 124 verification checks passed. Evaluation includes 1,260,168 eligible regular-session stock-days over 2022-2025. Valid checkpoints cover 99.94%; missing checkpoints use the unchanged daily forecast for every method.

| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed daily forecast | 0.46949 | 35.47% | -14.47% | 0.42901 | 32.83% |
| Checkpoint ratio rule | 0.76257 | 54.10% | -11.86% | 0.68849 | 52.32% |
| Ridge intraday update | 0.46459 | 34.85% | -13.02% | 0.41677 | 32.25% |
| Daily-only LightGBM control | 0.47902 | 35.96% | -13.34% | 0.42988 | 33.19% |
| 12:00 LightGBM update | 0.46910 | 35.24% | -13.10% | 0.42109 | 32.44% |

Development metrics average annual 2022-2024 values with equal year weight. The previously inspected 2025 year is confirmation evidence. These daily-reference metrics differ from earlier whole-year tables because early-close sessions are excluded for all candidates here.

## Forecast and information set

The outcome remains official closing-auction shares. The 12:00 update sees current cumulative trading volume, dollar volume, trade count, VWAP, recent 30-minute activity, and prior-only normalizations. It does not see final daily volume, closing returns, auction outcomes, order imbalance, or later checkpoints. The extraction time boundary relies on the upstream checkpoint files; these aggregates do not contain underlying tick timestamps for an independent cutoff re-audit.

The daily-only learned control receives the same original 58 daily features, saved daily forecast, prior price and historical checkpoint summaries used for normalization. It receives no current-session checkpoint inputs. Its sample rows, preprocessing, search budget, target and forecast bounds match the intraday LightGBM model. Beating this control is the evidence for additional current-session information within this experiment, rather than attributing every gain over the fixed daily forecast to intraday information.

The simple rule multiplies today's cumulative checkpoint volume by the prior-only 20-stock-row median auction/checkpoint ratio. Missing ratio history uses the fitting sample's median ratio. Ridge and LightGBM learn a log1p correction around the fixed daily forecast. Original daily predictions and model files are preserved.

## Historical training and validation

Update training uses only genuinely out-of-time daily forecasts from 2021 onward. It never uses fitted in-sample daily predictions. For the 2022 fold, tuning fits Jan-Sep 2021 and validates Oct-Dec 2021; the final model fits available 2021 rows. Later folds tune on the previous full year, train on earlier years from 2021, and refit through the previous year. All model, imputation and clipping parameters are fitted before the evaluation year.

This expanding 2021-onward update history differs from the daily model's five-year history. The initial quarterly validation is also distinct from later annual validation. These are disclosed limitations of this first bounded intraday study. The base model specification and 2025 results had already been inspected, so none of this is claimed as untouched production validation.

The preceding experiments' minimum-25-tree final-refit convention is retained. The noon daily-only control selected two trees in both 2023 and 2025, and was refitted with 25 in those folds. The corresponding 15:30 control selected five and four; these are separate fitted records. This can affect the learned-control comparison; it does not alter the fixed daily reference. Both learned LightGBM candidates follow the same rule. Exact selected and saved tree counts appear in `selected_hyperparameters.csv`. No post-result tree-count sensitivity or additional tuning was introduced.

Noon occurs before early-close auctions, but early-close sessions are excluded here to preserve the same evaluation universe as the 15:30 study. Missing or invalid regular-session checkpoints retain the daily forecast exactly. The available-checkpoint-only summary is separate from the primary all-regular-session comparison.

All learned/control candidates and the simple rule share bounds derived from update-fitting log targets. The fixed daily reference retains its saved bounds. The daily-only control helps isolate the effect of adding checkpoint inputs from differences in the extra modeling layer and these common update rules.

## Verification and uncertainty

Written predictions were checked for exact eligible keys and labels, unchanged daily reference, no early-close inclusion, finite nonnegative forecasts, exact missing-checkpoint fallback, metric recalculation, and reconstruction from saved models. Independent pandas calculations matched three prior-only checkpoint history features on 16,228 rows from 16 stocks.

The information gate requires at least 0.002 mean development log-RMSE improvement versus both the fixed daily forecast and the daily-only learned control, no annual deterioration beyond 0.002, no mean WAPE worsening, and paired log-MAE upper confidence bounds below zero against both. Whole-date bootstrap intervals preserve same-date dependence but not cross-date serial dependence and are not adjusted for multiple comparisons. The study measures forecast accuracy, not execution-cost savings.

## Outputs

- `annual_metrics.csv`, `intraday_comparison.csv`: common-row annual and development/confirmation results.
- `coverage_by_year.csv`, `available_checkpoint_development_metrics.csv`: coverage, exclusions and observed-checkpoint-only results.
- `intraday_information_gate.csv`, `paired_bootstrap.csv`: comparisons against both references.
- `predictions/`, `models/`, `samples/`: complete forecasts, fitted models, historical samples and preprocessing.
- `intraday_panel.parquet`, `feature_timing_catalog.csv`: engineered data and input availability contract.
- `segment_diagnostics.csv`: size, history, event, tail and availability diagnostics.
- `validation_checks.csv`, `input_fingerprints.csv`, `training_sample_fingerprints.csv`, `experiment_plan.json`, `experiment_summary.json`: checks and reproducibility records.

Run `.venv/bin/python notebooks/build_intraday_noon_update.py` with the preceding OpenMP runtime environment. Append `--summarize` to verify and rebuild reports from saved forecasts without model fitting.


## Noon versus 15:30

All evaluation keys, targets and daily reference forecasts match exactly across the two studies. Historical sample keys and order originate from experiment 19, with unavailable noon rows removed before fitting. Noon features and their prior histories are rebuilt from noon aggregates. Historical checkpoint features in each daily-only control also correspond to its own checkpoint. This compares the complete forecast procedures; training features and occasional missing-row exclusions differ.

```text
checkpoint             candidate  development_rmse_log  development_wape  development_bias  confirmation_rmse_log  confirmation_wape  confirmation_bias
     12:00 checkpoint_ratio_rule              0.762570          0.541024         -0.118642               0.688487           0.523167          -0.047689
     12:00           fixed_daily              0.469491          0.354676         -0.144684               0.429014           0.328298          -0.090003
     12:00    lgbm_daily_control              0.479021          0.359621         -0.133428               0.429881           0.331893          -0.071872
     12:00  lgbm_intraday_update              0.469099          0.352445         -0.131024               0.421091           0.324393          -0.070317
     12:00          ridge_update              0.464589          0.348529         -0.130164               0.416770           0.322457          -0.075256
     15:30 checkpoint_ratio_rule              0.691529          0.492621         -0.156346               0.614224           0.466934          -0.090475
     15:30           fixed_daily              0.469491          0.354676         -0.144684               0.429014           0.328298          -0.090003
     15:30    lgbm_daily_control              0.478712          0.359470         -0.133879               0.430338           0.332277          -0.071158
     15:30  lgbm_intraday_update              0.462146          0.347655         -0.119335               0.415594           0.321218          -0.058610
     15:30          ridge_update              0.459954          0.345919         -0.118001               0.409157           0.317815          -0.069106
```

Paired differences are noon minus 15:30; negative favors noon. Whole-date bootstrap intervals describe log absolute error and daily WAPE, not log RMSE; serial dependence and model-selection uncertainty are not covered. See `paired_checkpoint_time_bootstrap.csv`.


The initial noon build failed an independent prior-history check. Materializing the source join before window calculations corrected 137 false missing-checkpoint flags. All models were rerun on the corrected panel. Full source-availability audits then passed for both noon and the unchanged 15:30 panel.
