# First 15:30 intraday forecast update

The 15:30 LightGBM update passes the prespecified incremental-information gate against both references.

All 124 verification checks passed. Evaluation includes 1,260,168 eligible regular-session stock-days over 2022-2025. Valid checkpoints cover 99.98%; missing checkpoints use the unchanged daily forecast for every method.

| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed daily forecast | 0.46949 | 35.47% | -14.47% | 0.42901 | 32.83% |
| Checkpoint ratio rule | 0.69153 | 49.26% | -15.63% | 0.61422 | 46.69% |
| Ridge intraday update | 0.45995 | 34.59% | -11.80% | 0.40916 | 31.78% |
| Daily-only LightGBM control | 0.47871 | 35.95% | -13.39% | 0.43034 | 33.23% |
| 15:30 LightGBM update | 0.46215 | 34.77% | -11.93% | 0.41559 | 32.12% |

Development metrics average annual 2022-2024 values with equal year weight. The previously inspected 2025 year is confirmation evidence. These daily-reference metrics differ from earlier whole-year tables because early-close sessions are excluded for all candidates here.

## Forecast and information set

The outcome remains official closing-auction shares. The 15:30 update sees current cumulative trading volume, dollar volume, trade count, VWAP, recent 30-minute activity, and prior-only normalizations. It does not see final daily volume, closing returns, auction outcomes, order imbalance, or later checkpoints. The extraction time boundary relies on the upstream checkpoint files; these aggregates do not contain underlying tick timestamps for an independent cutoff re-audit.

The daily-only learned control receives the same original 58 daily features, saved daily forecast, prior price and historical checkpoint summaries used for normalization. It receives no current-session checkpoint inputs. Its sample rows, preprocessing, search budget, target and forecast bounds match the intraday LightGBM model. Beating this control is the evidence for additional current-session information within this experiment, rather than attributing every gain over the fixed daily forecast to intraday information.

The simple rule multiplies today's cumulative checkpoint volume by the prior-only 20-stock-row median auction/checkpoint ratio. Missing ratio history uses the fitting sample's median ratio. Ridge and LightGBM learn a log1p correction around the fixed daily forecast. Original daily predictions and model files are preserved.

## Historical training and validation

Update training uses only genuinely out-of-time daily forecasts from 2021 onward. It never uses fitted in-sample daily predictions. For the 2022 fold, tuning fits Jan-Sep 2021 and validates Oct-Dec 2021; the final model fits available 2021 rows. Later folds tune on the previous full year, train on earlier years from 2021, and refit through the previous year. All model, imputation and clipping parameters are fitted before the evaluation year.

This expanding 2021-onward update history differs from the daily model's five-year history. The initial quarterly validation is also distinct from later annual validation. These are disclosed limitations of this first bounded intraday study. The base model specification and 2025 results had already been inspected, so none of this is claimed as untouched production validation.

Early-close sessions are excluded because 15:30 occurs after their auction. Missing or invalid regular-session checkpoints retain the daily forecast exactly. No model claims a usable 15:30 forecast on early-close sessions. The available-checkpoint-only summary is separate from the primary all-regular-session comparison.

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

Run `.venv/bin/python notebooks/build_intraday_1530_update.py` with the preceding OpenMP runtime environment. Append `--summarize` to verify and rebuild reports from saved forecasts without model fitting.

## Final-tree-count convention

The first update retains the preceding experiments' minimum-25-tree final-refit convention. The daily-only control selected five trees for 2023 and four for 2025, and was refitted with 25 in those folds. This can affect the control comparison; it does not alter the fixed daily reference. Both learned LightGBM candidates follow the same rule. Exact selected and saved tree counts appear in `selected_hyperparameters.csv`. No tree-count sensitivity or additional tuning was introduced after seeing the results.
