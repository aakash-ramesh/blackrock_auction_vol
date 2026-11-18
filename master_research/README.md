# Auction-volume forecasting research

Read the executed notebooks in order. They contain the analysis, rationale, comparison tables and figures. Each notebook distinguishes saved historical studies, reconstructed forecasts and fresh fits.

| Notebook | Contents | Review stage |
| --- | --- | --- |
| [01](01_raw_data_eda_and_imputation.ipynb) | Raw data, EDA, quality and imputation | Checkpoint 1 |
| [02](02_target_variables.ipynb) | Target definitions and finalists | Checkpoint 1 |
| [03](03_features_and_transformations.ipynb) | Features, relationships and transformations | Checkpoint 1 |
| [04](04_ridge_models_and_segmentation.ipynb) | Ridge, evaluation metrics and segmentation | Checkpoint 2 |
| [05](05_lightgbm_and_hyperparameter_tuning.ipynb) | LightGBM and tuning | Checkpoint 3 |
| [06](06_catboost_xgboost_and_tuning.ipynb) | Tuned CatBoost and XGBoost | Checkpoint 4 |
| [07](07_regimes_hmm_neural_networks_and_blends.ipynb) | Regimes, HMMs, neural networks and blends | Checkpoint 4 extension |
| [08](08_splines_tail_mixtures_tweedie_and_events.ipynb) | Splines, tail mixtures, Tweedie and event features | Checkpoint 5 |
| [09](09_daily_model_results_and_decision.ipynb) | Consolidated daily results and model decision | Daily summary |
| [10](10_intraday_forecasts_controls_and_pairings.ipynb) | Noon/15:30 corrections, controls, robustness and daily pairings | Checkpoint 6 |
| [11](11_sequential_intraday_and_kalman_filters.ipynb) | Five-checkpoint ridge controls, Kalman variants and uncertainty diagnostics | Sequential exploration |

The daily stock-level champion remains tuned 61-source LightGBM; compact global ridge remains the linear benchmark. Event inputs warrant a matched champion-extension test, while Tweedie serves a separate aggregate-volume objective. The preferred intraday research pairing is tuned daily LightGBM plus annual 15:30 ridge; the neural blend remains a challenger. Notebook 11's fresh Kalman and multi-checkpoint exploration retains this pairing: correlated measurement errors matter, but no tested sequential candidate establishes a qualifying improvement.

Planned next:

- **12 — Stochastic optimal control and execution simulations:** ideation deferred at the user's request. The earlier simulation remains in `outputs/experiments/24_execution_control`; notebook 11 changes no execution policy.

The resumed control ideation now has a [literature review and mathematical formulation memo](../reports/optimal_control_literature_review_2026_11_18/research_memo.md). It audits available price data, distinguishes close-targeting from arrival-price risk, and proposes historical joint price/liquidity scenarios. No new control simulation has been implemented.

Run notebooks from the repository's project environment with their stated dependencies and source artifacts available. Earlier fitting notebooks can take several minutes and save new model artifacts under `outputs/experiments`. Notebook 09 reads the completed studies without training models and exports a source-hashed results ledger to [the summary directory](../outputs/experiments/31_daily_model_summary/).
