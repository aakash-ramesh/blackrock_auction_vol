# Intraday master audit — Checkpoint 6

[Notebook 10](../../../master_research/10_intraday_forecasts_controls_and_pairings.ipynb) consolidates the original intraday experiments and daily-anchor pairings. It rebuilds forecasts from saved models, independently checks scoring and historical feature timing, and performs one fresh 2022 ridge fit. The original study files remain unchanged.

## Outputs

- `plan.json`: source/panel fingerprints, execution versions and reconstruction scope. It pins the raw checkpoint files and original daily forecast files before reusing the noon and fresh-fit caches.
- `noon_panel.parquet`: reconstructed noon inputs, using the corrected materialised source join and prior-only checkpoint histories.
- `independent_history_checks.csv`: pandas checks of availability and three historical checkpoint features for 16 stocks at both times, covering 32,570 rows.
- `predictions/reconstructed_STUDY_YEAR.parquet`: restored original noon/15:30 forecasts; 15:30 files also include matched daily-only ridge, eight ablations, four partial corrections and quarterly ridge.
- `verified_annual_scores.csv`: 172 model/year/sample rows checked against the source studies.
- `checkpoint_comparison.csv`: aligned noon and 15:30 development and confirmation results.
- `restored_daily_sufficient_statistics.csv`: daily totals used to independently reproduce the original pairing's primary frozen inference.
- `verified_frozen_primary_intervals.csv`: eight directly recomputed metric interval rows, using the saved source seeds and 5,000 paired 20-session circular-block replicates.
- `verified_pairing_scores.csv`: 28 annual daily-anchor/update score rows, with saved ridge coefficients reconstructed and exact missing-checkpoint fallback checked.
- `fresh_ridge_2022/`: fresh matched-original ridge fit, seven-penalty historical validation, sample fingerprints, preprocessing, coefficients and forecast reconciliation. Its 2022 forecast matches the original matched refit exactly in this execution.
- `summary_arithmetic_checks.csv`: 46 development/confirmation reconciliations for ridge variants.
- `source_manifest.csv` and `verification.json`: source hashes and new verification scope/completion record.

## Interpretation

The preferred main research pairing is tuned 61-source daily LightGBM plus an annual 15:30 ridge correction. Development log RMSE is approximately 0.45471, stock WAPE 33.98%, and 2025 confirmation log RMSE 0.40418. The neural-blend pairing has a small uncertain incremental gain and slightly worse development WAPE.

Noon provides an optional earlier ridge update around the original daily anchor. Original-anchor daily-only controls support the predictive contribution of checkpoint inputs. No partial-correction variant passes every saved adoption condition. Quarterly ridge has a small original-anchor development advantage with uncertainty spanning zero.

Earlier controls, ablations, partial corrections and quarterly fits use the original 58-source daily anchor. Their gains cannot be added to the tuned pairing without a new matched test. Missing checkpoints use each candidate's own daily forecast exactly. Early-close sessions are excluded from both checkpoint comparisons. The 1,260,168 regular-session stock-days cover already inspected 2022–2025 history; no new forward period is scored.

Complete multi-year model training is not repeated here. Saved-model reconstructions cover all reported folds; the fresh fitting example covers 2022 only. Most uncertainty tables are original-study calculations with checked point estimates. The original frozen pairing's primary metric intervals are directly recomputed. Conditional intervals omit full refitting and repeated research-selection uncertainty.

Work stops at Checkpoint 6 for review before notebook 11's dynamic/Kalman exploration.
