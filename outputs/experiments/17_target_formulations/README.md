# Controlled target-formulation comparison

No alternative target passes the prespecified target-change gate.

All 92 forecast verification checks passed across 1,271,378 common stock-days. The experiment fits or reuses exactly the six planned model/target combinations. Historical notebooks and results remain unchanged.

| Model | Target | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| historical_reference | median20_log_residual | 0.48764 | 36.80% | -14.38% | 0.44512 | 34.56% |
| lgbm | median20_log_residual | 0.46941 | 35.45% | -14.40% | 0.43065 | 32.94% |
| lgbm | lag1_log_growth | 0.47029 | 35.34% | -16.06% | 0.42996 | 32.84% |
| lgbm | log_level | 0.47068 | 35.69% | -16.05% | 0.43003 | 33.33% |
| ridge | median20_log_residual | 0.48778 | 36.74% | -14.52% | 0.44677 | 34.51% |
| ridge | lag1_log_growth | 0.49063 | 37.42% | -14.85% | 0.44880 | 35.00% |
| ridge | log_level | 0.49650 | 37.06% | -15.37% | 0.45037 | 34.91% |

Development statistics are equal-year averages for 2022-2024, not pooled metrics. The 2025 period was previously inspected and is confirmation evidence, not an untouched holdout. Selection uses development results only. The historical frozen ridge is an external reference; the controlled ridge refits use the shared step-03 samples, so they need not reproduce the earlier ridge coefficients or metrics.

## Controls and definitions

- The three estimation targets are direct log1p auction shares, log1p change from the previous stock row's auction, and log1p departure from the prior 20-row auction median.
- All forecasts reconstruct closing-auction shares and use the same training-derived final log-forecast bounds.
- Missing growth/median references use that fitting window's global median auction shares. No eligible row is removed for missing history. Reference missingness is recorded separately.
- Features are the same 58 sources. Shared preprocessing uses the prior 99-feature preparation and the same feature mask. No feature, calibration, anomaly-history, weighting, or clipping-policy changes were introduced.
- Exact tuning, validation and final-fitting samples from step 03 are reused for all targets. Their fingerprints are in `shared_sample_fingerprints.csv`.
- Each ridge target tests seven penalties. Each new LightGBM target tests 15, 31 and 63 leaves with at most 2,000 trees, the same seeds, and historical validation/early stopping. The verified median-relative LightGBM model and complete original validation grid are reused.
- Test labels are not used for tuning or fallback construction. Fold settings are saved under `models/`.

## Decision rule and limits

A target change is compared with the median-relative target within the same model family. It must improve mean development log RMSE by at least 0.002, avoid annual development deterioration above 0.002, and have a paired development log-MAE bootstrap upper interval below zero. Exact differences and intervals appear in `target_comparison_summary.csv` and `paired_target_bootstrap.csv`.

The bootstrap resamples whole dates, preserving same-date dependence but not serial dependence between dates. Intervals are not adjusted for multiple comparisons. Log RMSE, WAPE and aggregate bias answer different questions; a numerical winner need not improve every metric or difficult stock group. Top-actual-volume groups are ex-post diagnostics, not model inputs.

## Verification and outputs

Saved forecasts were checked for unique keys, completeness, nonnegative finite values, exact historical-reference alignment, and exact reuse of the previous median-relative LightGBM forecasts. Metrics were recomputed from written prediction files. Every model was independently reloaded to reconstruct 200 deterministic rows per year using the saved preprocessing and inverse transformation.

- `annual_metrics.csv`: every candidate/year, including the external historical ridge.
- `target_comparison_summary.csv`: development/confirmation results and target-change decisions.
- `hyperparameter_validation.csv`: every validation trial, with reused trials labeled.
- `predictions/`: full row-level forecasts for 2022-2025.
- `models/`: coefficients, boosters, preprocessors, clipping/fallback settings; cached median LightGBM models remain in the verified step-03 folder.
- `segment_diagnostics.csv`: pooled development size, history, and tail results.
- `reference_missingness.csv`: missing references by target, model and year.
- `validation_checks.csv`, `experiment_summary.json`, `experiment_plan.json`: checks, fingerprints and the plan frozen before fitting.

## Reproduce

Use `.venv/bin/python notebooks/compare_target_formulations.py` with the macOS OpenMP library environment used in step 03. To rebuild only summaries/checks from existing forecasts, append `--summarize`. This reuses existing source data and samples; it does not change prior experiments.
