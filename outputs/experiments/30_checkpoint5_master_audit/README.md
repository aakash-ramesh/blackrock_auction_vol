# Checkpoint 5 audit: splines, tail mixtures, Tweedie and event features

The executable research narrative is [master notebook 08](../../../master_research/08_splines_tail_mixtures_tweedie_and_events.ipynb).

This directory holds new verification and worked-fit artifacts. The original studies remain in experiments 21, 22 and 23.

- `plan.json`: source/panel hashes, versions, reconstruction scope and fresh-fit settings. The notebook checks this plan before using cached fresh fits.
- `predictions/reconstructed_YEAR.parquet`: restored spline, stock-effect spline, two tail-mixture and verified 58-feature LightGBM predictions for all eligible 2022–2025 stock-days.
- `reconstructed_metrics.csv`: scores recomputed from the restored models and checked against the original study tables.
- `ridge_reference_reconciliation.csv`: the original studies' ridge reference scores compared with authoritative saved ridge forecasts. These source-reference discrepancies are disclosed rather than treated as successful exact reproductions.
- `verified_tweedie_annual_metrics.csv`: all 120 original model/year/segment scores recomputed from saved row-level forecasts.
- `fresh_tweedie_2022/`: six validation configurations and four final boosters, refit preprocessing, sample fingerprints and predictions from a fresh 2022 replication of the original four-way loss/event study.
- `fresh_2022_comparison.csv`: fresh results compared with the original 2022 forecasts and scores.
- `verification.json`: completion record for successful notebook checks.

Model reconstructions cover four historical folds. Fresh model training here covers 2022 only. The historical spline/tail confidence intervals resample individual whole dates; the historical Tweedie/event intervals use 20-session circular date blocks. They condition on fitted predictions and are labelled as original-study calculations in the notebook.

The earlier 58-feature LightGBM reference differs from the tuned 61-feature daily champion. An event-feature improvement over the matched historical log control does not establish an improvement over that champion. Tweedie improves aggregate volume forecasts while worsening stock-level log RMSE; the notebook reports both objectives.
