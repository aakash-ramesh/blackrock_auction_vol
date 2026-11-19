# Consolidated daily-model results

[Master notebook 09](../../../master_research/09_daily_model_results_and_decision.ipynb) builds and verifies this summary without training new models.

- `annual_results_ledger.csv`: annual 2022–2025 metrics, preserving cohort, candidate, source and evidence type. Repeated reference names remain separate across studies.
- `development_and_confirmation.csv`: equal-year 2022–2024 development averages and separate 2025 confirmation scores.
- `matched_contrasts.csv`: selected candidate-minus-reference development log-RMSE differences within their original cohorts.
- `source_gate_checks.csv`: independently checked point-score arithmetic and source adoption decisions for the LightGBM, HMM and neural series.
- `verified_forecast_scores.csv`: sixteen independently recomputed annual scores for the authoritative frozen ridge and three freshly fitted booster families, with matching stock/date labels.
- `source_manifest.csv`: exact source paths, SHA-256 hashes and sizes.
- `verification.json`: summary scope and successful execution record.

The tuned 61-source LightGBM remains the daily stock-level champion. Its published study and fresh selected-recipe refit are explicitly separate. Across-study differences in samples, preprocessing, references and uncertainty methods prevent treating every ledger row as one matched leaderboard. Historical ridge-reference discrepancies are disclosed in notebooks 08–09.

The previously inspected 2025 year supplies retrospective confirmation. Historical interval calculations condition on fitted forecasts and do not account for repeated research choices. Event features improve their matched historical log control; improvement over the tuned champion has not yet been established. Tweedie improves daily-total WAPE while worsening stock-level errors.
