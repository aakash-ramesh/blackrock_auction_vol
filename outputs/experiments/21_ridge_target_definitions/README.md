# Ridge-only target-definition comparison

13 ridge target definitions use the same 58 source features and identical rows. All 341 verification checks passed across 1,271,378 evaluation stock-days per specification. Scoring uses reconstructed auction shares.

## Metric-specific winners

- **rmse_log**: `median20_log_residual`, score 0.488111; 0.00% lower than current target when both tune penalties for that metric.
- **wape**: `mean60_log_residual`, score 0.36711; 0.05% lower than current target when both tune penalties for that metric.
- **rmse_raw**: `mean60_log_residual`, score 656902; 0.04% lower than current target when both tune penalties for that metric.

These are descriptive development rankings, not a declaration that one target wins every metric. A zero improvement means the current target wins. RMSE and R² share a ranking within a fixed evaluation year; equal-year averages can differ in ranking. WAPE and MAE share a ranking within a year, but equal-year averaging can also change their rankings.

## Common tuning rule: select each penalty by historical log RMSE

All metrics below are calculated after inverse transformation to shares. Log RMSE is then computed by applying log1p to actual and predicted shares, so unlike transformed-target training errors it is comparable across candidates.

| Target | Log RMSE ↓ | WAPE ↓ | MAE shares ↓ | RMSE shares ↓ | R² shares ↑ | Bias |
|---|---:|---:|---:|---:|---:|---:|
| median20_log_residual | 0.48811 | 36.74% | 82,272 | 657,160 | 0.350 | -14.50% |
| median60_log_residual | 0.48818 | 36.76% | 82,296 | 657,302 | 0.349 | -14.49% |
| mean20_log_residual | 0.48914 | 36.81% | 82,429 | 657,269 | 0.350 | -14.62% |
| mean60_log_residual | 0.48916 | 36.72% | 82,218 | 656,867 | 0.350 | -14.61% |
| lag1_log_growth | 0.49098 | 37.43% | 83,831 | 659,832 | 0.344 | -14.80% |
| log_volume | 0.49648 | 37.04% | 82,970 | 659,995 | 0.344 | -15.48% |
| adv20_log_residual | 0.49721 | 37.17% | 83,215 | 659,824 | 0.344 | -15.23% |
| median20_ratio | 0.75599 | 54.31% | 121,440 | 710,191 | 0.241 | +7.41% |
| adv20_ratio | 0.92217 | 57.00% | 126,707 | 730,260 | 0.193 | +1.67% |
| lag1_pct_growth | 0.96283 | 63.98% | 142,939 | 743,523 | 0.167 | +14.18% |
| sqrt_volume | 1.08089 | 65.51% | 146,951 | 792,837 | 0.053 | -43.39% |
| median20_share_residual | 1.31573 | 53.44% | 119,410 | 719,728 | 0.220 | -2.09% |
| raw_volume | 1.90740 | 88.59% | 198,232 | 798,723 | 0.037 | -11.99% |

## Design and interpretation

- Five-year historical tuning samples, the immediately preceding calendar year for validation, and five-year historical refits. Evaluation is the full eligible 2022–2025 universe. Development ranking uses 2022–2024 only; 2025 was previously inspected and is confirmation evidence.
- Tuning/refit/validation samples contain 250,000/350,000/120,000 rows. Deleted historical sample files were rebuilt as fresh deterministic single-thread reservoir draws; results are a new controlled comparison and need not reproduce earlier coefficients exactly. Sample files, hashes, source hashes, environments and models are saved.
- Ridge penalties are [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0], with an unpenalized intercept and the existing n-normalized ridge convention. Each target separately selects a penalty using historical log RMSE, WAPE, and share RMSE. Those are three declared selection strategies, not a post-hoc switch on evaluation data.
- Shared preprocessing fits the original 99-source feature contract on training data, then selects the compact 58 sources. Target transforms and denominator fallbacks use historical training data only. Historical reference windows exclude the forecast day.
- The comparison includes raw, square-root and log levels, six log differences, two volume ratios, a raw share residual, and simple day-over-day growth. Simple growth is `A / B - 1` and reconstructs shares as `B * (1 + predicted growth)`, where `B` is the preceding stock row's auction shares or the fitting sample's median auction shares when that reference is missing. Negative reconstructed share forecasts are floored at zero before applying the common bounds.
- Common training-derived 0.1%/99.9% auction-share bounds retain the established forecast policy. `nonnegative_only` metrics show sensitivity to removing these bounds, using the same fitted models and selected penalties. This sensitivity is not independently optimized. `bounded_fraction` records the affected share of rows.
- Square-root predictions are floored at zero before squaring. Log predictions are inverse-transformed without a smearing correction. Such corrections would be an additional calibration experiment. Raw and ratio targets are fitted without target winsorization.
- Reported R² is predictive out-of-time R², not training fit; it can be negative. `tail_wape` measures the largest 1% of realized auctions within each year, an ex-post diagnostic only. Bias is predicted total / realized total minus one.
- Rankings are point estimates without a paired uncertainty test. Prefer a change that is stable by year, improves the metric tied to the intended use, and remains credible under the bound sensitivity. Repeated development experiments can overfit this period; reserve genuinely new dates for a final decision.

## Files and reproduction

- `development_metrics.csv`: equal-year means for all targets, tuning strategies and bound policies.
- `annual_metrics.csv`: annual share errors, log errors, R², correlations, bias and tail WAPE, including 2025 confirmation.
- `metric_winners.csv`: metric-specific development winners and improvement over the current target.
- `hyperparameter_validation.csv`, `selected_penalties.csv`: complete grid scores and choices; boundary choices are flagged.
- `models/`, `predictions/`, `samples/`: saved coefficients, preprocessors, reconstruction parameters, complete forecasts and shared historical samples.
- `verification_checks.csv`: panel eligibility/counts, date separation, unique and complete keys, finite/nonnegative predictions, agreement with the existing ridge solver, and reconstruction from every saved model.
- [Target stability diagnostics](target_stability.md): plots and within-stock summaries for all targets on a fixed 2022–2025 cohort.

Run `.venv/bin/python -u notebooks/compare_ridge_targets.py`. Use `--summarize` to rebuild the compact tables, figure and reader notebook from annual metrics. Large generated Parquet files remain local and ignored by Git.
