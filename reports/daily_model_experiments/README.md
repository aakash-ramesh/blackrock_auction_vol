# Daily model experiments

A running report of daily-model experiments. New experiments are appended here rather than starting a new dated folder. It currently covers Notebooks 16a–16e: LightGBM improvements, CatBoost, the market-regime analysis, HMM regime features and the CNN + inception + LSTM sequence networks. For each one it explains why it was tried, how it was run, and how it compared with the other models. It also explains why the 50/50 network–LightGBM blend has a lower RMSE than LightGBM.

| File | Purpose |
| --- | --- |
| `daily_model_experiments_report.md` | Report text (the source of the Word file) |
| `NYSE_Closing_Auction_Daily_Model_Experiments.docx` | Word version with figures and tables |
| `build_report.py` | Builds the Word file from the Markdown with python-docx |
| `analyse_blend.py` | Section 7 blend decomposition; needs the git-ignored 16e predictions |
| `make_figures.py` | Figures 1 and 2 from the compact notebook outputs and `data/` |
| `data/` | `all_models_summary.csv`, `blend_decomposition.csv`, `blend_weight_curve.csv` |
| `figures/` | Figures created for this report; the other figures come from `outputs/eda` |

To rebuild from the repository root with the project `.venv`:

```sh
python reports/daily_model_experiments/analyse_blend.py   # optional; needs 16e predictions
python reports/daily_model_experiments/make_figures.py
python reports/daily_model_experiments/build_report.py
```

Numbers come from each notebook's `candidate_metrics.csv`, `candidate_selection.csv` and `summary.json`; those files remain the primary evidence.

## Adding a new experiment

1. Add a numbered section before "All experiments compared": why it was tried, design, results and outcome.
2. Add its row to the executive-summary table and the all-experiments table, and its notebook to Appendix A.
3. If it should appear in Figure 1, add its candidates to `MODELS` in `make_figures.py` (and its metrics file to the list read there).
4. Update the "Last updated" line, then rerun `make_figures.py` and `build_report.py`.
