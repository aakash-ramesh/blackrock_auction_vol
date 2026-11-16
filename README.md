# blackrock_auction_vol

Closing-auction volume prediction with leakage-safe daily features and rolling out-of-time evaluation.

## Environment

The EDA notebooks use `requirements-eda.txt`. Nonlinear model challengers additionally use LightGBM and scikit-learn:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-models.txt
```

## Notebook sequence

Run the numbered notebooks from the repository root. Notebook 05 generates the local, Git-ignored `enriched_daily_panel.parquet`; Notebook 09 freezes the global ridge benchmark; Notebook 16 evaluates nonlinear residual-model challengers against that frozen benchmark.

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/16_nonlinear_model_challengers.ipynb \
  --output-dir /tmp/blackrock_auction_vol_executed \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1
```

Notebook 16 is generated from `notebooks/build_notebook_16.py`, matching the existing generated-notebook convention. Its compact CSV, JSON, and PNG results are written under `outputs/eda/16_nonlinear_challengers/`; yearly prediction Parquet files are stored in the nested `predictions/` directory.


## Verified local continuation through November 4 2026

The original research remains intact. New reproducibility audits and controlled experiments extend it with verified nonlinear daily models, target comparisons, historical calibration, and noon and 15:30 updates.

- [Detailed project progress report](reports/project_progress_2026_11_04/project_progress_report.md) and [editable Word report](reports/project_progress_2026_11_04/NYSE_Closing_Auction_Project_Progress_Report.docx).
- [Daily panel reconstruction](outputs/reproduction/01_daily_panel/README.md): 23 checks.
- [Saved benchmark reconciliation](outputs/reproduction/02_benchmark_reconciliation/README.md): 81 checks; records the historical Notebook 16 summary discrepancy.
- [Verified compact nonlinear rerun](outputs/reproduction/03_compact_nonlinear/README.md): 60 checks, complete daily evaluation, saved models and preprocessing.
- [Target formulation comparison](outputs/experiments/17_target_formulations/README.md): 92 checks; retains the median-relative log target over tested direct-level and one-day-growth alternatives.
- [Error diagnosis and calibration](outputs/experiments/18_error_diagnosis_calibration/README.md): 87 checks; no tested correction passes the combined gate.
- [15:30 intraday update](outputs/experiments/19_intraday_1530/README.md): 124 checks; ridge has the best aggregate point estimates and intraday LightGBM passes its information gate.
- [Noon update and matched timing comparison](outputs/experiments/20_intraday_noon/README.md): 124 checks plus timing/source audits; 15:30 improves on noon for both learned update families.

The daily squared-error LightGBM improves mean annual 2022–2024 log RMSE from 0.48764 to 0.46941 versus frozen ridge. On the separate, common regular-session universe, the 15:30 ridge update improves log RMSE from 0.46949 for the daily anchor to 0.45995. These two comparisons use different cohorts. The previously inspected 2025 period is confirmation evidence, not an untouched holdout. No production or economic execution claim is made.

### Reproduce the new stages

Use the existing project model environment and run these scripts sequentially from the repository root:

```sh
python notebooks/reproduce_daily_panel.py
python notebooks/reconcile_saved_benchmarks.py
python notebooks/reproduce_compact_nonlinear.py
python notebooks/compare_target_formulations.py
python notebooks/diagnose_and_calibrate.py
python notebooks/build_intraday_1530_update.py
python notebooks/build_intraday_noon_update.py
python notebooks/compare_intraday_times.py
```

For the macOS environment used in these runs, LightGBM required the local OpenMP library path:

```sh
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Use an OpenMP installation appropriate to your own platform if that local path does not exist. Recorded environment versions and lockfiles are retained in the reproduction outputs. Report authoring additionally uses `requirements-reports.txt`.

### What is committed and what stays local

The repository includes new scripts, reader and executed notebooks, reports, fitted model files, preprocessing/reconstruction settings, numerical summaries, diagnostics, figures, validation checks, source fingerprints, and sample fingerprints. Approximately 6.95 GB of regenerated Parquet files (panels, sampled training rows and row-level forecasts) and local databases are omitted from this commit. An exact path/size inventory is in [generated_artifacts_not_committed.json](outputs/generated_artifacts_not_committed.json). Recreate those files by running the stages above with the existing source data. Fingerprints document the original run; exact reproduction can depend on the recorded software versions and sampling behavior.

The reader notebooks display committed compact results without retraining. Scripts that rebuild summaries or verify row-level forecasts require the regenerated local Parquet files. The noon comparison reuses historical sample keys produced by the 15:30 stage, so the stated order matters. Local virtual environments and rendering/cache files are not committed.


## Intraday continuation through November 16 2026

The original annual forecasts remain preserved. New retrospective studies are documented here:

- [21 — Frozen forecast robustness](outputs/experiments/21_intraday_robustness/README.md).
- [22 — Matched daily-only ridge control](outputs/experiments/22_intraday_ridge_control/README.md).
- [23 — Feature-group ablation](outputs/experiments/23_intraday_feature_ablation/findings.md).
- [24 — Historical correction shrinkage](outputs/experiments/24_intraday_shrinkage/findings.md).
- [25 — Direct metric-matched evaluation and forward-data audit](outputs/experiments/25_frozen_intraday_evaluation/README.md).
- [26 — Quarterly walk-forward audit](outputs/experiments/26_quarterly_intraday_audit/README.md).
- [Complete intraday research report](reports/intraday_progress_2026_11_16/intraday_research_report.md) and [editable Word report](reports/intraday_progress_2026_11_16/NYSE_Intraday_Forecasting_Research_Report.docx).

Reader notebooks 22–26 are in `notebooks/`. Each study's README and notebook identify its fitting and summary scripts. Run fitting stages in numerical order using the existing project environment and source data. Quarterly fitting saves actual sample keys, preprocessors, coefficients and full forecasts locally; the committed fingerprints and fold manifest document the run. Generated Parquet files and local databases are not included in this update. See [the publication inventory](outputs/intraday_publication_2026_11_16.json) for omitted files.

The quarterly audit passes 178 checks and improves on the daily anchor in all 16 quarters of 2022–2025. Its small development advantage over the annual intraday ridge is not established by the paired intervals. All years were previously inspected, so this remains retrospective evidence. The local forward-data audit finds core source coverage ending December 31, 2025; no new forward period was scored.
