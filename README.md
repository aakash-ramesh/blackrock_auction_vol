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
