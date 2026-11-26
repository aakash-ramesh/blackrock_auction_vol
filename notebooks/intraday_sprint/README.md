# Intraday sprint

A shared harness so four people can try different intraday models in parallel and get directly comparable results.

Every experiment predicts the same thing: the **log correction to the daily forecast at 15:30**,
`z = log1p(actual auction) − log1p(daily anchor)`. The harness handles everything else: folds, training
samples, the daily anchor (tuned 61-source LightGBM), forecast bounds, missing-checkpoint fallback, and scoring
against the incumbent (that anchor plus the annual 15:30 ridge, from master notebook 10 / experiment 27).

## One-time setup

The harness needs the regenerated incumbent files, which are not in Git (they are large). From the repository root:

```sh
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-models.txt pyarrow
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name auction-local
.venv/bin/python notebooks/reproduce_daily_panel.py                 # daily panel, about 1 minute
export AUCTION_DAILY_PANEL="$PWD/outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
.venv/bin/python notebooks/compare_intraday_anchors.py tree         # tuned daily LightGBM anchors, 2021–2025
.venv/bin/python notebooks/compare_intraday_anchors.py compare      # checkpoint panel + incumbent ridge
```

`pyarrow` is needed for Parquet files but is not in `requirements-models.txt` yet. The `ipykernel` line registers
the `auction-local` Jupyter kernel that `reproduce_daily_panel.py` expects. If that script fails with
`No module named pip`, run `.venv/bin/python -m ensurepip` first. Do not run `reproduce_compact_nonlinear.py` or
`diagnose_and_calibrate.py`: they check byte-for-byte fingerprints that a rebuilt panel never matches, and the
harness doesn't need them.

These scripts rewrite some committed summary files under `outputs/`. Restore them afterwards with
`git restore outputs/reproduction outputs/experiments/27_intraday_anchor_comparison` so they don't end up in a commit.

Then run `00_harness_check.ipynb` once. It must report that the harness reproduces the incumbent exactly.

## Using it

```python
from intraday_harness import Harness, leaderboard

h = Harness()                 # Harness(quick=True) scores 2024 only, for fast iteration
for fold in h.folds():        # test years 2022, 2023, 2024 (development) and 2025 (confirmation)
    prep = fold.preprocessor(fold.refit)                     # incumbent preprocessing, 73 source features
    model = MyModel().fit(prep.transform(fold.refit), fold.target(fold.refit))
    h.submit(fold, model.predict(prep.transform(fold.test)), name="my_model", owner="Name",
             description="what this tries and why")
result = h.report("my_model")  # summary, annual, intervals, gate, segments; appends to the leaderboard
leaderboard()
```

What each fold gives you:

| Attribute | Contents |
|---|---|
| `fold.tune`, `fold.valid` | Historical rows for tuning (validation is the year before the test year; Q4 2021 for 2022) |
| `fold.refit` | Historical rows for the final fit, ending the day before the test year |
| `fold.test` | Every regular-session stock-day of the test year (scored in full) |
| `fold.features` | Feature lists: `daily`, `prior_checkpoint`, `intraday`, `control`, `all` |
| `fold.target(frame)` | The log correction to predict |
| `fold.preprocessor(frame, extra_continuous=[...])` | The incumbent preprocessing, plus any columns you add |
| `fold.to_shares(correction)` | Converts a correction to share forecasts (the harness does this on `submit`) |

Training samples are the incumbent's exact rows: 250,000 tune, 120,000 validation and 350,000 refit rows, all
with a valid 15:30 checkpoint, drawn with the incumbent's seeds.

## Adding your own features

```python
def my_features(frame, fold_year, role):        # role: tune / valid / refit / test
    return pd.DataFrame({"my_feature": ...}, index=frame.index)

h.add_features(my_features)                      # before calling h.folds()
...
prep = fold.preprocessor(fold.refit, extra_continuous=["my_feature"])
```

Rules:

- Anything **fitted** inside a feature function (a GMM, PCA, seasonal profile) must use rows before `fold_year` only.
- **Cross-sectional** features (across all stocks on a date at 15:30) must be computed on the full panel, not on
  the sampled fold rows: use `h.full_panel(start, end, columns)`, then merge onto the fold frames by `date` and `permno`.
- Today's auction and today's full-day volume are never inputs. The harness refuses known forbidden columns,
  but it cannot catch everything, so check your own features.

## How results are judged

`h.report(name)` scores the submission against the daily anchor and the incumbent on identical rows:

- **Metrics:** log RMSE (primary), WAPE, aggregate bias; development = equal-weight mean of 2022–2024, 2025 reported separately.
- **Paired intervals:** 20-trading-date circular block bootstrap, 5,000 replicates (same method as experiment 27).
- **Gate (the project's adoption rule):** at least 0.002 better development log RMSE than the incumbent, no
  development year worse by more than 0.002, no worse development WAPE, and a development log-RMSE interval
  entirely below zero.
- **Segments:** month-end, quarterly expiry, low liquidity, small size, short history, Russell, MSCI, S&P 500
  changes, FOMC days, and the largest 1% of auctions (an after-the-fact diagnostic, never an input).

A clean negative result is still a result. Record it with a clear `description`.

## Files

- `intraday_harness.py`: the harness.
- `00_harness_check.ipynb`: proves the harness reproduces the incumbent; run it first.
- `trackN_<topic>.ipynb`: one notebook per person.
- `outputs/experiments/34_intraday_sprint/leaderboard.csv`: committed results table.
- `outputs/experiments/34_intraday_sprint/submissions/`: row-level predictions (Git-ignored).
