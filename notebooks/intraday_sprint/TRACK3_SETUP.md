# Track 3: setup and run, step by step

Follows the team's *Intraday Sprint: Team Handoff*. Everything runs from the repository root unless a step says otherwise. Total time is about 20 minutes, mostly waiting.

## 1. Get the sprint branch

```sh
git clone https://github.com/aakash-ramesh/blackrock_auction_vol.git   # skip if you already have it
cd blackrock_auction_vol
git fetch origin
git switch intraday-sprint
git pull
```

## 2. Copy in the Track 3 files

Put these three files in `notebooks/intraday_sprint/`:

- `track3_features.py`: builds the features
- `track3_intraday_seasonality.ipynb`: the Track 3 notebook
- `TRACK3_SETUP.md`: this guide

## 3. Python environment (one time; Python 3.12 recommended)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-models.txt pyarrow
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name auction-local
export JUPYTER_PATH="$PWD/.venv/share/jupyter"
```

- If pip says `No module named pip`, run `.venv/bin/python -m ensurepip` first.
- `requirements-models.txt` also installs torch, catboost and xgboost, which are large. Track 3 doesn't need them. For a lighter install, use: `.venv/bin/python -m pip install duckdb ipykernel jupyter nbconvert nbformat nbclient numpy pandas matplotlib seaborn "lightgbm>=4.6,<5" "scikit-learn>=1.7,<2" pyarrow`
- On a Mac, LightGBM may need OpenMP: run `brew install libomp`, or use the `DYLD_LIBRARY_PATH` line from the main README.
- On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`, and `set` instead of `export`.

## 4. Rebuild the large files that aren't in Git (one time, about 15 minutes)

```sh
.venv/bin/python notebooks/reproduce_daily_panel.py            # about 1 min; must end with "all_checks_passed": true
export AUCTION_DAILY_PANEL="$PWD/outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
.venv/bin/python notebooks/compare_intraday_anchors.py tree    # about 9 min: daily LightGBM anchors 2021–2025
.venv/bin/python notebooks/compare_intraday_anchors.py compare # about 5 min: checkpoint panel + incumbent ridge
git restore outputs/reproduction outputs/experiments/27_intraday_anchor_comparison
```

The last line puts back the committed summary files that these scripts overwrite, so they don't end up in a commit. Don't run `reproduce_compact_nonlinear.py` or `diagnose_and_calibrate.py`; the harness doesn't need them.

Each new terminal needs the two `export` lines again (`JUPYTER_PATH` and `AUCTION_DAILY_PANEL`).

## 5. Check the harness (about 3 minutes)

Open Jupyter from the repository root (`.venv/bin/jupyter lab`), go to `notebooks/intraday_sprint/` and run `00_harness_check.ipynb` with the **auction-local** kernel. It must print:

- `PASS: harness reproduces the team script's incumbent forecasts on every row.`
- `PASS: scoring, intervals, gate and segments run end to end.`

Your incumbent numbers may differ from teammates' in the fourth decimal (for example 0.45427 here vs 0.4545 in the handoff), because LightGBM isn't bit-identical across machines. That's expected. Only compare runs made on the same machine. If either PASS line is missing, stop and post in the team chat.

## 6. Run Track 3 (about 3 minutes)

Open `track3_intraday_seasonality.ipynb` (kernel **auction-local**) and choose Run All.

- `QUICK = True` in the second cell scores 2024 only, for fast experiments, and doesn't write to the leaderboard.
- `QUICK = False` runs all four years and appends to `outputs/experiments/34_intraday_sprint/leaderboard.csv`.

Expected development log RMSE (incumbent 0.45427 on the reference run):

| Variant | Dev log RMSE | Passes gate |
|---|---|---|
| `track3_rebalance_flags` (**the candidate**) | 0.45044 | yes |
| `track3_cal_all` | 0.45038 | yes, but the WAPE gain isn't significant |
| `track3_flags_projected_v2` | 0.45057 | yes, but adds nothing over the flags |
| `track3_projected_v2` | 0.45294 | no (gain 0.0013, under 0.002) |
| `track3_cal_all_norb` | 0.45366 | no |
| `track3_cal_surprise` | 0.45383 | no |

## 7. Commit (team workflow: everyone commits to `intraday-sprint`)

```sh
git switch intraday-sprint
git add notebooks/intraday_sprint/track3_features.py \
        notebooks/intraday_sprint/track3_intraday_seasonality.ipynb \
        notebooks/intraday_sprint/TRACK3_SETUP.md \
        outputs/experiments/34_intraday_sprint/leaderboard.csv
git status        # only these files should be staged
git commit -m "Track 3: calendar-adjusted surprises, projected full-day volume, rebalance-flag control"
git pull
git push
```

- Add files by name. Never use `git add .` or `git add -A`.
- Never commit Parquet files, `daily_tree_*.txt`, `.venv/`, anything under `submissions/`, changes under `outputs/reproduction/` or `outputs/experiments/27_intraday_anchor_comparison/`, or changes to `intraday_harness.py`.
- If `git pull` reports a conflict in `leaderboard.csv`, keep every row from both sides: delete only the `<<<<<<<`, `=======` and `>>>>>>>` lines, then `git add` the file and `git commit`.
- At the end of day 2, one team pull request goes from `intraday-sprint` into `main`. You don't need your own PR.

## 8. What to post in the team chat

> Track 3: calendar-adjusted surprises and projected full-day volume don't pass the gate by themselves. The win is telling the 15:30 ridge which days are Russell/MSCI rebalance days: `track3_rebalance_flags` passes (dev log RMSE −0.0038, WAPE better, 2025 better), with the gains on Russell, MSCI and month-end days. Track 2 should compare against incumbent + these flags.
