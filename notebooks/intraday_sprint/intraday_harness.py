"""Shared harness for the intraday sprint: one data contract, one scoring contract.

Every experiment predicts the same quantity, the log correction to the daily
anchor at the 15:30 checkpoint,

    z = log1p(actual auction shares) - log1p(daily anchor forecast),

and hands that correction to the harness. The harness owns the folds, the
training samples, the daily anchor, forecast bounds, the missing-checkpoint
fallback and all scoring, so results from different people are comparable.

The data and sampling contract copies ``notebooks/compare_intraday_anchors.py``
(experiment 27), whose tuned-LightGBM anchor plus 15:30 ridge is the incumbent
pairing in master notebook 10. ``00_harness_check.ipynb`` re-fits that ridge
through this harness and must reproduce its published annual scores.

Typical use::

    from intraday_harness import Harness
    h = Harness()                          # quick=True scores 2024 only
    for fold in h.folds():
        prep = fold.preprocessor(fold.refit)
        model = MyModel().fit(prep.transform(fold.refit), fold.target(fold.refit))
        h.submit(fold, model.predict(prep.transform(fold.test)), name="my_model", owner="me")
    h.report("my_model")
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.build_intraday_1530_update import (  # noqa: E402  (feature lists and seed of the incumbent)
    CONTROL_FEATURES, DAILY_FEATURES, INTRADAY_FEATURES, MODEL_FEATURES, PRIOR_CHECKPOINT_FEATURES, SEED,
)
from notebooks.evaluate_frozen_intraday import circular_weights, statistics  # noqa: E402
from notebooks.modeling_utils import (  # noqa: E402
    apply_preprocessor, fit_preprocessor, fit_ridge, forecast_metrics, predict_ridge,
)

ANCHOR_DIR = ROOT / "outputs/experiments/27_intraday_anchor_comparison"
INCUMBENT_PREPROCESSORS = ROOT / "outputs/experiments/19_intraday_1530/models"
PANEL = Path(os.environ.get(
    "AUCTION_DAILY_PANEL", ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"))
REBALANCE = ROOT / "blackrock_data/rebalancing/stock_rebalance_features_2011_2025.parquet"
FOMC = ROOT / "blackrock_data/events/fomc_decisions.csv"
OUT = ROOT / "outputs/experiments/34_intraday_sprint"
LEADERBOARD = OUT / "leaderboard.csv"

YEARS = (2022, 2023, 2024, 2025)
DEVELOPMENT_YEARS = (2022, 2023, 2024)
CONFIRMATION_YEAR = 2025
RIDGE_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SAMPLE_ROWS = {"tune": 250_000, "valid": 120_000, "refit": 350_000}
SEED_OFFSET = {"tune": 0, "valid": 100, "refit": 200}
BOOTSTRAP = {"block": 20, "reps": 5000, "seed": 20261113}
GATE = {"min_dev_rmse_gain": 0.002, "max_annual_rmse_worsening": 0.002}
INCUMBENT = "incumbent_tree_ridge"
ANCHOR = "tree"  # tuned 61-source daily LightGBM

# Same-day outcomes and identifiers that a model must never use as inputs.
FORBIDDEN = {"auction_volume", "log1p_auction_volume", "anchor_forecast", "date", "permno", "year"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


@dataclass
class Prep:
    """A fitted preprocessing state: ``transform`` applies the same encoding to any frame."""

    params: dict
    continuous: list
    boolean: list
    categorical: list

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return apply_preprocessor(frame, self.params, self.continuous, self.boolean, self.categorical)

    @property
    def names(self) -> list:
        return self.params["output_names"]


@dataclass
class Fold:
    """One evaluation year: historical samples to fit on and the full test year to score."""

    year: int
    tune: pd.DataFrame
    valid: pd.DataFrame
    refit: pd.DataFrame
    test: pd.DataFrame
    types: dict
    features: dict = field(default_factory=dict)

    @property
    def frames(self) -> dict:
        return {"tune": self.tune, "valid": self.valid, "refit": self.refit, "test": self.test}

    @staticmethod
    def target(frame: pd.DataFrame) -> np.ndarray:
        """Log correction to the daily anchor: the quantity every experiment predicts."""
        return (frame.log1p_auction_volume - frame.log_daily_forecast).to_numpy(float)

    @staticmethod
    def bounds(frame: pd.DataFrame) -> np.ndarray:
        """Forecast bounds in log1p shares, learned from the fitting frame only."""
        return np.quantile(frame.log1p_auction_volume, [0.001, 0.999])

    def preprocessor(self, fit_on: pd.DataFrame, continuous=None, boolean=None, categorical=None,
                     extra_continuous=(), extra_boolean=(), extra_categorical=()) -> Prep:
        """Fit the incumbent's preprocessing (log/clip/median-fill/scale + missing flags).

        Defaults to the incumbent's 73 source features. Pass ``extra_*`` to add your own
        columns, or full lists to replace the feature set. ``fit_on`` must be historical.
        """
        if fit_on["date"].max() >= pd.Timestamp(f"{self.year}-01-01"):
            raise ValueError("Preprocessing may only be fitted on rows before the test year.")
        continuous = list(self.types["continuous"] if continuous is None else continuous) + list(extra_continuous)
        boolean = list(self.types["boolean"] if boolean is None else boolean) + list(extra_boolean)
        categorical = list(self.types["categorical"] if categorical is None else categorical) + list(extra_categorical)
        used = set(continuous + boolean + categorical)
        if used & FORBIDDEN:
            raise ValueError(f"Forbidden input columns: {sorted(used & FORBIDDEN)}")
        _, params = fit_preprocessor(fit_on, continuous, boolean, categorical)
        return Prep(params, continuous, boolean, categorical)

    def to_shares(self, correction, frame: pd.DataFrame | None = None, bounds=None) -> np.ndarray:
        """Anchor + correction -> clipped log -> shares, with exact daily fallback for missing checkpoints."""
        frame = self.test if frame is None else frame
        bounds = self.bounds(self.refit) if bounds is None else bounds
        correction = np.asarray(correction, dtype=float)
        if correction.shape != (len(frame),):
            raise ValueError(f"Expected {len(frame)} corrections, got shape {correction.shape}.")
        daily = frame.anchor_forecast.to_numpy(float)
        log_forecast = np.clip(np.log1p(daily) + correction, *bounds)
        shares = np.maximum(np.expm1(log_forecast), 0)
        return np.where(frame.cp_available.to_numpy(bool), shares, daily)


class Harness:
    """Loads the incumbent's data contract and scores submissions against the incumbent."""

    def __init__(self, checkpoint: str = "15:30", years=YEARS, quick: bool = False, extra_columns=()):
        if checkpoint != "15:30":
            raise NotImplementedError("Only the 15:30 checkpoint panel is built so far.")
        self.checkpoint = checkpoint
        self.years = (2024,) if quick else tuple(years)
        self.quick = quick
        self.con = duckdb.connect()
        self.con.execute("SET threads=1")  # reservoir sampling must match the incumbent exactly
        self._register_views()
        self._columns = list(dict.fromkeys([
            "date", "permno", "auction_volume", "log1p_auction_volume", "cp_available",
            *[f for f in MODEL_FEATURES if f != "log_daily_forecast"], ANCHOR, *extra_columns,
        ]))
        self._feature_functions = []
        self._submissions: dict[str, list[pd.DataFrame]] = {}
        self._incumbent: pd.DataFrame | None = None
        OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    def _register_views(self):
        panel = ANCHOR_DIR / "checkpoint_panel.parquet"
        trees = [ANCHOR_DIR / f"tree_{year}.parquet" for year in range(2021, 2026)]
        missing = [str(p.relative_to(ROOT)) for p in [panel, *trees] if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Regenerate the incumbent first (see README: compare_intraday_anchors.py tree/compare). "
                f"Missing: {missing}")
        files = ", ".join(f"'{p}'" for p in trees)
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW checkpoint_panel AS SELECT * FROM read_parquet('{panel}')")
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW tree_forecasts AS SELECT * FROM read_parquet([{files}])")
        self.con.execute("""
            CREATE OR REPLACE TEMP VIEW comparison_panel AS
            SELECT p.*, p.daily_forecast AS original, tree.forecast AS tree
            FROM checkpoint_panel p JOIN tree_forecasts tree USING (date, permno)
            WHERE NOT p.is_early_close
        """)

    def _load(self, start: str, end: str, rows: int | None = None, seed: int | None = None) -> pd.DataFrame:
        sample = f"USING SAMPLE {rows} ROWS (reservoir, {seed})" if rows else ""
        checkpoint = "AND cp_available" if rows else ""
        frame = self.con.sql(f"""
            SELECT * FROM (
                SELECT {', '.join(self._columns)} FROM comparison_panel
                WHERE date >= DATE '{start}' AND date < DATE '{end}' {checkpoint}
                ORDER BY date, permno
            ) {sample} ORDER BY date, permno
        """).df()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["anchor_forecast"] = frame[ANCHOR].astype(float)
        frame["log_daily_forecast"] = np.log1p(frame["anchor_forecast"])
        return frame

    def full_panel(self, start: str, end: str, columns=None) -> pd.DataFrame:
        """Every regular-session stock-day in [start, end), unsampled.

        Use this for cross-sectional features (e.g. PCA or regimes across all stocks on a
        date at 15:30); compute them here and merge onto fold frames by date and permno.
        """
        cols = ", ".join(columns) if columns else "*"
        frame = self.con.sql(f"""SELECT {cols} FROM comparison_panel
            WHERE date >= DATE '{start}' AND date < DATE '{end}' ORDER BY date, permno""").df()
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def add_features(self, function):
        """Register ``function(frame, fold_year, role) -> DataFrame`` of new columns (same index).

        Applied to every fold frame as folds are built. Anything *fitted* inside it must
        use rows before ``fold_year`` only; ``role`` is one of tune/valid/refit/test.
        """
        self._feature_functions.append(function)

    def folds(self):
        for year in self.years:
            prep = json.loads((INCUMBENT_PREPROCESSORS / f"preprocessor_{year}.json").read_text())
            types = {key: list(prep[key]) for key in ("continuous", "boolean", "categorical")}
            start = "2021-10-01" if year == 2022 else f"{year - 1}-01-01"
            windows = {"tune": ("2021-01-01", start), "valid": (start, f"{year}-01-01"),
                       "refit": ("2021-01-01", f"{year}-01-01"), "test": (f"{year}-01-01", f"{year + 1}-01-01")}
            frames = {}
            for role, (lo, hi) in windows.items():
                rows = SAMPLE_ROWS.get(role)
                frames[role] = self._load(lo, hi, rows, SEED + SEED_OFFSET[role] + year if rows else None)
            assert frames["tune"].date.max() < frames["valid"].date.min()
            assert frames["valid"].date.max() < frames["test"].date.min()
            assert frames["refit"].date.max() < frames["test"].date.min()
            for function in self._feature_functions:
                for role, frame in frames.items():
                    added = function(frame, year, role)
                    if added is not None and len(added.columns):
                        if set(added.columns) & FORBIDDEN:
                            raise ValueError(f"Feature function returned forbidden columns: {set(added.columns) & FORBIDDEN}")
                        frames[role] = pd.concat([frame, added.set_axis(frame.index)], axis=1)
            yield Fold(year=year, types=types, **frames, features={
                "daily": list(DAILY_FEATURES), "prior_checkpoint": list(PRIOR_CHECKPOINT_FEATURES),
                "intraday": list(INTRADAY_FEATURES), "control": list(CONTROL_FEATURES), "all": list(MODEL_FEATURES),
            })

    # ------------------------------------------------------------ incumbent
    def incumbent_correction(self, fold: Fold):
        """Re-fit the incumbent ridge exactly as experiment 27 does. Returns (correction, settings)."""
        prep_tune = fold.preprocessor(fold.tune)
        x_tune, x_valid = prep_tune.transform(fold.tune), prep_tune.transform(fold.valid)
        z_tune, bounds = fold.target(fold.tune), fold.bounds(fold.tune)
        trials = []
        for strength in RIDGE_GRID:
            beta = fit_ridge(x_tune, z_tune, strength)
            forecast = fold.to_shares(predict_ridge(x_valid, beta), fold.valid, bounds)
            error = np.log1p(forecast) - fold.valid.log1p_auction_volume.to_numpy(float)
            trials.append((np.sqrt(np.mean(error ** 2)), strength))
        strength = min(trials)[1]
        prep = fold.preprocessor(fold.refit)
        beta = fit_ridge(prep.transform(fold.refit), fold.target(fold.refit), strength)
        return predict_ridge(prep.transform(fold.test), beta), {"strength": strength}

    def _incumbent_predictions(self) -> pd.DataFrame:
        if self._incumbent is None:
            path = OUT / "submissions" / INCUMBENT / "predictions.parquet"
            if path.is_file():
                saved = pd.read_parquet(path)
                self._incumbent = saved[saved.date.dt.year.isin(self.years)].reset_index(drop=True)
            if self._incumbent is None or set(self._incumbent.date.dt.year) != set(self.years):
                parts = []
                for fold in self.folds():
                    correction, _ = self.incumbent_correction(fold)
                    parts.append(self._prediction_frame(fold, fold.to_shares(correction)))
                self._incumbent = pd.concat(parts, ignore_index=True)
                if not self.quick:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._incumbent.to_parquet(path, index=False)
        return self._incumbent

    # ------------------------------------------------------------ submitting
    @staticmethod
    def _prediction_frame(fold: Fold, forecast) -> pd.DataFrame:
        out = fold.test[["date", "permno", "auction_volume", "cp_available", "anchor_forecast"]].copy()
        out["forecast"] = np.asarray(forecast, dtype=float)
        return out

    def submit(self, fold: Fold, correction, name: str, owner: str = "", description: str = ""):
        """Hand in the predicted log correction for every row of ``fold.test``."""
        forecast = fold.to_shares(correction)
        if not np.isfinite(forecast).all():
            raise ValueError("Non-finite forecasts.")
        self._submissions.setdefault(name, [])
        self._submissions[name] = [f for f in self._submissions[name] if f.date.dt.year.iloc[0] != fold.year]
        self._submissions[name].append(self._prediction_frame(fold, forecast))
        self._meta = getattr(self, "_meta", {})
        self._meta[name] = {"owner": owner, "description": description}
        metrics = forecast_metrics(fold.test.auction_volume, forecast)
        print(f"{name} {fold.year}: log RMSE {metrics['rmse_log']:.5f}, WAPE {metrics['wape']:.4%}, "
              f"bias {metrics['aggregate_bias']:+.2%}")

    # ------------------------------------------------------------- scoring
    def _segments(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach segment labels used by the diagnostics (all known before the forecast)."""
        keys = frame[["date", "permno"]].drop_duplicates()
        self.con.register("segment_keys", keys)
        rebalance = f"""LEFT JOIN read_parquet('{REBALANCE}') r
                        ON CAST(r.date AS DATE) = CAST(k.date AS DATE) AND r.permno = k.permno""" if REBALANCE.is_file() else ""
        rebalance_cols = """, coalesce(r.russell_recon, 0) AS russell_recon, coalesce(r.msci_review, 0) AS msci_review,
                           coalesce(r.sp500_add_day, 0) + coalesce(r.sp500_del_day, 0) AS sp500_change_day""" if REBALANCE.is_file() else ""
        seg = self.con.sql(f"""
            SELECT k.date, k.permno, p.size_percentile, p.liquidity_percentile, p.prior_auction_observations,
                   p.quarterly_options_expiry, p.monthly_options_expiry, p.is_last_trading_day_month {rebalance_cols}
            FROM segment_keys k
            LEFT JOIN read_parquet('{PANEL}') p ON CAST(p.date AS DATE) = CAST(k.date AS DATE) AND p.permno = k.permno
            {rebalance}
        """).df()
        self.con.unregister("segment_keys")
        seg["date"] = pd.to_datetime(seg["date"])
        out = frame.merge(seg, on=["date", "permno"], how="left", validate="many_to_one")
        if FOMC.is_file():
            fomc = pd.to_datetime(pd.read_csv(FOMC)["announcement_date"]).dt.normalize()
            out["fomc_day"] = out.date.isin(set(fomc)).astype(int)
        return out

    def _annual(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        rows = []
        for year, part in frame.groupby(frame.date.dt.year):
            rows.append({"year": year, **forecast_metrics(part.auction_volume, part[column])})
        return pd.DataFrame(rows)

    def _paired_intervals(self, frame: pd.DataFrame, reference: str, candidate: str) -> pd.DataFrame:
        """Same 20-date circular block bootstrap as experiment 27 (candidate - reference)."""
        actual = frame.auction_volume.to_numpy(float)
        p, q = frame[reference].to_numpy(float), frame[candidate].to_numpy(float)
        totals = frame[["date"]].assign(
            year=frame.date.dt.year, n=1, auction_volume=actual,
            sse_p=(np.log1p(actual) - np.log1p(p)) ** 2, sse_q=(np.log1p(actual) - np.log1p(q)) ** 2,
            ae_p=np.abs(actual - p), ae_q=np.abs(actual - q), p=p, q=q,
        ).groupby(["year", "date"]).sum()
        rows = []
        for period, years in [("development", DEVELOPMENT_YEARS), ("confirmation", (CONFIRMATION_YEAR,))]:
            years = [y for y in years if y in totals.index.get_level_values(0)]
            if not years:
                continue
            daily = [totals.loc[y].to_numpy() for y in years]
            weights = [circular_weights(len(arr), BOOTSTRAP["block"], BOOTSTRAP["reps"], BOOTSTRAP["seed"] + y)
                       for y, arr in zip(years, daily)]
            point = np.mean([statistics(arr.sum(axis=0)) for arr in daily], axis=0)
            sampled = np.mean([statistics(w @ arr) for w, arr in zip(weights, daily)], axis=0)
            lower, upper = np.quantile(sampled[:, 1] - sampled[:, 0], [0.025, 0.975], axis=0)
            for i, metric in enumerate(("rmse_log", "wape", "signed_bias", "absolute_bias")):
                rows.append({"period": period, "metric": metric, "difference": point[1, i] - point[0, i],
                             "lower": lower[i], "upper": upper[i]})
        return pd.DataFrame(rows)

    def report(self, name: str, segments: bool = True, record: bool = True) -> dict:
        """Score a submission against the incumbent and the daily anchor; append to the leaderboard."""
        if name not in self._submissions:
            raise KeyError(f"No submission named {name!r}.")
        candidate = pd.concat(self._submissions[name], ignore_index=True)
        missing_years = set(self.years) - set(candidate.date.dt.year)
        if missing_years:
            raise ValueError(f"{name} has no predictions for {sorted(missing_years)}.")
        incumbent = self._incumbent_predictions()[["date", "permno", "forecast"]].rename(columns={"forecast": INCUMBENT})
        frame = candidate.rename(columns={"forecast": name}).merge(
            incumbent, on=["date", "permno"], how="inner", validate="one_to_one")
        assert len(frame) == len(candidate), "Submission rows do not match the incumbent's test rows."
        frame = frame.rename(columns={"anchor_forecast": "daily_anchor"})

        models = ["daily_anchor", INCUMBENT, name]
        annual = pd.concat([self._annual(frame, m).assign(model=m) for m in models], ignore_index=True)
        dev = annual[annual.year.isin(DEVELOPMENT_YEARS)]
        summary = pd.DataFrame({
            "development_rmse_log": dev.groupby("model").rmse_log.mean(),
            "development_wape": dev.groupby("model").wape.mean(),
            "development_bias": dev.groupby("model").aggregate_bias.mean(),
            "confirmation_rmse_log": annual[annual.year == CONFIRMATION_YEAR].set_index("model").rmse_log,
            "confirmation_wape": annual[annual.year == CONFIRMATION_YEAR].set_index("model").wape,
        }).reindex(models)
        intervals = self._paired_intervals(frame, INCUMBENT, name)

        gate = {}
        if set(DEVELOPMENT_YEARS) <= set(self.years):
            by_year = annual.pivot(index="year", columns="model", values="rmse_log")
            gain = summary.loc[INCUMBENT, "development_rmse_log"] - summary.loc[name, "development_rmse_log"]
            worst = (by_year.loc[list(DEVELOPMENT_YEARS), name] - by_year.loc[list(DEVELOPMENT_YEARS), INCUMBENT]).max()
            dev_ci = intervals.query("period == 'development' and metric == 'rmse_log'").iloc[0]
            gate = {
                "dev_rmse_gain": gain, "gain_ok": gain >= GATE["min_dev_rmse_gain"],
                "worst_year_change": worst, "annual_ok": worst <= GATE["max_annual_rmse_worsening"],
                "wape_ok": summary.loc[name, "development_wape"] <= summary.loc[INCUMBENT, "development_wape"],
                "interval_ok": dev_ci.upper < 0,
            }
            gate["passes_gate"] = all(gate[k] for k in ("gain_ok", "annual_ok", "wape_ok", "interval_ok"))

        segment_table = None
        if segments:
            seg = self._segments(frame)
            if not self.quick:  # segment diagnostics describe the development years
                seg = seg[seg.date.dt.year.isin(DEVELOPMENT_YEARS)]
            def label(df):
                masks = {
                    "all": np.ones(len(df), bool),
                    "month_end": df.is_last_trading_day_month.fillna(False).astype(bool).to_numpy(),
                    "quarterly_expiry": df.quarterly_options_expiry.fillna(False).astype(bool).to_numpy(),
                    "low_liquidity": (df.liquidity_percentile < 1 / 3).to_numpy(),
                    "small_size": (df.size_percentile < 1 / 3).to_numpy(),
                    "short_history": (df.prior_auction_observations < 20).to_numpy(),
                }
                for col in ("russell_recon", "msci_review", "sp500_change_day", "fomc_day"):
                    if col in df:
                        masks[col] = (df[col].fillna(0) > 0).to_numpy()
                threshold = df.groupby(df.date.dt.year).auction_volume.transform(lambda s: s.quantile(0.99))
                masks["largest_1pct_auctions (diagnostic)"] = (df.auction_volume >= threshold).to_numpy()
                return masks
            rows = []
            for segment, mask in label(seg).items():
                part = seg[mask]
                if len(part) < 500:
                    continue
                for m in models:
                    r = forecast_metrics(part.auction_volume, part[m])
                    rows.append({"segment": segment, "model": m, "rows": len(part),
                                 "rmse_log": r["rmse_log"], "wape": r["wape"], "bias": r["aggregate_bias"]})
            segment_table = pd.DataFrame(rows).pivot(index="segment", columns="model", values="rmse_log")[models]

        result = {"summary": summary, "annual": annual, "intervals": intervals, "gate": gate, "segments": segment_table}
        if record and not self.quick:
            path = OUT / "submissions" / name
            path.mkdir(parents=True, exist_ok=True)
            candidate.to_parquet(path / "predictions.parquet", index=False)
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "name": name, **getattr(self, "_meta", {}).get(name, {}), "git_commit": _git_commit(),
                **{k: summary.loc[name, k] for k in summary.columns},
                **{f"incumbent_{k}": summary.loc[INCUMBENT, k] for k in ("development_rmse_log", "development_wape")},
                **{k: v for k, v in gate.items()},
            }
            board = pd.concat([pd.read_csv(LEADERBOARD), pd.DataFrame([row])], ignore_index=True) \
                if LEADERBOARD.is_file() else pd.DataFrame([row])
            board.to_csv(LEADERBOARD, index=False)
        return result


def leaderboard() -> pd.DataFrame:
    """The shared results table, best development log RMSE first."""
    if not LEADERBOARD.is_file():
        return pd.DataFrame()
    return pd.read_csv(LEADERBOARD).sort_values("development_rmse_log").reset_index(drop=True)
