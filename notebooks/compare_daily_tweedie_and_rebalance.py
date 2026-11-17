"""Walk-forward test of raw-share Tweedie loss and prior-close event features."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "auction-matplotlib"))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from modeling_utils import apply_preprocessor, fit_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/23_tweedie_and_rebalance"
DAILY = ROOT / "outputs/reproduction/03_compact_nonlinear/models"
CANONICAL_PANEL = ROOT / "outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet"
CACHE_PANEL = ROOT / "midterm presentation/data_analysis/cache/enriched_daily_panel.parquet"
RIDGE = ROOT / "outputs/eda/09_global_finalists"
YEARS = (2022, 2023, 2024, 2025)
RUSSELL_CLOSE = (
    "2015-06-26", "2016-06-24", "2017-06-23", "2018-06-22", "2019-06-28",
    "2020-06-26", "2021-06-25", "2022-06-24", "2023-06-23", "2024-06-28", "2025-06-27",
)
EVENT_FEATURES = ["russell_close", "prior_event_log_lift", "prior_event_count", "event_size_percentile"]
MODELS = ["log_control", "log_events", "tweedie", "tweedie_events"]
SEGMENTS = ["all", "regular", "month_end", "quarter_end", "russell", "actual_top_1pct"]


def eligible_events(con, panel):
    schedule = ", ".join(f"DATE '{date}'" for date in RUSSELL_CLOSE)
    con.execute(f"""
        CREATE TEMP TABLE events AS
        WITH days AS (
            SELECT date, permno, auction_volume, median20_auction_volume, size_percentile,
                   date IN ({schedule}) AS russell_close,
                   CASE WHEN date IN ({schedule}) THEN 'russell'
                        WHEN is_last_trading_day_quarter THEN 'quarter_end'
                        WHEN is_last_trading_day_month THEN 'month_end'
                        ELSE 'regular' END AS event_type
            FROM read_parquet(?)
            WHERE eligible_target
              AND (is_last_trading_day_month OR is_last_trading_day_quarter OR date IN ({schedule}))
        ), event_outcomes AS (
            SELECT *, CASE WHEN median20_auction_volume > 0 THEN
                      greatest(-3.0, least(3.0, ln(1 + auction_volume) - ln(1 + median20_auction_volume)))
                      END AS lift
            FROM days
        )
        SELECT date, permno, event_type, russell_close,
               avg(lift) OVER history AS prior_event_log_lift,
               count(lift) OVER history AS prior_event_count,
               size_percentile AS event_size_percentile,
               max(date) OVER history AS last_event_date
        FROM event_outcomes
        WINDOW history AS (PARTITION BY permno, event_type ORDER BY date
                           ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
    """, [str(panel)])
    assert con.sql("SELECT count(*) FROM events WHERE last_event_date >= date").fetchone()[0] == 0
    dates = con.sql("SELECT DISTINCT date::DATE FROM events WHERE russell_close ORDER BY 1").fetchall()
    assert tuple(str(date) for date, in dates) == RUSSELL_CLOSE


def get_frame(con, panel, types, first_year, stop_year, cap=None, seed=None):
    columns = list(dict.fromkeys(["date", "permno", "auction_volume", *sum(types, [])]))
    select = ", ".join(f'"{col}"' for col in columns)
    sample = f" USING SAMPLE {cap} ROWS (reservoir, {seed})" if cap else ""
    return con.execute(f"""
        SELECT a.*, coalesce(event_type, 'regular') AS event_type,
               coalesce(russell_close, false) AS russell_close,
               CASE WHEN event_type IS NULL THEN 0.0 ELSE prior_event_log_lift END AS prior_event_log_lift,
               coalesce(prior_event_count, 0) AS prior_event_count,
               CASE WHEN event_type IS NULL THEN 0.0 ELSE event_size_percentile END AS event_size_percentile
        FROM (SELECT * FROM (
                SELECT {select} FROM read_parquet(?) WHERE eligible_target AND year >= ? AND year < ?
              ){sample}) a
        LEFT JOIN events USING (date, permno)
        ORDER BY date, permno
    """, [str(panel), first_year, stop_year]).df()


def matrix(frame, preprocessing, types, features):
    approved = np.isin(preprocessing["output_sources"], list(features))
    x = np.ascontiguousarray(apply_preprocessor(frame, preprocessing, *types)[:, approved])
    events = np.ascontiguousarray(frame[EVENT_FEATURES].to_numpy(dtype=np.float32))
    return x, np.column_stack((x, events))


def fit_context(frame):
    actual = frame.auction_volume.to_numpy(dtype=float)
    return float(np.median(actual)), np.quantile(np.log1p(actual), [0.001, 0.999])


def shares_and_offset(frame, fallback):
    actual = frame.auction_volume.to_numpy(dtype=float)
    offset = np.log1p(frame.median20_auction_volume.to_numpy(dtype=float, na_value=np.nan))
    offset = np.where(np.isfinite(offset), offset, np.log1p(fallback))
    return actual, offset


def dataset(x, frame, fallback, power, reference=None):
    actual, offset = shares_and_offset(frame, fallback)
    if power is None:
        return lgb.Dataset(x, label=np.log1p(actual) - offset, reference=reference)
    # The starting mean is the same trailing stock scale as in the log model.
    return lgb.Dataset(x, label=actual / (1 + fallback),
                       init_score=offset - np.log1p(fallback), reference=reference)


def predict(model, x, frame, fallback, bounds, power):
    _, offset = shares_and_offset(frame, fallback)
    eta = offset + model.predict(x, raw_score=True)
    if power is not None:
        eta = np.logaddexp(0, eta)
    return np.expm1(np.clip(eta, *bounds))


def score(actual, forecast, dates):
    y, p = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    _, day = np.unique(dates, return_inverse=True)
    total_y, total_p = [np.bincount(day, weights=value) for value in (y, p)]
    return {
        "rows": len(y), "sessions": len(total_y), "log_rmse": np.mean((np.log1p(y) - np.log1p(p)) ** 2) ** .5,
        "share_wape": np.abs(y - p).sum() / y.sum(), "daily_total_wape": np.abs(total_y - total_p).sum() / y.sum(),
        "share_bias": p.sum() / y.sum() - 1,
    }


def log_evaluation(frame, fallback, bounds):
    actual, offset = shares_and_offset(frame, fallback)
    actual_log = np.log1p(actual)

    def evaluate(prediction, _dataset):
        loss = np.mean((actual_log - np.clip(prediction + offset, *bounds)) ** 2) ** .5
        return "share_log_rmse", loss, False

    return evaluate


def fit_model(year, power, threads, train, validation=None, evaluate=None, rounds=1000):
    params = {
        "objective": "regression" if power is None else "tweedie",
        "metric": "None" if power is None else "tweedie",
        "learning_rate": 0.035, "num_leaves": 31, "min_data_in_leaf": 350, "max_bin": 127,
        "feature_fraction": 0.85, "lambda_l1": 0.1, "lambda_l2": 5.0,
        "seed": 20261018 + year, "num_threads": threads, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }
    if power is not None:
        params["tweedie_variance_power"] = power
    return lgb.train(params, train, num_boost_round=rounds,
                     valid_sets=[validation] if validation is not None else None,
                     feval=evaluate, callbacks=[lgb.early_stopping(75, verbose=False)] if validation is not None else [])


def tweedie_loss(actual, prediction, power, scale):
    actual = np.asarray(actual, dtype=float) / scale
    prediction = np.asarray(prediction, dtype=float) / scale
    return np.mean(prediction ** (2 - power) / (2 - power)
                   - actual * prediction ** (1 - power) / (1 - power))


def reference_forecast(con, test, year, base_dir, types):
    if base_dir:
        saved = con.execute("SELECT date, permno, auction_volume, forecast FROM read_parquet(?) ORDER BY date, permno",
                            [str(base_dir / f"base_{year}.parquet")]).df()
        pd.testing.assert_frame_equal(test[["date", "permno", "auction_volume"]],
                                      saved[["date", "permno", "auction_volume"]], check_dtype=False)
        return saved.forecast.to_numpy()
    prep = json.loads((DAILY / f"preprocessor_{year}.json").read_text())
    settings = json.loads((DAILY / f"reconstruction_{year}.json").read_text())
    names = json.loads((DAILY / f"feature_columns_lgbm_residual_58_l2_{year}.json").read_text())["output_names"]
    pos = [prep["output_names"].index(name) for name in names]
    model = lgb.Booster(model_file=str(DAILY / f"lgbm_residual_58_l2_{year}.txt"))
    design = np.ascontiguousarray(apply_preprocessor(test, prep, *types)[:, pos])
    return predict(model, design, test, settings["fallback"], settings["clip_log"], None)


def run_fold(year, panel, threads, base_dir):
    template = json.loads((DAILY / "preprocessor_2022.json").read_text())
    types = [list(template[key]) for key in ("continuous", "boolean", "categorical")]
    features = json.loads((RIDGE / "frozen_global_benchmark.json").read_text())["features"]
    con = duckdb.connect(config={"threads": "1"})
    eligible_events(con, panel)
    train = get_frame(con, panel, types, year - 6, year - 1, 180_000, 20261018 + year)
    validation = get_frame(con, panel, types, year - 1, year, 100_000, 20261118 + year)
    refit = get_frame(con, panel, types, year - 5, year, 300_000, 20261218 + year)
    test = get_frame(con, panel, types, year, year + 1)
    assert len(train) == 180_000 and len(validation) == 100_000 and len(refit) == 300_000
    assert train.date.max() < validation.date.min() <= refit.date.max() < test.date.min()
    assert all(not frame.duplicated(["date", "permno"]).any() for frame in (train, validation, refit, test))
    frozen = con.execute("SELECT date, permno, auction_volume FROM read_parquet(?) ORDER BY date, permno",
                         [str(RIDGE / f"predictions/global_finalist_predictions_{year}.parquet")]).df()
    pd.testing.assert_frame_equal(test[["date", "permno", "auction_volume"]], frozen, check_dtype=False)
    sample_metadata = {}
    for role, frame in [("tune", train), ("validation", validation), ("refit", refit), ("test", test)]:
        keys = np.column_stack((frame.date.to_numpy(dtype="datetime64[D]").astype("int64"), frame.permno)).astype("<i8")
        sample_metadata[role] = {"rows": len(frame), "first": str(frame.date.iloc[0].date()),
                                 "last": str(frame.date.iloc[-1].date()), "keys_sha256": hashlib.sha256(keys).hexdigest()}

    fallback, bounds = fit_context(train)
    _, prep = fit_preprocessor(train, *types)
    matrices = [matrix(frame, prep, types, features) for frame in (train, validation)]
    validation_rows, selected = [], {}
    # All Tweedie powers are compared using the same validation deviance index.
    for name in MODELS:
        is_tweedie = name.startswith("tweedie")
        x, x_valid = [mat[int(name.endswith("events"))] for mat in matrices]
        for power in ((1.3, 1.7) if is_tweedie else (None,)):
            print(f"{year} tune {name} {power}", flush=True)
            d = dataset(x, train, fallback, power)
            valid = dataset(x_valid, validation, fallback, power, d)
            evaluate = log_evaluation(validation, fallback, bounds) if power is None else None
            model = fit_model(year, power, threads, d, valid, evaluate)
            prediction = predict(model, x_valid, validation, fallback, bounds, power)
            measured = score(validation.auction_volume, prediction, validation.date)
            criterion = (measured["log_rmse"] if power is None else
                         tweedie_loss(validation.auction_volume, prediction, 1.5, 1 + fallback))
            result = {"model": name, "power": power, "rounds": model.best_iteration,
                      "criterion": criterion, **measured}
            validation_rows.append(result)
            if name not in selected or criterion < selected[name]["criterion"]:
                selected[name] = result
    del matrices, x, x_valid, train, validation, d, valid

    output = test[["date", "permno", "auction_volume", "event_type"]].copy()
    output["published_log"] = reference_forecast(con, test, year, base_dir, types)
    baseline = pd.read_csv(ROOT / "outputs/experiments/18_error_diagnosis_calibration/annual_metrics.csv")
    expected = baseline.loc[(baseline.test_year == year) & baseline.method.eq("uncalibrated")].iloc[0]
    baseline_scores = score(output.auction_volume, output.published_log, output.date)
    for computed, prior in (("log_rmse", "rmse_log"), ("share_wape", "wape"), ("share_bias", "aggregate_bias")):
        np.testing.assert_allclose(baseline_scores[computed], expected[prior], rtol=1e-9, atol=1e-10)

    fallback, bounds = fit_context(refit)
    _, prep = fit_preprocessor(refit, *types)
    matrices = [matrix(frame, prep, types, features) for frame in (refit, test)]
    for name, best in selected.items():
        print(f"{year} refit {name} {best['power']} {best['rounds']} rounds", flush=True)
        x, x_test = [mat[int(name.endswith("events"))] for mat in matrices]
        model = fit_model(year, best["power"], threads, dataset(x, refit, fallback, best["power"]),
                          rounds=best["rounds"])
        output[name] = predict(model, x_test, test, fallback, bounds, best["power"])
        assert np.isfinite(output[name]).all() and output[name].ge(0).all()

    valid_path = OUT / f"validation_{year}.csv"
    pd.DataFrame(validation_rows).to_csv(valid_path, index=False)
    metadata = {"panel": str(panel.relative_to(ROOT)) if panel.is_relative_to(ROOT) else str(panel),
                "lightgbm": lgb.__version__, "samples": sample_metadata,
                "reference": "verified cache" if base_dir else "saved LightGBM model",
                "fallback": fallback, "share_bounds": np.expm1(bounds).tolist(), "selected": selected}
    (OUT / f"fit_{year}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    con.register("forecast_output", output)
    destination = OUT / f"predictions_{year}.parquet"
    con.execute("COPY forecast_output TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(destination)])
    print(f"{year} verified and saved {len(output):,} forecasts", flush=True)


def summarize():
    annual, paired = [], []
    paths = [str(OUT / f"predictions_{year}.parquet") for year in YEARS]
    con = duckdb.connect()
    for year, path in zip(YEARS, paths, strict=True):
        frame = con.execute("SELECT * FROM read_parquet(?) ORDER BY date, permno", [path]).df()
        n_tail = int(np.ceil(.01 * len(frame)))
        tail = frame.auction_volume.nlargest(n_tail).index
        segments = {"all": frame, "actual_top_1pct": frame.loc[tail]}
        segments.update({name: frame[frame.event_type == name] for name in SEGMENTS[1:-1]})
        for name in ["published_log", *MODELS]:
            for segment, subset in segments.items():
                annual.append({"year": year, "model": name, "segment": segment,
                               **score(subset.auction_volume, subset[name], subset.date)})
        paired.append(frame)
    annual = pd.DataFrame(annual)
    annual.to_csv(OUT / "annual_metrics.csv", index=False)
    annual[annual.year.lt(2025)].groupby(["model", "segment"]).mean(numeric_only=True).drop(columns="year").to_csv(
        OUT / "development_metrics.csv"
    )

    rng = np.random.default_rng(20261116)
    deltas = {}
    for frame in paired[:3]:
        dates, day = np.unique(frame.date, return_inverse=True)
        n = len(dates)
        idx = ((rng.integers(n, size=(4000, int(np.ceil(n / 20)), 1)) + np.arange(20)) % n).reshape(4000, -1)[:, :n]
        y = frame.auction_volume.to_numpy(dtype=float)
        y_total = np.bincount(day, weights=y)
        denominator = y_total[idx].sum(axis=1)
        counts = np.bincount(day)[idx].sum(axis=1)
        measures = {}
        for model in ["published_log", *MODELS]:
            pred = frame[model].to_numpy(dtype=float)
            pred_total = np.bincount(day, weights=pred)
            squared_log = np.bincount(day, weights=(np.log1p(y) - np.log1p(pred)) ** 2)
            abs_stock = np.bincount(day, weights=np.abs(y - pred))
            measures[model] = np.array([
                np.sqrt(squared_log[idx].sum(axis=1) / counts),
                abs_stock[idx].sum(axis=1) / denominator,
                np.abs(y_total - pred_total)[idx].sum(axis=1) / denominator,
                pred_total[idx].sum(axis=1) / denominator - 1,
            ])
        contrasts = [(m, "log_control") for m in ["log_events", "tweedie", "tweedie_events"]]
        contrasts.extend([("tweedie_events", "tweedie"), ("tweedie_events", "log_events")])
        for model, control in contrasts:
            deltas.setdefault((model, control), []).append(measures[model] - measures[control])
    for (model, control), folds in deltas.items():
        quantiles = np.quantile(np.mean(folds, axis=0), [.025, .975], axis=1).T
        for metric, (low, high) in zip(("log_rmse", "share_wape", "daily_total_wape", "share_bias"), quantiles, strict=True):
            annual_row = annual.loc[annual.year.lt(2025) & annual.segment.eq("all")].groupby("model")[metric].mean()
            yield {"model": model, "control": control, "metric": metric, "delta": annual_row[model] - annual_row[control],
                   "lower_95": low, "upper_95": high}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=YEARS, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--panel", type=Path, default=CANONICAL_PANEL if CANONICAL_PANEL.exists() else CACHE_PANEL)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summarize:
        for year in (args.year,) if args.year else YEARS:
            run_fold(year, args.panel.resolve(), args.threads, args.reference_cache)
    if args.summarize or not args.year:
        pd.DataFrame(summarize()).to_csv(OUT / "paired_development_intervals.csv", index=False)


if __name__ == "__main__":
    main()
