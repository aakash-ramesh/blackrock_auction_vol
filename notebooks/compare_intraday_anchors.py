"""Compare daily forecast anchors with the same annual 15:30 ridge procedure."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
default_panel = ROOT / "outputs/eda/05_feature_engineering/enriched_daily_panel.parquet"
cached_panel = ROOT / "midterm presentation/data_analysis/cache/enriched_daily_panel.parquet"
if not default_panel.is_file() and cached_panel.is_file():
    os.environ.setdefault("AUCTION_DAILY_PANEL", str(cached_panel))

import challenger_utils as cu
from notebooks.modeling_utils import apply_preprocessor, fit_preprocessor

OUT = ROOT / "outputs/experiments/27_intraday_anchor_comparison"
TREE = "lgbm_61_ratios_large_tuned"
HYBRID = "nn_cnn_lstm_hybrid"
YEARS = range(2021, 2026)
HISTORY = cu.EDA / "16e_sequence_neural_networks"


def save_predictions(frame, path):
    temporary = path.with_suffix(".partial")
    cu.con.register("anchor_output", frame)
    try:
        cu.con.execute("COPY anchor_output TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)])
    finally:
        cu.con.unregister("anchor_output")
    temporary.replace(path)


def fit_tree(x, y, config, iterations, seed):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=iterations, random_state=seed, n_jobs=6, deterministic=True,
        force_col_wise=True, verbosity=-1, **config,
    ).fit(x, y)


def tune_tree():
    import lightgbm as lgb

    year = 2021
    data = cu.prepare_tier(year, "large")
    grid = pd.read_csv(cu.EDA / "16a_lightgbm_improvements/tuned_search_space.csv")
    trials = []
    for row in grid.to_dict("records"):
        config_id = row.pop("config_id")
        model = lgb.LGBMRegressor(
            n_estimators=3000, random_state=cu.SEED + year + config_id,
            n_jobs=6, deterministic=True, force_col_wise=True, verbosity=-1, **row,
        )
        model.fit(
            data["x_tune"], data["y_tune"],
            eval_set=[(data["x_validation"], data["y_validation_residual"])],
            eval_metric="rmse", callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        prediction = cu.final_log_prediction(
            model.predict(data["x_validation"]), data["validation_base"], data["tune_log"],
        )
        rmse = np.sqrt(np.mean((data["y_validation_log"] - prediction) ** 2))
        trials.append({
            "config_id": config_id, "config": json.dumps(row),
            "best_iteration": model.best_iteration_, "validation_rmse_log": rmse,
        })
        print(f"2021 tree config {config_id}: validation RMSE {rmse:.6f}", flush=True)
    results = pd.DataFrame(trials)
    results.to_csv(OUT / "tree_2021_validation.csv", index=False)
    return results.loc[results.validation_rmse_log.idxmin()]


def daily_refit(year):
    refit = cu.load_period(year - 5, year, cu.TIERS["large"]["refit"], cu.SEED + 200 + year)
    test = cu.load_period(year, year + 1)
    types = (cu.CONTINUOUS, cu.BOOLEAN, cu.CATEGORICAL)
    x_refit, prep = fit_preprocessor(refit, *types)
    x_test = apply_preprocessor(test, prep, *types)
    y_refit, base = cu.residual_target_and_base(refit, test)
    return refit, test, x_refit, x_test, y_refit, base


def write_daily(test, year, anchor, residual, base, refit):
    forecast = np.expm1(cu.final_log_prediction(residual, base, refit.log1p_auction_volume))
    output = test[["date", "permno", "auction_volume"]].copy()
    output["forecast"] = forecast
    assert np.isfinite(forecast).all() and (forecast >= 0).all()
    assert not output.duplicated(["date", "permno"]).any()
    save_predictions(output, OUT / f"{anchor}_{year}.parquet")
    rmse = np.sqrt(np.mean((test.log1p_auction_volume - np.log1p(forecast)) ** 2))
    print(f"{year} {anchor}: {len(output):,} forecasts, log RMSE {rmse:.8f}", flush=True)


def build_trees(years):
    history = pd.read_csv(cu.EDA / "16a_lightgbm_improvements/hyperparameter_validation.csv")
    for year in years:
        if (OUT / f"tree_{year}.parquet").is_file():
            continue
        print(f"{year}: preparing tuned LightGBM", flush=True)
        if year == 2021:
            trials = OUT / "tree_2021_validation.csv"
            if trials.is_file():
                historical = pd.read_csv(trials)
                chosen = historical.loc[historical.validation_rmse_log.idxmin()]
            else:
                chosen = tune_tree()
        else:
            historical = history.query("test_year == @year and candidate == @TREE")
            chosen = historical.loc[historical.validation_rmse_log.idxmin()]
        refit, test, xr, xe, target, base = daily_refit(year)
        model = fit_tree(xr, target, json.loads(chosen["config"]),
                         max(25, int(chosen["best_iteration"])), cu.SEED + 300 + year)
        model.booster_.save_model(str(OUT / f"daily_tree_{year}.txt"))
        write_daily(test, year, "tree", model.predict(xe), base, refit)


def build_hybrids(years):
    import sequence_utils as su

    su.torch.set_num_threads(6)
    print(f"Hybrid training device: {su.DEVICE}", flush=True)
    store = su.build_history_store()
    history = pd.read_csv(HISTORY / "nn_metrics.csv").query("candidate == @HYBRID").set_index("test_year")
    for year in years:
        if (OUT / f"hybrid_{year}.parquet").is_file():
            continue
        if year == 2021:
            data = su.prepare_fold(year, store)
            validation = {
                "x_tab": data["x_validation"], "index": data["index_validation"],
                "base": data["validation_base"], "y_log": data["y_validation_log"],
                "train_log": data["tune_log"],
            }
            _, epochs, trials = su.train(
                "hybrid", data["x_tune"], data["y_tune"], store, data["index_tune"],
                cu.SEED + year + 2, su.TRAIN["max_epochs"], validation,
            )
            pd.DataFrame(trials).to_csv(OUT / "hybrid_2021_validation.csv", index=False)
            del data
        else:
            epochs = int(history.loc[year, "iterations"])
        refit, test, xr, xe, target, base = daily_refit(year)
        print(f"{year}: refitting hybrid for {epochs} epochs", flush=True)
        model, _, _ = su.train(
            "hybrid", xr, target, store, su.sequence_index(store, refit),
            cu.SEED + 300 + year + 2, epochs,
        )
        su.torch.save(model.state_dict(), OUT / f"daily_hybrid_{year}.pt")
        residual = su.predict(model, xe, store, su.sequence_index(store, test))
        write_daily(test, year, "hybrid", residual, base, refit)


def build_original():
    import lightgbm as lgb
    from notebooks.build_intraday_1530_update import DAILY, DAILY_FEATURES

    historical = ROOT / "outputs/experiments/18_error_diagnosis_calibration/historical_2021"
    for year in YEARS:
        path = OUT / f"original_{year}.parquet"
        if path.is_file():
            continue
        if year == 2021:
            prep_path = historical / "preprocessor.json"
            settings_path = historical / "manifest.json"
            model_path = historical / "model.txt"
        else:
            prep_path = DAILY / f"models/preprocessor_{year}.json"
            settings_path = DAILY / f"models/reconstruction_{year}.json"
            model_path = DAILY / f"models/lgbm_residual_58_l2_{year}.txt"
        prep = json.loads(prep_path.read_text())
        settings = json.loads(settings_path.read_text())
        types = tuple(list(prep[key]) for key in ("continuous", "boolean", "categorical"))
        columns = list(dict.fromkeys(["date", "permno", "auction_volume", "median20_auction_volume", *sum(types, [])]))
        frame = cu.sql(f"SELECT {', '.join(columns)} FROM read_parquet('{cu.PANEL}') WHERE eligible_target AND year = {year}")
        matrix = apply_preprocessor(frame, prep, *types)[:, cu.feature_mask(prep, DAILY_FEATURES)]
        residual = lgb.Booster(model_file=str(model_path)).predict(matrix, num_threads=6)
        base = np.log1p(frame.median20_auction_volume.fillna(settings["fallback"]).to_numpy(float))
        forecast = np.maximum(np.expm1(np.clip(base + residual, *settings["clip_log"])), 0)
        save_predictions(frame[["date", "permno", "auction_volume"]].assign(forecast=forecast), path)
        print(f"{year}: reconstructed original daily anchor", flush=True)


def checkpoint_panel():
    from notebooks.build_intraday_1530_update import checkpoint_panel_query

    path = OUT / "checkpoint_panel.parquet"
    if not path.is_file():
        from notebooks.build_intraday_1530_update import MODEL_FEATURES

        build_original()
        files = str(OUT / "original_*.parquet")
        cu.con.execute(f"CREATE OR REPLACE TEMP VIEW daily_predictions AS SELECT date, permno, forecast AS daily_forecast FROM read_parquet('{files}')")
        query = checkpoint_panel_query(cu.con, cu.PANEL)
        fields = ["date", "permno", "year", "auction_volume", "log1p_auction_volume", "cp_available",
                  "is_early_close", "daily_forecast", *MODEL_FEATURES]
        projection = ", ".join(dict.fromkeys(fields))
        sample = cu.con.sql(f"SELECT {projection} FROM ({query})")
        staging_path = path.with_suffix(".partial")
        sample.write_parquet(str(staging_path), compression="zstd")
        staging_path.replace(path)

    cu.con.execute(f"CREATE OR REPLACE TEMP VIEW checkpoint_panel AS SELECT * FROM read_parquet('{path}')")
    coverage = cu.sql("""
        SELECT year, count(*) AS eligible_targets,
               count(*) FILTER (WHERE NOT is_early_close) AS regular_session_rows,
               count(*) FILTER (WHERE NOT is_early_close AND cp_available) AS valid_checkpoint_rows
        FROM checkpoint_panel GROUP BY year ORDER BY year
    """)
    published = pd.read_csv(OUT.parent / "19_intraday_1530/coverage_by_year.csv")
    np.testing.assert_array_equal(coverage, published[coverage.columns])


def available_anchors():
    anchor_expressions = {"original": "p.daily_forecast"}
    joins = []
    for stem in ("tree", "hybrid"):
        paths = [OUT / f"{stem}_{year}.parquet" for year in YEARS]
        if not all(path.is_file() for path in paths):
            break
        cu.con.read_parquet([str(path) for path in paths]).create_view(f"{stem}_forecasts", replace=True)
        totals = cu.con.sql(f"SELECT count(*), count(DISTINCT(date, permno)) FROM {stem}_forecasts").fetchone()
        assert totals[0] == totals[1], f"Duplicate keys in {stem} forecasts"
        missing = cu.con.sql(f"""
            SELECT count(*) FROM checkpoint_panel p FULL JOIN {stem}_forecasts f USING (date, permno)
            WHERE p.auction_volume IS DISTINCT FROM f.auction_volume
               OR f.forecast IS NULL OR NOT isfinite(f.forecast) OR f.forecast < 0
        """).fetchone()[0]
        assert missing == 0, f"{stem} has {missing} invalid forecasts or mismatched keys or labels"
        joins.append(f"JOIN {stem}_forecasts {stem} USING (date, permno)")
        if stem == "tree":
            anchor_expressions["tree"] = "tree.forecast"
        else:
            anchor_expressions["blend"] = "exp(0.5 * ln(1 + tree.forecast) + 0.5 * ln(1 + hybrid.forecast)) - 1"
    projections = ", ".join(f"{expression} AS {anchor}" for anchor, expression in anchor_expressions.items())
    cu.con.execute(f"""
        CREATE OR REPLACE TEMP VIEW comparison_panel AS
        SELECT p.*, {projections} FROM checkpoint_panel p {' '.join(joins)} WHERE NOT p.is_early_close
    """)
    return list(anchor_expressions)


def audit_daily_refits(anchors):
    from notebooks.modeling_utils import forecast_metrics

    published_tree = pd.read_csv(cu.EDA / "16a_lightgbm_improvements/candidate_metrics.csv").query("candidate == @TREE")
    published_blend = pd.read_csv(HISTORY / "candidate_metrics.csv").query("candidate == 'blend_hybrid_lgbm'")
    rows = []
    for anchor, published in (("tree", published_tree), ("blend", published_blend)):
        if anchor not in anchors:
            continue
        for entry in published.itertuples(index=False):
            year = entry.test_year
            if anchor == "tree":
                query = "SELECT auction_volume, forecast FROM tree_forecasts"
            else:
                query = """SELECT t.auction_volume,
                      exp(0.5 * ln(1 + t.forecast) + 0.5 * ln(1 + h.forecast)) - 1 AS forecast
                      FROM tree_forecasts t JOIN hybrid_forecasts h USING (date, permno)"""
            frame = cu.sql(f"SELECT * FROM ({query} WHERE year(date) = {year})")
            actual = forecast_metrics(frame.auction_volume, frame.forecast)
            assert actual["observations"] == entry.observations
            rows.append({
                "year": year, "anchor": anchor, "observations": actual["observations"],
                "published_log_rmse": entry.rmse_log, "refit_log_rmse": actual["rmse_log"],
                "published_wape": entry.wape, "refit_wape": actual["wape"],
            })
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "daily_refit_audit.csv", index=False)


def load_update_rows(start, end, columns, rows=None, seed=None):
    sample = f"USING SAMPLE {rows} ROWS (reservoir, {seed})" if rows else ""
    checkpoint = "AND cp_available" if rows else ""
    return cu.sql(f"""
        SELECT * FROM (
            SELECT {', '.join(columns)} FROM comparison_panel
            WHERE date >= DATE '{start}' AND date < DATE '{end}' {checkpoint}
            ORDER BY date, permno
        ) {sample} ORDER BY date, permno
    """)


def ridge_prediction(frame, anchor, matrix, coefficient, bounds):
    from notebooks.modeling_utils import predict_ridge

    daily = frame[anchor].to_numpy(float)
    log_update = np.clip(np.log1p(daily) + predict_ridge(matrix, coefficient), *bounds)
    return np.where(frame.cp_available, np.maximum(np.expm1(log_update), 0), daily)


def fit_update(frame, validation, anchor, types):
    from notebooks.modeling_utils import fit_ridge

    for rows in (frame, validation):
        rows["log_daily_forecast"] = np.log1p(rows[anchor])
    train_matrix, prep = fit_preprocessor(frame, *types)
    validation_matrix = apply_preprocessor(validation, prep, *types)
    target = frame.log1p_auction_volume - frame.log_daily_forecast
    bounds = np.quantile(frame.log1p_auction_volume, [0.001, 0.999])
    trials = []
    for strength in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        beta = fit_ridge(train_matrix, target, strength)
        forecast = ridge_prediction(validation, anchor, validation_matrix, beta, bounds)
        error = np.log1p(forecast) - validation.log1p_auction_volume
        trials.append({"strength": strength, "validation_log_rmse": np.sqrt(np.mean(error**2))})
    return min(trials, key=lambda trial: trial["validation_log_rmse"])["strength"], trials


def published_reference(test, year, template, types):
    source = OUT.parent / "19_intraday_1530/models"
    reconstruction = json.loads((source / f"reconstruction_{year}.json").read_text())
    coefficient = np.load(source / f"ridge_update_{year}.npy")
    matrix = apply_preprocessor(test, template, *types)
    return ridge_prediction(test, "original", matrix, coefficient, reconstruction["clip_log"])


def score_pairs(anchors):
    from notebooks.build_intraday_1530_update import MODEL_FEATURES, SEED
    from notebooks.modeling_utils import fit_ridge, forecast_metrics

    source = OUT.parent / "19_intraday_1530"
    published = pd.read_csv(source / "annual_metrics.csv").query("sample == 'all_regular_sessions'")
    metrics, trials, sample_manifest = [], [], []
    all_predictions = []
    columns = ["date", "permno", "auction_volume", "log1p_auction_volume", "cp_available",
               *[f for f in MODEL_FEATURES if f != "log_daily_forecast"], *anchors]
    cu.con.execute("SET threads=1")
    for year in range(2022, 2026):
        start = "2021-10-01" if year == 2022 else f"{year-1}-01-01"
        end = f"{year}-01-01"
        train = load_update_rows("2021-01-01", start, columns, 250000, SEED + year) if len(anchors) > 1 else None
        validation = load_update_rows(start, end, columns, 120000, SEED + 100 + year) if train is not None else None
        refit = load_update_rows("2021-01-01", end, columns, 350000, SEED + 200 + year) if train is not None else None
        test = load_update_rows(end, f"{year+1}-01-01", columns)
        if train is not None:
            assert train.date.max() < validation.date.min() <= validation.date.max() < test.date.min()
            assert refit.date.max() < test.date.min()
            for name, frame in (("tune", train), ("validation", validation), ("refit", refit)):
                keys = pd.util.hash_pandas_object(frame[["date", "permno"]], index=False)
                sample_manifest.append({
                    "year": year, "sample": name, "rows": len(frame),
                    "first_date": str(frame.date.min()), "last_date": str(frame.date.max()),
                    "key_hash": hashlib.sha256(keys.to_numpy().tobytes()).hexdigest(),
                })
        prep = json.loads((source / f"models/preprocessor_{year}.json").read_text())
        types = tuple(list(prep[key]) for key in ("continuous", "boolean", "categorical"))
        assert set(sum(types, [])) == set(MODEL_FEATURES)
        output = test[["date", "permno", "auction_volume", "cp_available"]].copy()
        for anchor in anchors:
            test["log_daily_forecast"] = np.log1p(test[anchor])
            if anchor == "original":
                update = published_reference(test, year, prep, types)
            if train is not None:
                strength, grid = fit_update(train, validation, anchor, types)
                trials.extend({"year": year, "anchor": anchor, "selected": trial["strength"] == strength, **trial} for trial in grid)
                refit["log_daily_forecast"] = np.log1p(refit[anchor])
                matrix, fitted_prep = fit_preprocessor(refit, *types)
                target = refit.log1p_auction_volume - refit.log_daily_forecast
                coefficient = fit_ridge(matrix, target, strength)
                bounds = np.quantile(refit.log1p_auction_volume, [0.001, 0.999])
                refit_update = ridge_prediction(test, anchor, apply_preprocessor(test, fitted_prep, *types), coefficient, bounds)
                if anchor != "original":
                    update = refit_update
                settings = {"preprocessor": fitted_prep, "coefficient": coefficient.tolist(),
                            "clip_log": bounds.tolist(), "strength": strength, "latest_refit_date": str(refit.date.max())}
                (OUT / f"{anchor}_ridge_{year}.json").write_text(json.dumps(settings, indent=2))
            assert np.isfinite(update).all() and (update >= 0).all()
            np.testing.assert_array_equal(update[~test.cp_available], test.loc[~test.cp_available, anchor])
            stages = {"daily": test[anchor], "ridge_1530": update}
            if anchor == "original" and train is not None:
                stages["ridge_1530_matched_refit"] = refit_update
            for stage, forecast in stages.items():
                output[f"{anchor}_{stage}"] = forecast
                result = forecast_metrics(test.auction_volume, forecast)
                if anchor == "original" and stage != "ridge_1530_matched_refit":
                    method = "fixed_daily" if stage == "daily" else "ridge_update"
                    reference = published.query("test_year == @year and candidate == @method").iloc[0]
                    keys = ["observations", "rmse_log", "wape", "aggregate_bias"]
                    np.testing.assert_allclose([result[key] for key in keys], reference[keys].to_numpy(float), rtol=1e-8, atol=1e-9)
                metrics.append({"year": year, "anchor": anchor, "stage": stage, **result})
        all_predictions.append(output)
        print(f"{year}: verified and scored {len(test):,} common stock-days for {', '.join(anchors)}", flush=True)
    cu.con.execute("SET threads=4")
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "annual_metrics.csv", index=False)
    combined = metrics.assign(period=np.where(metrics.year == 2025, "2025", "2022–24"))
    summary = combined.groupby(["period", "anchor", "stage"], sort=False)[["rmse_log", "wape", "aggregate_bias"]].mean()
    summary.to_csv(OUT / "period_metrics.csv")
    print(summary.to_string(), flush=True)
    if trials:
        pd.DataFrame(trials).to_csv(OUT / "ridge_validation.csv", index=False)
        pd.DataFrame(sample_manifest).to_csv(OUT / "ridge_sample_manifest.csv", index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    save_predictions(predictions, OUT / "matched_predictions.parquet")
    paired_intervals(predictions, anchors)
    (OUT / "verification_summary.json").write_text(json.dumps({
        "anchors": anchors, "years": [2022, 2023, 2024, 2025],
        "evaluation_stock_days": len(predictions),
        "checks": [
            "common date/PERMNO keys and exact auction labels",
            "unique keys and finite nonnegative daily forecasts",
            "original annual daily and ridge scores reproduced",
            "original checkpoint coverage reproduced",
            "chronological update tuning and refitting",
            "exact daily fallback for unavailable checkpoints",
            "same historical samples across refitted ridge anchors",
            "finite nonnegative ridge forecasts",
        ],
        "status": "passed",
    }, indent=2))


def paired_intervals(frame, anchors):
    from notebooks.evaluate_frozen_intraday import circular_weights, statistics

    contrasts = [(f"{a}_daily", f"{a}_ridge_1530") for a in anchors]
    contrasts += [("original_ridge_1530", f"{a}_ridge_1530") for a in anchors[1:]]
    if len(anchors) > 1:
        contrasts += [("original_ridge_1530_matched_refit", f"{a}_ridge_1530") for a in anchors[1:]]
    if "blend" in anchors:
        contrasts.append(("tree_ridge_1530", "blend_ridge_1530"))
    actual = frame.auction_volume.to_numpy(float)
    frame = frame.assign(n=1, year=frame.date.dt.year)
    weights = {year: circular_weights(rows.date.nunique(), 20, 5000, 20261113 + year)
               for year, rows in frame.groupby("year")}
    rows = []
    for reference, candidate in contrasts:
        p, q = frame[reference].to_numpy(float), frame[candidate].to_numpy(float)
        totals = frame[["year", "date", "n", "auction_volume"]].assign(
            sse_p=(np.log1p(actual) - np.log1p(p))**2, sse_q=(np.log1p(actual) - np.log1p(q))**2,
            ae_p=np.abs(actual - p), ae_q=np.abs(actual - q), p=p, q=q,
        ).groupby(["year", "date"]).sum()
        for period, years in [("2022–24", (2022, 2023, 2024)), ("2025", (2025,))]:
            daily = [totals.loc[year].to_numpy() for year in years]
            point = np.mean([statistics(arr.sum(axis=0)) for arr in daily], axis=0)
            sampled = np.mean([statistics(weights[year] @ arr) for year, arr in zip(years, daily)], axis=0)
            difference = point[1] - point[0]
            intervals = np.quantile(sampled[:, 1] - sampled[:, 0], [0.025, 0.975], axis=0)
            for index, metric in enumerate(("rmse_log", "wape", "signed_bias", "absolute_bias")):
                rows.append({"period": period, "reference": reference, "candidate": candidate, "metric": metric,
                             "difference": difference[index], "lower": intervals[0, index], "upper": intervals[1, index]})
    pd.DataFrame(rows).to_csv(OUT / "paired_intervals.csv", index=False)


def compare():
    checkpoint_panel()
    anchors = available_anchors()
    print(f"Available complete daily anchors: {anchors}", flush=True)
    audit_daily_refits(anchors)
    score_pairs(anchors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tree", "hybrid", "compare"])
    parser.add_argument("--years", type=int, nargs="+", choices=list(YEARS), default=list(YEARS))
    args = parser.parse_args()
    if cu.SMOKE:
        parser.error("The comparison uses the full daily model training budgets.")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "tree":
        build_trees(args.years)
    elif args.stage == "hybrid":
        build_hybrids(args.years)
    else:
        compare()


if __name__ == "__main__":
    main()
