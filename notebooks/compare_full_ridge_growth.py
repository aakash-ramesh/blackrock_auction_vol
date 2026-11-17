"""Fit a matched 58/99-feature by median-residual/log-growth ridge comparison."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd

from notebooks.compare_ridge_targets import (
    BASE, GRID, OUT as TARGET_RUN, ROOT, YEARS, digest, extra_metrics,
    fit_context, invert, ridge_grid, transform,
)
from notebooks.modeling_utils import apply_preprocessor, fit_preprocessor, predict_ridge


OUT = TARGET_RUN.parent / "22_full_feature_ridge_growth"
BENCHMARK = ROOT / "outputs/eda/09_global_finalists"
GROWTH = "lag1_log_growth"
TARGETS = (BASE, GROWTH)
CONTROL = "compact58_median20"
CHALLENGER = "full99_log_growth"
FROZEN = "frozen_compact58_median20"
MODELS = {
    (58, BASE): CONTROL,
    (58, GROWTH): "compact58_log_growth",
    (99, BASE): "full99_median20",
    (99, GROWTH): CHALLENGER,
}
NAMES = {
    CONTROL: "58 features · 20-day median log residual",
    "compact58_log_growth": "58 features · one-day log growth",
    "full99_median20": "99 features · 20-day median log residual",
    CHALLENGER: "99 features · one-day log growth",
    FROZEN: "Frozen Notebook 09 ridge",
}


def design_matrices(train, score, types, compact_features):
    x, prep = fit_preprocessor(train, *types)
    z = apply_preprocessor(score, prep, *types)
    mask = np.isin(prep["output_sources"], compact_features)
    return x, z, prep, {58: mask, 99: np.ones(len(mask), dtype=bool)}


def target_matrix(frame, fallback):
    result = np.column_stack([transform(frame, target, fallback) for target in TARGETS])
    assert np.isfinite(result).all()
    return result


def paired_intervals(blocks):
    """Resample the same calendar months for both models, within each year."""
    rng = np.random.default_rng(20261022)
    year_draws = []
    for _, group in blocks.groupby("year"):
        sample = group[["n", "shares", "control_squared", "challenger_squared",
                        "control_absolute", "challenger_absolute"]].to_numpy()
        totals = sample[rng.integers(len(sample), size=(4000, len(sample)))].sum(axis=1)
        n, shares, control_sq, challenger_sq, control_abs, challenger_abs = totals.T
        year_draws.append(np.column_stack([
            np.sqrt(challenger_sq / n) - np.sqrt(control_sq / n),
            (challenger_abs - control_abs) / shares,
        ]))
    draws = np.mean(year_draws, axis=0)
    return pd.DataFrame({
        "metric": ["rmse_log", "wape"],
        "lower_95": np.quantile(draws, .025, axis=0),
        "upper_95": np.quantile(draws, .975, axis=0),
    })


def run():
    for folder in [OUT, OUT / "models", OUT / "predictions"]:
        folder.mkdir(exist_ok=True)
    config = json.loads((BENCHMARK / "frozen_global_benchmark.json").read_text())
    compact = config["features"]
    catalog = pd.read_csv(ROOT / "outputs/eda/05_feature_engineering/approved_enriched_daily_features.csv")
    features = catalog.variable.tolist()
    schema = json.loads((TARGET_RUN / "models/preprocessor_2022.json").read_text())
    types = [list(schema[key]) for key in ["continuous", "boolean", "categorical"]]
    assert len(set(compact)) == 58 and len(set(features)) == 99
    assert set(compact) <= set(features) == set(sum(types, []))
    provenance = pd.read_csv(TARGET_RUN / "sample_fingerprints.csv")
    panel = TARGET_RUN / "daily_panel.parquet"
    assert digest(panel) == json.loads((TARGET_RUN / "experiment_plan.json").read_text())["panel_sha256"]
    previous = pd.read_csv(TARGET_RUN / "annual_metrics.csv")
    previous = previous.query("selection == 'log_rmse' and bounds_policy == 'common_bounds'")
    columns = list(dict.fromkeys(["date", "permno", "auction_volume"] + features))
    projection = ", ".join(f'"{column}"' for column in columns)
    annual, validation, bootstrap, checks = [], [], [], []

    def check(name, condition):
        assert condition, name
        checks.append({"check": name, "passed": True})

    with duckdb.connect() as con:
        con.execute("SET threads=4")

        def read_sample(year, role, first, last):
            row = provenance.loc[provenance.year.eq(year) & provenance.role.eq(role)].iloc[0]
            path = ROOT / row.path
            check(f"{year} {role} matches saved sample fingerprint", digest(path) == row.sha256)
            frame = con.execute("SELECT * FROM read_parquet(?) ORDER BY date, permno", [str(path)]).df()
            check(f"{year} {role} chronology and unique keys", len(frame) == row.rows
                  and frame.date.dt.year.between(first, last).all()
                  and not frame.duplicated(["date", "permno"]).any())
            return frame

        for year in YEARS:
            print(f"{year} tuning four ridge specifications", flush=True)
            train = read_sample(year, "tune", year-6, year-2)
            val = read_sample(year, "validation", year-1, year-1)
            x, xv, _, masks = design_matrices(train, val, types, compact)
            fallback, bounds = fit_context(train)
            z = target_matrix(train, fallback)
            yv = np.log1p(val.auction_volume.to_numpy(dtype=float))
            selected = {}
            for size, mask in masks.items():
                coeffs = ridge_grid(x[:, mask], z)
                val_x = xv[:, mask].astype(float)
                scores = np.empty((len(GRID), len(TARGETS)))
                for k, (alpha, coef) in enumerate(zip(GRID, coeffs)):
                    components = coef[0] + val_x @ coef[1:]
                    for j, target in enumerate(TARGETS):
                        pred = invert(components[:, j], val, target, fallback, bounds)
                        scores[k, j] = np.sqrt(np.mean((yv - np.log1p(pred))**2))
                        validation.append({"year": year, "model": MODELS[size, target],
                                           "penalty": alpha, "validation_log_rmse": scores[k, j]})
                for j, target in enumerate(TARGETS):
                    selected[size, target] = int(scores[:, j].argmin())
            del train, val, x, xv, val_x, z

            print(f"{year} refitting and scoring the common stock-days", flush=True)
            refit = read_sample(year, "refit", year-5, year-1)
            test = con.execute(f"""
                SELECT {projection} FROM read_parquet(?) WHERE eligible_target AND year = ?
                ORDER BY date, permno
            """, [str(panel), year]).df()
            x, xe, prep, masks = design_matrices(refit, test, types, compact)
            fallback, bounds = fit_context(refit)
            z = target_matrix(refit, fallback)
            output = test[["date", "permno", "auction_volume"]].copy()
            y = test.auction_volume.to_numpy(dtype=float)
            checkpoint = {"fallbacks": fallback, "bounds": bounds, "models": {}}
            for size, mask in masks.items():
                coeffs = ridge_grid(x[:, mask], z)
                score_x = xe[:, mask].astype(float)
                for j, target in enumerate(TARGETS):
                    key = MODELS[size, target]
                    k = selected[size, target]
                    coef = coeffs[k][:, j]
                    pred = invert(coef[0] + score_x @ coef[1:], test, target, fallback, bounds)
                    check(f"{year} {key} finite valid share predictions",
                          np.isfinite(pred).all() and (pred >= 0).all())
                    output[key] = pred
                    stats = extra_metrics(y, pred)
                    if size == 58:
                        old = previous.loc[previous.year.eq(year) & previous.target.eq(target)].iloc[0]
                        check(f"{year} {key} reproduces Notebook 21 metrics",
                              all(np.isclose(v, old[name], rtol=1e-8, atol=1e-9) for name, v in stats.items()))
                    annual.append({"year": year, "model": key, "sources": size,
                                   "encoded_columns": int(mask.sum()), "penalty": GRID[k], **stats})
                    np.save(OUT / f"models/{key}_{year}.npy", coef)
                    checkpoint["models"][key] = {
                        "target": target, "penalty": GRID[k],
                        "columns": np.flatnonzero(mask).tolist(),
                    }
                del score_x
            del refit, x, xe, z
            checkpoint["preprocessor"] = prep
            checkpoint_file = OUT / f"models/reconstruction_{year}.json"
            checkpoint_file.write_text(json.dumps(checkpoint, indent=2))
            loaded = json.loads(checkpoint_file.read_text())
            subset = test.iloc[::max(1, len(test)//200)]
            xx = apply_preprocessor(subset, loaded["preprocessor"], *types)
            for key, metadata in loaded["models"].items():
                coef = np.load(OUT / f"models/{key}_{year}.npy")
                restored = invert(predict_ridge(xx[:, metadata["columns"]], coef), subset,
                                  metadata["target"], loaded["fallbacks"], loaded["bounds"])
                check(f"{year} {key} saved model reconstructs shares",
                      np.allclose(restored, output.loc[subset.index, key], rtol=1e-10, atol=1e-6))

            path = BENCHMARK / f"predictions/global_finalist_predictions_{year}.parquet"
            champion = config["champion"]
            frozen = con.execute(f"""
                SELECT date, permno, auction_volume, "{champion}" AS prediction
                FROM read_parquet(?) ORDER BY date, permno
            """, [str(path)]).df()
            check(f"{year} frozen benchmark matches evaluation keys and labels",
                  len(frozen) == len(output) and all(np.array_equal(frozen[col], output[col])
                                                   for col in ["date", "permno", "auction_volume"]))
            check(f"{year} frozen benchmark predictions are valid",
                  np.isfinite(frozen.prediction).all() and frozen.prediction.ge(0).all())
            output[FROZEN] = frozen.prediction.to_numpy()
            annual.append({"year": year, "model": FROZEN, "sources": 58,
                           **extra_metrics(y, output[FROZEN])})

            control, challenger = output[CONTROL].to_numpy(), output[CHALLENGER].to_numpy()
            monthly = pd.DataFrame({
                "month": output.date.dt.to_period("M"),
                "n": 1, "shares": y,
                "control_squared": (np.log1p(y) - np.log1p(control))**2,
                "challenger_squared": (np.log1p(y) - np.log1p(challenger))**2,
                "control_absolute": np.abs(y-control), "challenger_absolute": np.abs(y-challenger),
            }).groupby("month", as_index=False).sum()
            bootstrap.append(monthly.assign(year=year))
            pred_path = OUT / f"predictions/share_forecasts_{year}.parquet"
            con.register("predictions", output)
            con.execute(f"COPY predictions TO '{pred_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            con.unregister("predictions")
            check(f"{year} saved forecasts cover all unique stock-days",
                  con.execute("SELECT count(*), count(DISTINCT (date, permno)) FROM read_parquet(?)",
                              [str(pred_path)]).fetchone() == (len(test), len(test)))
            del test, output, frozen

    results = pd.DataFrame(annual)
    results.to_csv(OUT / "annual_metrics.csv", index=False)
    pd.DataFrame(validation).to_csv(OUT / "validation_grid.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "verification_checks.csv", index=False)
    provenance.to_csv(OUT / "sample_fingerprints.csv", index=False)
    blocks = pd.concat(bootstrap, ignore_index=True)
    intervals = paired_intervals(blocks[blocks.year.isin(YEARS[:-1])])
    intervals.to_csv(OUT / "paired_development_intervals.csv", index=False)
    groups = catalog.assign(added=~catalog.variable.isin(compact)).groupby("group").added.agg(["size", "sum"])
    groups.to_csv(OUT / "feature_groups.csv", index_label="group")
    report(results, intervals, len(checks))


def report(results, intervals, checks):
    metrics = ["rmse_log", "wape", "mae_raw", "rmse_raw", "r2_raw", "aggregate_bias"]
    dev = results[results.year.isin(YEARS[:-1])].groupby("model")[metrics].mean().reindex(NAMES)
    dev.to_csv(OUT / "development_metrics.csv")
    confirm = results[results.year.eq(YEARS[-1])].set_index("model").reindex(NAMES)
    tables = []
    for data in [dev, confirm]:
        rows = [
            "| Ridge specification | Log RMSE | WAPE | MAE shares | RMSE shares | Share R² | Bias |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key, row in data.iterrows():
            rows.append(f"| {NAMES[key]} | {row.rmse_log:.5f} | {row.wape:.2%} | "
                        f"{row.mae_raw:,.0f} | {row.rmse_raw:,.0f} | {row.r2_raw:.3f} | {row.aggregate_bias:+.2%} |")
        tables.append("\n".join(rows))
    difference = dev.loc[CHALLENGER] - dev.loc[CONTROL]
    ci = intervals.set_index("metric")
    log_interval = ci.loc["rmse_log"]
    wape_interval = 100 * ci.loc["wape"]
    observations = int(results.loc[results.model.eq(CONTROL), "observations"].sum())
    content = f"""# Full-feature ridge with one-day log growth

For stock-day auction shares `A_t`, the target is `log1p(A_t) - log1p(B_t)`, where `B_t` is the preceding stock row's auction shares. An unavailable or nonpositive reference uses the fitting sample's median auction shares. Ridge estimates the growth component, and `expm1(log1p(B_t) + predicted_growth)` recovers shares before applying the common training-based forecast bounds.

The fit solves the penalized normal equations in NumPy with an unpenalized intercept and uniform stock-day weights. All four fresh fits use the exact historical tuning, validation, and refit rows from Notebook 21. Each fold fits preprocessing using its training rows and picks the ridge penalty by validation log RMSE after reconstruction. The 99 approved daily source features include the compact 58 plus 41 from returns and price, market state, sector state, security and corporate actions, and size and history. The encoded column count can vary by fold.

## Development results (2022–2024 equal-year means)

{tables[0]}

The 99-feature growth model minus the matched current 58-feature fit was {difference.rmse_log:+.5f} log RMSE and {100*difference.wape:+.2f} percentage points of WAPE. Paired calendar-month resampling within years gave approximate 95% intervals of [{log_interval.lower_95:+.5f}, {log_interval.upper_95:+.5f}] for log RMSE and [{wape_interval.lower_95:+.2f}, {wape_interval.upper_95:+.2f}] points for WAPE. The bootstrap conditions on the fitted forecasts.

## 2025 confirmation

{tables[1]}

The project has previously inspected 2025. The frozen Notebook 09 forecast is included as a historical reference on identical stock-days. The fresh 58-feature median-residual fit is the control for isolating changes to the target and features.

All {checks} verification checks passed across {observations:,} evaluation stock-days per specification. Checks cover historical sample identity, fold chronology, matching compact-model metrics, reconstruction from saved coefficients, and alignment with the frozen benchmark.

Run `.venv/bin/python -u notebooks/compare_full_ridge_growth.py` with Notebook 21's panel and sample files available. The [annual metrics](annual_metrics.csv), [validation scores](validation_grid.csv), saved coefficients and share forecasts are in this directory.
"""
    (OUT / "README.md").write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    run()
