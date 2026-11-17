"""Ridge-only target comparison, with every metric evaluated in auction shares.

Run: .venv/bin/python -u notebooks/compare_ridge_targets.py
No trees are imported or fitted. Historical research outputs are left intact.
"""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/21_ridge_target_definitions"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
sys.path.insert(0, str(ROOT))

import duckdb
import nbformat
import numpy as np
import pandas as pd
from notebooks.modeling_utils import fit_preprocessor, apply_preprocessor, forecast_metrics, fit_ridge

YEARS = [2022, 2023, 2024, 2025]
GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1e3, 1e4, 1e5, 1e6]
SELECTIONS = {"log_rmse": "rmse_log", "wape": "wape", "share_rmse": "rmse_raw"}
TARGETS = {
    "raw_volume": ("raw", None, "A"),
    "sqrt_volume": ("sqrt", None, "sqrt(A)"),
    "log_volume": ("log", None, "log1p(A)"),
    "lag1_log_growth": ("log_residual", "lag1_auction_volume", "log1p(A) - log1p(lag1 auction)"),
    "lag1_pct_growth": ("growth", "lag1_auction_volume", "A / lag1 auction - 1 (simple growth rate)"),
    "median20_log_residual": ("log_residual", "median20_auction_volume", "log1p(A) - log1p(median20 auction); current target"),
    "median60_log_residual": ("log_residual", "median60_auction_volume", "log1p(A) - log1p(median60 auction)"),
    "mean20_log_residual": ("log_residual", "avg20_auction_volume", "log1p(A) - log1p(mean20 auction)"),
    "mean60_log_residual": ("log_residual", "avg60_auction_volume", "log1p(A) - log1p(mean60 auction)"),
    "median20_ratio": ("ratio", "median20_auction_volume", "A / median20 auction"),
    "adv20_ratio": ("ratio", "avg20_daily_volume", "A / mean20 daily volume"),
    "adv20_log_residual": ("log_residual", "avg20_daily_volume", "log1p(A) - log1p(mean20 daily volume)"),
    "median20_share_residual": ("raw_residual", "median20_auction_volume", "A - median20 auction"),
}
BASE = "median20_log_residual"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_panel(con):
    """Reuse the original point-in-time SQL, without executing plotting cells."""
    panel = OUT / "daily_panel.parquet"
    if not panel.exists():
        data = ROOT / "blackrock_data"
        paths = {"CRSP": "stock_daily/crsp_daily/year=*/crsp_daily_*.parquet",
                 "AUCTION": "stock_daily/auction_daily/year=*/auction_daily_*.parquet",
                 "MARKET": "market/market_daily.parquet", "SECTOR": "sector/sector_daily.parquet",
                 "EVENTS": "events/event_calendar.parquet", "INDUSTRY": "reference/industry_history.parquet",
                 "ANOMALIES": "quality/auction_source_anomalies.parquet"}
        ns = {k: str(data / v) for k, v in paths.items()}
        nb = nbformat.read(ROOT / "notebooks/05_feature_engineering_enriched_daily_panel.ipynb", as_version=4)
        source = next(c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("panel_sql ="))
        exec(source.split("escaped_path =")[0], ns)
        con.execute(f"COPY ({ns['panel_sql']}) TO '{panel}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    return panel


def reference(frame, column, fallbacks):
    if column is None:
        return np.zeros(len(frame))
    values = frame[column].to_numpy(dtype=float)
    return np.where(np.isfinite(values) & (values > 0), values, fallbacks[column])


def transform(frame, name, fallbacks):
    kind, column, _ = TARGETS[name]
    a = frame.auction_volume.to_numpy(dtype=float)
    b = reference(frame, column, fallbacks)
    if kind == "raw": return a
    if kind == "sqrt": return np.sqrt(a)
    if kind == "log": return np.log1p(a)
    if kind == "log_residual": return np.log1p(a) - np.log1p(b)
    if kind == "ratio": return a / b
    if kind == "growth": return a / b - 1
    return a - b


def invert(component, frame, name, fallbacks, bounds=None):
    kind, column, _ = TARGETS[name]
    b = reference(frame, column, fallbacks)
    if kind in ("log", "log_residual"):
        logp = component + (np.log1p(b) if column else 0)
        # Numerical guard only; empirical clipping is separately recorded.
        raw = np.expm1(np.clip(logp, -700, 700))
    elif kind == "sqrt":
        raw = np.maximum(component, 0) ** 2
    elif kind == "ratio": raw = component * b
    elif kind == "growth": raw = (1 + component) * b
    elif kind == "raw_residual": raw = component + b
    else: raw = component
    raw = np.maximum(raw, 0)
    return np.clip(raw, *bounds) if bounds is not None else raw


def extra_metrics(y, p):
    metrics = forecast_metrics(y, p)
    residual = y - p
    metrics["rmse_raw"] = float(np.sqrt(np.mean(residual**2)))
    metrics["r2_raw"] = float(1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2))
    ly, lp = np.log1p(y), np.log1p(p)
    metrics["r2_log"] = float(1 - np.sum((ly-lp)**2) / np.sum((ly-np.mean(ly))**2))
    metrics["pearson_raw"] = float(np.corrcoef(y, p)[0, 1])
    return metrics


def fit_context(frame):
    auction_fallback = float(frame.auction_volume.median())
    fallbacks = {col: auction_fallback for _, col, _ in TARGETS.values() if col}
    positive = frame.avg20_daily_volume[frame.avg20_daily_volume > 0]
    fallbacks["avg20_daily_volume"] = float(positive.median())
    bounds = np.quantile(frame.auction_volume, [.001, .999]).tolist()
    return fallbacks, bounds


def matrices(train, score, types, features):
    x, prep = fit_preprocessor(train, *types)
    z = apply_preprocessor(score, prep, *types)
    mask = np.array([source in set(features) for source in prep["output_sources"]])
    return x[:, mask], z[:, mask], prep, mask


def ridge_grid(x, targets):
    """Same unpenalized intercept and n-normalized penalty as fit_ridge.

    Reuse the Gram matrix across targets and penalties; columns are separate fits.
    """
    design = np.column_stack([np.ones(len(x)), x]).astype(float)
    gram = design.T @ design / len(design)
    rhs = design.T @ targets / len(design)
    eye = np.eye(gram.shape[0]); eye[0, 0] = 0
    return [np.linalg.solve(gram + alpha*eye, rhs) for alpha in GRID]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["models", "predictions", "samples"]: (OUT / sub).mkdir(exist_ok=True)
    con = duckdb.connect(); con.execute("SET threads=4"); con.execute("SET memory_limit='4GB'")
    panel = build_panel(con)
    catalog = pd.read_csv(ROOT / "outputs/eda/05_feature_engineering/approved_enriched_daily_features.csv")
    features = json.loads((ROOT / "outputs/eda/09_global_finalists/frozen_global_benchmark.json").read_text())["features"]
    oldprep = json.loads((ROOT / "outputs/reproduction/03_compact_nonlinear/models/preprocessor_2022.json").read_text())
    types = [list(oldprep[k]) for k in ["continuous", "boolean", "categorical"]]
    assert set(sum(types, [])) == set(catalog.variable)
    columns = list(dict.fromkeys(["date", "permno", "auction_volume", "log1p_auction_volume", "prior_auction_observations",
                                 "quarterly_options_expiry", "is_last_trading_day_month"] + catalog.variable.tolist()))
    projection = ",".join(f'"{c}"' for c in columns)
    plan = {"targets": {k: v[2] for k, v in TARGETS.items()}, "ridge_grid": GRID,
            "features": features, "development_years": YEARS[:3], "confirmation_year": 2025,
            "confirmation_previously_inspected": True, "selection_metrics": SELECTIONS,
            "samples": "Fresh deterministic single-thread DuckDB reservoir draws; shared by every target. Not the deleted historical samples.",
            "tune_rows": 250000, "validation_rows": 120000, "refit_rows": 350000,
            "chronology": "Tune on five years ending before prior-year validation; refit on five years before evaluation year.",
            "preprocessing": "Fit historical 99-source preprocessor, then mask the same compact 58 sources.",
            "bounds": "Historical refit auction share quantiles .001/.999, common across targets. Also report nonnegative-only sensitivity, without re-tuning.",
            "fallbacks": "Missing/nonpositive auction reference: fitting sample median auction. Missing/nonpositive ADV: median positive fitting ADV.",
            "label_policy": "Unchanged Notebook 05 eligible_target; source anomaly excluded; missing labels not filled.",
            "script_sha256": digest(__file__), "panel_sha256": digest(panel)}
    (OUT / "experiment_plan.json").write_text(json.dumps(plan, indent=2))
    (OUT / "environment.json").write_text(json.dumps({p: importlib.metadata.version(p) for p in ["numpy", "pandas", "duckdb", "matplotlib", "nbformat"]}, indent=2))
    source_files = sorted((ROOT / "blackrock_data").rglob("*.parquet"))
    pd.DataFrame([{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in source_files]).to_csv(OUT / "source_fingerprints.csv", index=False)
    names = list(TARGETS)
    annual, validations, chosen, checks, samples, missingness = [], [], [], [], [], []

    def check(name, passed, **detail):
        checks.append({"check": name, "passed": bool(passed), **detail})
        assert passed, name

    counts = con.sql(f"SELECT count(*),count(*) FILTER(WHERE eligible_target),count(*) FILTER(WHERE eligible_target AND date=DATE '2019-08-12') FROM read_parquet('{panel}')").fetchone()
    check("panel_counts_and_anomaly", counts == (3594404, 3499401, 0), observed=str(counts))

    def load(year, role, start, end, n, seed):
        # A fixed input order and one sampling thread make fresh draws repeatable.
        con.execute("SET threads=1")
        clause = f"USING SAMPLE {n} ROWS (reservoir, {seed})" if n else ""
        frame = con.sql(f"SELECT * FROM (SELECT {projection} FROM read_parquet('{panel}') WHERE eligible_target AND date>=DATE '{start}-01-01' AND date<DATE '{end}-01-01' ORDER BY date,permno) {clause} ORDER BY date,permno").df()
        con.execute("SET threads=4")
        check(f"{year}_{role}_dates_unique", frame.date.min().year >= start and frame.date.max().year < end and not frame.duplicated(["date", "permno"]).any())
        if n:
            path = OUT / f"samples/{year}_{role}.parquet"
            con.register("sample_rows", frame)
            con.execute(f"COPY sample_rows TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            con.unregister("sample_rows")
            samples.append({"year": year, "role": role, "rows": len(frame), "path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        return frame

    for year in YEARS:
        print(f"{year}: historical tuning for {len(names)} ridge targets", flush=True)
        train = load(year, "tune", year-6, year-1, 250000, 20261018+year)
        val = load(year, "validation", year-1, year, 120000, 20261118+year)
        xt, xv, prep, mask = matrices(train, val, types, features)
        fallback, bounds = fit_context(train)
        z = np.column_stack([transform(train, name, fallback) for name in names])
        check(f"{year}_finite_transformed_targets", np.isfinite(z).all())
        coefs = ridge_grid(xt, z)
        # Verify optimization against the existing independent single-target helper.
        expected = fit_ridge(xt, z[:, names.index(BASE)], 1e-3)
        check(f"{year}_ridge_matches_existing_helper", np.allclose(expected, coefs[4][:, names.index(BASE)], atol=1e-7, rtol=1e-6))
        yv = val.auction_volume.to_numpy(dtype=float)
        for alpha, coef in zip(GRID, coefs):
            components = coef[0] + xv.astype(float) @ coef[1:]
            for j, name in enumerate(names):
                p = invert(components[:, j], val, name, fallback, bounds)
                validations.append({"year": year, "target": name, "alpha": alpha,
                                    "rmse_log": float(np.sqrt(np.mean((np.log1p(yv)-np.log1p(p))**2))),
                                    "wape": float(np.abs(yv-p).sum()/yv.sum()),
                                    "rmse_raw": float(np.sqrt(np.mean((yv-p)**2)))})
        vf = pd.DataFrame(validations); vf = vf[vf.year == year]
        selected = {}
        for name in names:
            group = vf[vf.target == name]
            for selection, metric in SELECTIONS.items():
                best = group.sort_values([metric, "alpha"], kind="stable").iloc[0]
                selected[(name, selection)] = float(best.alpha)
                chosen.append({"year": year, "target": name, "selection": selection, "alpha": float(best.alpha), "validation_score": float(best[metric]), "grid_boundary": bool(best.alpha in [GRID[0], GRID[-1]])})
        del train, val, xt, xv, z, coefs, components
        gc.collect()

        print(f"{year}: refitting and evaluating on all eligible stock-days", flush=True)
        refit = load(year, "refit", year-5, year, 350000, 20261218+year)
        test = load(year, "test", year, year+1, None, 0)
        xr, xe, prep, mask = matrices(refit, test, types, features)
        fallback, bounds = fit_context(refit)
        z = np.column_stack([transform(refit, name, fallback) for name in names])
        coefs = ridge_grid(xr, z)
        output = test[["date", "permno", "auction_volume", "prior_auction_observations", "quarterly_options_expiry", "is_last_trading_day_month"]].copy()
        yf = test.auction_volume.to_numpy(dtype=float)
        tail = yf >= np.quantile(yf, .99)
        checkpoint = {"fallbacks": fallback, "bounds": bounds, "features": np.array(prep["output_names"])[mask].tolist(), "selected": {f"{n}__{s}": a for (n, s), a in selected.items()}}
        (OUT / f"models/reconstruction_{year}.json").write_text(json.dumps(checkpoint, indent=2))
        (OUT / f"models/preprocessor_{year}.json").write_text(json.dumps(prep, indent=2))
        for j, name in enumerate(names):
            col = TARGETS[name][1]
            missingness.append({"year": year, "target": name, "missing_test_reference": int((~np.isfinite(test[col].astype(float)) | (test[col] <= 0)).sum()) if col else 0})
            for selection in SELECTIONS:
                alpha = selected[(name, selection)]
                coef = coefs[GRID.index(alpha)][:, j]
                path = OUT / f"models/{name}__{selection}_{year}.npy"
                np.save(path, coef)
                component = coef[0] + xe.astype(float) @ coef[1:]
                uncapped = invert(component, test, name, fallback)
                p = np.clip(uncapped, *bounds)
                key = f"{name}__{selection}"
                output[key] = p
                check(f"{year}_{key}_valid", len(p) == len(test) and np.isfinite(p).all() and (p >= 0).all() and np.isfinite(uncapped).all())
                for policy, pred in [("common_bounds", p), ("nonnegative_only", uncapped)]:
                    annual.append({"year": year, "target": name, "selection": selection, "bounds_policy": policy,
                                   "tail_wape": float(np.abs(yf[tail]-pred[tail]).sum()/yf[tail].sum()),
                                   "bounded_fraction": float(np.mean((uncapped < bounds[0]) | (uncapped > bounds[1]))),
                                   **extra_metrics(yf, pred)})
        path = OUT / f"predictions/ridge_targets_{year}.parquet"
        con.register("predictions", output)
        con.execute(f"COPY predictions TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        con.unregister("predictions")
        saved = con.sql(f"SELECT * FROM read_parquet('{path}') ORDER BY date,permno").df()
        check(f"{year}_saved_keys_complete", saved[["date", "permno"]].equals(test[["date", "permno"]]) and saved.auction_volume.equals(test.auction_volume))
        # Reload every model and preprocessor and independently reconstruct saved rows.
        subset = test.iloc[::max(1, len(test)//200)].copy()
        loaded = json.loads((OUT / f"models/preprocessor_{year}.json").read_text())
        xx = apply_preprocessor(subset, loaded, *types)[:, mask].astype(float)
        for key in checkpoint["selected"]:
            name, _ = key.split("__")
            coef = np.load(OUT / f"models/{key}_{year}.npy")
            pred = invert(coef[0] + xx @ coef[1:], subset, name, fallback, bounds)
            check(f"{year}_{key}_reload", np.allclose(pred, saved[key].iloc[subset.index], atol=1e-6, rtol=1e-10))
        pd.DataFrame(annual).to_csv(OUT / "annual_metrics.csv", index=False)
        pd.DataFrame(validations).to_csv(OUT / "hyperparameter_validation.csv", index=False)
        pd.DataFrame(chosen).to_csv(OUT / "selected_penalties.csv", index=False)
        pd.DataFrame(samples).to_csv(OUT / "sample_fingerprints.csv", index=False)
        pd.DataFrame(missingness).to_csv(OUT / "reference_missingness.csv", index=False)
        pd.DataFrame(checks).to_csv(OUT / "verification_checks.csv", index=False)
        print(f"{year}: saved {len(test):,} common predictions per specification", flush=True)
        del refit, test, xr, xe, z, coefs, output, saved
        gc.collect()
    con.close()
    summarize()


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    annual = pd.read_csv(OUT / "annual_metrics.csv")
    metric_columns = ["mae_raw", "wape", "rmse_raw", "r2_raw", "rmse_log", "r2_log", "mae_log", "spearman", "pearson_raw", "aggregate_bias", "tail_wape", "bounded_fraction"]
    dev = annual[annual.year.isin(YEARS[:3])].groupby(["target", "selection", "bounds_policy"], as_index=False)[metric_columns].mean()
    dev.to_csv(OUT / "development_metrics.csv", index=False)
    main = dev[(dev.bounds_policy == "common_bounds") & (dev.selection == "log_rmse")].sort_values("rmse_log")
    winners = []
    for selection, metric in SELECTIONS.items():
        for policy in ["common_bounds", "nonnegative_only"]:
            part = dev[(dev.selection == selection) & (dev.bounds_policy == policy)]
            best = part.loc[part[metric].idxmin()]
            base = part[part.target == BASE].iloc[0]
            winners.append({"selection": selection, "bounds_policy": policy, "metric": metric, "winner": best.target,
                            "score": best[metric], "current_target_score": base[metric], "relative_improvement": 1-best[metric]/base[metric]})
    pd.DataFrame(winners).to_csv(OUT / "metric_winners.csv", index=False)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for ax, (selection, metric) in zip(axes, SELECTIONS.items()):
        part = dev[(dev.selection == selection) & (dev.bounds_policy == "common_bounds")].set_index("target").loc[main.target.iloc[::-1]].reset_index()
        values = part[metric] * (100 if metric == "wape" else 1)
        ax.barh(part.target, values, color=["#cb6b32" if t == BASE else "#347d9e" for t in part.target])
        ax.set_title({"log_rmse": "Log RMSE", "wape": "WAPE (%)", "share_rmse": "RMSE (shares)"}[selection])
        ax.grid(axis="x", alpha=.2)
        if selection != "log_rmse": ax.set_yticklabels([])
    fig.suptitle("Ridge targets after conversion to auction shares\n2022–2024 equal-year means; penalty tuned separately for each panel's metric")
    fig.tight_layout(); fig.savefig(OUT / "ridge_target_comparison.png", dpi=160); plt.close(fig)
    lines = ["| Target | Log RMSE ↓ | WAPE ↓ | MAE shares ↓ | RMSE shares ↓ | R² shares ↑ | Bias |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in main.itertuples():
        lines.append(f"| {r.target} | {r.rmse_log:.5f} | {r.wape:.2%} | {r.mae_raw:,.0f} | {r.rmse_raw:,.0f} | {r.r2_raw:.3f} | {r.aggregate_bias:+.2%} |")
    winlines = []
    for r in winners:
        if r["bounds_policy"] == "common_bounds":
            winlines.append(f"- **{r['metric']}**: `{r['winner']}`, score {r['score']:.6g}; {r['relative_improvement']:.2%} lower than current target when both tune penalties for that metric.")
    checks = pd.read_csv(OUT / "verification_checks.csv")
    totals = annual[(annual.selection == "log_rmse") & (annual.bounds_policy == "common_bounds") & (annual.target == BASE)].observations.sum()
    report = f"""# Ridge-only target-definition comparison

{len(TARGETS)} ridge target definitions use the same 58 source features and identical rows. All {len(checks)} verification checks passed across {totals:,} evaluation stock-days per specification. Scoring uses reconstructed auction shares.

## Metric-specific winners

{chr(10).join(winlines)}

These are descriptive development rankings, not a declaration that one target wins every metric. A zero improvement means the current target wins. RMSE and R² share a ranking within a fixed evaluation year; equal-year averages can differ in ranking. WAPE and MAE share a ranking within a year, but equal-year averaging can also change their rankings.

## Common tuning rule: select each penalty by historical log RMSE

All metrics below are calculated after inverse transformation to shares. Log RMSE is then computed by applying log1p to actual and predicted shares, so unlike transformed-target training errors it is comparable across candidates.

{chr(10).join(lines)}

## Design and interpretation

- Five-year historical tuning samples, the immediately preceding calendar year for validation, and five-year historical refits. Evaluation is the full eligible 2022–2025 universe. Development ranking uses 2022–2024 only; 2025 was previously inspected and is confirmation evidence.
- Tuning/refit/validation samples contain 250,000/350,000/120,000 rows. Deleted historical sample files were rebuilt as fresh deterministic single-thread reservoir draws; results are a new controlled comparison and need not reproduce earlier coefficients exactly. Sample files, hashes, source hashes, environments and models are saved.
- Ridge penalties are {GRID}, with an unpenalized intercept and the existing n-normalized ridge convention. Each target separately selects a penalty using historical log RMSE, WAPE, and share RMSE. Those are three declared selection strategies, not a post-hoc switch on evaluation data.
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
"""
    (OUT / "README.md").write_text(report)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(f"# 21 — Ridge-only target-definition comparison\n\n{len(TARGETS)} target definitions. All evaluation metrics follow conversion back to auction shares. See the saved README for design and limitations."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'blackrock_data').exists())\nout = root / 'outputs/experiments/21_ridge_target_definitions'"),
        nbformat.v4.new_code_cell("display(pd.read_csv(out / 'metric_winners.csv'))\ndisplay(Image(filename=str(out / 'ridge_target_comparison.png')))"),
        nbformat.v4.new_code_cell("development = pd.read_csv(out / 'development_metrics.csv')\ndisplay(development.query(\"selection == 'log_rmse' and bounds_policy == 'common_bounds'\").sort_values('rmse_log'))"),
        nbformat.v4.new_code_cell("annual = pd.read_csv(out / 'annual_metrics.csv')\ndisplay(annual.query(\"selection == 'log_rmse' and bounds_policy == 'common_bounds'\").pivot(index='target', columns='year', values=['rmse_log','wape','rmse_raw','r2_raw','aggregate_bias']))"),
        nbformat.v4.new_code_cell("display(development.query(\"bounds_policy == 'nonnegative_only'\"))\ndisplay(pd.read_csv(out / 'selected_penalties.csv'))"),
        nbformat.v4.new_code_cell("if (out / 'target_stability.csv').exists():\n    display(pd.read_csv(out / 'target_stability.csv'))\n    for figure in ('target_time_series.png', 'target_spread_and_tails.png'):\n        display(Image(filename=str(out / figure)))"),
        nbformat.v4.new_code_cell("display(Markdown((out / 'README.md').read_text()))")])
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(notebook, ROOT / "notebooks/21_ridge_target_definition_comparison.ipynb")
    print(report.split("## Design")[0], flush=True)


if __name__ == "__main__":
    summarize() if "--summarize" in sys.argv else run()
