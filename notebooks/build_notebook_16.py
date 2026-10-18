"""Build Notebook 16: nonlinear residual-model challengers."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "16_nonlinear_model_challengers.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r'''
# 16 — Nonlinear residual-model challengers

This notebook tests whether gradient-boosted trees add useful nonlinear structure beyond the frozen global ridge benchmark.

Experimental discipline:

- The benchmark from Notebook 09 remains frozen and is never refit here.
- Every challenger predicts the residual from the trailing 20-observation median auction volume.
- Five-year rolling training windows and immediately preceding validation years match the benchmark design.
- 2022–2024 are development years; 2025 is reported separately as confirmation evidence.
- Hyperparameters are selected using historical validation data only.
- A replacement must improve development log RMSE by at least 0.002, remain stable in every development year, and pass a paired date-block log-MAE test.
'''),
    code(r'''
from pathlib import Path
import gc
import json
import sys
import warnings

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 180)
pd.set_option("display.max_rows", 180)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (13, 7), "figure.dpi": 120})

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate blackrock_data.")
sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.modeling_utils import fit_preprocessor, apply_preprocessor, forecast_metrics

PANEL = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering" / "enriched_daily_panel.parquet"
CATALOG = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering" / "approved_enriched_daily_features.csv"
BENCHMARK_JSON = PROJECT_ROOT / "outputs" / "eda" / "09_global_finalists" / "frozen_global_benchmark.json"
BENCHMARK_PREDICTIONS = PROJECT_ROOT / "outputs" / "eda" / "09_global_finalists" / "predictions"
OUT = PROJECT_ROOT / "outputs" / "eda" / "16_nonlinear_challengers"
PREDICTIONS = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True)
PREDICTIONS.mkdir(parents=True, exist_ok=True)

if not PANEL.exists() or not BENCHMARK_JSON.exists():
    raise FileNotFoundError("Run Notebooks 05 and 09 before Notebook 16.")

con = duckdb.connect()
con.execute("PRAGMA threads=4")

def sql(query, params=None):
    return con.execute(query, params or []).df()

with open(BENCHMARK_JSON) as handle:
    frozen = json.load(handle)
CHAMPION = frozen["champion"]

catalog = pd.read_csv(CATALOG)
FEATURES = catalog["variable"].tolist()
FEATURE_GROUPS = catalog.groupby("group")["variable"].apply(list).to_dict()

BOOLEAN_FEATURES = [
    "auction_spike_lag1", "is_monday", "is_friday",
    "is_first_trading_day_month", "is_last_trading_day_month",
    "is_last_3_trading_days_month", "is_last_5_trading_days_month",
    "is_last_trading_day_quarter", "is_last_3_trading_days_quarter",
    "is_last_5_trading_days_quarter", "is_last_trading_day_year",
    "monthly_options_expiry", "quarterly_options_expiry",
    "after_long_market_gap", "before_long_market_gap", "is_early_close",
    "early_close_month_end_window", "expiry_month_end_window",
    "lag1_dividend_flag", "lag1_distribution_flag", "lag1_price_adjustment_flag",
]
CATEGORICAL_FEATURES = ["month", "day_of_week", "ff12_code", "size_decile", "liquidity_decile"]
CONTINUOUS_FEATURES = [f for f in FEATURES if f not in BOOLEAN_FEATURES and f not in CATEGORICAL_FEATURES]

CHAMPION_FEATURES = frozen["features"]
assert len(CHAMPION_FEATURES) == 58
assert len(FEATURES) == 99

print(f"Frozen benchmark: {CHAMPION}")
print(f"LightGBM version: {lgb.__version__}")
print(f"Feature sets: compact={len(CHAMPION_FEATURES)}, full={len(FEATURES)}")
'''),
    md('## Prespecified challenger design'),
    code(r'''
TEST_YEARS = [2022, 2023, 2024, 2025]
DEVELOPMENT_YEARS = [2022, 2023, 2024]
CONFIRMATION_YEAR = 2025
TUNE_ROWS = 250_000
VALIDATION_ROWS = 120_000
REFIT_ROWS = 350_000
SEED = 20261018
LEAF_GRID = [15, 31, 63]

CHALLENGERS = {
    "lgbm_residual_58_l2": {"features": CHAMPION_FEATURES, "objective": "regression"},
    "lgbm_residual_99_l2": {"features": FEATURES, "objective": "regression"},
    "lgbm_residual_58_huber": {"features": CHAMPION_FEATURES, "objective": "huber"},
}

CONTEXT = [
    "date", "permno", "auction_volume", "log1p_auction_volume",
    "median20_auction_volume", "size_percentile", "liquidity_percentile",
    "auction_intensity_percentile", "prior_auction_observations",
    "quarterly_options_expiry", "monthly_options_expiry",
    "is_last_trading_day_month", "is_early_close",
]
QUERY_COLUMNS = list(dict.fromkeys(CONTEXT + FEATURES))
QUOTED_COLUMNS = ", ".join(f'"{c}"' for c in QUERY_COLUMNS)

design = pd.DataFrame([
    {"candidate": name, "objective": spec["objective"], "source_feature_count": len(spec["features"])}
    for name, spec in CHALLENGERS.items()
])
display(design)
design.to_csv(OUT / "challenger_specifications.csv", index=False)
'''),
    md('## Leakage-safe loading and fitting helpers'),
    code(r'''
def load_period(start_date, end_date, sample_rows=None, seed=SEED):
    sample_clause = f"USING SAMPLE {sample_rows} ROWS (reservoir, {seed})" if sample_rows else ""
    return sql(f"""
        SELECT * FROM (
            SELECT {QUOTED_COLUMNS}
            FROM read_parquet('{PANEL}')
            WHERE eligible_target
              AND date >= DATE '{start_date}' AND date < DATE '{end_date}'
        ) {sample_clause}
    """)

def feature_mask(prep, source_features):
    allowed = set(source_features)
    return np.array([source in allowed for source in prep["output_sources"]])

def residual_target_and_base(train, score):
    fallback = float(np.median(train["auction_volume"]))
    train_base = train["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    score_base = score["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    target = train["log1p_auction_volume"].to_numpy(dtype=float) - np.log1p(train_base)
    return target, np.log1p(score_base)

def model_params(objective, leaves, seed, n_estimators=2000):
    return {
        "objective": objective,
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "num_leaves": leaves,
        "min_child_samples": 500,
        "max_bin": 127,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": 4,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }

def fit_validation_model(x_train, y_train, x_validation, y_validation, objective, leaves, seed):
    model = lgb.LGBMRegressor(**model_params(objective, leaves, seed))
    model.fit(
        x_train, y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model

def final_log_prediction(model, matrix, base_log, train_log):
    prediction = base_log + model.predict(matrix)
    lo, hi = np.quantile(train_log, [0.001, 0.999])
    return np.clip(prediction, lo, hi)

def load_benchmark_predictions(year):
    path = BENCHMARK_PREDICTIONS / f"global_finalist_predictions_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return sql(f'SELECT date, permno, "{CHAMPION}" AS frozen_prediction FROM read_parquet(\'{path}\')')
'''),
    md('## Nested validation and rolling out-of-time predictions'),
    code(r'''
metric_rows = []
validation_rows = []
importance_rows = []

for test_year in TEST_YEARS:
    print(f"Running nonlinear challengers for {test_year}...")
    validation_year = test_year - 1

    tune_train = load_period(
        f"{validation_year - 5}-01-01", f"{validation_year}-01-01",
        TUNE_ROWS, SEED + test_year,
    )
    validation = load_period(
        f"{validation_year}-01-01", f"{test_year}-01-01",
        VALIDATION_ROWS, SEED + 100 + test_year,
    )
    x_tune, prep_tune = fit_preprocessor(
        tune_train, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
    )
    x_validation = apply_preprocessor(
        validation, prep_tune, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
    )
    y_tune_residual, validation_base_log = residual_target_and_base(tune_train, validation)
    y_validation_log = validation["log1p_auction_volume"].to_numpy(dtype=float)
    y_validation_residual = y_validation_log - validation_base_log

    selected = {}
    for candidate, spec in CHALLENGERS.items():
        mask = feature_mask(prep_tune, spec["features"])
        best = None
        for leaves in LEAF_GRID:
            model = fit_validation_model(
                x_tune[:, mask], y_tune_residual,
                x_validation[:, mask], y_validation_residual,
                spec["objective"], leaves, SEED + test_year + leaves,
            )
            prediction_log = final_log_prediction(
                model, x_validation[:, mask], validation_base_log,
                tune_train["log1p_auction_volume"].to_numpy(dtype=float),
            )
            rmse = float(np.sqrt(np.mean((y_validation_log - prediction_log) ** 2)))
            row = {
                "test_year": test_year, "validation_year": validation_year,
                "candidate": candidate, "num_leaves": leaves,
                "best_iteration": int(model.best_iteration_), "validation_rmse_log": rmse,
            }
            validation_rows.append(row)
            if best is None or rmse < best["validation_rmse_log"]:
                best = row
        selected[candidate] = best

    del tune_train, validation, x_tune, x_validation, y_tune_residual, y_validation_residual
    gc.collect()

    refit = load_period(
        f"{test_year - 5}-01-01", f"{test_year}-01-01",
        REFIT_ROWS, SEED + 200 + test_year,
    )
    test = load_period(f"{test_year}-01-01", f"{test_year + 1}-01-01")
    x_refit, prep = fit_preprocessor(
        refit, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
    )
    x_test = apply_preprocessor(
        test, prep, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
    )
    y_refit_residual, test_base_log = residual_target_and_base(refit, test)
    refit_log = refit["log1p_auction_volume"].to_numpy(dtype=float)
    y_test_raw = test["auction_volume"].to_numpy(dtype=float)

    output = test[CONTEXT].copy()
    benchmark = load_benchmark_predictions(test_year)
    output = output.merge(benchmark, on=["date", "permno"], how="left", validate="one_to_one")
    if output["frozen_prediction"].isna().any():
        raise AssertionError("Frozen benchmark predictions did not align with the test panel.")
    metric_rows.append({
        "test_year": test_year, "candidate": CHAMPION, "model_family": "frozen ridge",
        **forecast_metrics(y_test_raw, output["frozen_prediction"]),
    })

    for candidate, spec in CHALLENGERS.items():
        choice = selected[candidate]
        mask = feature_mask(prep, spec["features"])
        iterations = max(25, choice["best_iteration"])
        model = lgb.LGBMRegressor(**model_params(
            spec["objective"], choice["num_leaves"], SEED + 300 + test_year,
            n_estimators=iterations,
        ))
        model.fit(x_refit[:, mask], y_refit_residual)
        prediction_log = final_log_prediction(model, x_test[:, mask], test_base_log, refit_log)
        prediction_raw = np.maximum(np.expm1(prediction_log), 0)
        output[candidate] = prediction_raw
        metric_rows.append({
            "test_year": test_year, "candidate": candidate, "model_family": "LightGBM",
            **forecast_metrics(y_test_raw, prediction_raw),
        })

        names = np.array(prep["output_names"])[mask]
        sources = np.array(prep["output_sources"])[mask]
        fold_importance = pd.DataFrame({
            "output_feature": names, "source_feature": sources,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }).groupby("source_feature", as_index=False)[["gain", "split"]].sum()
        fold_importance["test_year"] = test_year
        fold_importance["candidate"] = candidate
        importance_rows.extend(fold_importance.to_dict("records"))

        blend_name = candidate.replace("lgbm_", "blend50_")
        blend_log = 0.5 * np.log1p(output["frozen_prediction"].to_numpy(dtype=float)) + 0.5 * prediction_log
        blend_raw = np.maximum(np.expm1(blend_log), 0)
        output[blend_name] = blend_raw
        metric_rows.append({
            "test_year": test_year, "candidate": blend_name, "model_family": "50/50 log blend",
            **forecast_metrics(y_test_raw, blend_raw),
        })

    path = PREDICTIONS / f"nonlinear_challenger_predictions_{test_year}.parquet"
    con.register("prediction_output", output)
    con.execute(f"COPY prediction_output TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("prediction_output")
    print(f"  wrote {len(output):,} predictions to {path.name}")

    del refit, test, x_refit, x_test, y_refit_residual, output, benchmark
    gc.collect()

metrics = pd.DataFrame(metric_rows)
validation_results = pd.DataFrame(validation_rows)
importance = pd.DataFrame(importance_rows)
metrics.to_csv(OUT / "nonlinear_candidate_metrics.csv", index=False)
validation_results.to_csv(OUT / "nonlinear_hyperparameter_validation.csv", index=False)
importance.to_csv(OUT / "nonlinear_feature_importance.csv", index=False)

display(metrics.sort_values(["test_year", "rmse_log"]))
'''),
    md('## Paired date-block bootstrap and replacement gate'),
    code(r'''
PREDICTION_GLOB = str(PREDICTIONS / "nonlinear_challenger_predictions_*.parquet")
CANDIDATES = [c for c in metrics["candidate"].unique() if c != CHAMPION]

def bootstrap_mean_interval(values, seed, draws=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        means[draw] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()

bootstrap_rows = []
for candidate_number, candidate in enumerate(CANDIDATES):
    daily = sql(f"""
        SELECT date, year(date) AS test_year,
               avg(abs(ln(1 + "{candidate}") - log1p_auction_volume)
                 - abs(ln(1 + frozen_prediction) - log1p_auction_volume)) AS log_mae_difference
        FROM read_parquet('{PREDICTION_GLOB}')
        GROUP BY date, year(date)
        ORDER BY date
    """)
    for period, years in {
        "development_2022_2024": DEVELOPMENT_YEARS,
        "confirmation_2025": [CONFIRMATION_YEAR],
    }.items():
        values = daily.loc[daily["test_year"].isin(years), "log_mae_difference"].to_numpy()
        mean, low, high = bootstrap_mean_interval(values, SEED + candidate_number)
        bootstrap_rows.append({
            "candidate": candidate, "reference": CHAMPION, "period": period,
            "mean_difference": mean, "ci_2_5": low, "ci_97_5": high,
            "dates": len(values),
        })

bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(OUT / "nonlinear_date_block_bootstrap.csv", index=False)

development = metrics[metrics["test_year"].isin(DEVELOPMENT_YEARS)].groupby(
    ["candidate", "model_family"], as_index=False
).agg(
    development_rmse_log=("rmse_log", "mean"),
    development_wape=("wape", "mean"),
    development_spearman=("spearman", "mean"),
    development_bias=("aggregate_bias", "mean"),
    worst_development_rmse=("rmse_log", "max"),
)
confirmation = metrics[metrics["test_year"] == CONFIRMATION_YEAR][
    ["candidate", "rmse_log", "wape", "spearman", "aggregate_bias"]
].rename(columns={
    "rmse_log": "confirmation_rmse_log", "wape": "confirmation_wape",
    "spearman": "confirmation_spearman", "aggregate_bias": "confirmation_bias",
})
dev_bootstrap = bootstrap[bootstrap["period"] == "development_2022_2024"][
    ["candidate", "mean_difference", "ci_2_5", "ci_97_5"]
]
decision = development.merge(confirmation, on="candidate", how="left").merge(
    dev_bootstrap, on="candidate", how="left"
)
benchmark_rmse = float(decision.loc[decision["candidate"] == CHAMPION, "development_rmse_log"].iloc[0])
decision["rmse_change_vs_frozen"] = decision["development_rmse_log"] - benchmark_rmse

annual = metrics[metrics["test_year"].isin(DEVELOPMENT_YEARS)].pivot(
    index="candidate", columns="test_year", values="rmse_log"
)
benchmark_annual = annual.loc[CHAMPION]
stable = annual.sub(benchmark_annual, axis=1).max(axis=1).le(0.002)
decision["stable_each_development_year"] = decision["candidate"].map(stable)
decision["practical_rmse_improvement"] = decision["rmse_change_vs_frozen"] <= -0.002
decision["paired_log_mae_significant"] = decision["ci_97_5"] < 0
decision["eligible_replacement"] = (
    decision["practical_rmse_improvement"]
    & decision["stable_each_development_year"]
    & decision["paired_log_mae_significant"].fillna(False)
)
decision.to_csv(OUT / "nonlinear_candidate_selection.csv", index=False)
display(decision.sort_values("development_rmse_log"))
'''),
    md('## Segment and tail diagnostics'),
    code(r'''
prediction_columns = [CHAMPION] + CANDIDATES
segment_expressions = {
    "actual_tail": "CASE WHEN actual_percentile >= 0.99 THEN 'top_1pct' ELSE 'lower_99pct' END",
    "size": "CASE WHEN size_percentile < 0.33 THEN 'small' WHEN size_percentile < 0.67 THEN 'mid' ELSE 'large' END",
    "history": "CASE WHEN prior_auction_observations < 5 THEN 'cold_0_4' WHEN prior_auction_observations < 20 THEN 'developing_5_19' ELSE 'seasoned_20_plus' END",
    "event": "CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",
}

segment_queries = []
for candidate in prediction_columns:
    prediction = "frozen_prediction" if candidate == CHAMPION else f'"{candidate}"'
    for dimension, expression in segment_expressions.items():
        segment_queries.append(f"""
            SELECT '{candidate}' AS candidate, '{dimension}' AS dimension,
                   {expression} AS segment, count(*) AS observations,
                   sqrt(avg(pow(ln(1 + {prediction}) - log1p_auction_volume, 2))) AS rmse_log,
                   sum(abs({prediction} - auction_volume)) / sum(auction_volume) AS wape,
                   sum({prediction}) / sum(auction_volume) - 1 AS aggregate_bias
            FROM ranked GROUP BY 1, 2, 3
        """)

segment_diagnostics = sql(f"""
    WITH ranked AS (
        SELECT *, percent_rank() OVER (
            PARTITION BY year(date) ORDER BY auction_volume
        ) AS actual_percentile
        FROM read_parquet('{PREDICTION_GLOB}')
        WHERE year(date) IN (2022, 2023, 2024)
    )
    {" UNION ALL ".join(segment_queries)}
""")
segment_diagnostics.to_csv(OUT / "nonlinear_segment_diagnostics.csv", index=False)
display(segment_diagnostics.sort_values(["dimension", "segment", "rmse_log"]))
'''),
    md('## Comparison plot and conclusion'),
    code(r'''
plot_order = decision.sort_values("development_rmse_log")["candidate"].tolist()
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.lineplot(
    data=metrics, x="test_year", y="rmse_log", hue="candidate",
    hue_order=plot_order, marker="o", ax=axes[0],
)
axes[0].axvline(2024.5, color="black", linestyle="--", linewidth=1)
axes[0].set_title("Out-of-time log RMSE by year")
axes[0].set_xlabel("Test year")
axes[0].set_ylabel("Log RMSE")
axes[0].legend(fontsize=7, loc="best")

plot_data = decision.sort_values("development_rmse_log")
sns.barplot(data=plot_data, y="candidate", x="rmse_change_vs_frozen", ax=axes[1], color="#4c72b0")
axes[1].axvline(-0.002, color="green", linestyle="--", label="replacement threshold")
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Mean 2022–2024 RMSE change vs frozen ridge")
axes[1].set_xlabel("Change in log RMSE; lower is better")
axes[1].set_ylabel("")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT / "nonlinear_challenger_comparison.png", bbox_inches="tight")
plt.show()

eligible = decision.loc[decision["eligible_replacement"], "candidate"].tolist()
recommendation = {
    "frozen_benchmark": CHAMPION,
    "development_years": DEVELOPMENT_YEARS,
    "confirmation_year": CONFIRMATION_YEAR,
    "eligible_replacements": eligible,
    "retain_frozen_benchmark": len(eligible) == 0,
    "best_development_candidate": decision.sort_values("development_rmse_log").iloc[0]["candidate"],
}
with open(OUT / "nonlinear_model_recommendation.json", "w") as handle:
    json.dump(recommendation, handle, indent=2)
display(recommendation)
'''),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, DEST)
print(f"Wrote {DEST}")
