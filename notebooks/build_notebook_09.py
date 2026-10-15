"""Build Notebook 09: combine and freeze the global benchmark specification."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "09_global_finalists_and_benchmark_freeze.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r'''
# 09 — Global finalists and benchmark freeze

Notebook 08 tested robustness dimensions separately. This notebook combines the best choices and freezes the global benchmark that future interaction and separate-model experiments must beat.

Selection discipline:

- 2022–2024 are development evaluation years.
- 2025 is reported separately as a confirmation year. It is not perfectly pristine because earlier notebooks already inspected 2025, so results remain research evidence rather than a claim of untouched production validation.
- Every finalist tunes its ridge penalty using the immediately preceding historical year only.
- A model within 0.002 mean development log RMSE of the best is considered practically tied; among tied models, fewer source features are preferred.
- Raw WAPE, bias, tail behavior, and stability are reported alongside the primary log-RMSE criterion.
'''),
    code(r'''
from pathlib import Path
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 160)
pd.set_option("display.max_rows", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (12, 7), "figure.dpi": 120})

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate blackrock_data.")
sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.modeling_utils import (
    fit_preprocessor, apply_preprocessor, fit_ridge, predict_ridge, forecast_metrics
)

PANEL = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering" / "enriched_daily_panel.parquet"
CATALOG = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering" / "approved_enriched_daily_features.csv"
OUT = PROJECT_ROOT / "outputs" / "eda" / "09_global_finalists"
PREDICTIONS = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True)
PREDICTIONS.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute("PRAGMA threads=4")

def sql(query, params=None):
    return con.execute(query, params or []).df()

catalog = pd.read_csv(CATALOG)
FEATURES = catalog["variable"].tolist()
FEATURE_GROUPS = catalog.groupby("group")["variable"].apply(list).to_dict()
'''),
    md('## Predeclared finalists'),
    code(r'''
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

AUCTION_CALENDAR = list(dict.fromkeys(
    FEATURE_GROUPS["Auction history"] + FEATURE_GROUPS["Calendar"]
))
AUCTION_CALENDAR_LIQUIDITY = list(dict.fromkeys(
    AUCTION_CALENDAR + FEATURE_GROUPS["Liquidity"]
))

FINALISTS = {
    "expanding_log_full": {
        "window": "expanding", "target": "log_level", "weight": "uniform", "features": FEATURES,
    },
    "rolling5_log_full": {
        "window": "rolling5", "target": "log_level", "weight": "uniform", "features": FEATURES,
    },
    "rolling5_residual_full": {
        "window": "rolling5", "target": "median20_residual", "weight": "uniform", "features": FEATURES,
    },
    "rolling3_residual_full": {
        "window": "rolling3", "target": "median20_residual", "weight": "uniform", "features": FEATURES,
    },
    "rolling5_residual_recent_full": {
        "window": "rolling5", "target": "median20_residual", "weight": "recent", "features": FEATURES,
    },
    "rolling5_residual_auction_calendar": {
        "window": "rolling5", "target": "median20_residual", "weight": "uniform",
        "features": AUCTION_CALENDAR,
    },
    "rolling5_residual_auction_calendar_liquidity": {
        "window": "rolling5", "target": "median20_residual", "weight": "uniform",
        "features": AUCTION_CALENDAR_LIQUIDITY,
    },
}

finalist_table = pd.DataFrame([
    {"candidate": name, "window": spec["window"], "target": spec["target"],
     "weight": spec["weight"], "source_feature_count": len(spec["features"])}
    for name, spec in FINALISTS.items()
])
display(finalist_table)
finalist_table.to_csv(OUT / "finalist_specifications.csv", index=False)
'''),
    md('## Shared query and estimation helpers'),
    code(r'''
TEST_YEARS = [2022, 2023, 2024, 2025]
DEVELOPMENT_YEARS = [2022, 2023, 2024]
CONFIRMATION_YEAR = 2025
LAMBDA_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
TUNE_ROWS = 250_000
VALIDATION_ROWS = 120_000
REFIT_ROWS = 350_000
SEED = 20261012

CONTEXT = [
    "date", "permno", "auction_volume", "log1p_auction_volume",
    "median20_auction_volume", "avg20_daily_volume",
    "size_percentile", "liquidity_percentile", "auction_intensity_percentile",
    "prior_auction_observations", "quarterly_options_expiry", "monthly_options_expiry",
    "is_last_trading_day_month", "is_early_close",
]
QUERY_COLUMNS = list(dict.fromkeys(CONTEXT + FEATURES))
QUOTED_COLUMNS = ", ".join(f'"{c}"' for c in QUERY_COLUMNS)

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

def window_start(end_year, window):
    if window == "expanding":
        return "2015-01-01"
    years = int(window.replace("rolling", ""))
    return f"{end_year - years}-01-01"

def feature_mask(prep, source_features):
    allowed = set(source_features)
    return np.array([source in allowed for source in prep["output_sources"]])

def training_weights(frame, specification, prediction_year):
    if specification["weight"] == "uniform":
        return None
    age_years = (
        pd.Timestamp(f"{prediction_year}-01-01") - pd.to_datetime(frame["date"])
    ).dt.days.to_numpy() / 365.25
    return np.power(0.5, age_years / 2.0)

def model_target_and_base(train, score, specification):
    y_train_log = train["log1p_auction_volume"].to_numpy(dtype=float)
    if specification["target"] == "log_level":
        return y_train_log, np.zeros(len(score)), np.median(train["auction_volume"])
    fallback = float(np.median(train["auction_volume"]))
    train_base = train["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    score_base = score["median20_auction_volume"].fillna(fallback).to_numpy(dtype=float)
    return y_train_log - np.log1p(train_base), np.log1p(score_base), fallback

def fit_score_candidate(train, score, x_train, x_score, mask, specification, strength, prediction_year):
    target, score_base_log, fallback = model_target_and_base(train, score, specification)
    weights = training_weights(train, specification, prediction_year)
    coef = fit_ridge(x_train[:, mask], target, strength, weights)
    component = predict_ridge(x_score[:, mask], coef)
    final_log = component + score_base_log
    y_train_log = train["log1p_auction_volume"].to_numpy(dtype=float)
    lo, hi = np.quantile(y_train_log, [0.001, 0.999])
    return np.clip(final_log, lo, hi), coef, fallback
'''),
    md(r'''
## Nested tuning and out-of-time finalist forecasts

Matrices are shared among finalists with the same history window, while each candidate receives its own feature mask, target, weights, and selected penalty.
'''),
    code(r'''
metric_rows = []
lambda_rows = []
coefficient_rows = []

for test_year in TEST_YEARS:
    print(f"Running finalists for {test_year}...")
    test = load_period(f"{test_year}-01-01", f"{test_year + 1}-01-01")
    output = test[CONTEXT].copy()
    y_test_raw = test["auction_volume"].to_numpy(dtype=float)

    for window in sorted({spec["window"] for spec in FINALISTS.values()}):
        candidates = {name: spec for name, spec in FINALISTS.items() if spec["window"] == window}
        validation_year = test_year - 1
        tune_train = load_period(
            window_start(validation_year, window), f"{validation_year}-01-01",
            TUNE_ROWS, SEED + test_year + len(window),
        )
        validation = load_period(
            f"{validation_year}-01-01", f"{test_year}-01-01",
            VALIDATION_ROWS, SEED + 100 + test_year + len(window),
        )
        x_tune, prep_tune = fit_preprocessor(
            tune_train, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        x_validation = apply_preprocessor(
            validation, prep_tune, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )

        selected = {}
        for name, spec in candidates.items():
            mask = feature_mask(prep_tune, spec["features"])
            best_strength, best_rmse = None, np.inf
            for strength in LAMBDA_GRID:
                pred_log, _, _ = fit_score_candidate(
                    tune_train, validation, x_tune, x_validation, mask,
                    spec, strength, validation_year,
                )
                rmse = np.sqrt(np.mean(
                    (validation["log1p_auction_volume"].to_numpy(dtype=float) - pred_log) ** 2
                ))
                lambda_rows.append({
                    "test_year": test_year, "candidate": name,
                    "validation_year": validation_year, "ridge_strength": strength,
                    "validation_rmse_log": rmse,
                })
                if rmse < best_rmse:
                    best_strength, best_rmse = strength, rmse
            selected[name] = (best_strength, best_rmse)

        refit = load_period(
            window_start(test_year, window), f"{test_year}-01-01",
            REFIT_ROWS, SEED + 200 + test_year + len(window),
        )
        x_refit, prep = fit_preprocessor(
            refit, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )
        x_test = apply_preprocessor(
            test, prep, CONTINUOUS_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES
        )

        for name, spec in candidates.items():
            strength, validation_rmse = selected[name]
            mask = feature_mask(prep, spec["features"])
            pred_log, coef, fallback = fit_score_candidate(
                refit, test, x_refit, x_test, mask, spec, strength, test_year,
            )
            pred_raw = np.expm1(pred_log)
            output[name] = pred_raw
            metric_rows.append({
                "test_year": test_year, "candidate": name,
                "selected_lambda": strength, "validation_rmse_log": validation_rmse,
                **forecast_metrics(y_test_raw, pred_raw),
            })
            selected_names = np.asarray(prep["output_names"])[mask]
            selected_sources = np.asarray(prep["output_sources"])[mask]
            coefficient_rows.extend([
                {"test_year": test_year, "candidate": name, "output_feature": "INTERCEPT",
                 "source_feature": "INTERCEPT", "coefficient": coef[0]},
                *[
                    {"test_year": test_year, "candidate": name, "output_feature": out_name,
                     "source_feature": source, "coefficient": value}
                    for out_name, source, value in zip(selected_names, selected_sources, coef[1:])
                ],
            ])
        del tune_train, validation, refit, x_tune, x_validation, x_refit, x_test

    prediction_path = PREDICTIONS / f"global_finalist_predictions_{test_year}.parquet"
    con.register("global_finalist_output", output)
    con.execute(f"COPY global_finalist_output TO '{prediction_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("global_finalist_output")
    print(f"  wrote {len(output):,} predictions")

metrics = pd.DataFrame(metric_rows)
lambda_results = pd.DataFrame(lambda_rows)
coefficients = pd.DataFrame(coefficient_rows)
metrics.to_csv(OUT / "global_finalist_metrics.csv", index=False)
lambda_results.to_csv(OUT / "global_finalist_lambda_validation.csv", index=False)
coefficients.to_csv(OUT / "global_finalist_coefficients.csv", index=False)
display(metrics.sort_values(["test_year", "rmse_log"]))
'''),
    md(r'''
## Development selection and 2025 confirmation

The primary development metric is mean annual log RMSE over 2022–2024. Candidates no more than 0.002 above the best are considered practically tied; the candidate with the fewest source features is selected from that set, with RMSE breaking any remaining tie.
'''),
    code(r'''
dev = metrics[metrics["test_year"].isin(DEVELOPMENT_YEARS)].groupby("candidate").agg(
    development_rmse_log=("rmse_log", "mean"),
    development_wape=("wape", "mean"),
    development_spearman=("spearman", "mean"),
    development_bias=("aggregate_bias", "mean"),
    worst_development_rmse=("rmse_log", "max"),
).reset_index().merge(finalist_table[["candidate", "source_feature_count"]], on="candidate")

best_rmse = dev["development_rmse_log"].min()
dev["within_practical_tolerance"] = dev["development_rmse_log"] <= best_rmse + 0.002
eligible = dev[dev["within_practical_tolerance"]].sort_values(
    ["source_feature_count", "development_rmse_log"]
)
CHAMPION = eligible.iloc[0]["candidate"]

confirmation = metrics[metrics["test_year"] == CONFIRMATION_YEAR].merge(
    finalist_table[["candidate", "source_feature_count"]], on="candidate"
).sort_values("rmse_log")
selection_table = dev.merge(
    confirmation[["candidate", "rmse_log", "wape", "spearman", "aggregate_bias"]].rename(columns={
        "rmse_log": "confirmation_rmse_log", "wape": "confirmation_wape",
        "spearman": "confirmation_spearman", "aggregate_bias": "confirmation_bias",
    }), on="candidate"
).sort_values("development_rmse_log")
selection_table["selected_global_champion"] = selection_table["candidate"] == CHAMPION
selection_table.to_csv(OUT / "global_finalist_selection.csv", index=False)

print(f"Selected global champion: {CHAMPION}")
display(selection_table)
'''),
    md(r'''
## Paired date-block uncertainty

Every finalist is compared with the original expanding/full/log candidate on identical dates and stock-days. The champion is also compared with the best non-champion development candidate. Intervals are reported separately for development dates and the 2025 confirmation dates.
'''),
    code(r'''
REFERENCE = "expanding_log_full"
runner_up = selection_table[selection_table["candidate"] != CHAMPION].iloc[0]["candidate"]
PREDICTION_GLOB = str(PREDICTIONS / "global_finalist_predictions_*.parquet")

comparisons = [(candidate, REFERENCE) for candidate in FINALISTS if candidate != REFERENCE]
if (CHAMPION, runner_up) not in comparisons:
    comparisons.append((CHAMPION, runner_up))

rng = np.random.default_rng(SEED)
bootstrap_rows = []
for candidate, reference in comparisons:
    daily = sql(f"""
        SELECT date, year(date) AS test_year,
               avg(abs(ln(1 + \"{candidate}\") - log1p_auction_volume)
                   - abs(ln(1 + \"{reference}\") - log1p_auction_volume)) AS log_mae_difference,
               sum(abs(\"{candidate}\" - auction_volume)) / sum(auction_volume)
                   - sum(abs(\"{reference}\" - auction_volume)) / sum(auction_volume)
                   AS daily_wape_difference
        FROM read_parquet('{PREDICTION_GLOB}') GROUP BY date
    """)
    for period, subset in {
        "development_2022_2024": daily[daily["test_year"].isin(DEVELOPMENT_YEARS)],
        "confirmation_2025": daily[daily["test_year"] == CONFIRMATION_YEAR],
    }.items():
        for metric in ["log_mae_difference", "daily_wape_difference"]:
            values = subset[metric].dropna().to_numpy()
            draws = np.empty(3000)
            for i in range(len(draws)):
                draws[i] = rng.choice(values, size=len(values), replace=True).mean()
            bootstrap_rows.append({
                "candidate": candidate, "reference": reference, "period": period,
                "metric": metric, "mean_difference": values.mean(),
                "ci_2_5": np.quantile(draws, 0.025), "ci_97_5": np.quantile(draws, 0.975),
                "dates": len(values),
            })

bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(OUT / "global_finalist_date_block_bootstrap.csv", index=False)
display(bootstrap[(bootstrap["candidate"] == CHAMPION) | (bootstrap["reference"] == CHAMPION)])
'''),
    md(r'''
## Champion tail, segment, and sign diagnostics

Freezing the benchmark does not mean its errors are homogeneous. The final diagnostics identify where future interaction or specialist models must improve.
'''),
    code(r'''
segment_diagnostics = sql(f"""
WITH base AS (
    SELECT *,
           CASE WHEN size_percentile < 1.0/3 THEN 'small'
                WHEN size_percentile < 2.0/3 THEN 'mid' ELSE 'large' END AS size_segment,
           CASE WHEN liquidity_percentile < 1.0/3 THEN 'low'
                WHEN liquidity_percentile < 2.0/3 THEN 'mid' ELSE 'high' END AS liquidity_segment,
           CASE WHEN auction_intensity_percentile < 1.0/3 THEN 'low'
                WHEN auction_intensity_percentile < 2.0/3 THEN 'mid' ELSE 'high' END AS intensity_segment,
           CASE WHEN prior_auction_observations < 5 THEN 'cold_0_4'
                WHEN prior_auction_observations < 20 THEN 'developing_5_19'
                ELSE 'seasoned_20_plus' END AS history_segment,
           CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry'
                WHEN monthly_options_expiry THEN 'monthly_expiry'
                WHEN is_last_trading_day_month THEN 'month_end'
                WHEN is_early_close THEN 'early_close' ELSE 'regular' END AS event_segment,
           percent_rank() OVER (PARTITION BY year(date) ORDER BY auction_volume) AS actual_percentile
    FROM read_parquet('{PREDICTION_GLOB}')
), long_segments AS (
    SELECT date, auction_volume, log1p_auction_volume, "{CHAMPION}" AS prediction,
           'size' AS dimension, size_segment AS segment FROM base
    UNION ALL SELECT date, auction_volume, log1p_auction_volume, "{CHAMPION}",
           'liquidity', liquidity_segment FROM base
    UNION ALL SELECT date, auction_volume, log1p_auction_volume, "{CHAMPION}",
           'auction_intensity', intensity_segment FROM base
    UNION ALL SELECT date, auction_volume, log1p_auction_volume, "{CHAMPION}",
           'history', history_segment FROM base
    UNION ALL SELECT date, auction_volume, log1p_auction_volume, "{CHAMPION}",
           'event', event_segment FROM base
    UNION ALL SELECT date, auction_volume, log1p_auction_volume, "{CHAMPION}",
           'actual_tail', CASE WHEN actual_percentile >= 0.99 THEN 'top_1pct' ELSE 'lower_99pct' END FROM base
)
SELECT dimension, segment, count(*) AS observations,
       sqrt(avg(pow(ln(1 + prediction) - log1p_auction_volume, 2))) AS rmse_log,
       sum(abs(prediction - auction_volume)) / sum(auction_volume) AS wape,
       sum(prediction) / sum(auction_volume) - 1 AS aggregate_bias
FROM long_segments GROUP BY dimension, segment ORDER BY dimension, segment
""")
segment_diagnostics.to_csv(OUT / "champion_segment_and_tail_diagnostics.csv", index=False)
display(segment_diagnostics)

champion_coef = coefficients[
    (coefficients["candidate"] == CHAMPION)
    & (coefficients["output_feature"] == coefficients["source_feature"])
    & (coefficients["source_feature"] != "INTERCEPT")
]
champion_signs = champion_coef.groupby("source_feature").agg(
    estimates=("coefficient", "size"),
    mean_coefficient=("coefficient", "mean"),
    sign_consistency=("coefficient", lambda s: max((s > 0).mean(), (s < 0).mean())),
    minimum=("coefficient", "min"), maximum=("coefficient", "max"),
).reset_index()
champion_signs.to_csv(OUT / "champion_coefficient_stability.csv", index=False)
display(champion_signs.sort_values("sign_consistency").head(20))
'''),
    md('## Final comparison visualization'),
    code(r'''
plot_metrics = metrics.merge(finalist_table[["candidate", "source_feature_count"]], on="candidate")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.lineplot(data=plot_metrics, x="test_year", y="rmse_log", hue="candidate", marker="o", ax=axes[0])
axes[0].set(title="Global finalist log RMSE", xlabel="Test year", ylabel="Log RMSE")
axes[0].legend(fontsize=8, frameon=False)
selection_plot = selection_table.sort_values("development_rmse_log")
sns.scatterplot(
    data=selection_plot, x="development_rmse_log", y="confirmation_rmse_log",
    size="source_feature_count", hue="candidate", sizes=(80, 260), ax=axes[1], legend=False,
)
for _, row in selection_plot.iterrows():
    axes[1].annotate(row["candidate"], (row["development_rmse_log"], row["confirmation_rmse_log"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
axes[1].set(title="Development versus 2025 confirmation", xlabel="2022–2024 mean log RMSE",
            ylabel="2025 log RMSE")
fig.tight_layout()
fig.savefig(OUT / "global_finalist_comparison.png", bbox_inches="tight")
plt.show()
'''),
    md(r'''
## Frozen benchmark handoff

The selected candidate is the benchmark for subsequent experiments. Separate models or interactions must use the same target universe, folds, metrics, and paired date-level comparisons. A segmented approach is justified only if it improves the frozen benchmark consistently, not merely one historical rule or the earlier arbitrary ridge specification.
'''),
    code(r'''
champion_record = {
    "champion": CHAMPION,
    "selection_years": DEVELOPMENT_YEARS,
    "confirmation_year": CONFIRMATION_YEAR,
    "selection_tolerance_log_rmse": 0.002,
    **FINALISTS[CHAMPION],
}
champion_record["features"] = list(champion_record["features"])
import json
with open(OUT / "frozen_global_benchmark.json", "w") as handle:
    json.dump(champion_record, handle, indent=2)

artifacts = pd.DataFrame({
    "artifact": [
        "Finalist selection", "Paired date-block bootstrap", "Finalist metrics",
        "Lambda validation", "Champion segment/tail diagnostics",
        "Champion coefficient stability", "Frozen benchmark specification",
        "Prediction directory",
    ],
    "path": [
        str(OUT / "global_finalist_selection.csv"),
        str(OUT / "global_finalist_date_block_bootstrap.csv"),
        str(OUT / "global_finalist_metrics.csv"),
        str(OUT / "global_finalist_lambda_validation.csv"),
        str(OUT / "champion_segment_and_tail_diagnostics.csv"),
        str(OUT / "champion_coefficient_stability.csv"),
        str(OUT / "frozen_global_benchmark.json"), str(PREDICTIONS),
    ],
})
artifacts.to_csv(OUT / "artifact_manifest.csv", index=False)
print(f"Frozen global benchmark: {CHAMPION}")
display(artifacts)
'''),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (capstone-eda)",
            "language": "python",
            "name": "capstone-eda",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, DEST)
print(f"Wrote {DEST}")
