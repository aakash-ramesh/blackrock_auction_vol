"""Build Notebook 06: validation and role annotation for enriched features."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "06_feature_validation_and_role_audit.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r'''
# 06 — Feature validation and role audit

This notebook validates the 99 point-in-time-safe features in the enriched daily panel. **It does not drop features.** Instead, it annotates each feature for possible use as:

- a direct predictor,
- a segmentation or bifurcation variable,
- a context/control/interaction variable, or
- a retained feature that needs caution.

Validation is broader than a hypothesis-test p-value. With roughly 3.5 million labels, tiny and economically irrelevant effects can be statistically significant, while useful segmentation variables can have weak pooled relationships. The audit therefore emphasizes effect size, within-stock behavior, annual stability, nonlinear separation, and heterogeneity in the error of a historical-auction baseline.
'''),
    code(r'''
from pathlib import Path
from math import erfc, sqrt
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 140)
pd.set_option("display.max_rows", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (12, 7), "figure.dpi": 120})

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate blackrock_data.")

SOURCE = PROJECT_ROOT / "outputs" / "eda" / "05_feature_engineering"
PANEL = SOURCE / "enriched_daily_panel.parquet"
CATALOG = SOURCE / "approved_enriched_daily_features.csv"
OUT = PROJECT_ROOT / "outputs" / "eda" / "06_feature_validation"
OUT.mkdir(parents=True, exist_ok=True)
if not PANEL.exists() or not CATALOG.exists():
    raise FileNotFoundError("Run Notebook 05 before Notebook 06.")

con = duckdb.connect()
con.execute("PRAGMA threads=4")

def sql(query, params=None):
    return con.execute(query, params or []).df()

catalog = pd.read_csv(CATALOG)
FEATURES = catalog["variable"].tolist()
print(f"Features retained for validation: {len(FEATURES)}")
print(f"Panel: {PANEL}")
'''),
    md(r'''
## Validation policy

The main outcome is `log1p(auction_volume)`. The simple reference forecast is `log1p(median20_auction_volume)`. A feature may be useful for bifurcation when it separates the target level or the absolute error of this reference forecast, even if it has little pooled correlation with the target.

Approximate p-values below treat sampled stock-days as independent and are therefore optimistic because observations share dates and securities. They are reported for completeness, but role assignments use effect sizes and stability—not p-values alone.
'''),
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
ORDERED_OR_CONTINUOUS = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

assert set(BOOLEAN_FEATURES).issubset(FEATURES)
assert set(CATEGORICAL_FEATURES).issubset(FEATURES)
print(f"Ordered/continuous or binary: {len(ORDERED_OR_CONTINUOUS)}")
print(f"Explicit categorical: {len(CATEGORICAL_FEATURES)}")
'''),
    md('## Fixed analytical sample'),
    code(r'''
quoted_features = ",\n       ".join(f'"{f}"' for f in FEATURES)
sample = sql(f"""
SELECT date, year, permno,
       log1p_auction_volume AS target,
       ln(1 + median20_auction_volume) AS baseline_prediction,
       {quoted_features}
FROM read_parquet('{PANEL}')
WHERE eligible_target
USING SAMPLE 250000 ROWS (reservoir, 20261011)
""")
sample["baseline_error"] = sample["target"] - sample["baseline_prediction"]
sample["baseline_absolute_error"] = sample["baseline_error"].abs()

for feature in BOOLEAN_FEATURES:
    sample[feature] = sample[feature].astype("Float64")
for feature in FEATURES:
    if feature not in CATEGORICAL_FEATURES:
        sample[feature] = pd.to_numeric(sample[feature], errors="coerce")
sample.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"Sample rows: {len(sample):,}")
print(f"Stocks: {sample.permno.nunique():,}; dates: {sample.date.nunique():,}")
'''),
    md(r'''
## Pooled, within-stock, residual, and annual associations

All four measures use ranks, reducing sensitivity to heavy tails:

- **Pooled:** cross-sectional and time-series association combined.
- **Within stock:** relationship after removing each security's average rank.
- **Baseline residual:** association with what the 20-day auction-median forecast misses.
- **Annual stability:** direction and magnitude computed separately each year.
'''),
    code(r'''
y_rank = sample["target"].rank(pct=True)
residual_rank = sample["baseline_error"].rank(pct=True)
y_within = y_rank - y_rank.groupby(sample["permno"]).transform("mean")

association_rows = []
annual_rows = []
for feature in ORDERED_OR_CONTINUOUS:
    x = sample[feature]
    valid = x.notna() & sample["target"].notna()
    n = int(valid.sum())
    if n < 20 or x[valid].nunique() < 2:
        pooled = within = residual_corr = np.nan
    else:
        xr = x.rank(pct=True)
        pooled = xr.corr(y_rank)
        x_within = xr - xr.groupby(sample["permno"]).transform("mean")
        within = x_within.corr(y_within)
        residual_corr = xr.corr(residual_rank)

    year_values = []
    for year, idx in sample.groupby("year").groups.items():
        xy = sample.loc[idx, [feature, "target"]].dropna()
        if len(xy) >= 20 and xy[feature].nunique() >= 2:
            value = xy[feature].rank(pct=True).corr(xy["target"].rank(pct=True))
        else:
            value = np.nan
        year_values.append(value)
        annual_rows.append({"year": int(year), "feature": feature, "rank_correlation": value})

    year_values = np.asarray(year_values, dtype=float)
    valid_years = year_values[np.isfinite(year_values)]
    if len(valid_years):
        reference_sign = np.sign(pooled) if np.isfinite(pooled) and pooled != 0 else np.sign(np.nanmean(valid_years))
        sign_consistency = np.mean(np.sign(valid_years) == reference_sign)
        min_year = np.min(valid_years)
        max_year = np.max(valid_years)
        annual_std = np.std(valid_years, ddof=1) if len(valid_years) > 1 else 0.0
    else:
        sign_consistency = min_year = max_year = annual_std = np.nan

    if n > 3 and np.isfinite(pooled) and abs(pooled) < 1:
        z = np.arctanh(np.clip(pooled, -0.999999, 0.999999)) * np.sqrt(n - 3)
        p_approx = erfc(abs(z) / sqrt(2))
    else:
        p_approx = np.nan

    association_rows.append({
        "feature": feature, "association_n": n,
        "pooled_spearman": pooled, "within_stock_rank_corr": within,
        "baseline_residual_rank_corr": residual_corr,
        "annual_sign_consistency": sign_consistency,
        "annual_min_corr": min_year, "annual_max_corr": max_year,
        "annual_corr_std": annual_std, "p_value_approx_iid": p_approx,
    })

associations = pd.DataFrame(association_rows)
annual = pd.DataFrame(annual_rows)

# Benjamini-Hochberg annotation; descriptive only because the IID assumption is not valid here.
valid_p = associations["p_value_approx_iid"].notna()
rank = associations.loc[valid_p, "p_value_approx_iid"].rank(method="max")
m = valid_p.sum()
associations.loc[valid_p, "p_value_bh_iid"] = np.minimum(
    1.0, associations.loc[valid_p, "p_value_approx_iid"] * m / rank
)
associations["significant_bh_iid_5pct"] = associations["p_value_bh_iid"] < 0.05

annual.to_csv(OUT / "annual_rank_relationships.csv", index=False)
display(associations.sort_values("within_stock_rank_corr", key=lambda s: s.abs(), ascending=False).head(25))
'''),
    md(r'''
## Nonlinear separation and segmentation evidence

Continuous and ordered features are divided into quintiles. Categorical and binary variables use their observed levels. For each feature, we measure:

- range in mean log target across groups,
- range in median log auction volume,
- range in mean absolute baseline error,
- smallest group share.

Large baseline-error differences are especially relevant to bifurcation because they indicate that a single historical rule performs differently across regimes.
'''),
    code(r'''
separation_rows = []
group_detail_rows = []

for feature in FEATURES:
    valid = sample[[feature, "target", "baseline_absolute_error"]].dropna().copy()
    if feature in CATEGORICAL_FEATURES or feature in BOOLEAN_FEATURES:
        valid["validation_group"] = valid[feature].astype(str)
        method = "observed_level"
    else:
        try:
            valid["validation_group"] = pd.qcut(valid[feature], 5, labels=False, duplicates="drop") + 1
            method = "sample_quintile"
        except ValueError:
            valid["validation_group"] = "single_level"
            method = "single_level"

    grouped = valid.groupby("validation_group", observed=True).agg(
        observations=("target", "size"),
        mean_log_target=("target", "mean"),
        median_log_target=("target", "median"),
        mean_baseline_absolute_error=("baseline_absolute_error", "mean"),
    ).reset_index()
    grouped["feature"] = feature
    grouped["grouping_method"] = method
    grouped["group_share"] = grouped["observations"] / grouped["observations"].sum()
    group_detail_rows.append(grouped)

    separation_rows.append({
        "feature": feature,
        "grouping_method": method,
        "group_count": len(grouped),
        "minimum_group_share": grouped["group_share"].min(),
        "mean_log_target_range": grouped["mean_log_target"].max() - grouped["mean_log_target"].min(),
        "median_log_target_range": grouped["median_log_target"].max() - grouped["median_log_target"].min(),
        "baseline_mae_range": (
            grouped["mean_baseline_absolute_error"].max()
            - grouped["mean_baseline_absolute_error"].min()
        ),
        "largest_group_baseline_mae": grouped["mean_baseline_absolute_error"].max(),
        "smallest_group_baseline_mae": grouped["mean_baseline_absolute_error"].min(),
    })

separation = pd.DataFrame(separation_rows)
group_details = pd.concat(group_detail_rows, ignore_index=True)
separation.to_csv(OUT / "nonlinear_and_segmentation_separation.csv", index=False)
group_details.to_csv(OUT / "feature_group_diagnostics.csv", index=False)

display(separation.sort_values("baseline_mae_range", ascending=False).head(25))
'''),
    md(r'''
## Missingness and variation diagnostics

Missingness can itself carry information, but it can also signal cold-start stocks or data availability bias. All affected features remain available; the audit marks them for explicit missing indicators and training-window imputation.
'''),
    code(r'''
missingness = pd.read_csv(SOURCE / "enriched_feature_missingness.csv")[[
    "feature", "missing_rate", "distinct_values"
]]
missing_effect_rows = []
for feature in FEATURES:
    is_missing = sample[feature].isna()
    if is_missing.any() and (~is_missing).any():
        missing_effect = sample.loc[is_missing, "target"].mean() - sample.loc[~is_missing, "target"].mean()
        missing_error_effect = (
            sample.loc[is_missing, "baseline_absolute_error"].mean()
            - sample.loc[~is_missing, "baseline_absolute_error"].mean()
        )
    else:
        missing_effect = missing_error_effect = np.nan
    missing_effect_rows.append({
        "feature": feature,
        "missing_vs_present_log_target_difference": missing_effect,
        "missing_vs_present_baseline_mae_difference": missing_error_effect,
    })
missing_effects = pd.DataFrame(missing_effect_rows)
'''),
    md(r'''
## Role annotation without feature deletion

The thresholds below are screening rules, not immutable scientific cutoffs:

- **Direct prediction evidence:** meaningful pooled, within-stock, or residual association with reasonably stable annual direction.
- **Segmentation evidence:** meaningful variation in target or baseline error across groups, including rare calendar/event groups.
- **Context/interaction:** safe variables without strong standalone evidence that may still control regimes or participate in interactions.
- **Retain with caution:** unstable, sparse, near-constant, or higher-missingness variables.

Every row is marked `retain_in_master_panel = True`.
'''),
    code(r'''
catalog_validation = catalog.assign(feature=catalog["variable"])
validation = (
    catalog_validation.merge(associations, on="feature", how="left")
           .merge(separation, on="feature", how="left")
           .merge(missingness, on="feature", how="left")
           .merge(missing_effects, on="feature", how="left")
)

# Categorical variables are assessed through group separation, not rank correlation.
stable_numeric = validation["annual_sign_consistency"].fillna(0) >= 0.70
numeric_direct = (
    (validation["pooled_spearman"].abs().fillna(0) >= 0.05)
    | (validation["within_stock_rank_corr"].abs().fillna(0) >= 0.03)
    | (validation["baseline_residual_rank_corr"].abs().fillna(0) >= 0.03)
) & stable_numeric
categorical_direct = (
    validation["variable"].isin(CATEGORICAL_FEATURES + BOOLEAN_FEATURES)
    & (validation["mean_log_target_range"].fillna(0) >= 0.15)
    & (validation["minimum_group_share"].fillna(0) >= 0.002)
)
validation["direct_prediction_evidence"] = numeric_direct | categorical_direct

adequate_group = validation["minimum_group_share"].fillna(0) >= 0.002
validation["segmentation_evidence"] = adequate_group & (
    (validation["baseline_mae_range"].fillna(0) >= 0.10)
    | (validation["mean_log_target_range"].fillna(0) >= 0.25)
)

validation["quality_caution"] = (
    (validation["missing_rate"].fillna(0) >= 0.05)
    | (validation["distinct_values"].fillna(0) <= 1)
    | (
        validation["variable"].isin(CATEGORICAL_FEATURES + BOOLEAN_FEATURES)
        & (validation["minimum_group_share"].fillna(0) < 0.001)
    )
    | (
        validation["annual_sign_consistency"].notna()
        & (validation["annual_sign_consistency"] < 0.55)
        & (validation["pooled_spearman"].abs().fillna(0) >= 0.02)
    )
)

def assign_role(row):
    if row["quality_caution"]:
        return "retain_with_caution"
    if row["direct_prediction_evidence"] and row["segmentation_evidence"]:
        return "predictor_and_segmentation_candidate"
    if row["direct_prediction_evidence"]:
        return "direct_predictor_candidate"
    if row["segmentation_evidence"]:
        return "segmentation_candidate"
    return "context_control_or_interaction"

validation["recommended_role"] = validation.apply(assign_role, axis=1)
validation["retain_in_master_panel"] = True
validation["needs_missing_indicator"] = validation["missing_rate"].fillna(0) > 0
validation["significance_warning"] = (
    "IID p-value is descriptive only; repeated stock/date observations overstate precision"
)

validation = validation.sort_values([
    "recommended_role", "baseline_mae_range", "within_stock_rank_corr"
], ascending=[True, False, False])
validation.to_csv(OUT / "feature_validation_and_roles.csv", index=False)

assert len(validation) == len(FEATURES)
assert validation["retain_in_master_panel"].all()
display(validation.groupby(["recommended_role", "group"]).size().rename("features").to_frame())
display(validation[[
    "variable", "group", "recommended_role", "pooled_spearman",
    "within_stock_rank_corr", "baseline_residual_rank_corr",
    "annual_sign_consistency", "mean_log_target_range", "baseline_mae_range",
    "missing_rate", "quality_caution", "retain_in_master_panel",
]].head(40))
'''),
    md('## Visual summary'),
    code(r'''
plot = validation.copy()
plot["absolute_within_stock"] = plot["within_stock_rank_corr"].abs().fillna(0)

fig, ax = plt.subplots(figsize=(13, 8))
sns.scatterplot(
    data=plot, x="absolute_within_stock", y="baseline_mae_range",
    hue="group", style="recommended_role", s=85, alpha=0.8, ax=ax,
)
label_candidates = pd.concat([
    plot.nlargest(1, "absolute_within_stock"),
    plot.nlargest(8, "baseline_mae_range"),
]).drop_duplicates("variable")
for _, row in label_candidates.iterrows():
    ax.annotate(row["variable"], (row["absolute_within_stock"], row["baseline_mae_range"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set(
    title="Direct within-stock evidence versus segmentation evidence",
    xlabel="Absolute within-stock rank correlation",
    ylabel="Range in baseline MAE across feature groups",
)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "prediction_vs_segmentation_evidence.png", bbox_inches="tight")
plt.show()

top_features = validation.assign(
    absolute_within=validation["within_stock_rank_corr"].abs()
).nlargest(20, "absolute_within", keep="all")["variable"].tolist()
heat = annual[annual["feature"].isin(top_features)].pivot(
    index="feature", columns="year", values="rank_correlation"
)
heat = heat.loc[heat.abs().mean(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(heat, cmap="vlag", center=0, ax=ax)
ax.set(title="Annual rank relationships for strongest within-stock candidates", xlabel="Year", ylabel="")
fig.tight_layout()
fig.savefig(OUT / "annual_relationship_stability.png", bbox_inches="tight")
plt.show()
'''),
    md(r'''
## Conclusions and handoff

This audit preserves the complete feature set. The role labels guide—not dictate—the next experiments. In particular:

- weak standalone features remain available for interactions and bifurcation;
- segmentation candidates must still pass training-only cutoff stability and out-of-time performance tests;
- correlated variables remain in the master panel, although individual model pipelines may regularize or select among them;
- transformations, clipping, imputation, and encoding must be fitted inside each training window.
'''),
    code(r'''
role_summary = validation.groupby("recommended_role").agg(
    features=("variable", "size"),
    median_absolute_within=("within_stock_rank_corr", lambda s: s.abs().median()),
    median_baseline_mae_range=("baseline_mae_range", "median"),
).reset_index()
display(role_summary)
role_summary.to_csv(OUT / "role_summary.csv", index=False)

print("No features dropped.")
print(f"Retained master features: {validation.retain_in_master_panel.sum()} / {len(validation)}")
print(f"Direct-prediction evidence: {validation.direct_prediction_evidence.sum()}")
print(f"Segmentation evidence: {validation.segmentation_evidence.sum()}")
print(f"Quality cautions: {validation.quality_caution.sum()}")
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
