"""Build Notebook 13: unsmoothed daily plots for representative stock-years."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "13_representative_stock_year_daily_predictions.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r'''
# 13 — Representative stock-year daily predictions

This notebook shows unsmoothed daily actual and predicted closing-auction volumes for the six stocks in Notebook 12.

To avoid cherry-picking either an unusually good or unusually bad year, each stock's representative year is selected mechanically: compute annual daily log RMSE for 2016–2025, find the stock's median annual RMSE, and choose the year closest to that median. The five largest absolute log errors in the selected year are marked on each chart.

The logarithmic vertical axis is only an axis transformation; no rolling mean, median, or other smoothing is applied.
'''),
    code(r'''
from pathlib import Path
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 130})

cwd = Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate blackrock_data.")
sys.path.insert(0, str(PROJECT_ROOT))

from notebooks.modeling_utils import forecast_metrics

SOURCE = (
    PROJECT_ROOT / "outputs" / "eda" / "12_selected_stock_predictions"
    / "selected_stock_daily_walk_forward_predictions.parquet"
)
OUT = PROJECT_ROOT / "outputs" / "eda" / "13_representative_stock_year_daily"
OUT.mkdir(parents=True, exist_ok=True)

PROFILES = {
    47896: ("JPM", "large / high liquidity"),
    87267: ("BLK", "large / moderate liquidity"),
    80432: ("AEO", "mid-size / high liquidity"),
    11018: ("FBP", "mid-size / moderate liquidity"),
    76732: ("DIN", "small / moderate liquidity"),
    11267: ("CATO", "small / low liquidity"),
}
PERMNO_ORDER = list(PROFILES)

con = duckdb.connect()
predictions = con.execute(f"SELECT * FROM read_parquet('{SOURCE}') WHERE prediction IS NOT NULL").df()
predictions["date"] = pd.to_datetime(predictions["date"])
predictions["year"] = predictions["date"].dt.year
predictions["absolute_log_error"] = np.abs(
    np.log1p(predictions["prediction"]) - np.log1p(predictions["auction_volume"])
)
'''),
    md('## Select a representative year for each stock'),
    code(r'''
annual_rows = []
for (permno, year), frame in predictions.groupby(["permno", "year"]):
    ticker, profile = PROFILES[permno]
    annual_rows.append({
        "permno": permno, "ticker": ticker, "profile": profile, "year": year,
        **forecast_metrics(frame["auction_volume"], frame["prediction"]),
    })

annual_metrics = pd.DataFrame(annual_rows)
annual_metrics["median_stock_rmse"] = annual_metrics.groupby("permno")["rmse_log"].transform("median")
annual_metrics["distance_from_median_rmse"] = np.abs(
    annual_metrics["rmse_log"] - annual_metrics["median_stock_rmse"]
)
selected_years = (
    annual_metrics.sort_values(["permno", "distance_from_median_rmse", "year"])
    .groupby("permno", as_index=False).first()
)
selected_years.to_csv(OUT / "representative_stock_years.csv", index=False)
annual_metrics.to_csv(OUT / "stock_year_daily_metrics.csv", index=False)
display(selected_years[[
    "ticker", "profile", "year", "observations", "rmse_log", "wape",
    "spearman", "aggregate_bias", "median_stock_rmse",
]])
'''),
    md(r'''
## Unsmoothed daily actual versus predicted values

Every line segment below is a daily value. Red crosses identify the five dates with the largest absolute error in log volume for that stock-year. Because the model is optimized in log space, these are the largest proportional-scale misses rather than necessarily the largest share-count misses.
'''),
    code(r'''
fig, axes = plt.subplots(3, 2, figsize=(19, 15))
largest_miss_rows = []

for ax, permno in zip(axes.flat, PERMNO_ORDER):
    selected = selected_years[selected_years["permno"] == permno].iloc[0]
    year = int(selected["year"])
    ticker, profile = PROFILES[permno]
    frame = predictions[(predictions["permno"] == permno) & (predictions["year"] == year)].copy()
    largest = frame.nlargest(5, "absolute_log_error")

    ax.plot(frame["date"], frame["auction_volume"], color="#1F4E79", linewidth=1.05,
            alpha=0.85, label="Actual daily volume")
    ax.plot(frame["date"], frame["prediction"], color="#E76F00", linewidth=1.05,
            alpha=0.85, label="Predicted daily volume")
    ax.scatter(largest["date"], largest["auction_volume"], marker="x", s=55,
               linewidths=1.8, color="#D62728", label="Five largest log errors", zorder=5)
    ax.set_yscale("log")
    ax.set_title(
        f"{ticker} — representative year {year}\n{profile} | "
        f"RMSE={selected['rmse_log']:.3f}, WAPE={selected['wape']:.1%}, "
        f"bias={selected['aggregate_bias']:.1%}"
    )
    ax.set_ylabel("Auction volume (shares, log scale)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for _, row in largest.iterrows():
        largest_miss_rows.append({
            "permno": permno, "ticker": ticker, "year": year, "date": row["date"],
            "auction_volume": row["auction_volume"], "prediction": row["prediction"],
            "absolute_log_error": row["absolute_log_error"],
            "prediction_to_actual_ratio": row["prediction"] / row["auction_volume"]
                if row["auction_volume"] > 0 else np.nan,
        })

fig.suptitle("Unsmoothed daily closing-auction forecasts in representative years", fontsize=18, y=1.01)
fig.tight_layout()
figure_path = OUT / "representative_stock_year_daily_actual_vs_predicted.png"
fig.savefig(figure_path, bbox_inches="tight")
plt.show()

largest_misses = pd.DataFrame(largest_miss_rows).sort_values(["ticker", "absolute_log_error"], ascending=[True, False])
largest_misses.to_csv(OUT / "representative_year_largest_daily_misses.csv", index=False)
display(largest_misses)
'''),
    code(r'''
artifacts = pd.DataFrame({
    "artifact": ["Representative years", "All stock-year metrics", "Largest daily misses", "Daily figure"],
    "path": [
        OUT / "representative_stock_years.csv", OUT / "stock_year_daily_metrics.csv",
        OUT / "representative_year_largest_daily_misses.csv", figure_path,
    ],
})
artifacts["path"] = artifacts["path"].astype(str)
artifacts.to_csv(OUT / "artifact_manifest.csv", index=False)
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
