"""Figures for the daily-model experiments report (Notebooks 16a-16e).

Reads compact outputs under outputs/eda and data/blend_*.csv (from analyse_blend.py).
Usage (from the repository root):
    python reports/daily_model_experiments/make_figures.py
"""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

EDA = ROOT / "outputs/eda"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
DEV = [2022, 2023, 2024]

# Reference palette, fixed categorical order (blue, orange, aqua, yellow); neutral gray for the baseline.
COLORS = {"Ridge benchmark": "#8a8984", "Tree models": "#2a78d6", "Regime features": "#eb6834",
          "Neural networks": "#1baf7a", "Blend": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def dev_and_confirmation(path):
    m = pd.read_csv(path)
    pivot = m.pivot_table(index="candidate", columns="test_year", values="rmse_log")
    return pd.DataFrame({"development": pivot[DEV].mean(axis=1), "confirmation": pivot[2025]})


frames = [dev_and_confirmation(EDA / f) for f in [
    "16a_lightgbm_improvements/candidate_metrics.csv", "16b_catboost_challenger/candidate_metrics.csv",
    "16d_hmm_regime_features/candidate_metrics.csv", "16e_sequence_neural_networks/candidate_metrics.csv"]]
scores = pd.concat(frames)
scores = scores[~scores.index.duplicated()]

MODELS = [  # (candidate, label, family)
    ("rolling5_residual_auction_calendar_liquidity", "Frozen ridge (benchmark)", "Ridge benchmark"),
    ("lgbm_58_control", "16a  LightGBM, Notebook 16 setup", "Tree models"),
    ("lgbm_61_ratios", "16a  + 3 ratio features", "Tree models"),
    ("lgbm_61_ratios_large", "16a  + 3x training rows", "Tree models"),
    ("lgbm_61_ratios_large_tuned", "16a  + wider tuning (champion)", "Tree models"),
    ("lgbm_61_ratios_large_linear", "16a  linear trees", "Tree models"),
    ("catboost_61_ratios_large", "16b  CatBoost", "Tree models"),
    ("lgbm_plus_regime_inputs", "16d  + raw regime inputs", "Regime features"),
    ("lgbm_plus_hmm_bic", "16d  + HMM, BIC states", "Regime features"),
    ("lgbm_plus_hmm_k3", "16d  + HMM, 3 states", "Regime features"),
    ("lgbm_plus_inputs_and_hmm_bic", "16d  + inputs and HMM", "Regime features"),
    ("nn_mlp_tabular", "16e  MLP on 61 features", "Neural networks"),
    ("nn_cnn_lstm_sequence", "16e  CNN-inception-LSTM, history only", "Neural networks"),
    ("nn_cnn_lstm_hybrid", "16e  CNN-inception-LSTM + features", "Neural networks"),
    ("blend_hybrid_lgbm", "16e  50/50 blend hybrid + LightGBM", "Blend"),
]
table = pd.DataFrame([{"candidate": c, "label": l, "family": f, **scores.loc[c].to_dict()} for c, l, f in MODELS])
ridge = table.loc[0, "development"]
table["change_vs_ridge"] = table["development"] - ridge
table.to_csv(HERE / "data" / "all_models_summary.csv", index=False)

# Figure 1: every model's 2022-2024 log RMSE change vs the frozen ridge.
fig, ax = plt.subplots(figsize=(7.4, 5.6))
plot = table.iloc[1:].iloc[::-1]
ax.barh(plot["label"], plot["change_vs_ridge"], color=[COLORS[f] for f in plot["family"]], height=0.62,
        edgecolor="white", linewidth=2)
champion = table.set_index("candidate").loc["lgbm_61_ratios_large_tuned", "change_vs_ridge"]
ax.axvline(0, color=MUTED, lw=0.8)
ax.axvline(champion, color=MUTED, lw=0.8, ls="--", zorder=0)
ax.text(champion, len(plot) - 0.35, "champion level ", fontsize=7.5, color=MUTED, va="bottom", ha="right")
for y, (value, dev) in enumerate(zip(plot["change_vs_ridge"], plot["development"])):
    ax.text(value - 0.0005, y, f"{dev:.4f}", va="center", ha="right", fontsize=7.5, color=INK,
            bbox=dict(fc="white", ec="none", pad=0.6))
ax.set_xlim(-0.031, 0.001)
ax.set_xlabel("Change vs frozen ridge (0.4879), mean 2022–2024 log RMSE; left is better\n"
              "Number beside each bar: the model's own log RMSE")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f]) for f in list(COLORS)[1:]]
ax.legend(handles, list(COLORS)[1:], loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=4, frameon=False, fontsize=8)
ax.set_title("Every daily-model experiment against the frozen ridge", loc="left", fontsize=11, color=INK)
plt.tight_layout()
plt.savefig(FIG / "fig1_all_models_vs_ridge.png", dpi=200)
plt.close()

# Figure 2: blend weight curves by year (hindsight diagnostic), marking the 50/50 and pure-LightGBM points.
curve = pd.read_csv(HERE / "data" / "blend_weight_curve.csv")
decomposition = pd.read_csv(HERE / "data" / "blend_decomposition.csv").set_index("period")
periods = ["2022", "2023", "2024", "2025"]
fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharex=True)
for ax, period in zip(axes, periods):
    c = curve[curve.period == period]
    ax.plot(c.weight_on_lightgbm, c.rmse_log, color=COLORS["Tree models"], lw=2)
    for w, name, dy in [(0.5, "50/50", 1), (1.0, "LightGBM", -1)]:
        r = c.loc[(c.weight_on_lightgbm - w).abs().idxmin(), "rmse_log"]
        ax.plot(w, r, "o", ms=6, color=COLORS["Blend"] if w == 0.5 else INK, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{name}\n{r:.4f}", (w, r), textcoords="offset points", xytext=(0, 10 * dy),
                    ha="center" if w == 0.5 else "right", va="bottom" if dy > 0 else "top", fontsize=7, color=INK)
    d = decomposition.loc[period]
    ax.set_title(f"{period}  (error correlation {d.error_correlation:.3f})", fontsize=8.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("weight on LightGBM")
    spread = c.rmse_log.max() - c.rmse_log.min()
    ax.set_ylim(c.rmse_log.min() - 0.35 * spread, c.rmse_log.max() + 0.15 * spread)
axes[0].set_ylabel("log RMSE")
fig.suptitle("Blend error vs weight: 0 = hybrid network only, 1 = LightGBM only", x=0.01, ha="left", fontsize=10.5, color=INK)
plt.tight_layout()
plt.savefig(FIG / "fig2_blend_weight_curves.png", dpi=200)
plt.close()
print(table[["label", "development", "confirmation", "change_vs_ridge"]].round(4).to_string(index=False))
