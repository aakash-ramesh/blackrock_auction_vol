"""Why the 50/50 blend of the CNN-LSTM hybrid and tuned LightGBM beats LightGBM (Notebook 16e).

Reads the 16e row-level predictions (git-ignored; rebuild by running Notebook 16e) and writes two
small CSVs used by the report and make_figures.py:
- data/blend_decomposition.csv: per-year RMSEs, error correlation, the break-even correlations and
  the variance formula check
- data/blend_weight_curve.csv: log RMSE for blend weights 0..1 on LightGBM (hindsight diagnostic)

Usage (from the repository root):
    python reports/daily_model_experiments/analyse_blend.py [predictions_dir]
"""

from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREDICTIONS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/eda/16e_sequence_neural_networks/predictions"
TREE, NET, BLEND = "lgbm_61_ratios_large_tuned", "nn_cnn_lstm_hybrid", "blend_hybrid_lgbm"

errors = duckdb.sql(f"""
    SELECT year(date) AS year,
           ln(1 + {TREE}) - log1p_auction_volume AS e_tree,
           ln(1 + {NET}) - log1p_auction_volume AS e_net,
           ln(1 + {BLEND}) - log1p_auction_volume AS e_blend
    FROM read_parquet('{PREDICTIONS}/sequence_nn_predictions_*.parquet')
""").df()

rows, curves = [], []
groups = [(str(y), g) for y, g in errors.groupby("year")] + [("2022-2024 pooled", errors[errors.year < 2025])]
for label, g in groups:
    s1, s2 = np.sqrt(np.mean(g.e_tree ** 2)), np.sqrt(np.mean(g.e_net ** 2))
    rho = np.mean(g.e_tree * g.e_net) / (s1 * s2)  # correlation of errors about zero (what RMSE uses)
    formula = np.sqrt(0.25 * s1 ** 2 + 0.25 * s2 ** 2 + 0.5 * rho * s1 * s2)
    weights = np.round(np.linspace(0, 1, 101), 2)
    rmse = [np.sqrt(np.mean((w * g.e_tree + (1 - w) * g.e_net) ** 2)) for w in weights]
    curves += [{"period": label, "weight_on_lightgbm": w, "rmse_log": r} for w, r in zip(weights, rmse)]
    rows.append({
        "period": label, "rows": len(g),
        "rmse_lightgbm": s1, "rmse_hybrid": s2, "error_correlation": rho,
        "breakeven_any_blend": s1 / s2,                                  # some blend beats LightGBM if rho < this
        "breakeven_50_50": (3 * s1 ** 2 - s2 ** 2) / (2 * s1 * s2),       # the 50/50 blend beats LightGBM if rho < this
        "rmse_blend_50_50": np.sqrt(np.mean(g.e_blend ** 2)), "rmse_blend_formula": formula,
        "hindsight_best_weight_on_lightgbm": weights[int(np.argmin(rmse))], "hindsight_best_rmse": min(rmse),
        "mean_error_lightgbm": g.e_tree.mean(), "mean_error_hybrid": g.e_net.mean(), "mean_error_blend": g.e_blend.mean(),
        "share_opposite_sign_errors": float(np.mean(np.sign(g.e_tree) != np.sign(g.e_net))),
    })

out = HERE / "data"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "blend_decomposition.csv", index=False)
pd.DataFrame(curves).to_csv(out / "blend_weight_curve.csv", index=False)
print(pd.DataFrame(rows).set_index("period").T.round(5).to_string())
