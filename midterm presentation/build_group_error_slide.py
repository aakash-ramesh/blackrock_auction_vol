"""Summarize saved ridge forecasts, baseline comparisons, and calibration.

Use development years 2022--2024 for the slide. Reproduce the original
2022--2025 diagnostic first to check definitions and aggregation.
"""
from pathlib import Path
import json
import sys

try:
    import duckdb
    import matplotlib
except ImportError:
    sys.path.insert(0, '/tmp/blackrock_midterm_deps')
    import duckdb
    import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'assets'
MODEL = 'rolling5_residual_auction_calendar_liquidity'
SOURCE = REPO / 'outputs/eda/09_global_finalists'
con = duckdb.connect()
con.read_parquet(str(SOURCE / 'predictions/global_finalist_predictions_*.parquet')).create_view('raw_forecasts')
manifest_path = REPO / 'outputs/eda/07_temporal_baselines/preprocessing_manifest.json'
manifest = json.loads(manifest_path.read_text())
fallbacks = pd.DataFrame([
    {'test_year': int(year), 'fallback': values['training_global_median_raw_target']}
    for year, values in manifest.items()
])
con.register('baseline_fallbacks', fallbacks)
con.execute('''CREATE VIEW forecasts AS SELECT f.*,
    coalesce(median20_auction_volume, b.fallback) AS median_rule
    FROM raw_forecasts f JOIN baseline_fallbacks b ON year(f.date) = b.test_year''')
fallbacks.to_csv(OUT / 'group_error_baseline_fallbacks.csv', index=False)


def summarize(last_year):
    return con.execute(f"""
    WITH base AS (
      SELECT *,
        CASE WHEN size_percentile < 1.0/3 THEN 'small'
             WHEN size_percentile < 2.0/3 THEN 'mid' ELSE 'large' END AS size_group,
        CASE WHEN prior_auction_observations < 5 THEN 'cold_0_4'
             WHEN prior_auction_observations < 20 THEN 'developing_5_19'
             ELSE 'seasoned_20_plus' END AS history_group
      FROM forecasts WHERE year(date) BETWEEN 2022 AND {last_year}
    ), grouped AS (
      SELECT *, 'size' AS dimension, size_group AS segment FROM base
      UNION ALL
      SELECT *, 'history' AS dimension, history_group AS segment FROM base
    )
    SELECT dimension, segment, count(*) AS observations,
      count(DISTINCT permno) AS stocks,
      sqrt(avg(pow(ln(1 + "{MODEL}") - log1p_auction_volume, 2))) AS rmse_log,
      sum(abs("{MODEL}" - auction_volume)) / sum(auction_volume) AS wape,
      sum("{MODEL}") / sum(auction_volume) - 1 AS aggregate_bias,
      sum(abs(median_rule - auction_volume)) / sum(auction_volume) AS median_rule_wape,
      sum(median_rule) / sum(auction_volume) - 1 AS median_rule_bias
    FROM grouped GROUP BY dimension, segment ORDER BY dimension, segment
    """).fetchdf()


original = pd.read_csv(SOURCE / 'champion_segment_and_tail_diagnostics.csv')
original = original[original.dimension.isin(['size', 'history'])].set_index(['dimension', 'segment'])
reproduced = summarize(2025).set_index(['dimension', 'segment']).loc[original.index]
for col in ['observations', 'rmse_log', 'wape', 'aggregate_bias']:
    assert np.allclose(reproduced[col], original[col], rtol=1e-10, atol=1e-12), col

data = summarize(2024)
data.insert(0, 'period', '2022–2024')
data['wape_reduction_pp'] = 100 * (data.median_rule_wape - data.wape)
data['relative_wape_reduction_pct'] = 100 * (1 - data.wape / data.median_rule_wape)
data.to_csv(OUT / 'group_error_development_inputs.csv', index=False)
assert data.groupby('dimension').observations.sum().nunique() == 1
assert con.execute(f'SELECT count(*) FROM forecasts WHERE year(date) BETWEEN 2022 AND 2024 AND '
                   f'("{MODEL}" IS NULL OR auction_volume IS NULL OR size_percentile IS NULL '
                   'OR prior_auction_observations IS NULL)').fetchone()[0] == 0

# Recover the exact deployable baseline used by the existing annual comparison,
# including saved historical fallback values; do not silently drop cold starts.
annual = con.execute(f'''SELECT year(date) AS test_year, count(*) AS observations,
    count(*) FILTER(WHERE median20_auction_volume IS NULL) AS fallback_rows,
    sum(abs(median_rule - auction_volume))/sum(auction_volume) AS baseline_wape,
    sum(abs("{MODEL}" - auction_volume))/sum(auction_volume) AS ridge_wape,
    sum("{MODEL}")/sum(auction_volume)-1 AS ridge_bias
    FROM forecasts GROUP BY 1 ORDER BY 1''').fetchdf()
original_baseline = pd.read_csv(REPO / 'outputs/eda/07_temporal_baselines/overall_baseline_metrics.csv')
original_baseline = original_baseline[original_baseline.model.eq('median20_fallback')].set_index('test_year')
assert np.allclose(annual.baseline_wape, original_baseline.loc[annual.test_year, 'wape'], rtol=1e-10)
assert np.array_equal(annual.observations, original_baseline.loc[annual.test_year, 'observations'])
annual.to_csv(OUT / 'forecast_diagnostic_annual_checks.csv', index=False)

navy, teal, muted = '#102C43', '#007F82', '#637482'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 17, 'axes.titlesize': 19,
    'axes.titleweight': 'bold', 'axes.labelsize': 16, 'xtick.labelsize': 16,
    'ytick.labelsize': 15, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': navy, 'text.color': navy, 'axes.edgecolor': muted,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'pdf.fonttype': 42,
})
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.7), layout='constrained', sharey=True)
for ax, dimension, groups, labels, title in [
    (axes[0], 'size', ['small', 'mid', 'large'], ['Small', 'Mid', 'Large'], 'Stock size'),
    (axes[1], 'history', ['cold_0_4', 'developing_5_19', 'seasoned_20_plus'],
     ['0–4', '5–19', '20+'], 'Prior observed auctions'),
]:
    d = data[data.dimension.eq(dimension)].set_index('segment').loc[groups]
    x = np.arange(3)
    for offset, values, color, label in [
        (-.18, 100*d.median_rule_wape, navy, '20-day-median rule'),
        (.18, 100*d.wape, teal, 'Ridge benchmark'),
    ]:
        bars = ax.bar(x+offset, values, color=color, width=.33, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+2, f'{value:.1f}',
                    ha='center', va='bottom', weight='bold', fontsize=13)
    ax.set_xticks(np.arange(3), [f'{label}\nn = {n:,}' for label, n in zip(labels, d.observations)])
    ax.set_title(title, pad=16)
    ax.set(ylim=(0, 160), yticks=[0, 40, 80, 120, 160])
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Pooled WAPE, 2022–2024 (%)\nLower is better')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2,
           frameon=False, fontsize=14)
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'backtest_group_errors.{ext}', dpi=170, bbox_inches='tight', pad_inches=.08)
plt.close(fig)
history = data[data.dimension.eq('history')]
metadata = {
    'model': MODEL, 'period': [2022, 2024], 'aggregation': 'pooled stock-day WAPE',
    'total_stock_days': int(history.observations.sum()),
    'under_20_history_share': float(history.loc[~history.segment.eq('seasoned_20_plus'), 'observations'].sum()
                                    / history.observations.sum()),
    'size_definition': 'daily cross-sectional prior-day market-cap percentile, cut at 1/3 and 2/3',
    'history_definition': 'count of nonmissing observed auction targets before forecast day',
    'original_2022_2025_diagnostics_reproduced': True,
    'baseline_annual_metrics_reproduced': True,
    'baseline_fallback_rows_2022_2024': int(annual.loc[annual.test_year.le(2024), 'fallback_rows'].sum()),
}
(OUT / 'group_error_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(data.to_string(index=False))
print(json.dumps(metadata, indent=2))

# Calibration groups are defined by predictions, not by realized outcomes.
# Rank once in the pooled development sample, with date/key tie-breaks.
calibration = con.execute(f'''
    WITH ranked AS (
      SELECT *, ntile(10) OVER(ORDER BY "{MODEL}", date, permno) AS prediction_decile
      FROM forecasts WHERE year(date) BETWEEN 2022 AND 2024
    ) SELECT prediction_decile, count(*) AS observations,
      avg("{MODEL}") AS mean_prediction, avg(auction_volume) AS mean_actual,
      median("{MODEL}") AS median_prediction, median(auction_volume) AS median_actual,
      sum("{MODEL}")/sum(auction_volume) AS predicted_actual_ratio
    FROM ranked GROUP BY 1 ORDER BY 1''').fetchdf()
calibration.to_csv(OUT / 'calibration_development_inputs.csv', index=False)
pooled = con.execute(f'''SELECT count(*) AS observations, sum(auction_volume) AS actual_shares,
    sum("{MODEL}") AS predicted_shares,
    sum("{MODEL}")/sum(auction_volume)-1 AS aggregate_bias,
    sum(abs("{MODEL}"-auction_volume))/sum(auction_volume) AS wape
    FROM forecasts WHERE year(date) BETWEEN 2022 AND 2024''').fetchdf().iloc[0].to_dict()
pooled['observations'] = int(pooled['observations'])
pooled['all_decile_means_underpredicted'] = bool((calibration.mean_prediction < calibration.mean_actual).all())
(OUT / 'calibration_development_metrics.json').write_text(json.dumps(pooled, indent=2)+'\n')
assert np.isclose((calibration.mean_actual*calibration.observations).sum(), pooled['actual_shares'])
assert np.isclose((calibration.mean_prediction*calibration.observations).sum(), pooled['predicted_shares'])

from matplotlib.ticker import FuncFormatter
def shares(value, _):
    return f'{value/1e6:g}m' if value >= 1e6 else f'{value/1e3:g}k'

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.7), layout='constrained')
ax = axes[0]
limits = (1000, 2_000_000)
ax.plot(limits, limits, ls='--', color=muted, lw=1.8, label='Perfect mean calibration')
ax.plot(calibration.mean_prediction, calibration.mean_actual, color=teal, marker='o', lw=2.1,
        label='Prediction-decile averages')
for decile in [1, 5, 10]:
    row = calibration[calibration.prediction_decile.eq(decile)].iloc[0]
    ax.annotate(str(decile), (row.mean_prediction, row.mean_actual), xytext=(8, 4),
                textcoords='offset points', fontsize=14, color=navy)
ax.set(xscale='log', yscale='log', xlim=limits, ylim=limits,
       title='Mean volume by prediction decile', xlabel='Mean forecast (shares, log scale)',
       ylabel='Mean actual (shares, log scale)')
ax.xaxis.set_major_formatter(FuncFormatter(shares))
ax.yaxis.set_major_formatter(FuncFormatter(shares))
ax.legend(frameon=False, loc='lower right', fontsize=12)
ax.grid(alpha=.15)
ax = axes[1]
ax.plot(calibration.prediction_decile, calibration.predicted_actual_ratio,
        color=teal, marker='o', lw=2.5)
ax.axhline(1, ls='--', color=muted, lw=1.8)
ax.text(1.2, 1.015, 'Perfect mean calibration = 1', fontsize=13, color=muted)
ax.set(title='Forecast / actual volume', xlabel='Prediction decile (1 = lowest forecast)',
       ylabel='Total forecast / total actual shares', xticks=[1, 2, 4, 6, 8, 10],
       ylim=(.35, 1.1), xlim=(.7, 10.3))
ax.grid(axis='y', alpha=.15)
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'backtest_calibration.{ext}', dpi=170, bbox_inches='tight', pad_inches=.08)
plt.close(fig)
print('Calibration metrics:', json.dumps(pooled))
