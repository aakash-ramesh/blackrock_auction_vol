"""Build presentation exhibits from saved diagnostics and forward model results.

No models are fitted here. Selection comparisons use 2022--2024 only.
"""
from pathlib import Path
import json
import sys

try:
    import matplotlib
except ImportError:
    sys.path.insert(0, '/tmp/blackrock_midterm_deps')
    import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'assets'
TABLES = ROOT / 'data_analysis' / 'tables'
NAVY, TEAL, MUTED = '#102C43', '#007F82', '#637482'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 17, 'axes.titlesize': 18,
    'axes.titleweight': 'bold', 'axes.labelsize': 16,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': NAVY, 'text.color': NAVY,
    'axes.edgecolor': MUTED, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'pdf.fonttype': 42,
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


curves = pd.read_csv(TABLES / 'panel_bin_curves.csv')
features = ['median20_auction_volume', 'prior_raw_daily_volume']
selected_curves = curves[curves.feature.isin(features)].copy()
selected_curves.to_csv(OUT / 'analysis_relationship_inputs.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), layout='constrained', sharey=True)
for ax, feature, title in zip(axes, features, [
    'Trailing 20-day auction median', 'Prior-day daily trading volume'
]):
    d = selected_curves[selected_curves.feature.eq(feature)].sort_values('bin_order')
    assert len(d) == 10 and d.bin_order.is_unique
    ax.plot(d.bin_order, d.mean_auction_shares, color=NAVY, marker='o', lw=2.5, label='Mean auction')
    ax.plot(d.bin_order, d.median_auction_shares, color=TEAL, marker='s', lw=2.5, label='Median auction')
    ax.set(title=title, xlabel='Input decile (1 = lowest)', ylim=(0, 1_100_000))
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y / 1000:,.0f}k'))
    ax.grid(axis='y', alpha=.15)
axes[0].set_ylabel('Actual auction volume (shares)')
axes[0].legend(frameon=False, loc='upper left', fontsize=15)
save(fig, 'analysis_individual_relationships')

family = pd.read_csv(REPO / 'outputs/eda/08_global_robustness/economic_feature_group_metrics.csv')
development = family[family.test_year.between(2022, 2024)]
assert set(development.test_year) == {2022, 2023, 2024}
means = development.groupby('specification')[['rmse_log', 'wape', 'rmse_delta_vs_full', 'wape_delta_vs_full']].mean()
means.to_csv(OUT / 'analysis_family_development.csv')
families = ['Auction history', 'Calendar', 'Liquidity', 'Market state',
            'Returns and price', 'Sector state', 'Size and history',
            'Security and corporate actions']
labels = ['Auction history', 'Calendar', 'Liquidity', 'Market context',
          'Returns / price', 'Sector context', 'Size / history', 'Security / corp. actions']
delta = means.loc[['full_drop__' + name for name in families], 'rmse_delta_vs_full'].to_numpy()
fig, ax = plt.subplots(figsize=(13.6, 4.6), layout='constrained')
y = np.arange(len(families))
ax.barh(y, delta, color=[TEAL, TEAL] + [MUTED] * 6, height=.62)
ax.axvline(0, color=NAVY, lw=1)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set(xlim=(-.004, .225), xlabel='Change in log RMSE after removing a family (higher = worse)')
ax.set_xticks([0, .05, .10, .15, .20])
for pos, value in zip(y, delta):
    ax.text(max(value, 0) + .003, pos, f'{value:+.5f}', va='center', fontsize=15)
ax.grid(axis='x', alpha=.12)
save(fig, 'analysis_family_ablation')

finalists = pd.read_csv(REPO / 'outputs/eda/09_global_finalists/global_finalist_selection.csv').set_index('candidate')
chosen = finalists.loc['rolling5_residual_auction_calendar_liquidity']
assert bool(chosen.selected_global_champion)
assert chosen.source_feature_count == 58
assert chosen.development_rmse_log - finalists.development_rmse_log.min() < .002
assert finalists.loc['rolling5_residual_auction_calendar', 'development_rmse_log'] - finalists.development_rmse_log.min() > .002
finalists.to_csv(OUT / 'analysis_finalist_inputs.csv')

performance = pd.read_csv(TABLES / 'historical_baseline_comparison.csv')
assert (performance.observations_baseline == performance.observations_selected).all()
performance.to_csv(OUT / 'analysis_performance_inputs.csv', index=False)
dev = performance[performance.test_year.between(2022, 2024)]
metrics = {
    'selection_years': [2022, 2023, 2024], 'confirmation_year': 2025,
    'development_mean_annual_log_rmse_baseline': float(dev.rmse_log_baseline.mean()),
    'development_mean_annual_log_rmse_selected': float(dev.rmse_log_selected.mean()),
    'development_mean_annual_wape_baseline': float(dev.wape_baseline.mean()),
    'development_mean_annual_wape_selected': float(dev.wape_selected.mean()),
    'direct_log_level_99_feature_rmse': float(finalists.loc['rolling5_log_full', 'development_rmse_log']),
    'residual_99_feature_rmse': float(finalists.loc['rolling5_residual_full', 'development_rmse_log']),
    'best_99_feature_residual_rmse': float(finalists.development_rmse_log.min()),
    'selected_58_feature_residual_rmse': float(chosen.development_rmse_log),
    'auction_calendar_42_feature_residual_rmse': float(finalists.loc['rolling5_residual_auction_calendar', 'development_rmse_log']),
    'practical_tolerance': .002,
}
(OUT / 'analysis_slide_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
fig, ax = plt.subplots(figsize=(13.6, 4.6), layout='constrained')
x = np.arange(len(performance))
for offset, column, color, label in [
    (-.18, 'wape_baseline', NAVY, 'Recent-median rule'),
    (.18, 'wape_selected', TEAL, 'Ridge benchmark'),
]:
    values = 100 * performance[column]
    ax.bar(x + offset, values, width=.34, color=color, label=label)
    for pos, value in zip(x + offset, values):
        ax.text(pos, value + .8, f'{value:.1f}%', ha='center', fontsize=16)
ax.axvline(2.5, color=MUTED, ls='--', lw=1.2)
ax.set(title='Annual WAPE (lower is better)', ylabel='WAPE (%)', ylim=(0, 59))
ax.set_xticks(x, ['2022', '2023', '2024', '2025\nconfirmation'])
ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=15)
ax.grid(axis='y', alpha=.15)
save(fig, 'analysis_forward_performance')
print(json.dumps(metrics, indent=2))
