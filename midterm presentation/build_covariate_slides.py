"""Create three additional individual-covariate exhibits from saved diagnostics.

Targets remain in shares. No fitting, feature selection, or causal estimation.
"""
from pathlib import Path
import sys

try:
    import matplotlib
except ImportError:
    sys.path.insert(0, '/tmp/blackrock_midterm_deps')
    import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'data_analysis/tables'
OUT = ROOT / 'assets'
NAVY, TEAL = '#102C43', '#007F82'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 17, 'axes.titlesize': 18,
    'axes.titleweight': 'bold', 'axes.labelsize': 16,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.edgecolor': '#637482',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'pdf.fonttype': 42,
})
panel = pd.read_csv(TABLES / 'panel_bin_curves.csv')
native = pd.read_csv(TABLES / 'native_bin_curves.csv')
specs = [
    ('covariates_size_spread', 'stock_date', [
        ('prior_raw_market_cap_usd', 'Prior-day market capitalization', 'Input decile (1 = smallest)', 800_000),
        ('lag1_quoted_spread', 'Prior-day relative quoted spread', 'Input decile (1 = tightest spread)', 300_000),
    ]),
    ('covariates_returns', 'stock_date', [
        ('prior_raw_return_total', 'Prior-day signed stock return', 'Return decile (1 = most negative)', 240_000),
        ('lag1_abs_return', 'Prior-day absolute stock return', 'Return-size decile (1 = smallest)', 240_000),
    ]),
    ('covariates_market_sector', 'native', [
        ('prior_raw_market_total_volume', 'Prior-day market trading volume', 'Market-volume decile (1 = lowest)', 320_000),
        ('prior_raw_sector_total_volume', 'Prior-day sector trading volume', 'Sector-volume decile (1 = lowest)', 320_000),
    ]),
]
inputs = []
features = []
for name, unit, pairs in specs:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), layout='constrained')
    for ax, (feature, title, xlabel, ceiling) in zip(axes, pairs):
        d = (panel if unit == 'stock_date' else native)
        d = d[d.feature.eq(feature)].sort_values('bin_order').copy()
        assert len(d) == 10 and d.bin_order.tolist() == list(range(1, 11)), feature
        d['slide_asset'] = name
        d['curve_source'] = 'panel_bin_curves' if unit == 'stock_date' else 'native_bin_curves'
        inputs.append(d)
        features.append(feature)
        mean_label = 'Mean auction' if unit == 'stock_date' else 'Mean group outcome'
        median_label = 'Median auction' if unit == 'stock_date' else 'Median group outcome'
        ax.plot(d.bin_order, d.mean_auction_shares, color=NAVY, marker='o', lw=2.5, label=mean_label)
        ax.plot(d.bin_order, d.median_auction_shares, color=TEAL, marker='s', lw=2.5, label=median_label)
        ax.set(title=title, xlabel=xlabel, ylim=(0, ceiling))
        ax.set_xticks([1, 2, 4, 6, 8, 10])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y / 1000:,.0f}k'))
        ax.set_ylabel('Actual auction volume (shares)' if unit == 'stock_date'
                      else 'Average auction shares per stock')
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(frameon=False, loc='upper left', fontsize=13)
    for extension in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=170, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)

pd.concat(inputs, ignore_index=True).to_csv(OUT / 'covariate_extension_plot_inputs.csv', index=False)
associations = pd.read_csv(TABLES / 'feature_associations.csv')
associations[associations.feature.isin(features)].to_csv(OUT / 'covariate_extension_associations.csv', index=False)
print('Built three two-panel covariate figures from saved diagnostics; no model fitting.')
