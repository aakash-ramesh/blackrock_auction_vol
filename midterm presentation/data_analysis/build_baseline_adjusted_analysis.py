"""Retrospective covariate associations with the existing log-residual target.

Reads the isolated panel and saved 2022--2024 forecast keys. No fitting or
changes to the research pipeline. Missing stock medians are excluded here;
the main baseline comparison separately retains their historical fallback.
"""
from pathlib import Path
import sys
import json

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
REPO = ROOT.parents[1]
ASSETS = ROOT.parent / 'assets'
TABLES = ROOT / 'tables'
con = duckdb.connect()
con.execute('SET threads=4')
con.read_parquet(str(ROOT / 'cache/analysis_panel.parquet')).create_view('panel')
con.read_parquet(str(REPO / 'outputs/eda/09_global_finalists/predictions/global_finalist_predictions_*.parquet')).create_view('forecasts')
con.execute('''CREATE VIEW base AS
    SELECT p.*, f.auction_volume AS saved_actual
    FROM panel p JOIN forecasts f USING(date, permno)
    WHERE year(p.date) BETWEEN 2022 AND 2024''')
rows, keys, mismatches, missing = con.execute('''SELECT count(*), count(DISTINCT (date,permno)),
    count(*) FILTER(WHERE auction_volume != saved_actual),
    count(*) FILTER(WHERE median20_auction_volume IS NULL) FROM base''').fetchone()
assert rows == keys == 963525 and mismatches == 0 and missing == 616

features = [
    ('prior_raw_daily_volume', 'Prior-day trading volume'),
    ('prior_raw_market_cap_usd', 'Prior-day market capitalization'),
    ('lag1_quoted_spread', 'Prior-day relative quoted spread'),
    ('lag1_abs_return', 'Prior-day absolute return'),
]
curves, summaries = [], []
for feature, title in features:
    d = con.execute(f'''SELECT "{feature}" AS x, auction_volume AS actual,
        median20_auction_volume AS baseline,
        ln(1+auction_volume)-ln(1+median20_auction_volume) AS residual
        FROM base WHERE median20_auction_volume IS NOT NULL AND "{feature}" IS NOT NULL''').fetchdf()
    assert np.isfinite(d.to_numpy()).all()
    # Cut at observed quantiles: all identical input values remain together.
    # Duplicate boundaries are removed rather than arbitrarily splitting ties.
    edges = np.unique(d.x.quantile(np.linspace(.1, .9, 9)).to_numpy())
    d['decile'] = np.searchsorted(edges, d.x.to_numpy(), side='left') + 1
    g = d.groupby('decile', observed=True).agg(
        observations=('actual', 'size'), input_min=('x', 'min'), input_max=('x', 'max'),
        mean_actual_shares=('actual', 'mean'), median_actual_shares=('actual', 'median'),
        mean_raw_departure_shares=('actual', lambda x: float((x-d.loc[x.index, 'baseline']).mean())),
        mean_log_residual=('residual', 'mean'), median_log_residual=('residual', 'median'),
    ).reset_index()
    g.insert(0, 'feature', feature)
    assert g.observations.sum() == len(d)
    assert g.input_min.is_monotonic_increasing
    curves.append(g)
    xrank = d.x.rank(method='average')
    summaries.append({
        'feature': feature, 'label': title, 'observations': len(d), 'bins': len(g),
        'spearman_raw_target': xrank.corr(d.actual.rank(method='average')),
        'spearman_log_residual': xrank.corr(d.residual.rank(method='average')),
        'mean_log_residual_lowest_bin': g.iloc[0].mean_log_residual,
        'mean_log_residual_highest_bin': g.iloc[-1].mean_log_residual,
        'highest_minus_lowest_mean_log_residual': g.iloc[-1].mean_log_residual-g.iloc[0].mean_log_residual,
    })
curves = pd.concat(curves, ignore_index=True)
summary = pd.DataFrame(summaries)
curves.to_csv(TABLES / 'baseline_adjusted_covariate_curves.csv', index=False)
summary.to_csv(TABLES / 'baseline_adjusted_covariate_summary.csv', index=False)
metadata = {'years': [2022, 2024], 'forecast_stock_days': rows,
    'missing_stock_median_excluded': missing, 'baseline_available_rows': rows-missing,
    'target': 'log(1 + actual shares) - log(1 + prior-20-observation median shares)',
    'causal_or_incremental_prediction_test': False, 'models_fitted': 0,
    'binning': 'observed input quantiles; ties kept together; duplicate edges removed',
    'selection': 'four illustrative prior-close covariates, not a feature-selection screen'}
(ROOT / 'baseline_adjusted_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

navy, teal, muted = '#102C43', '#007F82', '#637482'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 16, 'axes.titlesize': 18,
    'axes.titleweight': 'bold', 'axes.labelsize': 15, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': navy, 'text.color': navy, 'axes.edgecolor': muted,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'pdf.fonttype': 42,
})
low = float(min(curves.mean_log_residual.min(), curves.median_log_residual.min(), 0))-.025
high = float(max(curves.mean_log_residual.max(), curves.median_log_residual.max(), 0))+.025


def render(specs, shape, name, figsize):
    fig, axes = plt.subplots(*shape, figsize=figsize, layout='constrained', sharey=True)
    for ax, (feature, title) in zip(np.ravel(axes), specs):
        d = curves[curves.feature.eq(feature)]
        ax.axhline(0, color=muted, ls='--', lw=1.4, label='Actual equals recent median')
        ax.plot(d.decile, d.mean_log_residual, marker='o', color=navy, lw=2.2, label='Mean log deviation')
        ax.plot(d.decile, d.median_log_residual, marker='s', color=teal, lw=2.2, label='Median log deviation')
        ax.set(title=title, xlabel='Input group (1 = lowest)', ylabel='Log deviation from recent median',
               xticks=[1, 2, 4, 6, 8, 10], ylim=(low, high))
        ax.grid(axis='y', alpha=.15)
    fig.legend(*np.ravel(axes)[0].get_legend_handles_labels(), ncol=3, frameon=False,
               loc='outside upper center', fontsize=12)
    for ext in ('pdf', 'png'):
        fig.savefig(ASSETS / f'{name}.{ext}', dpi=170, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


render(features[:2], (1, 2), 'analysis_baseline_adjusted', (13.6, 4.7))
render(features, (2, 2), 'analysis_baseline_adjusted_all', (13.6, 8.0))
lines = [
    '# Covariates and deviations from the recent auction median', '',
    'This supporting diagnostic uses the saved 2022–2024 development forecast universe. '
    'Of 963,525 stock-days, 616 lack a stock-specific trailing median and are excluded here. '
    'The main baseline-versus-ridge evaluation retains those cases with the saved historical fallback. '
    'Each input uses its own complete pairs, so its sample count can be smaller.', '',
    'The outcome is `r = log(1 + actual auction shares) − log(1 + prior-only 20-observation auction median)`. '
    'Zero means equality to the recent median; positive/negative values mean above/below it. '
    'This is the model’s existing target formulation before fallback and forecast clipping. '
    'It is neither a fitted-model error nor a causal estimate.', '',
    '| Prior-close input | Pairs | Rank correlation with auction shares | Rank correlation with log deviation |',
    '| --- | ---: | ---: | ---: |',
]
for row in summary.itertuples():
    lines.append(f'| {row.label} | {row.observations:,} | {row.spearman_raw_target:.3f} | {row.spearman_log_residual:.3f} |')
lines += ['', '## What the comparisons show', '',
    '- Prior-day market capitalization has a strong raw-volume rank association (0.634), but little association with the log deviation (0.012). Its adjusted curve is also much flatter than the raw-share curve. This is consistent with a substantial scale component in this sample, without identifying a causal mechanism.',
    '- Prior-day trading volume retains a modest increasing relationship with the adjusted outcome: rank correlation 0.075, compared with 0.857 for raw shares. A strong level relationship does not translate into the same strength for departures from the baseline.',
    '- The spread association also becomes weak on the adjusted outcome (−0.015). Absolute returns have a modest positive adjusted association (0.067), while their raw-share rank association is −0.053. This motivates testing the outcome relevant to the model without establishing incremental forecast gains.',
    '', '![Four baseline-adjusted covariate curves](../assets/analysis_baseline_adjusted_all.png)', '',
    '## Interpretation limits', '',
    '- These plots examine departures from the stock-specific baseline, supplementing the existing natural-unit target plots.',
    '- Removing a recent median does not control for calendar, stock, or market confounders. Differences across stocks and dates remain.',
    '- Inputs are correlated with each other and with the historical baseline. The curves do not estimate partial effects or incremental out-of-time predictive value.',
    '- Rank correlations with raw shares and log deviations use the same complete pairs. This makes the scale-versus-departure comparison consistent.',
    '- Log deviations are not share-volume percentage errors. Exponentiating an average log deviation does not yield an arithmetic mean volume ratio.',
    '- This is a retrospective diagnostic of development years already used in research. No new validation period, calibration correction, feature selection, or model fit is introduced.',
    '', '## Reproduction', '',
    'Run `build_baseline_adjusted_analysis.py` with pandas, NumPy, Matplotlib, and DuckDB. '
    'It reads the existing isolated `cache/analysis_panel.parquet` and saved global-forecast keys. '
    'If the cache is unavailable, rebuild it with `build_panels.py`. All outputs stay within the presentation folder.', '']
(ROOT / 'baseline_adjusted_covariates.md').write_text('\n'.join(lines))
print(summary.to_string(index=False))
