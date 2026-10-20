"""Render saved forecasts and split metrics without refitting research models."""
from pathlib import Path
import sys

try:
    import matplotlib
    import duckdb
except ImportError:
    sys.path.insert(0, '/tmp/blackrock_midterm_deps')
    import matplotlib
    import duckdb
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'assets'
NAVY, TEAL, MUTED, RED = '#102C43', '#007F82', '#637482', '#B24738'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 17, 'axes.titlesize': 18,
    'axes.titleweight': 'bold', 'axes.labelsize': 16,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.edgecolor': MUTED,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'pdf.fonttype': 42,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=170, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


selection = pd.read_csv(REPO / 'outputs/eda/13_representative_stock_year_daily/representative_stock_years.csv')
examples = selection.set_index('ticker').loc[['JPM', 'CATO']].reset_index()
path = REPO / 'outputs/eda/12_selected_stock_predictions/selected_stock_daily_walk_forward_predictions.parquet'
con = duckdb.connect()
frames = []
case_metrics = []
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), layout='constrained')
for ax, row in zip(axes, examples.itertuples()):
    d = con.execute('''SELECT date, permno, ticker, auction_volume, prediction, median20_auction_volume
        FROM read_parquet(?) WHERE permno = ? AND year(date) = ?
        AND prediction IS NOT NULL ORDER BY date''', [str(path), int(row.permno), int(row.year)]).fetchdf()
    assert len(d) == row.observations
    err = np.log1p(d.prediction) - np.log1p(d.auction_volume)
    wape = np.abs(d.prediction - d.auction_volume).sum() / d.auction_volume.sum()
    rmse = np.sqrt(np.mean(err ** 2))
    assert np.isclose(wape, row.wape) and np.isclose(rmse, row.rmse_log)
    assert d.median20_auction_volume.notna().all()
    baseline_wape = np.abs(d.median20_auction_volume - d.auction_volume).sum() / d.auction_volume.sum()
    case_metrics.append({'ticker': row.ticker, 'year': row.year, 'observations': len(d),
                         'ridge_wape': wape, 'median_rule_wape': baseline_wape,
                         'relative_wape_improvement_pct': 100 * (1 - wape / baseline_wape)})
    d['absolute_log_error'] = np.abs(err)
    d['five_largest_log_errors'] = False
    worst = d.nlargest(5, 'absolute_log_error')
    d.loc[worst.index, 'five_largest_log_errors'] = True
    frames.append(d)
    ax.plot(d.date, d.auction_volume, color=NAVY, lw=1.2, label='Actual')
    ax.plot(d.date, d.prediction, color=TEAL, lw=1.5, label='Forecast')
    ax.scatter(worst.date, worst.auction_volume, color=RED, marker='x', s=40,
               linewidths=1.6, zorder=4, label='Five largest log errors')
    ax.set_yscale('log')
    ax.set_title(f'{row.ticker}, {row.year}')
    ax.set_xlim(pd.Timestamp(int(row.year), 1, 1), pd.Timestamp(int(row.year), 12, 31))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: f'{y / 1e6:g}m' if y >= 1e6 else f'{y / 1e3:g}k'))
    ax.grid(axis='y', which='major', alpha=.15)
    ax.set_xlabel('Trading date')
    ax.set_ylabel('Auction shares, logarithmic scale')
axes[0].legend(frameon=False, loc='upper right', fontsize=12)
pd.concat(frames, ignore_index=True).to_csv(OUT / 'backtest_daily_examples_inputs.csv', index=False)
examples.to_csv(OUT / 'backtest_example_selection.csv', index=False)
pd.DataFrame(case_metrics).to_csv(OUT / 'backtest_example_baseline_comparison.csv', index=False)
save(fig, 'backtest_daily_examples')

# Preserve the original display smoother: trailing 20 stock observations,
# at least five non-missing values, calculated separately for each series.
history = con.execute('''SELECT date, permno, ticker, auction_volume, prediction
    FROM read_parquet(?) WHERE permno IN (?, ?) ORDER BY permno, date''',
    [str(path), int(examples.iloc[0].permno), int(examples.iloc[1].permno)]).fetchdf()
for source, target in [('auction_volume', 'actual_median20'), ('prediction', 'forecast_median20')]:
    history[target] = history.groupby('permno')[source].transform(
        lambda values: values.rolling(20, min_periods=5).median())
history.to_csv(OUT / 'backtest_rolling_median_inputs.csv', index=False)
assert history.loc[history.date.dt.year.eq(2015), 'prediction'].isna().all()
assert history.loc[history.date.dt.year.ge(2016), 'prediction'].notna().all()
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), layout='constrained')
for ax, ticker in zip(axes, examples.ticker):
    d = history[history.ticker.eq(ticker)]
    ax.plot(d.date, d.actual_median20, color=NAVY, lw=2, label='Actual: 20-day median')
    ax.plot(d.date, d.forecast_median20, color=TEAL, lw=2, label='Forecast: 20-day median')
    ax.axvspan(pd.Timestamp(2015, 1, 1), pd.Timestamp(2016, 1, 1), color=MUTED, alpha=.13)
    ax.text(pd.Timestamp(2015, 7, 1), .96, 'Warm-up', ha='center', va='top', fontsize=12,
            color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: f'{y / 1e6:g}m' if y >= 1e6 else f'{y / 1e3:g}k'))
    ax.set(title=ticker, ylabel='Auction shares, logarithmic scale', xlabel='Year',
           xlim=(pd.Timestamp(2015, 1, 1), pd.Timestamp(2025, 12, 31)))
    ax.set_xticks([pd.Timestamp(year, 1, 1) for year in [2015, 2017, 2019, 2021, 2023, 2025]])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', which='major', alpha=.15)
axes[0].legend(frameon=False, loc='upper right', fontsize=12)
save(fig, 'backtest_rolling_medians')

metrics = pd.read_csv(REPO / 'outputs/eda/11_frozen_global_backtest/frozen_global_train_validation_test_metrics.csv')
metrics.to_csv(OUT / 'backtest_split_metrics_inputs.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), layout='constrained')
styles = [('Train (in-sample)', NAVY, 'o', 'Training (in sample)'),
          ('Validation', '#BC7938', '^', 'Historical validation'),
          ('Test', TEAL, 's', 'Forward evaluation')]
for ax, metric, factor, title, ylabel in [
    (axes[0], 'rmse_log', 1, 'Log-space forecast error', 'RMSE of log(1 + auction shares)'),
    (axes[1], 'wape', 100, 'Error measured in shares', 'WAPE (%)'),
]:
    for split, color, marker, label in styles:
        d = metrics[metrics.split.eq(split)].sort_values('test_year')
        assert d.test_year.tolist() == [2022, 2023, 2024, 2025]
        ax.plot(d.test_year, factor * d[metric], color=color, marker=marker, lw=2.3, label=label)
    ax.axvline(2024.5, color=MUTED, ls='--', lw=1.2)
    ax.set_xticks([2022, 2023, 2024, 2025], ['2022', '2023', '2024', '2025\nconfirmation'])
    ax.set(title=title, ylabel=ylabel, xlim=(2021.85, 2025.2))
    ax.grid(axis='y', alpha=.15)
axes[0].set_ylim(.41, .68)
axes[1].set_ylim(32, 50)
axes[0].legend(frameon=False, loc='upper right', fontsize=12)
save(fig, 'backtest_train_validation_test')
print('Built three backtest figures from saved forecasts and metrics. No model fitting.')
