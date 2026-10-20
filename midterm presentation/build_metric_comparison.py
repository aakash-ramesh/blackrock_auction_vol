"""Add MAPE/MSPE sensitivity to the saved 2022--2024 comparisons; no fitting."""
from pathlib import Path
import json
import sys

try:
    import duckdb
except ImportError:
    sys.path.insert(0, '/tmp/blackrock_midterm_deps')
    import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'assets'
SOURCE = REPO / 'outputs/eda/09_global_finalists'
MODELS = [
    'rolling5_log_full', 'rolling5_residual_full',
    'rolling5_residual_recent_full',
    'rolling5_residual_auction_calendar_liquidity',
    'rolling5_residual_auction_calendar',
]
con = duckdb.connect()
con.read_parquet(str(SOURCE / 'predictions/global_finalist_predictions_*.parquet')).create_view('forecasts')
annual = []
for model in MODELS:
    d = con.execute(f'''SELECT '{model}' AS candidate, year(date) AS test_year,
        count(*) AS observations, count(*) FILTER(WHERE auction_volume=0) AS zero_actuals,
        count(*) FILTER(WHERE auction_volume>0) AS relative_error_observations,
        sqrt(avg(pow(ln(1+"{model}")-ln(1+auction_volume),2))) AS rmse_log,
        100*avg(abs("{model}"-auction_volume)/nullif(auction_volume,0)) AS mape_pct,
        avg(pow(("{model}"-auction_volume)/nullif(auction_volume,0),2)) AS mspe_ratio_squared,
        sum(abs("{model}"-auction_volume))/sum(auction_volume) AS wape
        FROM forecasts WHERE year(date) BETWEEN 2022 AND 2024 GROUP BY 2 ORDER BY 2''').fetchdf()
    assert d.test_year.tolist() == [2022, 2023, 2024]
    assert d.observations.tolist() == [328634, 318679, 316212]
    assert d.zero_actuals.eq(0).all()
    assert d.relative_error_observations.eq(d.observations).all()
    assert con.execute(f'''SELECT count(*) FROM forecasts WHERE year(date) BETWEEN 2022 AND 2024
        AND ("{model}" IS NULL OR NOT isfinite("{model}") OR auction_volume<0)''').fetchone()[0] == 0
    annual.append(d)
annual = pd.concat(annual, ignore_index=True)
saved = pd.read_csv(SOURCE / 'global_finalist_metrics.csv')
paired = annual.merge(saved[['candidate', 'test_year', 'rmse_log', 'wape']],
                      on=['candidate', 'test_year'], suffixes=('', '_saved'), validate='one_to_one')
for metric in ['rmse_log', 'wape']:
    assert np.allclose(paired[metric], paired[metric+'_saved'], atol=1e-12, rtol=1e-10)
summary = annual.groupby('candidate', sort=False)[['rmse_log', 'mape_pct', 'mspe_ratio_squared', 'wape']].mean().reset_index()
annual.to_csv(OUT / 'metric_comparison_annual.csv', index=False)
summary.to_csv(OUT / 'metric_comparison_summary.csv', index=False)
metadata = {
    'period': [2022, 2024], 'observations_per_candidate': 963525,
    'aggregation': 'arithmetic average of three annual metrics; not pooled metrics',
    'mape': '100 * mean(abs(prediction - actual) / actual)',
    'mspe': 'mean(((prediction - actual) / actual)^2); unscaled ratio-squared units',
    'zero_actual_policy': 'undefined relative errors excluded if present; zero actuals observed = 0',
    'new_fits': 0, 'saved_rmse_and_wape_reproduced': True,
}
(OUT / 'metric_comparison_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

# Quantify low-denominator sensitivity for supporting interpretation only.
# This concentration diagnostic pools development rows; it is not the table average.
sensitivity = []
for model in MODELS:
    d = con.execute(f'''SELECT '{model}' AS candidate,
        count(*) FILTER(WHERE auction_volume<=100) AS small_actual_observations,
        count(*) FILTER(WHERE auction_volume<=100)::DOUBLE/count(*) AS small_actual_share,
        sum(CASE WHEN auction_volume<=100 THEN pow(("{model}"-auction_volume)/auction_volume,2)
                 ELSE 0 END)/sum(pow(("{model}"-auction_volume)/auction_volume,2)) AS squared_relative_error_share
        FROM forecasts WHERE year(date) BETWEEN 2022 AND 2024''').fetchdf()
    sensitivity.append(d)
pd.concat(sensitivity, ignore_index=True).to_csv(OUT / 'metric_comparison_denominator_sensitivity.csv', index=False)

# Match the displayed stock-source counts to the requested common analysis period.
window = con.execute('''SELECT 'CRSP daily' AS source, count(*) AS rows,
    count(DISTINCT permno) AS securities, min(date) AS first_date, max(date) AS last_date
    FROM read_parquet(?, union_by_name=true) WHERE year(date) BETWEEN 2015 AND 2025''',
    [str(REPO / 'blackrock_data/stock_daily/crsp_daily/year=*/*.parquet')]).fetchdf()
window.to_csv(ROOT / 'sources/data_source_analysis_window.csv', index=False)
assert window.iloc[0].rows == 3594404 and window.iloc[0].securities == 2149
print(summary.to_string(index=False))
print(window.to_string(index=False))
