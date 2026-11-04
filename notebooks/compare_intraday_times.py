"""Compare saved noon and 15:30 predictions on identical stock-days."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/20_intraday_noon'
LATE=ROOT/'outputs/experiments/19_intraday_1530'

def run():
    con=duckdb.connect()
    paths=[OUT,LATE]
    metrics=[]
    for time,path in zip(['12:00','15:30'],paths):
        frame=pd.read_csv(path/'intraday_comparison.csv')
        frame.insert(0,'checkpoint',time)
        metrics.append(frame)
    pd.concat(metrics).to_csv(OUT/'checkpoint_time_comparison.csv',index=False)
    availability_checks=[]
    for checkpoint,path in zip(['12:00','15:30'],paths):
        # Materialize independently before comparing, avoiding window-query fusion.
        con.execute(f"""CREATE OR REPLACE TEMP TABLE source_availability AS
          SELECT p.date,p.permno,coalesce(c.checkpoint_data_found,false)
          AND NOT coalesce(c.checkpoint_after_market_close,true)
          AND NOT coalesce(p.is_early_close,true) AND c.cum_volume IS NOT NULL
          AND c.cum_volume>0 AS expected
          FROM read_parquet('{ROOT}/outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet') p
          LEFT JOIN (SELECT * FROM read_parquet('{ROOT}/blackrock_data/stock_intraday/trade_checkpoints/year=*/trade_checkpoints_*.parquet',union_by_name=true)
          WHERE checkpoint='{checkpoint}') c USING(date,permno)""")
        mismatches=con.sql(f"SELECT count(*) FROM source_availability a JOIN read_parquet('{path}/intraday_panel.parquet') n USING(date,permno) WHERE a.expected<>n.cp_available").fetchone()[0]
        assert mismatches==0
        availability_checks.append(dict(checkpoint=checkpoint,source_availability_mismatches=mismatches))
    pd.DataFrame(availability_checks).to_csv(OUT/'full_checkpoint_availability_audit.csv',index=False)
    rows=[]
    for year in range(2022,2026):
        noon=OUT/f'predictions/intraday_predictions_{year}.parquet'
        late=LATE/f'predictions/intraday_predictions_{year}.parquet'
        n=con.sql(f"SELECT count(*) FROM read_parquet('{noon}')").fetchone()[0]
        joined=con.sql(f'''SELECT count(*) FROM read_parquet('{noon}') n FULL JOIN read_parquet('{late}') l USING(date,permno)
            WHERE n.date IS NULL OR l.date IS NULL OR n.auction_volume<>l.auction_volume OR n.fixed_daily<>l.fixed_daily''').fetchone()[0]
        assert joined==0
        rows.append({'year':year,'rows':n,'keys_labels_daily_reference_exact':joined==0})
    pd.DataFrame(rows).to_csv(OUT/'timing_alignment_checks.csv',index=False)
    boot=[]
    for method in ['ridge_update','lgbm_intraday_update']:
        daily=con.sql(f'''SELECT n.date,year(n.date) test_year,
          avg(abs(ln(1+n.{method})-n.log1p_auction_volume)-abs(ln(1+l.{method})-l.log1p_auction_volume)) log_mae_delta,
          (sum(abs(n.{method}-n.auction_volume))-sum(abs(l.{method}-l.auction_volume)))/sum(n.auction_volume) daily_wape_delta
          FROM read_parquet('{OUT}/predictions/intraday_predictions_*.parquet') n
          JOIN read_parquet('{LATE}/predictions/intraday_predictions_*.parquet') l USING(date,permno)
          GROUP BY n.date ORDER BY n.date''').df()
        for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
            for metric in ['log_mae_delta','daily_wape_delta']:
                values=daily.loc[daily.test_year.isin(years),metric].to_numpy()
                rng=np.random.default_rng(20261018)
                draws=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
                lo,hi=np.quantile(draws,[.025,.975])
                boot.append(dict(method=method,period=period,metric=metric,noon_minus_1530=values.mean(),ci_lower=lo,ci_upper=hi))
    pd.DataFrame(boot).to_csv(OUT/'paired_checkpoint_time_bootstrap.csv',index=False)
    with (OUT/'README.md').open('a') as f:
        f.write('\n\n## Noon versus 15:30\n\nAll evaluation keys, targets and daily reference forecasts match exactly across the two studies. Historical sample keys and order originate from experiment 19, with unavailable noon rows removed before fitting. Noon features and their prior histories are rebuilt from noon aggregates. Historical checkpoint features in each daily-only control also correspond to its own checkpoint. This compares the complete forecast procedures; training features and occasional missing-row exclusions differ.\n\n'+'```text\n'+pd.concat(metrics).to_string(index=False)+'\n```'+'\n\nPaired differences are noon minus 15:30; negative favors noon. Whole-date bootstrap intervals describe log absolute error and daily WAPE, not log RMSE; serial dependence and model-selection uncertainty are not covered. See `paired_checkpoint_time_bootstrap.csv`.\n')
    print(pd.concat(metrics).to_string(index=False))
if __name__=='__main__': run()
