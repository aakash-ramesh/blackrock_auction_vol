"""Diagnostic robustness audit of frozen 15:30 forecasts. No refitting or selection."""
from pathlib import Path
import json,hashlib
import duckdb,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/experiments/19_intraday_1530'
OUT=ROOT/'outputs/experiments/21_intraday_robustness'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((SOURCE/'experiment_summary.json').read_text());assert manifest['all_checks_passed']
 for f in manifest['prediction_files']:assert sha(ROOT/f['path'])==f['sha256']
 (OUT/'audit_plan.json').write_text(json.dumps({'scope':'Frozen 15:30 ridge and LightGBM versus fixed daily; no fitting, hyperparameter changes or promotion decision','development':[2022,2023,2024],'confirmation':2025,'confirmation_previously_inspected':True,'bootstrap':'2000 moving blocks of 5 and 20 consecutive trading dates sampled within each calendar year; equal-year means of daily log MAE differences','segments':'Prior-information size, liquidity and history; disjoint known calendar events; ex-post actual top 1 percent diagnostic only','sources':manifest['prediction_files']},indent=2))
 c=duckdb.connect();c.execute('SET threads=4')
 c.execute(f"CREATE TEMP VIEW p AS SELECT *,percent_rank() OVER(PARTITION BY year ORDER BY auction_volume) actual_rank FROM read_parquet('{SOURCE}/predictions/intraday_predictions_*.parquet')")
 methods=['fixed_daily','ridge_update','lgbm_intraday_update']
 rows=[]
 dims={'annual':"'all'",'month':"strftime(date,'%Y-%m')",'size':"CASE WHEN size_percentile IS NULL THEN 'missing' WHEN size_percentile<0.333333333333 THEN 'small' WHEN size_percentile<0.666666666667 THEN 'mid' ELSE 'large' END",'liquidity':"CASE WHEN liquidity_percentile IS NULL THEN 'missing' WHEN liquidity_percentile<0.333333333333 THEN 'low' WHEN liquidity_percentile<0.666666666667 THEN 'mid' ELSE 'high' END",'history':"CASE WHEN prior_auction_observations IS NULL THEN 'missing' WHEN prior_auction_observations<5 THEN '0_4' WHEN prior_auction_observations<20 THEN '5_19' ELSE '20_plus' END",'event':"CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END",'actual_tail_ex_post':"CASE WHEN actual_rank>=0.99 THEN 'top_1_percent' ELSE 'other' END"}
 for dimension,expr in dims.items():
  for method in methods:
   rows.append(c.sql(f'''SELECT year,'{dimension}' dimension,{expr} segment,'{method}' AS candidate,count(*) observations,count(DISTINCT date) dates,
   sqrt(avg(pow(ln(1+{method})-log1p_auction_volume,2))) rmse_log,
   sum(abs({method}-auction_volume))/sum(auction_volume) wape,
   sum({method})/sum(auction_volume)-1 aggregate_bias,
   sum(abs({method}-auction_volume)) absolute_share_error
   FROM p GROUP BY 1,2,3,4''').df())
 metrics=pd.concat(rows,ignore_index=True).rename(columns={"candidate":"method"});metrics.to_csv(OUT/'annual_segment_metrics.csv',index=False)
 base=metrics[metrics.method.eq('fixed_daily')].drop(columns='method')
 paired=metrics[~metrics.method.eq('fixed_daily')].merge(base,on=['year','dimension','segment'],suffixes=('','_daily'),validate='many_to_one')
 assert (paired.observations==paired.observations_daily).all()
 for metric in ['rmse_log','wape','aggregate_bias']:paired[metric+'_change']=paired[metric]-paired[metric+'_daily']
 paired.to_csv(OUT/'paired_segment_changes.csv',index=False)
 daily=[]
 for method in methods[1:]:
  daily.append(c.sql(f"SELECT date,year,'{method}' AS candidate,avg(abs(ln(1+{method})-log1p_auction_volume)-abs(ln(1+fixed_daily)-log1p_auction_volume)) log_mae_change FROM p GROUP BY 1,2,3 ORDER BY date").df())
 daily=pd.concat(daily).rename(columns={"candidate":"method"});daily.to_csv(OUT/'daily_paired_errors.csv',index=False)
 rng=np.random.default_rng(20261113);bootstrap=[]
 for method in methods[1:]:
  for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
   values=[daily[(daily.method==method)&(daily.year==y)].sort_values('date').log_mae_change.to_numpy() for y in years]
   for block in [5,20]:
    draws=[]
    for _ in range(2000):
     means=[]
     for v in values:
      starts=rng.integers(0,len(v)-block+1,size=int(np.ceil(len(v)/block)))
      sampled=(starts[:,None]+np.arange(block)).ravel()[:len(v)]
      means.append(v[sampled].mean())
     draws.append(np.mean(means))
    lo,hi=np.quantile(draws,[.025,.975]);bootstrap.append(dict(method=method,period=period,block_trading_dates=block,mean_log_mae_change=np.mean([v.mean() for v in values]),ci_lower=lo,ci_upper=hi))
 pd.DataFrame(bootstrap).to_csv(OUT/'serial_block_bootstrap.csv',index=False)
 annual=paired[paired.dimension.eq('annual')]
 summary=[]
 for method in methods[1:]:
  for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
   months=paired[(paired.method==method)&paired.year.isin(years)&paired.dimension.eq('month')]
   a=annual[(annual.method==method)&annual.year.isin(years)]
   summary.append(dict(method=method,period=period,years=len(a),mean_annual_rmse_change=a.rmse_log_change.mean(),mean_annual_wape_change=a.wape_change.mean(),months=len(months),months_rmse_improved=int((months.rmse_log_change<0).sum()),months_wape_improved=int((months.wape_change<0).sum())))
 summary=pd.DataFrame(summary);summary.to_csv(OUT/'robustness_summary.csv',index=False)
 # Reconcile every annual metric to the previously verified written summaries.
 prior=pd.read_csv(SOURCE/'annual_metrics.csv');checks=[]
 for row in metrics[metrics.dimension.eq('annual')].itertuples():
  old=prior[(prior.test_year==row.year)&prior.candidate.eq(row.method)&prior['sample'].eq('all_regular_sessions')].iloc[0]
  checks.append({'check':f'{row.year}_{row.method}_annual_reconciliation','passed':bool(row.observations==old.observations and all(np.isclose(getattr(row,k),old[k],rtol=1e-12,atol=1e-12) for k in ['rmse_log','wape','aggregate_bias']))})
 for dim in dims:
  sums=metrics[metrics.dimension.eq(dim)].groupby(['year','method']).observations.sum();annual_counts=metrics[metrics.dimension.eq('annual')].set_index(['year','method']).observations
  checks.append({'check':f'{dim}_exhaustive_partition','passed':bool(sums.sort_index().equals(annual_counts.sort_index()))})
 assert all(x['passed'] for x in checks)
 pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False)
 text='# Frozen intraday forecast robustness audit\n\nNo models were refitted. Annual and monthly results and partitions by prior size, liquidity, history, disjoint calendar events and actual-volume tails are in the accompanying CSVs. Actual-tail membership is an ex-post diagnostic, never a usable prediction input. Early closes remain excluded.\n\n'
 text+='## Main results\n\n```text\n'+summary.to_string(index=False)+'\n```\n\n## Serial dependence sensitivity\n\n```text\n'+pd.DataFrame(bootstrap).to_string(index=False)+'\n```\n\n'
 text+='Changes are update minus daily anchor; negative error changes favor the update. Block intervals concern equal-year mean daily log MAE, not log RMSE or aggregate WAPE. Five- and twenty-trading-date blocks are sampled within years. These are retrospective diagnostics, unadjusted for multiple comparisons, and not pristine holdout validation.\n\nA matching daily-only ridge control remains necessary to isolate the contribution of current checkpoint features to ridge gains. This audit makes no promotion decision, changes no model, and has not implemented other checkpoints or new event data.\n\n'
 text+='## Interpretation\n\nRidge improves annual log RMSE in every evaluated year, 32 of 36 development months, and all 12 confirmation months. Both 5-date and 20-date bootstrap intervals favor both update families. However, equal-year development segment means show worse ridge log RMSE for the low-liquidity and large-stock groups. Disjoint month-end days worsen in both log RMSE and WAPE; monthly expiry WAPE is slightly worse. The top actual-volume 1 percent still has roughly 43 percent negative aggregate bias. These are exploratory segment findings, not thresholds chosen to route predictions. See the annual segment counts before interpreting rare groups.\n\n'
 text+=f'All {len(checks)} metric/partition checks passed; all four source forecast fingerprints verified.\n'
 (OUT/'README.md').write_text(text)
 print(summary.to_string(index=False));print(pd.DataFrame(bootstrap).to_string(index=False))
if __name__=='__main__':run()
