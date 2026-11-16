"""Historically tuned shrinkage of frozen 15:30 ridge toward its daily anchor."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,gc
import duckdb,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from notebooks.modeling_utils import fit_preprocessor,apply_preprocessor,fit_ridge,predict_ridge,forecast_metrics
SOURCE=ROOT/'outputs/experiments/19_intraday_1530';OUT=ROOT/'outputs/experiments/24_intraday_shrinkage'
VARIANTS=['global_shrink','month_end_shrink','low_liquidity_shrink','month_end_liquidity_shrink'];GRID=np.linspace(0,1,11)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return duckdb.read_parquet(str(p)).df()
def month_end(f):return (f.is_last_trading_day_month.fillna(False)&~f.monthly_options_expiry.fillna(False)&~f.quarterly_options_expiry.fillna(False)).to_numpy(bool)
def low(f):return f.liquidity_percentile.lt(1/3).fillna(False).to_numpy(bool)
def cells(f,variant):
 m,l=month_end(f),low(f)
 if variant=='global_shrink':return {'global':np.ones(len(f),bool)}
 if variant=='month_end_shrink':return {'month_end':m}
 if variant=='low_liquidity_shrink':return {'low_liquidity':l}
 return {'month_end_only':m&~l,'low_liquidity_only':l&~m,'month_end_and_low':m&l}
def blend(anchor,full,alpha):
 p=np.maximum(np.expm1(np.log1p(anchor)+alpha*(np.log1p(full)-np.log1p(anchor))),0)
 return np.where(alpha==0,anchor,np.where(alpha==1,full,p))
def run():
 for p in [OUT,OUT/'parameters',OUT/'predictions']:p.mkdir(parents=True,exist_ok=True)
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'source_experiment':19,'variants':VARIANTS,'alpha_grid':GRID.tolist(),'formula':'log1p(daily)+alpha*(log1p(bounded full ridge)-log1p(daily)); no second clipping; exact endpoints','month_end':'is_last_trading_day_month AND NOT monthly_options_expiry AND NOT quarterly_options_expiry, matching previous disjoint event diagnostics','low_liquidity':'prior-information liquidity_percentile < 1/3; missing percentile treated as not low and reported','segments':'Targeted variants retain full update outside target. Combined has separate month_end_only, low_only, intersection strengths. Sparse target cells use global strength.','fitting':'Rebuild original tuning ridge on saved tune rows with original selected penalty, predict saved historical validation rows; choose alpha minimizing validation mean squared log error; ties prefer alpha 1 (less change). Same historical validation selects original penalty and shrinkage, not a new nested calibration split. Final full ridge and anchor are unchanged saved forecasts.','minimum_cell_dates':8,'minimum_cell_rows':500,'chronology':'2022 validation Q4 2021; later years previous full year, using saved reservoir sample; only earlier labels fit alpha','fallback':'Exact daily fallback when checkpoint missing, inherited from both endpoints','evaluation':'Equal-year 2022-2024 development decision saved before scoring previously inspected 2025. No 2025 choice of variants or strength.','adoption_gate':'Versus full ridge: development mean log RMSE gain >=0.002, no annual worsening >0.002, no mean WAPE worsening, 20-date paired log-MAE upper CI <0. Versus daily: no annual log RMSE worsening >0.002, no mean WAPE worsening. Pick lowest development RMSE among passers, else retain full.','bootstrap':'2000 within-year moving 20-trading-date blocks; equal-year mean daily log-MAE difference; unadjusted exploratory multiple comparisons','limits':'Hypotheses were motivated by inspected development diagnostics, so this is retrospective research. No production claim, new external data or new ridge fitting policy. Q4 validation may force sparse month-end cell fallback.'}
 (OUT/'experiment_plan.json').write_text(json.dumps(plan,indent=2));checks=[];inputs=[];trials=[];parameters=[];annual=[];boot=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)));assert v,name
 manifest=json.loads((SOURCE/'experiment_summary.json').read_text());assert manifest['all_checks_passed']
 for r in manifest['prediction_files']:
  check('hash_'+r['path'],sha(ROOT/r['path'])==r['sha256']);inputs.append(r)
 for r in pd.read_csv(SOURCE/'training_sample_fingerprints.csv').itertuples():
  check('hash_'+r.path,sha(ROOT/r.path)==r.sha256);inputs.append(dict(path=r.path,sha256=r.sha256))
 c=duckdb.connect();c.execute('SET threads=4')
 def score_period(period,years):
  nonlocal boot
  a=pd.DataFrame(annual);a=a[a.test_year.isin(years)]
  summary=a.groupby('candidate')[['rmse_log','wape','aggregate_bias']].mean().reset_index();summary['period']=period
  paths=[str(OUT/f'predictions/shrinkage_predictions_{y}.parquet') for y in years]
  rng=np.random.default_rng(20261113)
  for name in VARIANTS:
   d=c.sql(f"SELECT date,year,avg(abs(ln(1+{name})-log1p_auction_volume)-abs(ln(1+ridge_update)-log1p_auction_volume)) delta FROM read_parquet({paths!r}) GROUP BY date,year ORDER BY date").df()
   d.to_csv(OUT/f'daily_errors_{name}_{period}.csv',index=False);vs=[d[d.year==y].delta.to_numpy() for y in years];draws=[]
   for _ in range(2000):
    means=[]
    for v in vs:
     starts=rng.integers(0,len(v)-19,size=int(np.ceil(len(v)/20)));ix=(starts[:,None]+np.arange(20)).ravel()[:len(v)];means.append(v[ix].mean())
    draws.append(np.mean(means))
   lo,hi=np.quantile(draws,[.025,.975]);boot.append(dict(candidate=name,period=period,reference='ridge_update',mean_log_mae_delta=float(np.mean([v.mean() for v in vs])),ci_lower=lo,ci_upper=hi))
  return summary
 def fit_year(year):
  print(f'{year}: fitting strengths on historical validation only',flush=True)
  train=read(SOURCE/f'samples/{year}_tune.parquet');val=read(SOURCE/f'samples/{year}_validation.parquet')
  prep=json.loads((SOURCE/f'models/preprocessor_{year}.json').read_text());types=[list(prep[k]) for k in ['continuous','boolean','categorical']];rec=json.loads((SOURCE/f'models/reconstruction_{year}.json').read_text())
  check(f'{year}_chronology',train.date.max()<val.date.min() and val.date.max()<pd.Timestamp(f'{year}-01-01'));check(f'{year}_validation_checkpoint_available',val.cp_available.all())
  xt,tp=fit_preprocessor(train,*types);xv=apply_preprocessor(val,tp,*types);coef=fit_ridge(xt,(train.log1p_auction_volume-train.log_daily_forecast).to_numpy(),rec['selected']['ridge_update']['parameter']);bounds=np.quantile(train.log1p_auction_volume,[.001,.999])
  v_log=np.clip(val.log_daily_forecast.to_numpy()+predict_ridge(xv,coef),*bounds)
  check(f'{year}_original_validation_reproduced',np.isclose(np.sqrt(np.mean((val.log1p_auction_volume.to_numpy()-v_log)**2)),rec['selected']['ridge_update']['validation_rmse_log'],rtol=1e-12))
  anchor=val.daily_forecast.to_numpy();full=np.maximum(np.expm1(v_log),0);truth=val.log1p_auction_volume.to_numpy()
  def fit_cell(label,mask):
   rows=[]
   for alpha in GRID:
    pred=blend(anchor[mask],full[mask],alpha);mse=float(np.mean((np.log1p(pred)-truth[mask])**2))
    row=dict(test_year=year,cell=label,alpha=float(alpha),validation_mse_log=mse,observations=int(mask.sum()),dates=int(val.loc[mask,'date'].nunique()));rows.append(row);trials.append(row)
   best=min(rows,key=lambda r:(r['validation_mse_log'],-r['alpha']));return best['alpha']
  global_alpha=fit_cell('global',np.ones(len(val),bool));fits={}
  for name in VARIANTS:
   fits[name]={}
   for label,mask in cells(val,name).items():
    count=int(mask.sum());dates=int(val.loc[mask,'date'].nunique());sufficient=count>=500 and dates>=8
    alpha=global_alpha if label=='global' or not sufficient else fit_cell(label,mask)
    record=dict(test_year=year,candidate=name,cell=label,alpha=alpha,observations=count,dates=dates,used_global_fallback=label!='global' and not sufficient,validation_start=str(val.date.min()),validation_end=str(val.date.max()))
    parameters.append(record);fits[name][label]=alpha
  (OUT/f'parameters/strengths_{year}.json').write_text(json.dumps({'year':year,'global_alpha':global_alpha,'strengths':fits,'validation_end':str(val.date.max()),'formula':plan['formula']},indent=2))
  del train,val,xt,xv;gc.collect()
  saved=read(SOURCE/f'predictions/intraday_predictions_{year}.parquet');n=len(saved);check(f'{year}_unique_keys',not saved.duplicated(['date','permno']).any());check(f'{year}_log_labels',np.allclose(np.log1p(saved.auction_volume),saved.log1p_auction_volume,rtol=1e-12))
  anchor=saved.fixed_daily.to_numpy();full=saved.ridge_update.to_numpy()
  check(f'{year}_alpha_zero_exact',np.array_equal(blend(anchor,full,0),anchor));check(f'{year}_alpha_one_exact',np.array_equal(blend(anchor,full,1),full))
  for name in VARIANTS:
   alpha=np.ones(n)
   for label,mask in cells(saved,name).items():alpha[mask]=fits[name][label]
   saved[name]=blend(anchor,full,alpha)
   check(f'{year}_{name}_strength_grid',np.isin(alpha,GRID).all())
   if name!='global_shrink':
    target_mask=np.zeros(n,bool)
    for group_mask in cells(saved,name).values():target_mask|=group_mask
    check(f'{year}_{name}_outside_target_exact',np.array_equal(saved.loc[~target_mask,name],saved.loc[~target_mask,'ridge_update']))
   check(f'{year}_{name}_finite',np.isfinite(saved[name]).all() and saved[name].ge(0).all());check(f'{year}_{name}_between_endpoints',((saved[name]>=np.minimum(anchor,full)-1e-8)&(saved[name]<=np.maximum(anchor,full)+1e-8)).all());check(f'{year}_{name}_fallback_exact',np.array_equal(saved.loc[~saved.cp_available,name],saved.loc[~saved.cp_available,'fixed_daily']))
   # Independently reload strengths and recompute in log space on the whole evaluation year.
   loaded=json.loads((OUT/f'parameters/strengths_{year}.json').read_text());alphas=np.ones(n)
   for label,mask in cells(saved,name).items():alphas[mask]=loaded['strengths'][name][label]
   expected=(1-alphas)*np.log1p(anchor)+alphas*np.log1p(full)
   check(f'{year}_{name}_parameter_reconstruction',np.allclose(np.log1p(saved[name]),expected,rtol=1e-12,atol=1e-12))
  dest=OUT/f'predictions/shrinkage_predictions_{year}.parquet';c.register('shrunk_output',saved);c.execute(f"COPY shrunk_output TO '{dest}' (FORMAT PARQUET,COMPRESSION ZSTD)");c.unregister('shrunk_output');written=read(dest);check(f'{year}_written_exact',written.equals(saved))
  for name in ['fixed_daily','ridge_update',*VARIANTS]:
   metric=forecast_metrics(written.auction_volume,written[name])
   if name in ['fixed_daily','ridge_update']:
    original_metrics=pd.read_csv(SOURCE/'annual_metrics.csv');r=original_metrics[(original_metrics.test_year==year)&original_metrics.candidate.eq(name)&original_metrics['sample'].eq('all_regular_sessions')].iloc[0]
    check(f'{year}_{name}_reference_metrics_exact',metric['observations']==r.observations and all(np.isclose(metric[k],r[k],rtol=1e-12,atol=1e-12) for k in ['rmse_log','wape','aggregate_bias']))
   annual.append(dict(test_year=year,candidate=name,**metric))
  print(f'{year}: global strength {global_alpha}; parameters saved',flush=True)
 for year in [2022,2023,2024]:fit_year(year)
 dev=score_period('development',[2022,2023,2024]);d=dev.set_index('candidate');a=pd.DataFrame(annual);pivot=a.pivot(index='test_year',columns='candidate',values='rmse_log');gates=[]
 for name in VARIANTS:
  row=dict(candidate=name,rmse_change=float(d.loc[name,'rmse_log']-d.loc['ridge_update','rmse_log']),wape_change=float(d.loc[name,'wape']-d.loc['ridge_update','wape']),worst_annual_change=float((pivot[name]-pivot.ridge_update).max()),worst_annual_vs_daily=float((pivot[name]-pivot.fixed_daily).max()),wape_vs_daily=float(d.loc[name,'wape']-d.loc['fixed_daily','wape']),paired_ci_upper=next(x['ci_upper'] for x in boot if x['candidate']==name and x['period']=='development'))
  row['passes']=bool(row['rmse_change']<=-.002 and row['wape_change']<=0 and row['worst_annual_change']<=.002 and row['worst_annual_vs_daily']<=.002 and row['wape_vs_daily']<=0 and row['paired_ci_upper']<0);gates.append(row)
 pd.DataFrame(gates).to_csv(OUT/'development_adoption_gate.csv',index=False)
 passers=[r['candidate'] for r in gates if r['passes']];selected=min(passers,key=lambda name:d.loc[name,'rmse_log']) if passers else 'ridge_update'
 decision={'selected_candidate':selected,'decided_before_scoring_2025':True,'development_only':True,'decision_utc':datetime.now(timezone.utc).isoformat(),'gate':plan['adoption_gate'],'development_metrics':dev.to_dict(orient='records')};(OUT/'development_decision.json').write_text(json.dumps(decision,indent=2));print('Development decision frozen: '+selected,flush=True)
 decision_hash=sha(OUT/'development_decision.json');fit_year(2025);confirmation=score_period('confirmation',[2025]);check('development_decision_unchanged_after_confirmation',sha(OUT/'development_decision.json')==decision_hash)
 summary=pd.concat([dev,confirmation]);summary.to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_metrics.csv',index=False);pd.DataFrame(parameters).to_csv(OUT/'selected_strengths.csv',index=False);pd.DataFrame(trials).drop_duplicates().to_csv(OUT/'validation_strength_grid.csv',index=False);pd.DataFrame(boot).to_csv(OUT/'paired_block_bootstrap.csv',index=False)
 # Diagnose targeted regimes without selecting new rules from their results.
 liq="CASE WHEN liquidity_percentile IS NULL THEN 'missing' WHEN liquidity_percentile<1.0/3 THEN 'low' WHEN liquidity_percentile<2.0/3 THEN 'mid' ELSE 'high' END";event="CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END";seg=[]
 for name in ['fixed_daily','ridge_update',*VARIANTS]:
  for dim,expr in {'event':event,'liquidity':liq,'month_end_x_liquidity':f"concat(CASE WHEN is_last_trading_day_month AND NOT monthly_options_expiry AND NOT quarterly_options_expiry THEN 'month_end' ELSE 'other' END,'_',({liq}))"}.items():
   seg.append(c.sql(f"SELECT year,'{name}' candidate,'{dim}' dimension,{expr} segment,count(*) observations,count(DISTINCT date) dates,sqrt(avg(pow(ln(1+{name})-log1p_auction_volume,2))) rmse_log,sum(abs({name}-auction_volume))/sum(auction_volume) wape,sum({name})/sum(auction_volume)-1 aggregate_bias FROM read_parquet('{OUT}/predictions/shrinkage_predictions_*.parquet') GROUP BY 1,2,3,4").df())
 seg=pd.concat(seg);seg.to_csv(OUT/'annual_segment_metrics.csv',index=False)
 for dim in seg.dimension.unique():
  got=seg[seg.dimension.eq(dim)].groupby(['year','candidate']).observations.sum();expected=pd.DataFrame(annual).set_index(['test_year','candidate']).observations
  check(dim+'_exhaustive_partition',np.array_equal(got.sort_index().to_numpy(),expected.sort_index().to_numpy()))
 seg[seg.year.isin([2022,2023,2024])].groupby(['candidate','dimension','segment'])[['rmse_log','wape','aggregate_bias']].mean().reset_index().to_csv(OUT/'development_segment_means.csv',index=False)
 pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False);pd.DataFrame(inputs).to_csv(OUT/'input_fingerprints.csv',index=False)
 (OUT/'experiment_summary.json').write_text(json.dumps({'all_checks_passed':all(r['passed'] for r in checks),'checks':len(checks),'rows':int(pd.DataFrame(annual).query("candidate=='fixed_daily'").observations.sum()),'selected_candidate':selected,'script_sha256':sha(Path(__file__)),'plan_sha256':sha(OUT/'experiment_plan.json'),'development_decision_sha256':decision_hash,'prediction_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((OUT/'predictions').glob('*.parquet'))]},indent=2))
 (OUT/'README.md').write_text('# Historical intraday correction shrinkage\n\nThe development-only decision retains **'+selected+'**. Model strength is chosen on prior historical validation rows, never evaluation-year labels. The development decision was saved and fingerprinted before scoring 2025, and verified unchanged afterward. The specification itself is motivated by previously inspected diagnostics and is retrospective research.\n\n```text\n'+summary.to_string(index=False)+'\n```\n\n## Design and interpretation\n\nStrength 0 retains the daily anchor exactly; 1 retains the full bounded intraday ridge exactly. Intermediate strengths interpolate log1p share forecasts without re-clipping, preserving endpoints. Each year uses saved historical tuning and validation samples to reconstruct the original validation ridge forecast and then fit strength; the same validation year selected the original ridge penalty, so this is joint historical tuning rather than an independent nested calibration split. Full evaluation-year ridge and anchor forecasts are unchanged.\n\nGlobal, disjoint month-end, low-liquidity and combined segment variants were fixed before fitting. Non-targeted rows retain full ridge. Target cells with fewer than 8 dates or 500 rows use the global strength. Missing liquidity is treated as not low and separately reported. Read selected_strengths.csv for sparse-cell fallbacks and counts.\n\nDevelopment means weight years equally. Gates, all annual metrics, and separate 2025 confirmation results are saved. A lower subgroup error alone cannot promote a candidate. Confidence intervals concern paired daily log MAE using 2000 within-year moving 20-date blocks, not headline log RMSE or WAPE; multiple comparisons are unadjusted. 2025 was already inspected in earlier research. Early closes remain excluded; missing checkpoints retain daily forecasts exactly.\n\nAll '+str(len(checks))+' checks passed, including input hashes, original validation forecast reproduction, chronology, exact alpha endpoints, forecast bounds between endpoints, fallback, saved-parameter reconstruction, written forecasts, exhaustive group counts and the frozen development decision.\n')
 print(summary.to_string(index=False));print(pd.DataFrame(parameters)[['test_year','candidate','cell','alpha','dates','used_global_fallback']].to_string(index=False))
if __name__=='__main__':run()
