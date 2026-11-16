"""Controlled add-one/drop-one ablation of current-session 15:30 ridge inputs."""
from pathlib import Path
import sys,json,hashlib,gc
import duckdb,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from notebooks.modeling_utils import fit_preprocessor,apply_preprocessor,forecast_metrics,predict_ridge
SOURCE=ROOT/'outputs/experiments/19_intraday_1530';CONTROL=ROOT/'outputs/experiments/22_intraday_ridge_control';OUT=ROOT/'outputs/experiments/23_intraday_feature_ablation'
GROUPS={'volume':['cp_cum_volume','cp_cum_dollar_volume','cp_volume_to_adv20','cp_volume_surprise20'],'trades':['cp_cum_trade_count','cp_trade_count_surprise20'],'recent_activity':['cp_recent_30m_volume','cp_recent_30m_volume_share'],'price':['cp_vwap','cp_vwap_return_prior_close']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return duckdb.read_parquet(str(p)).df()
def run():
 for p in [OUT,OUT/'models',OUT/'predictions']:p.mkdir(parents=True,exist_ok=True)
 oldplan=json.loads((SOURCE/'experiment_plan.json').read_text());base=oldplan['daily_features']+['log_daily_forecast']+oldplan['additional_prior_close_control_features'];full=base+oldplan['intraday_features']
 assert len(sum(GROUPS.values(),[]))==len(set(sum(GROUPS.values(),[]))) and set(sum(GROUPS.values(),[]))==set(oldplan['intraday_features'])
 specs={**{'add_'+g:base+features for g,features in GROUPS.items()},**{'drop_'+g:[f for f in full if f not in features] for g,features in GROUPS.items()}}
 grid=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
 (OUT/'experiment_plan.json').write_text(json.dumps({'groups':GROUPS,'candidate_sources':specs,'penalty_grid':grid,'protocol':'Exact experiment 19 historical samples, full shared preparation, residual target, bounds, checkpoint fallback; each ablation tunes its own penalty historically','references':'Unchanged experiment 22 full ridge, daily-only ridge and fixed daily forecasts','interpretation':'Add-one versus daily-only control; drop-one versus full ridge. Conditional predictive contribution, not additive attribution or causality. Shared historical checkpoint features remain in every candidate.','evaluation':'Equal-year 2022-2024 development. Previously inspected 2025 separate confirmation, never used to choose groups. Diagnostic study; no automatic feature selection or production promotion.','diagnostics':'Disjoint calendar events, liquidity thirds and month-end by liquidity; year-specific sample counts; no segment routing fitted','bootstrap':'2000 within-year moving 20-trading-date blocks, equal-year mean daily log-MAE differences; unadjusted exploratory intervals'},indent=2))
 checks=[];inputs=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)));assert v,name
 for folder in [SOURCE,CONTROL]:
  m=json.loads((folder/'experiment_summary.json').read_text());assert m['all_checks_passed']
  for r in m['prediction_files']:
   check('hash_'+r['path'],sha(ROOT/r['path'])==r['sha256']);inputs.append(r)
 for r in pd.read_csv(SOURCE/'training_sample_fingerprints.csv').itertuples():
  check('sample_'+r.path,sha(ROOT/r.path)==r.sha256);inputs.append(dict(path=r.path,sha256=r.sha256))
 c=duckdb.connect();c.execute('SET threads=4');metrics=[];trials=[];selections=[]
 for year in range(2022,2026):
  print(f'{year}: shared historical preparation and eight ablation fits',flush=True)
  train=read(SOURCE/f'samples/{year}_tune.parquet');val=read(SOURCE/f'samples/{year}_validation.parquet');refit=read(SOURCE/f'samples/{year}_refit.parquet')
  prep=json.loads((SOURCE/f'models/preprocessor_{year}.json').read_text());types=[list(prep[k]) for k in ['continuous','boolean','categorical']]
  check(f'{year}_chronology',train.date.max()<val.date.min() and val.date.max()<pd.Timestamp(f'{year}-01-01') and refit.date.max()<pd.Timestamp(f'{year}-01-01'))
  xt,tp=fit_preprocessor(train,*types);xv=apply_preprocessor(val,tp,*types);z=(train.log1p_auction_volume-train.log_daily_forecast).to_numpy();bounds=np.quantile(train.log1p_auction_volume,[.001,.999])
  # One shared sufficient-statistic calculation; subsets solve the identical ridge objective.
  def sufficient(x,y):
   design=np.column_stack([np.ones(len(x),dtype=np.float32),x]).astype(np.float64)
   return design.T@design/len(x),design.T@y/len(x)
  def solve(gram,rhs,mask,strength):
   ix=np.r_[0,np.where(mask)[0]+1];g=gram[np.ix_(ix,ix)].copy();pen=np.eye(len(ix))*strength;pen[0,0]=0
   return np.linalg.solve(g+pen,rhs[ix])
  gram,rhs=sufficient(xt,z);chosen={}
  from notebooks.modeling_utils import fit_ridge
  for name,features in specs.items():
   mask=np.isin(tp['output_sources'],features);choices=[]
   for strength in grid:
    coef=solve(gram,rhs,mask,strength);p=np.clip(val.log_daily_forecast.to_numpy()+predict_ridge(xv[:,mask],coef),*bounds)
    row=dict(test_year=year,candidate=name,strength=strength,validation_rmse_log=float(np.sqrt(np.mean((val.log1p_auction_volume.to_numpy()-p)**2))));choices.append(row);trials.append(row)
   choice=min(choices,key=lambda x:x['validation_rmse_log']);chosen[name]=choice;selections.append(choice)
   # Verify optimized fitting against the existing helper at each selected penalty.
   check(f'{year}_{name}_ridge_solver_matches',np.allclose(solve(gram,rhs,mask,choice['strength']),fit_ridge(xt[:,mask],z,choice['strength']),rtol=1e-8,atol=1e-8))
  del train,val,xt,xv,gram,rhs;gc.collect()
  xr,rp=fit_preprocessor(refit,*types);check(f'{year}_preprocessor_exact',rp==prep);bounds=np.quantile(refit.log1p_auction_volume,[.001,.999]);rec=json.loads((SOURCE/f'models/reconstruction_{year}.json').read_text());check(f'{year}_bounds_exact',np.array_equal(bounds,rec['clip_log']))
  z=(refit.log1p_auction_volume-refit.log_daily_forecast).to_numpy();gram,rhs=sufficient(xr,z)
  test=c.sql(f"SELECT * FROM read_parquet('{SOURCE}/intraday_panel.parquet') WHERE year={year} AND NOT is_early_close ORDER BY date,permno").df();saved=read(CONTROL/f'predictions/ridge_control_predictions_{year}.parquet').sort_values(['date','permno']).reset_index(drop=True)
  check(f'{year}_keys_labels',test[['date','permno','auction_volume']].reset_index(drop=True).equals(saved[['date','permno','auction_volume']]))
  xe=apply_preprocessor(test,rp,*types);available=test.cp_available.to_numpy();anchor=test.log_daily_forecast.to_numpy()
  def reconstruct(component):return np.where(available,np.maximum(np.expm1(np.clip(anchor+component,*bounds)),0),test.daily_forecast)
  for name,features in [('ridge_update',full),('ridge_daily_control',base)]:
   mask=np.isin(rp['output_sources'],features)
   prior=np.load((SOURCE/f'models/ridge_update_{year}.npy') if name=='ridge_update' else (CONTROL/f'models/ridge_daily_control_{year}.npy'))
   check(f'{year}_{name}_unchanged',np.allclose(reconstruct(predict_ridge(xe[:,mask],prior)),saved[name],rtol=1e-10,atol=1e-8))
  (OUT/f'models/preprocessor_{year}.json').write_text(json.dumps(rp,indent=2))
  for name,features in specs.items():
   mask=np.isin(rp['output_sources'],features);coef=solve(gram,rhs,mask,chosen[name]['strength']);np.save(OUT/f'models/{name}_{year}.npy',coef)
   saved[name]=reconstruct(predict_ridge(xe[:,mask],coef));check(f'{year}_{name}_finite',np.isfinite(saved[name]).all() and saved[name].ge(0).all());check(f'{year}_{name}_fallback',np.array_equal(saved.loc[~saved.cp_available,name],saved.loc[~saved.cp_available,'fixed_daily']))
   reconstruction={'source_features':features,'matrix_columns':np.array(rp['output_names'])[mask].tolist(),'selected':chosen[name],'clip_log':bounds.tolist(),'refit_end':str(refit.date.max())};(OUT/f'models/{name}_{year}.json').write_text(json.dumps(reconstruction,indent=2))
   rx=apply_preprocessor(test.iloc[:200],json.loads((OUT/f'models/preprocessor_{year}.json').read_text()),*types)
   slice_pred=np.where(available[:200],np.maximum(np.expm1(np.clip(anchor[:200]+predict_ridge(rx[:,mask],np.load(OUT/f'models/{name}_{year}.npy')),*bounds)),0),test.daily_forecast.to_numpy()[:200])
   check(f'{year}_{name}_reload',np.allclose(slice_pred,saved[name].to_numpy()[:200],rtol=1e-12,atol=1e-8))
  dest=OUT/f'predictions/ablation_predictions_{year}.parquet';c.register('forecast_output',saved);c.execute(f"COPY forecast_output TO '{dest}' (FORMAT PARQUET,COMPRESSION ZSTD)");c.unregister('forecast_output')
  written=read(dest);check(f'{year}_written_exact',written.equals(saved))
  for name in ['fixed_daily','ridge_daily_control','ridge_update',*specs]:metrics.append(dict(test_year=year,candidate=name,**forecast_metrics(written.auction_volume,written[name])))
  del refit,test,saved,written,xr,xe,gram,rhs;gc.collect()
 pd.DataFrame(trials).to_csv(OUT/'hyperparameter_validation.csv',index=False);pd.DataFrame(selections).to_csv(OUT/'selected_hyperparameters.csv',index=False)
 a=pd.DataFrame(metrics);a.to_csv(OUT/'annual_metrics.csv',index=False);summ=[]
 for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
  t=a[a.test_year.isin(years)].groupby('candidate')[['rmse_log','wape','aggregate_bias']].mean().reset_index();t['period']=period;summ.append(t)
 summary=pd.concat(summ);summary.to_csv(OUT/'comparison.csv',index=False)
 glob=f'{OUT}/predictions/ablation_predictions_*.parquet';segments=[];boot=[];rng=np.random.default_rng(20261113)
 liq="CASE WHEN liquidity_percentile IS NULL THEN 'missing' WHEN liquidity_percentile<1.0/3 THEN 'low' WHEN liquidity_percentile<2.0/3 THEN 'mid' ELSE 'high' END"
 event="CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END"
 for name in ['fixed_daily','ridge_daily_control','ridge_update',*specs]:
  for dimension,expr in {'event':event,'liquidity':liq,'month_end_x_liquidity':f"concat(CASE WHEN is_last_trading_day_month THEN 'month_end' ELSE 'other' END,'_',({liq}))"}.items():
   segments.append(c.sql(f"SELECT year,'{name}' candidate,'{dimension}' dimension,{expr} segment,count(*) observations,count(DISTINCT date) dates,sqrt(avg(pow(ln(1+{name})-log1p_auction_volume,2))) rmse_log,sum(abs({name}-auction_volume))/sum(auction_volume) wape,sum({name})/sum(auction_volume)-1 aggregate_bias FROM read_parquet('{glob}') GROUP BY 1,2,3,4").df())
 for name in specs:
  ref='ridge_daily_control' if name.startswith('add_') else 'ridge_update'
  d=c.sql(f"SELECT date,year,avg(abs(ln(1+{name})-log1p_auction_volume)-abs(ln(1+{ref})-log1p_auction_volume)) delta FROM read_parquet('{glob}') GROUP BY date,year ORDER BY date").df()
  for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
   vs=[d[d.year==y].delta.to_numpy() for y in years];draws=[];block=20
   for _ in range(2000):
    means=[]
    for v in vs:
     starts=rng.integers(0,len(v)-block+1,size=int(np.ceil(len(v)/block)));ix=(starts[:,None]+np.arange(block)).ravel()[:len(v)];means.append(v[ix].mean())
    draws.append(np.mean(means))
   lo,hi=np.quantile(draws,[.025,.975]);boot.append(dict(candidate=name,reference=ref,period=period,mean_log_mae_delta=float(np.mean([v.mean() for v in vs])),ci_lower=lo,ci_upper=hi))
 s=pd.concat(segments);s.to_csv(OUT/'annual_segment_metrics.csv',index=False)
 for dim in s.dimension.unique():
  counts=s[s.dimension.eq(dim)].groupby(['year','candidate']).observations.sum();expected=a.set_index(['test_year','candidate']).observations
  check(f'{dim}_exhaustive_counts',np.array_equal(counts.sort_index().to_numpy(),expected.sort_index().to_numpy()))
 segmean=s[s.year.isin([2022,2023,2024])].groupby(['candidate','dimension','segment'])[['rmse_log','wape','aggregate_bias']].mean().reset_index();segmean.to_csv(OUT/'development_segment_means.csv',index=False)
 pd.DataFrame(boot).to_csv(OUT/'paired_group_bootstrap.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False);pd.DataFrame(inputs).to_csv(OUT/'input_fingerprints.csv',index=False)
 (OUT/'experiment_summary.json').write_text(json.dumps({'all_checks_passed':all(r['passed'] for r in checks),'checks':len(checks),'rows':int(a[a.candidate.eq('fixed_daily')].observations.sum()),'script_sha256':sha(Path(__file__)),'prediction_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((OUT/'predictions').glob('*.parquet'))]},indent=2))
 (OUT/'README.md').write_text('# Intraday ridge feature group ablation\n\nFour disjoint current-session groups are added individually to the daily-only control and removed individually from the full intraday ridge. Historical checkpoint information is retained throughout. Each candidate is refitted and tunes its own penalty historically. Source features carry their missingness flags together.\n\n## Results\n\n```text\n'+summary.to_string(index=False)+'\n```\n\nThe group definitions and frozen design are in experiment_plan.json. Segment diagnostics include disjoint calendar events, liquidity thirds, and month-end crossed with liquidity. Headline and development segment metrics are equal-year averages; annual segment counts are retained separately.\n\nAdd-one changes are compared with daily-only ridge; drop-one changes with full intraday ridge. These are conditional predictive comparisons, not a causal or additive allocation of importance. Correlated groups may substitute for one another. Twenty-date block intervals concern daily log MAE, not headline log RMSE; unadjusted multiple comparisons are exploratory. 2025 is previously inspected confirmation and is not used to choose groups. No specialized routing or production promotion is performed.\n\nAll '+str(len(checks))+' checks passed. Existing references were preserved and reconstructed, historical sample and forecast hashes verified, preprocessing and bounds matched, optimized ridge fitting compared to the original helper, and saved forecasts checked for exact keys, nonnegativity, fallback, model reconstruction and exhaustive segment counts.\n')
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':run()
