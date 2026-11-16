"""Matched daily-only ridge control for frozen experiment 19, without changing it."""
from pathlib import Path
import sys,json,hashlib,gc
import numpy as np,pandas as pd,duckdb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from notebooks.modeling_utils import fit_preprocessor,apply_preprocessor,fit_ridge,predict_ridge,forecast_metrics
SOURCE=ROOT/'outputs/experiments/19_intraday_1530';OUT=ROOT/'outputs/experiments/22_intraday_ridge_control'
def read_parquet(p):return duckdb.read_parquet(str(p)).df()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 for folder in [OUT,OUT/'models',OUT/'predictions']:folder.mkdir(parents=True,exist_ok=True)
 source_manifest=json.loads((SOURCE/'experiment_summary.json').read_text());assert source_manifest['all_checks_passed']
 plan=json.loads((SOURCE/'experiment_plan.json').read_text())
 control=plan['daily_features']+['log_daily_forecast']+plan['additional_prior_close_control_features']
 full=control+plan['intraday_features'];grid=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
 (OUT/'experiment_plan.json').write_text(json.dumps({'scope':'Matched daily-only ridge control versus frozen full intraday ridge and fixed daily forecast','source_experiment':'19_intraday_1530','control_sources':control,'full_sources':full,'penalty_grid':grid,'sampling':'Exact saved tune, validation and refit samples; no new sampling','preprocessing':'Full 73 source preparation followed by source mask, matching experiment 19; verify final preparation exactly','forecast':'Same residual target, training-derived bounds and exact daily fallback when checkpoint missing','evaluation':'2022-2024 equal-year development; previously inspected 2025 confirmation','uncertainty':'2000 moving blocks of 5 and 20 consecutive trading dates sampled within each year; equal-year daily log MAE differences','interpretation':'Retrospective matched experiment, not pristine validation or an automatic production promotion'},indent=2))
 checks=[];inputs=[];trials=[];annual=[];selected=[]
 def check(name,condition):
  checks.append(dict(check=name,passed=bool(condition)))
  assert condition,name
 fingerprints=pd.read_csv(SOURCE/'training_sample_fingerprints.csv')
 for r in fingerprints.itertuples():
  check('fingerprint_'+Path(r.path).name,sha(ROOT/r.path)==r.sha256)
  inputs.append(dict(path=r.path,sha256=r.sha256))
 c=duckdb.connect();c.execute('SET threads=4')
 for year in range(2022,2026):
  print(f'Fold {year}: fitting matched control',flush=True)
  train=read_parquet(SOURCE/f'samples/{year}_tune.parquet');val=read_parquet(SOURCE/f'samples/{year}_validation.parquet');refit=read_parquet(SOURCE/f'samples/{year}_refit.parquet')
  oldprep=json.loads((SOURCE/f'models/preprocessor_{year}.json').read_text());types=[list(oldprep[k]) for k in ['continuous','boolean','categorical']]
  check(f'{year}_historical_chronology',train.date.max()<val.date.min() and val.date.max()<pd.Timestamp(f'{year}-01-01') and refit.date.max()<pd.Timestamp(f'{year}-01-01'))
  xt,tp=fit_preprocessor(train,*types);xv=apply_preprocessor(val,tp,*types)
  mask=np.array([f in control for f in tp['output_sources']]);bounds=np.quantile(train.log1p_auction_volume,[.001,.999]);z=(train.log1p_auction_volume-train.log_daily_forecast).to_numpy()
  choices=[]
  for strength in grid:
   coef=fit_ridge(xt[:,mask],z,strength)
   pred=np.clip(val.log_daily_forecast.to_numpy()+predict_ridge(xv[:,mask],coef),*bounds)
   row=dict(test_year=year,strength=strength,validation_rmse_log=float(np.sqrt(np.mean((val.log1p_auction_volume.to_numpy()-pred)**2))))
   choices.append(row);trials.append(row)
  choice=min(choices,key=lambda r:r['validation_rmse_log']);selected.append(choice)
  # Recreate the full ridge's historical selection, before trusting the matched contrast.
  original=json.loads((SOURCE/f'models/reconstruction_{year}.json').read_text())
  full_trials=[]
  for strength in grid:
   coef=fit_ridge(xt,z,strength);pred=np.clip(val.log_daily_forecast.to_numpy()+predict_ridge(xv,coef),*bounds)
   full_trials.append((float(np.sqrt(np.mean((val.log1p_auction_volume.to_numpy()-pred)**2))),strength))
  best=min(full_trials,key=lambda r:r[0]);prior=original['selected']['ridge_update']
  check(f'{year}_full_ridge_selection_reproduced',best[1]==prior['parameter'] and np.isclose(best[0],prior['validation_rmse_log'],rtol=1e-12))
  del train,val,xt,xv;gc.collect()
  xr,rp=fit_preprocessor(refit,*types);check(f'{year}_preprocessor_exact',rp==oldprep)
  mask=np.array([f in control for f in rp['output_sources']]);check(f'{year}_no_current_features',set(np.array(rp['output_sources'])[mask])<=set(control) and not(set(control)&set(plan['intraday_features'])))
  z=(refit.log1p_auction_volume-refit.log_daily_forecast).to_numpy();bounds=np.quantile(refit.log1p_auction_volume,[.001,.999]);check(f'{year}_bounds_exact',np.array_equal(bounds,original['clip_log']))
  coef=fit_ridge(xr[:,mask],z,choice['strength']);np.save(OUT/f'models/ridge_daily_control_{year}.npy',coef)
  fullcoef=fit_ridge(xr,z,prior['parameter']);stored=np.load(SOURCE/f'models/ridge_update_{year}.npy');check(f'{year}_full_coefficients_reproduced',np.allclose(fullcoef,stored,rtol=1e-10,atol=1e-10))
  predpath=SOURCE/f'predictions/intraday_predictions_{year}.parquet';expected=next(x for x in source_manifest['prediction_files'] if x['path'].endswith(predpath.name));check(f'{year}_prediction_fingerprint',sha(predpath)==expected['sha256']);inputs.append(dict(path=str(predpath.relative_to(ROOT)),sha256=sha(predpath)))
  test=c.sql(f"SELECT * FROM read_parquet('{SOURCE}/intraday_panel.parquet') WHERE year={year} AND NOT is_early_close ORDER BY date,permno").df()
  saved=read_parquet(predpath).sort_values(['date','permno']).reset_index(drop=True)
  check(f'{year}_keys_labels_exact',test[['date','permno','auction_volume']].reset_index(drop=True).equals(saved[['date','permno','auction_volume']]))
  xe=apply_preprocessor(test,rp,*types)
  def reconstruct(component):return np.where(test.cp_available,np.maximum(np.expm1(np.clip(test.log_daily_forecast.to_numpy()+component,*bounds)),0),test.daily_forecast)
  check(f'{year}_full_forecast_reproduced',np.allclose(reconstruct(predict_ridge(xe,fullcoef)),saved.ridge_update,rtol=1e-10,atol=1e-8))
  saved['ridge_daily_control']=reconstruct(predict_ridge(xe[:,mask],coef))
  check(f'{year}_finite_nonnegative',np.isfinite(saved.ridge_daily_control).all() and saved.ridge_daily_control.ge(0).all())
  check(f'{year}_fallback_exact',np.array_equal(saved.loc[~saved.cp_available,'ridge_daily_control'],saved.loc[~saved.cp_available,'fixed_daily']))
  (OUT/f'models/preprocessor_{year}.json').write_text(json.dumps(rp,indent=2));(OUT/f'models/reconstruction_{year}.json').write_text(json.dumps({'clip_log':bounds.tolist(),'source_features':control,'matrix_columns':np.array(rp['output_names'])[mask].tolist(),'selected':choice,'refit_end':str(refit.date.max())},indent=2))
  reloadcoef=np.load(OUT/f'models/ridge_daily_control_{year}.npy');reloaded_prep=json.loads((OUT/f'models/preprocessor_{year}.json').read_text());slice_x=apply_preprocessor(test.iloc[:200],reloaded_prep,*types)
  slice_pred=np.where(test.cp_available.to_numpy()[:200],np.maximum(np.expm1(np.clip(test.log_daily_forecast.to_numpy()[:200]+predict_ridge(slice_x[:,mask],reloadcoef),*bounds)),0),test.daily_forecast.to_numpy()[:200])
  check(f'{year}_saved_model_reconstruction',np.array_equal(slice_pred,saved.ridge_daily_control.to_numpy()[:200]))
  target=OUT/f'predictions/ridge_control_predictions_{year}.parquet';c.register('forecast_output',saved);c.execute(f"COPY forecast_output TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)");c.unregister('forecast_output');written=read_parquet(target)
  for method in ['fixed_daily','ridge_daily_control','ridge_update']:
   m=forecast_metrics(written.auction_volume,written[method]);annual.append(dict(test_year=year,candidate=method,**m))
  del refit,test,saved,written,xr,xe;gc.collect()
 pd.DataFrame(trials).to_csv(OUT/'hyperparameter_validation.csv',index=False);pd.DataFrame(selected).to_csv(OUT/'selected_hyperparameters.csv',index=False)
 a=pd.DataFrame(annual);a.to_csv(OUT/'annual_metrics.csv',index=False)
 summary=[]
 for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
  t=a[a.test_year.isin(years)].groupby('candidate')[['rmse_log','wape','aggregate_bias']].mean().reset_index();t['period']=period;summary.append(t)
 summary=pd.concat(summary);summary.to_csv(OUT/'comparison.csv',index=False)
 bootstrap=[];rng=np.random.default_rng(20261113)
 for ref in ['fixed_daily','ridge_daily_control']:
  daily=c.sql(f"SELECT date,year,avg(abs(ln(1+ridge_update)-log1p_auction_volume)-abs(ln(1+{ref})-log1p_auction_volume)) delta FROM read_parquet('{OUT}/predictions/*.parquet') GROUP BY date,year ORDER BY date").df()
  daily.to_csv(OUT/f'daily_errors_vs_{ref}.csv',index=False)
  for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
   vs=[daily[daily.year==y].delta.to_numpy() for y in years]
   for block in [5,20]:
    draws=[]
    for _ in range(2000):
     means=[]
     for v in vs:
      starts=rng.integers(0,len(v)-block+1,size=int(np.ceil(len(v)/block)));ix=(starts[:,None]+np.arange(block)).ravel()[:len(v)];means.append(v[ix].mean())
     draws.append(np.mean(means))
    lo,hi=np.quantile(draws,[.025,.975]);bootstrap.append(dict(reference=ref,period=period,block_dates=block,mean_log_mae_delta=float(np.mean([v.mean() for v in vs])),ci_lower=lo,ci_upper=hi))
 b=pd.DataFrame(bootstrap);b.to_csv(OUT/'paired_block_bootstrap.csv',index=False)
 pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False);pd.DataFrame(inputs).to_csv(OUT/'input_fingerprints.csv',index=False)
 (OUT/'experiment_summary.json').write_text(json.dumps({'all_checks_passed':all(x['passed'] for x in checks),'checks':len(checks),'rows':int(a[a.candidate.eq('fixed_daily')].observations.sum()),'script_sha256':sha(Path(__file__)),'prediction_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted((OUT/'predictions').glob('*.parquet'))]},indent=2))
 (OUT/'README.md').write_text('# Matched daily-only ridge control\n\nThe frozen full 15:30 ridge and daily anchor remain unchanged. The new control uses the same samples, residual target, preprocessing, penalty grid, clipping and missing-checkpoint fallback but excludes all ten current checkpoint inputs. Prior checkpoint history remains in both models. Each control chooses its own penalty using historical validation.\n\n```text\n'+summary.to_string(index=False)+'\n```\n\nPaired differences below are full intraday ridge minus reference; negative favors the intraday update. Confidence intervals concern equal-year mean daily log MAE, not log RMSE or WAPE.\n\n```text\n'+b.to_string(index=False)+'\n```\n\nThis retrospective study uses previously inspected 2025 confirmation results. The shared requirement for a valid checkpoint in historical training and daily fallback on missing checkpoints is retained for a matched comparison, rather than an independently optimized daily-only forecast. Blocks preserve some serial dependence; intervals do not adjust for multiple comparisons. No model is promoted automatically.\n\nAll '+str(len(checks))+' checks passed, including historical sample hashes, original full-ridge validation selection and coefficient reconstruction, original forecasts, final preprocessing and bounds, exact keys and fallback, and saved control coefficients.\n')
 print(summary.to_string(index=False),flush=True);print(b.to_string(index=False),flush=True)
if __name__=='__main__':run()
