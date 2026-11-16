"""Quarterly historical tuning and refitting of the fixed 15:30 ridge specification."""
from pathlib import Path
import sys,json,hashlib,gc
import duckdb,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from notebooks.modeling_utils import fit_preprocessor,apply_preprocessor,fit_ridge,predict_ridge,forecast_metrics
from notebooks.evaluate_frozen_intraday import statistics,circular_weights,COLS,METRICS
SOURCE=ROOT/'outputs/experiments/19_intraday_1530';OUT=ROOT/'outputs/experiments/26_quarterly_intraday_audit'
GRID=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1];SEED=20261116

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 for p in [OUT,OUT/'models',OUT/'sample_keys',OUT/'predictions']:p.mkdir(parents=True,exist_ok=True)
 config=json.loads((SOURCE/'experiment_plan.json').read_text());features=config['daily_features']+['log_daily_forecast']+config['additional_prior_close_control_features']+config['intraday_features']
 plan={'design':'16 quarterly forward folds 2022Q1-2025Q4. Fixed 73-source full 15:30 ridge specification; no shrinkage or feature selection.','training_start':'2021-01-01','inner_validation':'Immediately preceding complete calendar quarter; tune on all earlier update history from 2021. Example 2022Q1 fits Jan-Sep 2021, validates Q4 2021.','outer_refit':'All update history from 2021 through immediately preceding quarter, never the evaluation quarter.','sample_caps':{'tune':250000,'validation':120000,'refit':350000},'sample_keys':'Saved in actual fitting order, with source panel hash. Reservoir sampling on ordered input with one DuckDB thread and fixed fold seeds.','penalty_grid':GRID,'objective':'Validation reconstructed log RMSE after tuning-training-derived bounds; exact ties prefer smaller penalty in grid order.','preprocessing':'Relearn transforms, clipping, missingness, scaling and encoding on tuning or refit sample only.','target':'log1p(auction shares)-log1p(saved annual out-of-time daily anchor)','daily_anchor':'Unchanged annually fitted saved daily LightGBM forecasts, not quarterly daily retraining. Lags continue updating through each quarter.','bounds':'Relearn 0.1/99.9 percentiles of historical refit log volume; missing checkpoint uses exact daily anchor.','training_universe':'Same regular-session and checkpoint-available policy as experiment 19; full regular-session evaluation with exact fallback.','comparisons':'Primary quarterly intraday versus daily. Secondary quarterly versus original annual intraday; changed samples/windows/bounds mean this does not isolate refit frequency alone.','reporting':'Quarterly, annual and equal-year 2022-2024 development / 2025 previously inspected confirmation; fixed event and liquidity subgroup reports.','uncertainty':'5000 circular date-block replicates within years, 20-date primary, 5/60 sensitivity; directly recompute annual RMSE/WAPE/bias then equal-year means. Conditional on fitted quarterly forecasts, not complete refitting or selection uncertainty.','limits':'All evaluation years have been inspected. Nested historical tuning improves procedure audit but does not create a new holdout. No shrinkage calibration stage, new data, automated promotion or quarterly-independent significance test.','features':features}
 (OUT/'experiment_plan.json').write_text(json.dumps(plan,indent=2));plan_hash=sha(OUT/'experiment_plan.json');checks=[];inputs=[];folds=[];trials=[];qmetrics=[];key_files=[]
 def check(name,v):checks.append(dict(check=name,passed=bool(v)));assert v,name
 panel=SOURCE/'intraday_panel.parquet';inputs.append(dict(path=str(panel.relative_to(ROOT)),sha256=sha(panel)))
 manifest=json.loads((SOURCE/'experiment_summary.json').read_text());assert manifest['all_checks_passed']
 for r in manifest['prediction_files']:
  check('hash_'+r['path'],sha(ROOT/r['path'])==r['sha256']);inputs.append(r)
 c=duckdb.connect();c.execute('SET threads=1');c.execute(f"CREATE TEMP VIEW panel AS SELECT * FROM read_parquet('{panel}')")
 template=json.loads((SOURCE/'models/preprocessor_2022.json').read_text());types=[list(template[k]) for k in ['continuous','boolean','categorical']];check('source_allowlist',set(sum(types,[]))==set(features))
 def load(start,end,cap,seed,available=True):
  extra='AND cp_available' if available else ''
  query=f"SELECT * FROM panel WHERE date>=DATE '{start}' AND date<DATE '{end}' AND NOT is_early_close {extra} ORDER BY date,permno"
  if cap:query=f'SELECT * FROM ({query}) USING SAMPLE {cap} ROWS (reservoir,{seed})'
  return c.sql(query).df().sort_values(['date','permno']).reset_index(drop=True)
 for year in range(2022,2026):
  for quarter in range(1,5):
   origin=pd.Timestamp(year,1+(quarter-1)*3,1);end=origin+pd.DateOffset(months=3);vstart=origin-pd.DateOffset(months=3);label=f'{year}Q{quarter}';seed=SEED+year*10+quarter
   print(label+': historical tuning and refit',flush=True)
   train=load('2021-01-01',str(vstart.date()),250000,seed);val=load(str(vstart.date()),str(origin.date()),120000,seed+10000)
   check(label+'_inner_chronology',train.date.max()<val.date.min() and val.date.max()<origin)
   for part,frame in [('tune',train),('validation',val)]:
    p=OUT/f'sample_keys/{label}_{part}.parquet';c.register('keys',frame[['date','permno']]);c.execute(f"COPY keys TO '{p}' (FORMAT PARQUET,COMPRESSION ZSTD)");c.unregister('keys');key_files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=len(frame)))
   xt,tp=fit_preprocessor(train,*types);xv=apply_preprocessor(val,tp,*types);z=(train.log1p_auction_volume-train.log_daily_forecast).to_numpy();bounds=np.quantile(train.log1p_auction_volume,[.001,.999])
   design=np.column_stack([np.ones(len(xt),dtype=np.float32),xt]).astype(float);gram=design.T@design/len(design);rhs=design.T@z/len(design);choices=[]
   for strength in GRID:
    penalty=np.eye(len(rhs))*strength;penalty[0,0]=0;coef=np.linalg.solve(gram+penalty,rhs)
    pred=np.clip(val.log_daily_forecast.to_numpy()+predict_ridge(xv,coef),*bounds)
    row=dict(fold=label,strength=strength,validation_rmse_log=float(np.sqrt(np.mean((val.log1p_auction_volume.to_numpy()-pred)**2))));choices.append(row);trials.append(row)
   best=min(choices,key=lambda x:x['validation_rmse_log']);chosen=fit_ridge(xt,z,best['strength']);penalty=np.eye(len(rhs))*best['strength'];penalty[0,0]=0
   check(label+'_solver_matches_helper',np.allclose(chosen,np.linalg.solve(gram+penalty,rhs),rtol=1e-8,atol=1e-8))
   (OUT/f'models/tuning_preprocessor_{label}.json').write_text(json.dumps(tp,indent=2))
   train_dates=[str(train.date.min()),str(train.date.max())];validation_dates=[str(val.date.min()),str(val.date.max())];counts=[len(train),len(val)]
   del train,val,xt,xv,design,gram,rhs;gc.collect()
   refit=load('2021-01-01',str(origin.date()),350000,seed+20000);test=load(str(origin.date()),str(end.date()),None,seed,False)
   check(label+'_outer_chronology',refit.date.max()<test.date.min() and test.date.min()>=origin and test.date.max()<end)
   check(label+'_training_checkpoint_policy',refit.cp_available.all())
   p=OUT/f'sample_keys/{label}_refit.parquet';c.register('keys',refit[['date','permno']]);c.execute(f"COPY keys TO '{p}' (FORMAT PARQUET,COMPRESSION ZSTD)");c.unregister('keys');key_files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=len(refit)))
   xr,rp=fit_preprocessor(refit,*types);xe=apply_preprocessor(test,rp,*types);z=(refit.log1p_auction_volume-refit.log_daily_forecast).to_numpy();coef=fit_ridge(xr,z,best['strength']);clip=np.quantile(refit.log1p_auction_volume,[.001,.999]);pred=np.where(test.cp_available,np.maximum(np.expm1(np.clip(test.log_daily_forecast.to_numpy()+predict_ridge(xe,coef),*clip)),0),test.daily_forecast)
   old=c.sql(f"SELECT * FROM read_parquet('{SOURCE}/predictions/intraday_predictions_{year}.parquet') WHERE date>=DATE '{origin.date()}' AND date<DATE '{end.date()}' ORDER BY date,permno").df()
   check(label+'_exact_reference_keys_labels',test[['date','permno','auction_volume']].equals(old[['date','permno','auction_volume']]))
   check(label+'_daily_anchor_exact',np.array_equal(test.daily_forecast,old.fixed_daily));check(label+'_nonnegative_finite',np.isfinite(pred).all() and (pred>=0).all());check(label+'_fallback_exact',np.array_equal(pred[~test.cp_available],test.daily_forecast[~test.cp_available]))
   np.save(OUT/f'models/quarterly_ridge_{label}.npy',coef);(OUT/f'models/preprocessor_{label}.json').write_text(json.dumps(rp,indent=2))
   model_record={'fold':label,'selected':best,'clip_log':clip.tolist(),'train_dates':train_dates,'validation_dates':validation_dates,'refit_dates':[str(refit.date.min()),str(refit.date.max())],'test_dates':[str(test.date.min()),str(test.date.max())],'sample_rows':{'tune':counts[0],'validation':counts[1],'refit':len(refit)},'test_rows':len(test),'features':features}
   (OUT/f'models/reconstruction_{label}.json').write_text(json.dumps(model_record,indent=2));folds.append(model_record)
   xx=apply_preprocessor(test.iloc[:200],json.loads((OUT/f'models/preprocessor_{label}.json').read_text()),*types);cc=np.load(OUT/f'models/quarterly_ridge_{label}.npy');again=np.where(test.cp_available.iloc[:200],np.maximum(np.expm1(np.clip(test.log_daily_forecast.to_numpy()[:200]+predict_ridge(xx,cc),*clip)),0),test.daily_forecast.iloc[:200])
   check(label+'_serialized_reconstruction',np.allclose(again,pred[:200],rtol=1e-12,atol=1e-8))
   keep=['date','permno','year','auction_volume','log1p_auction_volume','cp_available','size_percentile','liquidity_percentile','prior_auction_observations','quarterly_options_expiry','monthly_options_expiry','is_last_trading_day_month','fixed_daily']
   result=old[keep].copy();result['annual_ridge']=old.ridge_update;result['quarterly_ridge']=pred;result['fold']=label
   dest=OUT/f'predictions/quarterly_predictions_{label}.parquet';c.register('predictions',result);c.execute(f"COPY predictions TO '{dest}' (FORMAT PARQUET,COMPRESSION ZSTD)");c.unregister('predictions');written=c.sql(f"SELECT * FROM read_parquet('{dest}')").df();check(label+'_written_exact',written.equals(result))
   for name in ['fixed_daily','annual_ridge','quarterly_ridge']:qmetrics.append(dict(fold=label,year=year,quarter=quarter,candidate=name,**forecast_metrics(written.auction_volume,written[name])))
   pd.DataFrame(qmetrics).to_csv(OUT/'quarter_metrics.csv',index=False)
   del refit,test,xr,xe,old,result,written;gc.collect()
 pd.DataFrame(trials).to_csv(OUT/'hyperparameter_validation.csv',index=False);(OUT/'fold_manifest.json').write_text(json.dumps(folds,indent=2));pd.DataFrame(key_files).to_csv(OUT/'sample_key_fingerprints.csv',index=False)
 paths=[str(OUT/f'predictions/quarterly_predictions_{y}Q{q}.parquet') for y in range(2022,2026) for q in range(1,5)];c.execute(f"CREATE TEMP VIEW scored AS SELECT * FROM read_parquet({paths!r})")
 check('all_keys_unique',c.sql('SELECT count(*)-count(DISTINCT(date,permno)) FROM scored').fetchone()[0]==0);check('all_rows_complete',c.sql('SELECT count(*) FROM scored').fetchone()[0]==manifest['regular_session_rows'])
 annual=[]
 for y in range(2022,2026):
  f=c.sql(f'SELECT * FROM scored WHERE year={y}').df()
  for name in ['fixed_daily','annual_ridge','quarterly_ridge']:annual.append(dict(year=y,candidate=name,**forecast_metrics(f.auction_volume,f[name])))
 a=pd.DataFrame(annual);a.to_csv(OUT/'annual_metrics.csv',index=False);summary=[];intervals=[]
 for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
  s=a[a.year.isin(years)].groupby('candidate')[['rmse_log','wape','aggregate_bias']].mean().reset_index();s['period']=period;summary.append(s)
  for reference in ['fixed_daily','annual_ridge']:
   daily=c.sql(f"SELECT date,year,count(*) n,sum(auction_volume) actual,sum(pow(ln(1+{reference})-log1p_auction_volume,2)) sse_daily,sum(pow(ln(1+quarterly_ridge)-log1p_auction_volume,2)) sse_update,sum(abs({reference}-auction_volume)) ae_daily,sum(abs(quarterly_ridge-auction_volume)) ae_update,sum({reference}) pred_daily,sum(quarterly_ridge) pred_update FROM scored WHERE year IN ({','.join(map(str,years))}) GROUP BY date,year ORDER BY date").df();daily.to_csv(OUT/f'daily_statistics_{period}_vs_{reference}.csv',index=False)
   mats=[daily[daily.year.eq(y)][COLS].to_numpy(float) for y in years];point=np.mean([statistics(v.sum(axis=0)) for v in mats],axis=0);delta=point[1]-point[0]
   for block in ([5,20,60] if reference=='fixed_daily' else [20]):
    draws=np.mean([statistics(circular_weights(len(v),block,5000,SEED+block*10000+y)@v) for y,v in zip(years,mats)],axis=0);differences=draws[:,1,:]-draws[:,0,:];check(f'{period}_{reference}_{block}_finite_bootstrap',np.isfinite(differences).all())
    lo,hi=np.quantile(differences,[.025,.975],axis=0)
    for k,metric in enumerate(METRICS):intervals.append(dict(period=period,reference=reference,metric=metric,block_dates=block,difference=delta[k],ci_lower=lo[k],ci_upper=hi[k]))
 summary=pd.concat(summary);summary.to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'metric_intervals.csv',index=False)
 seg=[];liq="CASE WHEN liquidity_percentile IS NULL THEN 'missing' WHEN liquidity_percentile<1.0/3 THEN 'low' WHEN liquidity_percentile<2.0/3 THEN 'mid' ELSE 'high' END";event="CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END"
 for name in ['fixed_daily','annual_ridge','quarterly_ridge']:
  for dim,expression in {'liquidity':liq,'event':event}.items():
   seg.append(c.sql(f"SELECT year,'{name}' candidate,'{dim}' dimension,{expression} segment,count(*) observations,count(DISTINCT date) dates,sqrt(avg(pow(ln(1+{name})-log1p_auction_volume,2))) rmse_log,sum(abs({name}-auction_volume))/sum(auction_volume) wape,sum({name})/sum(auction_volume)-1 aggregate_bias FROM scored GROUP BY 1,2,3,4").df())
 seg=pd.concat(seg);seg.to_csv(OUT/'annual_segment_metrics.csv',index=False)
 for dim in seg.dimension.unique():check(dim+'_exhaustive',np.array_equal(seg[seg.dimension.eq(dim)].groupby(['year','candidate']).observations.sum().sort_index(),a.set_index(['year','candidate']).observations.sort_index()))
 check('protocol_unchanged',sha(OUT/'experiment_plan.json')==plan_hash)
 pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False);pd.DataFrame(inputs).to_csv(OUT/'input_fingerprints.csv',index=False)
 (OUT/'experiment_summary.json').write_text(json.dumps({'all_checks_passed':all(x['passed'] for x in checks),'checks':len(checks),'folds':16,'rows':manifest['regular_session_rows'],'script_sha256':sha(Path(__file__)),'plan_sha256':plan_hash,'prediction_files':[{'path':str(Path(p).relative_to(ROOT)),'sha256':sha(Path(p))} for p in paths]},indent=2))
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':run()
