"""Direct metric-matched inference for the frozen daily versus 15:30 contrast.

Historical evaluation only. Does not fit models or accept forward-year forecasts.
"""
from pathlib import Path
import json,hashlib
import duckdb,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'outputs/experiments/19_intraday_1530';OUT=ROOT/'outputs/experiments/25_frozen_intraday_evaluation'
COLS=['n','actual','sse_daily','sse_update','ae_daily','ae_update','pred_daily','pred_update']
METRICS=['rmse_log','wape','signed_bias','absolute_annual_bias']
REPS=5000;SEED=20261113

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def statistics(t):
 """Totals (...,8) -> (...,2 models,4 metrics); take annual stats before averaging."""
 n,a,sd,su,ed,eu,pd_,pu=np.moveaxis(np.asarray(t,dtype=float),-1,0)
 with np.errstate(divide='ignore',invalid='ignore'):
  bd=pd_/a-1;bu=pu/a-1
  daily=np.stack([np.sqrt(sd/n),ed/a,bd,np.abs(bd)],axis=-1)
  update=np.stack([np.sqrt(su/n),eu/a,bu,np.abs(bu)],axis=-1)
 return np.stack([daily,update],axis=-2)
def circular_weights(n,block,reps,seed):
 rng=np.random.default_rng(seed);starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
 ix=((starts[...,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
 w=np.zeros((reps,n),dtype=float);np.add.at(w,(np.arange(reps)[:,None],ix),1)
 return w

def run():
 OUT.mkdir(parents=True,exist_ok=True);assert (OUT/'evaluation_protocol.md').exists()
 protocol_hash=sha(OUT/'evaluation_protocol.md');checks=[];inputs=[]
 def check(name,v):checks.append(dict(check=name,passed=bool(v)));assert v,name
 # Unequal counts and share denominators catch mean-daily-WAPE and wrong RMSE aggregation.
 toy=np.array([[2,100,8,2,20,10,80,90],[1,900,9,1,90,180,810,1080]],float)
 st=statistics(toy.sum(axis=0));check('synthetic_root_after_sum',np.isclose(st[0,0],np.sqrt(17/3)))
 check('synthetic_ratio_of_sums',np.isclose(st[1,1],190/1000) and not np.isclose(st[1,1],np.mean([.1,.2])))
 check('synthetic_bias_and_absolute_bias',np.isclose(st[0,2],-.11) and np.isclose(st[0,3],.11))
 check('synthetic_year_weighting',np.isclose(statistics(toy).mean(axis=0)[0,0],(2+3)/2) and not np.isclose(statistics(toy).mean(axis=0)[0,0],st[0,0]))
 w=circular_weights(7,3,30,SEED);check('bootstrap_preserves_date_count',np.all(w.sum(axis=1)==7));check('bootstrap_reproducible',np.array_equal(w,circular_weights(7,3,30,SEED)))
 # Aggregating sampled daily totals must equal explicit resampling of those dates.
 ix=np.array([0,0,1]);counts=np.bincount(ix,minlength=2);check('synthetic_resample_exact',np.array_equal(statistics(counts@toy),statistics(toy[ix].sum(axis=0))))
 manifest=json.loads((SOURCE/'experiment_summary.json').read_text());assert manifest['all_checks_passed']
 paths=[]
 for r in manifest['prediction_files']:
  p=ROOT/r['path'];check('source_hash_'+p.name,sha(p)==r['sha256']);inputs.append(r);paths.append(str(p))
 c=duckdb.connect();c.execute('SET threads=4');c.execute(f"CREATE TEMP TABLE p AS SELECT * FROM read_parquet({paths!r})")
 check('historical_years_only',c.sql('SELECT list(DISTINCT year ORDER BY year) FROM p').fetchone()[0]==[2022,2023,2024,2025])
 check('exact_source_rows',c.sql('SELECT count(*) FROM p').fetchone()[0]==manifest['regular_session_rows'])
 check('no_duplicate_keys',c.sql('SELECT count(*)-count(DISTINCT(date,permno)) FROM p').fetchone()[0]==0)
 check('no_invalid_scored_rows',c.sql('SELECT count(*) FROM p WHERE auction_volume IS NULL OR auction_volume<0 OR NOT isfinite(fixed_daily) OR fixed_daily<0 OR NOT isfinite(ridge_update) OR ridge_update<0 OR date IS NULL OR permno IS NULL').fetchone()[0]==0)
 check('missing_checkpoint_exact_fallback',c.sql('SELECT count(*) FROM p WHERE NOT cp_available AND fixed_daily IS DISTINCT FROM ridge_update').fetchone()[0]==0)
 check('source_year_matches_date',c.sql('SELECT count(*) FROM p WHERE year<>year(date)').fetchone()[0]==0)
 liq="CASE WHEN liquidity_percentile IS NULL THEN 'missing' WHEN liquidity_percentile<1.0/3 THEN 'low' WHEN liquidity_percentile<2.0/3 THEN 'mid' ELSE 'high' END"
 size="CASE WHEN size_percentile IS NULL THEN 'missing' WHEN size_percentile<1.0/3 THEN 'small' WHEN size_percentile<2.0/3 THEN 'mid' ELSE 'large' END"
 event="CASE WHEN quarterly_options_expiry THEN 'quarterly_expiry' WHEN monthly_options_expiry THEN 'monthly_expiry' WHEN is_last_trading_day_month THEN 'month_end' ELSE 'regular' END"
 history="CASE WHEN prior_auction_observations IS NULL THEN 'missing' WHEN prior_auction_observations<5 THEN '0_4' WHEN prior_auction_observations<20 THEN '5_19' ELSE '20_plus' END"
 dimensions={'overall':("'all'",['all']),'size':(size,['small','mid','large','missing']),'liquidity':(liq,['low','mid','high','missing']),'history':(history,['0_4','5_19','20_plus','missing']),'event':(event,['quarterly_expiry','monthly_expiry','month_end','regular']),'month_end_x_liquidity':(f"concat(CASE WHEN ({event})='month_end' THEN 'month_end' ELSE 'other' END,'_',({liq}))",[f'{e}_{l}' for e in ['month_end','other'] for l in ['low','mid','high','missing']])}
 pieces=[]
 for dim,(expression,_) in dimensions.items():
  pieces.append(c.sql(f'''SELECT date,year,'{dim}' dimension,{expression} segment,count(*) n,sum(auction_volume) actual,
   sum(pow(ln(1+fixed_daily)-log1p_auction_volume,2)) sse_daily,sum(pow(ln(1+ridge_update)-log1p_auction_volume,2)) sse_update,
   sum(abs(fixed_daily-auction_volume)) ae_daily,sum(abs(ridge_update-auction_volume)) ae_update,sum(fixed_daily) pred_daily,sum(ridge_update) pred_update
   FROM p GROUP BY 1,2,3,4 ORDER BY year,date''').df())
 daily=pd.concat(pieces);daily.to_csv(OUT/'daily_sufficient_statistics.csv',index=False)
 calendars={y:pd.DatetimeIndex(daily[daily.dimension.eq('overall')&daily.year.eq(y)].sort_values('date').date) for y in [2022,2023,2024,2025]}
 arrays={};annual=[];raw_annual=pd.read_csv(SOURCE/'annual_metrics.csv')
 for dim,(_,labels) in dimensions.items():
  for seg in labels:
   for y in calendars:
    part=daily[daily.dimension.eq(dim)&daily.segment.eq(seg)&daily.year.eq(y)].set_index('date').reindex(calendars[y]);arr=part[COLS].fillna(0).to_numpy(float);arrays[(dim,seg,y)]=arr
    totals=arr.sum(axis=0);s=statistics(totals);dates=int((arr[:,0]>0).sum())
    for i,name in enumerate(['fixed_daily','ridge_update']):
     annual.append(dict(year=y,dimension=dim,segment=seg,candidate=name,observations=int(totals[0]),dates=dates,actual_share_total=totals[1],**dict(zip(METRICS,s[i]))))
     if dim=='overall':
      old=raw_annual[raw_annual.test_year.eq(y)&raw_annual.candidate.eq(name)&raw_annual['sample'].eq('all_regular_sessions')].iloc[0]
      check(f'{y}_{name}_annual_reconciliation',int(totals[0])==old.observations and np.allclose(s[i,:3],[old.rmse_log,old.wape,old.aggregate_bias],rtol=1e-11,atol=1e-12))
  for y in calendars:
   check(f'{dim}_{y}_exhaustive_daily_counts',np.array_equal(sum(arrays[(dim,l,y)][:,0] for l in labels),arrays[('overall','all',y)][:,0]))
 a=pd.DataFrame(annual);a.to_csv(OUT/'annual_subgroup_metrics.csv',index=False)
 intervals=[];point=[];weight_records=[]
 for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
  weights={}
  for block in [5,20,60]:
   for y in years:
    weights[(block,y)]=circular_weights(len(calendars[y]),block,REPS,SEED+block*10000+y)
    check(f'{period}_{block}_{y}_weight_sums',np.all(weights[(block,y)].sum(axis=1)==len(calendars[y])))
    weight_records.append(dict(period=period,block_dates=block,year=y,seed=SEED+block*10000+y,dates=len(calendars[y]),replicates=REPS,sha256=hashlib.sha256(weights[(block,y)].tobytes()).hexdigest()))
  for dim,(_,labels) in dimensions.items():
   for seg in labels:
    mats=[arrays[(dim,seg,y)] for y in years];valid=all(m[:,0].sum()>0 and m[:,1].sum()>0 for m in mats)
    metrics=np.mean([statistics(m.sum(axis=0)) for m in mats],axis=0) if valid else np.full((2,4),np.nan)
    for k,metric in enumerate(METRICS):
     point.append(dict(period=period,dimension=dim,segment=seg,metric=metric,baseline=metrics[0,k],candidate=metrics[1,k],difference=metrics[1,k]-metrics[0,k],years_present=sum(m[:,0].sum()>0 for m in mats),years_required=len(years),observations=int(sum(m[:,0].sum() for m in mats)),min_annual_dates=int(min((m[:,0]>0).sum() for m in mats)),min_annual_observations=int(min(m[:,0].sum() for m in mats))))
    eligible=valid and all((m[:,0]>0).sum()>=30 and m[:,0].sum()>=500 for m in mats)
    blocks=[5,20,60] if dim=='overall' else [20]
    for block in blocks:
     reason='';draws=None
     if not eligible:reason='insufficient dates/rows, missing year, or zero actual-share denominator'
     else:
      stats=[statistics(weights[(block,y)]@m) for y,m in zip(years,mats)]
      if not all(np.isfinite(v).all() for v in stats):reason='at least one bootstrap replicate has an empty/invalid denominator'
      else:
       b=np.mean(stats,axis=0);draws=b[:,1,:]-b[:,0,:]
       check(f'{period}_{dim}_{seg}_{block}_finite_replicates',np.isfinite(draws).all())
     for k,metric in enumerate(METRICS):
      lo,hi=np.quantile(draws[:,k],[.025,.975]) if draws is not None else (np.nan,np.nan)
      intervals.append(dict(period=period,dimension=dim,segment=seg,metric=metric,block_dates=block,replicates=REPS if draws is not None else 0,difference=metrics[1,k]-metrics[0,k],ci_lower=lo,ci_upper=hi,status='reported' if draws is not None else 'suppressed',reason=reason,interpretation='primary retrospective contrast' if dim=='overall' and metric=='rmse_log' and block==20 else 'descriptive unadjusted interval'))
 pd.DataFrame(point).to_csv(OUT/'period_metric_differences.csv',index=False);i=pd.DataFrame(intervals);i.to_csv(OUT/'metric_intervals.csv',index=False);pd.DataFrame(weight_records).to_csv(OUT/'bootstrap_weight_fingerprints.csv',index=False)
 # Independent row-level metric check on a deterministic slice.
 sl=c.sql('SELECT * FROM p ORDER BY date,permno LIMIT 333').df()
 totals=np.array([len(sl),sl.auction_volume.sum(),((np.log1p(sl.fixed_daily)-np.log1p(sl.auction_volume))**2).sum(),((np.log1p(sl.ridge_update)-np.log1p(sl.auction_volume))**2).sum(),abs(sl.fixed_daily-sl.auction_volume).sum(),abs(sl.ridge_update-sl.auction_volume).sum(),sl.fixed_daily.sum(),sl.ridge_update.sum()])
 s=statistics(totals)
 for j,name in enumerate(['fixed_daily','ridge_update']):
  truth=sl.auction_volume.to_numpy();pred=sl[name].to_numpy();expected=[np.sqrt(np.mean((np.log1p(pred)-np.log1p(truth))**2)),abs(pred-truth).sum()/truth.sum(),pred.sum()/truth.sum()-1]
  check(name+'_independent_row_level_formula',np.allclose(s[j,:3],expected,rtol=1e-12))
 check('protocol_unchanged',sha(OUT/'evaluation_protocol.md')==protocol_hash)
 pd.DataFrame(checks).to_csv(OUT/'validation_checks.csv',index=False);pd.DataFrame(inputs).to_csv(OUT/'input_fingerprints.csv',index=False)
 (OUT/'evaluation_summary.json').write_text(json.dumps({'all_checks_passed':all(x['passed'] for x in checks),'checks':len(checks),'rows':manifest['regular_session_rows'],'replicates':REPS,'primary_block_dates':20,'sensitivity_blocks':[5,60],'protocol_sha256':protocol_hash,'script_sha256':sha(Path(__file__)),'forward_period_scored':False,'retrospective_only':True},indent=2))
 print(i[(i.dimension=='overall')&(i.block_dates==20)][['period','metric','difference','ci_lower','ci_upper']].to_string(index=False),flush=True)
 print('Checks passed:',len(checks),flush=True)
if __name__=='__main__':run()
