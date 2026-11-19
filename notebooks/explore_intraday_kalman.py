"""Chronological multi-checkpoint ridge and Kalman research; no execution control.

The annual daily anchors are already out of time. Within each evaluation fold,
checkpoint components are selected on Q2 of the preceding year, refit through
June, then frozen. Q3 calibrates filter measurements; Q4 selects filter settings.
After selection, measurement calibration uses Q3+Q4; components remain frozen.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gc
import hashlib
import json
from datetime import datetime, timezone
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from notebooks.modeling_utils import fit_preprocessor, apply_preprocessor, predict_ridge
from notebooks.build_intraday_1530_update import DAILY_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/experiments/33_intraday_kalman'
ANCHORS = ROOT / 'outputs/experiments/27_intraday_anchor_comparison'
PANEL = next(p for p in [ROOT/'outputs/reproduction/01_daily_panel/enriched_daily_panel.parquet',
    ROOT/'outputs/eda/05_feature_engineering/enriched_daily_panel.parquet',
    ROOT/'midterm presentation/data_analysis/cache/enriched_daily_panel.parquet'] if p.is_file())
TIMES = ['10:30','12:00','14:00','15:00','15:30']
TAGS = ['1030','1200','1400','1500','1530']
YEARS = [2022,2023,2024,2025]
DEV = [2022,2023,2024]
GRID = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
CP = ['cp_cum_volume','cp_cum_dollar_volume','cp_cum_trade_count','cp_recent_30m_volume',
      'cp_vwap','cp_recent_30m_volume_share','checkpoint_avg_volume20','checkpoint_avg_trades20',
      'ratio20_auction_to_checkpoint','cp_volume_to_adv20','cp_volume_surprise20',
      'cp_trade_count_surprise20','cp_vwap_return_prior_close']
# lag1_price is common, giving 58+log forecast+lag price+13 per checkpoint = 73.
COMMON = list(dict.fromkeys([*DAILY_FEATURES,'log_daily_forecast','lag1_price']))
BOOLEAN = {'auction_spike_lag1','is_monday','is_friday','is_first_trading_day_month','is_last_trading_day_month',
 'is_last_3_trading_days_month','is_last_5_trading_days_month','is_last_trading_day_quarter',
 'is_last_3_trading_days_quarter','is_last_5_trading_days_quarter','is_last_trading_day_year',
 'monthly_options_expiry','quarterly_options_expiry','after_long_market_gap','before_long_market_gap',
 'is_early_close','early_close_month_end_window','expiry_month_end_window'}
CATEGORICAL = {'month','day_of_week'}
SEED=20261118
FAMILIES=['kalman_diagonal','kalman_inflated','kalman_correlated','kalman_robust']
METHODS=['daily','latest_ridge','multi_checkpoint_ridge','signal_stack_ridge',*FAMILIES,'incumbent_1530']


def progress(message):
    OUT.mkdir(parents=True,exist_ok=True)
    line=f'{datetime.now(timezone.utc).isoformat()} {message}'
    print(line,flush=True)
    with (OUT/'progress.log').open('a') as f:f.write(line+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    p=Path(path);temporary=p.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2));temporary.replace(p)


def save_frame(con,frame,path):
    temporary=Path(path).with_suffix('.partial');temporary.unlink(missing_ok=True)
    con.register('kalman_output',frame)
    try:con.execute('COPY kalman_output TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(temporary)])
    finally:con.unregister('kalman_output')
    temporary.replace(path)


def score(actual,pred):
    actual=np.asarray(actual,float);pred=np.asarray(pred,float)
    assert np.isfinite(actual).all() and np.isfinite(pred).all() and (pred>=0).all()
    err=np.log1p(pred)-np.log1p(actual)
    return {'observations':len(actual),'rmse_log':float(np.sqrt(np.mean(err**2))),
            'mae_log':float(np.mean(np.abs(err))),'wape':float(np.abs(pred-actual).sum()/actual.sum()),
            'aggregate_bias':float(pred.sum()/actual.sum()-1)}


def fit_grid(matrix,target):
    """Reuse the exact ridge normal equations across penalties; intercept unpenalised."""
    x=np.column_stack([np.ones(len(matrix),np.float32),matrix]).astype(float)
    gram=x.T@x/len(x);rhs=x.T@np.asarray(target,float)/len(x)
    result=[]
    for strength in GRID:
        penalty=np.eye(x.shape[1])*strength;penalty[0,0]=0
        result.append(np.linalg.solve(gram+penalty,rhs))
    return result


def encode(frame,features):
    types=([f for f in features if f not in BOOLEAN|CATEGORICAL],
           [f for f in features if f in BOOLEAN],[f for f in features if f in CATEGORICAL])
    return fit_preprocessor(frame,*types)


def transform(frame,prep):
    return apply_preprocessor(frame,prep,*[list(prep[k]) for k in ['continuous','boolean','categorical']])


def forecast(frame,residual,bounds,available):
    daily=frame.daily_forecast.to_numpy(float)
    pred=np.maximum(np.expm1(np.clip(np.log1p(daily)+residual,*bounds)),0)
    return np.where(available,pred,daily)


def prepare():
    for p in [OUT,OUT/'panels',OUT/'models',OUT/'predictions',OUT/'samples']:p.mkdir(parents=True,exist_ok=True)
    plan={'version':1,'years':YEARS,'times':TIMES,'target':'log1p(auction)-log1p(tuned daily)',
      'component_tune':'2021-01-01 through March preceding evaluation year',
      'component_validation':'Q2 preceding evaluation year','component_final_fit':'2021 onward through June; frozen thereafter',
      'meta_calibration':'Q3 preceding evaluation year','meta_validation':'Q4 preceding evaluation year',
      'meta_final_fit':'Q3+Q4, using genuinely out-of-time frozen component measurements',
      'row_caps':{'component_tune':250000,'validation':120000,'component_refit':350000},
      'ridge_grid':GRID,'kalman_selection':'Q4 log RMSE at 15:30; grid fixed before scoring',
      'within_day_state':'fixed terminal log residual; correlated static filter has Q=0',
      'inflation':'diagonal-only sensitivity; Q/P0 fractions 0.01,0.1,0.5 per elapsed hour',
      'missing_policy':'latest-only uses daily if latest missing; sequential methods retain earlier measurements; all-missing exact daily',
      'gate':{'mean_log_rmse_gain':.002,'maximum_annual_deterioration':.002,'no_mean_wape_worsening':True,
              'paired_log_rmse_upper_below_zero':True,'references':['latest_ridge','incumbent_1530']},
      'uncertainty':{'block_dates':20,'replicates':5000,'conditional_on_fits':True},
      'code_hash':sha(Path(__file__)),'panel_hash':sha(PANEL),
      'raw_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'blackrock_data/stock_intraday/trade_checkpoints').glob('year=*/*.parquet'))},
      'anchor_hashes':{p.name:sha(p) for p in sorted(ANCHORS.glob('tree_*.parquet'))},
      'incumbent_hash':sha(ANCHORS/'matched_predictions.parquet')}
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan,'Changed plan: use a new run or explicitly rebuild generated outputs.'
    else:write_json(OUT/'plan.json',plan)
    con=duckdb.connect(config={'threads':4,'memory_limit':'2GB'})
    raw=str(ROOT/'blackrock_data/stock_intraday/trade_checkpoints/year=*/trade_checkpoints_*.parquet')
    for tag,time in zip(TAGS,TIMES):
        destination=OUT/f'panels/checkpoint_{tag}.parquet'
        if destination.exists():continue
        progress(f'Building prior-only checkpoint histories {time}')
        con.execute(f"CREATE OR REPLACE TEMP VIEW raw_cp AS SELECT * FROM read_parquet('{raw}',union_by_name=true) WHERE checkpoint='{time}'")
        assert con.sql('SELECT count(*)-count(DISTINCT(date,permno)) FROM raw_cp').fetchone()[0]==0
        valid='coalesce(c.checkpoint_data_found,false) AND NOT coalesce(c.checkpoint_after_market_close,true) AND NOT coalesce(p.is_early_close,true) AND c.cum_volume>0'
        source_fields=[('cum_volume','cp_cum_volume'),('cum_dollar_volume','cp_cum_dollar_volume'),
                       ('cum_trade_count','cp_cum_trade_count'),('recent_30m_volume','cp_recent_30m_volume'),
                       ('vwap','cp_vwap'),('recent_30m_volume_share','cp_recent_30m_volume_share')]
        expressions=','.join(f'CASE WHEN {valid} THEN c.{a} END AS {b}' for a,b in source_fields)
        query=f"""WITH joined AS MATERIALIZED(
          SELECT p.date,p.permno,p.year,p.eligible_target,p.auction_volume,p.avg20_daily_volume,p.lag1_price,
                 coalesce({valid},false) AS available,{expressions}
          FROM read_parquet('{PANEL}') p LEFT JOIN raw_cp c USING(date,permno)
        ), history AS(
          SELECT *,avg(cp_cum_volume) OVER w AS checkpoint_avg_volume20,
          avg(cp_cum_trade_count) OVER w AS checkpoint_avg_trades20,
          median(CASE WHEN eligible_target AND available THEN auction_volume/cp_cum_volume END) OVER w AS ratio20_auction_to_checkpoint
          FROM joined WINDOW w AS(PARTITION BY permno ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
        ) SELECT date,permno,available,{','.join(CP[:9])},
          cp_cum_volume/nullif(avg20_daily_volume,0) cp_volume_to_adv20,
          cp_cum_volume/nullif(checkpoint_avg_volume20,0) cp_volume_surprise20,
          cp_cum_trade_count/nullif(checkpoint_avg_trades20,0) cp_trade_count_surprise20,
          cp_vwap/nullif(lag1_price,0)-1 cp_vwap_return_prior_close
          FROM history WHERE eligible_target AND year BETWEEN 2021 AND 2025"""
        temporary=destination.with_suffix('.partial');temporary.unlink(missing_ok=True)
        con.execute(f'COPY ({query}) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(temporary)]);temporary.replace(destination)
    destination=OUT/'panels/wide_panel.parquet'
    if not destination.exists():
        anchors=[str(ANCHORS/f'tree_{y}.parquet') for y in range(2021,2026)]
        con.read_parquet(anchors).create_view('daily_anchors')
        fields=list(dict.fromkeys(['date','permno','year','auction_volume','log1p_auction_volume','size_percentile','liquidity_percentile',
                                  'quarterly_options_expiry','monthly_options_expiry','is_last_trading_day_month',*COMMON]))
        # The existing original-anchor log forecast is replaced by its tuned counterpart.
        base=','.join('p."'+f+'"' for f in fields if f!='log_daily_forecast')
        projections=[base,'d.forecast AS daily_forecast','ln(1+d.forecast) AS log_daily_forecast']
        joins=[]
        for tag in TAGS:
            path=OUT/f'panels/checkpoint_{tag}.parquet'
            joins.append(f"JOIN read_parquet('{path}') c{tag} USING(date,permno)")
            projections.extend([f'c{tag}.available AS available_{tag}',*[f'c{tag}.{f} AS "{tag}_{f}"' for f in CP]])
        query=f"""SELECT {','.join(projections)} FROM read_parquet('{ANCHORS/'checkpoint_panel.parquet'}') p
                 JOIN daily_anchors d USING(date,permno) {' '.join(joins)} WHERE NOT p.is_early_close ORDER BY p.date,p.permno"""
        temporary=destination.with_suffix('.partial');temporary.unlink(missing_ok=True)
        con.execute(f'COPY ({query}) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(temporary)]);temporary.replace(destination)
    con.read_parquet(str(destination)).create_view('panel')
    assert con.sql('SELECT count(*)-count(DISTINCT(date,permno)) FROM panel').fetchone()[0]==0
    assert con.sql('SELECT count(*) FROM panel WHERE year BETWEEN 2022 AND 2025').fetchone()[0]==1260168
    coverage=con.sql('SELECT year,count(*) AS rows,'+','.join(f'count(*) FILTER(WHERE available_{t}) AS valid_{t}' for t in TAGS)+' FROM panel GROUP BY year ORDER BY year').df()
    coverage.to_csv(OUT/'coverage.csv',index=False)
    con.close();return coverage


def measure_calibration(target,measurements):
    """Fit m_j=a_j+H_j*r+epsilon_j on time-held-out component predictions."""
    r=np.asarray(target,float);m=np.asarray(measurements,float)
    mu=float(np.mean(r));p=max(float(np.var(r)),1e-6)
    h=[];a=[]
    for j in range(m.shape[1]):
        valid=np.isfinite(m[:,j]);rv=r[valid];mv=m[valid,j]
        loading=float(np.mean((rv-rv.mean())*(mv-mv.mean()))/max(np.var(rv),1e-6))
        h.append(loading);a.append(float(mv.mean()-loading*rv.mean()))
    h=np.array(h);a=np.array(a)
    eps=m-a-r[:,None]*h
    complete=np.isfinite(eps).all(axis=1)
    assert complete.sum()>1000
    covariance=np.cov(eps[complete],rowvar=False,bias=True)
    covariance+=np.eye(len(h))*1e-6
    return {'mu':mu,'p':p,'h':h.tolist(),'a':a.tolist(),'R':covariance.tolist(),
            'rows':len(r),'complete_rows':int(complete.sum())}


def kalman(measurements,calibration,config):
    """Sequential scalar updates after causal Cholesky whitening, grouped by missingness.

    For correlated measurement noise, each whitened observation uses only earlier
    observed measurements. Q inflation is restricted to diagonal-R experiments.
    """
    m=np.asarray(measurements,float);n,k=m.shape
    h=np.array(calibration['h']);a=np.array(calibration['a']);r=np.array(calibration['R'])
    shrinkage=config.get('shrinkage',1.0);r=(1-shrinkage)*r+shrinkage*np.diag(np.diag(r))
    mu=calibration['mu'];p0=calibration['p']*config.get('prior_scale',1.0)
    q=config.get('q_fraction',0)*p0
    assert q==0 or shrinkage==1
    clip=config.get('innovation_clip',None)
    estimates=np.full((n,k),mu);variances=np.full((n,k),p0)
    mask=np.isfinite(m);codes=mask@ (1<<np.arange(k))
    elapsed=np.array([0,1.5,2.0,1.0,.5])[:k]
    for pattern in np.unique(codes):
        rows=np.flatnonzero(codes==pattern);observed=np.flatnonzero((pattern&(1<<np.arange(k)))>0)
        if len(observed)==0:continue
        chol=np.linalg.cholesky(r[np.ix_(observed,observed)])
        inv=np.linalg.solve(chol,np.eye(len(observed)))
        whitened=(m[np.ix_(rows,observed)]-a[observed])@inv.T
        hw=inv@h[observed]
        state=np.full(len(rows),mu);p=p0;index=0
        for j in range(k):
            p+=q*elapsed[j]
            if index<len(observed) and observed[index]==j:
                loading=hw[index];innovation=whitened[:,index]-loading*state
                s=loading**2*p+1;gain=p*loading/s
                if clip is not None:innovation=np.clip(innovation,-clip*np.sqrt(s),clip*np.sqrt(s))
                state+=gain*innovation
                p=(1-gain*loading)**2*p+gain**2
                index+=1
            estimates[rows,j]=state;variances[rows,j]=p
    return estimates,variances


def filter_grid():
    rows=[]
    for scale in [.5,1.,2.]:rows.append(('kalman_diagonal',{'shrinkage':1.,'prior_scale':scale}))
    for q in [.01,.1,.5]:rows.append(('kalman_inflated',{'shrinkage':1.,'q_fraction':q}))
    for shrinkage in [0.,.25,.5,.75]:
        for scale in [.5,1.,2.]:rows.append(('kalman_correlated',{'shrinkage':shrinkage,'prior_scale':scale}))
    for shrinkage in [.25,.75]:
        for clip in [2.5,4.]:rows.append(('kalman_robust',{'shrinkage':shrinkage,'innovation_clip':clip}))
    return rows


def stack_design(measurements):
    valid=np.isfinite(measurements)
    return np.column_stack([np.where(valid,measurements,0),~valid]).astype(np.float32)


def fit_year(year):
    complete=OUT/f'predictions/forecasts_{year}.parquet'
    if complete.exists() and (OUT/f'models/meta_{year}.json').exists():
        progress(f'{year}: complete cache found; reconstructed by verify_year')
        return verify_year(year)
    progress(f'{year}: fitting checkpoint components with chronological Q2 validation')
    con=duckdb.connect(config={'threads':1,'memory_limit':'2GB'})
    path=OUT/'panels/wide_panel.parquet'
    def load(start,end,cap=None,offset=0):
        sample=f'USING SAMPLE {cap} ROWS (reservoir,{SEED+year+offset})' if cap else ''
        return con.execute(f"SELECT * FROM (SELECT * FROM read_parquet(?) WHERE date>=DATE '{start}' AND date<DATE '{end}' ORDER BY date,permno) {sample} ORDER BY date,permno",[str(path)]).df()
    q2=f'{year-1}-04-01';q3=f'{year-1}-07-01';q4=f'{year-1}-10-01';end=f'{year}-01-01'
    train=load('2021-01-01',q2,250000,0)
    validation=load(q2,q3,120000,100)
    refit=load('2021-01-01',q3,350000,200)
    calibration=load(q3,q4);selection=load(q4,end)
    test=load(end,f'{year+1}-01-01')
    assert train.date.max()<validation.date.min()<=validation.date.max()<calibration.date.min()
    assert refit.date.max()<calibration.date.min()<=calibration.date.max()<selection.date.min()<=selection.date.max()<test.date.min()
    manifest=[]
    for role,frame in [('component_tune',train),('component_validation',validation),('component_refit',refit),
                       ('meta_calibration',calibration),('meta_validation',selection),('test',test)]:
        manifest.append({'year':year,'role':role,'rows':len(frame),'first':str(frame.date.min()),'last':str(frame.date.max()),
            'key_hash':hashlib.sha256(pd.util.hash_pandas_object(frame[['date','permno']],index=False).to_numpy().tobytes()).hexdigest()})
    pd.DataFrame(manifest).to_csv(OUT/f'samples/manifest_{year}.csv',index=False)
    bounds=np.quantile(refit.log1p_auction_volume,[.001,.999]).tolist()
    tune_bounds=np.quantile(train.log1p_auction_volume,[.001,.999]).tolist()
    target_train=(train.log1p_auction_volume-train.log_daily_forecast).to_numpy()
    target_refit=(refit.log1p_auction_volume-refit.log_daily_forecast).to_numpy()
    matrices={'calibration':np.full((len(calibration),5),np.nan),'selection':np.full((len(selection),5),np.nan),'test':np.full((len(test),5),np.nan)}
    latest=np.empty((len(test),5));multi=np.empty((len(test),5))
    trials=[];model_records=[]
    for j,tag in enumerate(TAGS):
        for family in ['latest','multi']:
            if family=='multi' and j==0:
                multi[:,0]=latest[:,0];continue
            prefix=[tag] if family=='latest' else TAGS[:j+1]
            features=COMMON+[f'{t}_{f}' for t in prefix for f in CP]
            fit_available=train[[f'available_{t}' for t in prefix]].any(axis=1).to_numpy()
            validation_available=validation[[f'available_{t}' for t in prefix]].any(axis=1).to_numpy()
            refit_available=refit[[f'available_{t}' for t in prefix]].any(axis=1).to_numpy()
            xt,prep=encode(train.loc[fit_available],features)
            xv=transform(validation,prep)
            coefficients=fit_grid(xt,target_train[fit_available])
            choices=[]
            for strength,coefficient in zip(GRID,coefficients):
                pred=forecast(validation,predict_ridge(xv,coefficient),tune_bounds,validation_available)
                choices.append({'year':year,'family':family,'checkpoint':TIMES[j],'strength':strength,
                                'validation_log_rmse':score(validation.auction_volume,pred)['rmse_log']})
            trials.extend(choices);best=min(choices,key=lambda x:x['validation_log_rmse'])
            del xt,xv,coefficients
            xr,prep=encode(refit.loc[refit_available],features)
            # Compute all regularised systems together; retain the historical selection.
            beta=fit_grid(xr,target_refit[refit_available])[GRID.index(best['strength'])]
            settings={'year':year,'family':family,'checkpoint':TIMES[j],'features':features,'preprocessor':prep,
                      'coefficient':beta.tolist(),'bounds':bounds,'strength':best['strength'],
                      'refit_end':str(refit.date.max()),'refit_rows':int(refit_available.sum())}
            write_json(OUT/f'models/{family}_{tag}_{year}.json',settings);model_records.append(best)
            for role,frame in [('calibration',calibration),('selection',selection),('test',test)]:
                if family=='multi' and role!='test':continue
                residual=predict_ridge(transform(frame,prep),beta)
                if family=='latest':
                    available=frame[f'available_{tag}'].to_numpy(bool)
                    matrices[role][:,j]=np.where(available,residual,np.nan)
                    if role=='test':latest[:,j]=forecast(frame,residual,bounds,available)
                else:
                    available=frame[[f'available_{t}' for t in prefix]].any(axis=1).to_numpy()
                    multi[:,j]=forecast(frame,residual,bounds,available)
            del xr;gc.collect()
        progress(f'{year}: checkpoint {TIMES[j]} components fitted')
    pd.DataFrame(trials).to_csv(OUT/f'component_validation_{year}.csv',index=False)
    del train,validation,refit;gc.collect()
    target_cal=(calibration.log1p_auction_volume-calibration.log_daily_forecast).to_numpy()
    target_select=(selection.log1p_auction_volume-selection.log_daily_forecast).to_numpy()
    cal=measure_calibration(target_cal,matrices['calibration']);meta_trials=[];chosen={}
    select_available=np.isfinite(matrices['selection']).any(axis=1)
    for family,config in filter_grid():
        estimates,_=kalman(matrices['selection'],cal,config)
        pred=forecast(selection,estimates[:,-1],bounds,select_available)
        row={'family':family,'config':config,'validation_log_rmse':score(selection.auction_volume,pred)['rmse_log']}
        meta_trials.append(row)
        if family not in chosen or row['validation_log_rmse']<chosen[family]['validation_log_rmse']:chosen[family]=row
    # Final measurement calibration adds Q4 after selecting settings, keeping component models fixed.
    full_target=np.r_[target_cal,target_select];full_measurements=np.vstack([matrices['calibration'],matrices['selection']])
    final_cal=measure_calibration(full_target,full_measurements)
    output=test[['date','permno','year','auction_volume','daily_forecast','size_percentile','liquidity_percentile',
                 'quarterly_options_expiry','monthly_options_expiry','is_last_trading_day_month']].copy()
    for j,tag in enumerate(TAGS):
        output[f'available_{tag}']=test[f'available_{tag}']
        output[f'measurement_{tag}']=matrices['test'][:,j]
        output[f'daily_{tag}']=test.daily_forecast
        output[f'latest_ridge_{tag}']=latest[:,j]
        output[f'multi_checkpoint_ridge_{tag}']=multi[:,j]
    stack_models=[]
    for j,tag in enumerate(TAGS):
        xc=stack_design(matrices['calibration'][:,:j+1]);xv=stack_design(matrices['selection'][:,:j+1])
        choices=[];valid=np.isfinite(matrices['selection'][:,:j+1]).any(axis=1)
        for strength,beta in zip(GRID,fit_grid(xc,target_cal)):
            pred=forecast(selection,predict_ridge(xv,beta),bounds,valid)
            choices.append({'family':'signal_stack_ridge','checkpoint':TIMES[j],'strength':strength,
                            'validation_log_rmse':score(selection.auction_volume,pred)['rmse_log']})
        best=min(choices,key=lambda x:x['validation_log_rmse']);meta_trials.extend(choices)
        beta=fit_grid(stack_design(full_measurements[:,:j+1]),full_target)[GRID.index(best['strength'])]
        stack_models.append({'tag':tag,'strength':best['strength'],'coefficient':beta.tolist()})
        residual=predict_ridge(stack_design(matrices['test'][:,:j+1]),beta)
        output[f'signal_stack_ridge_{tag}']=forecast(test,residual,bounds,np.isfinite(matrices['test'][:,:j+1]).any(axis=1))
    for family,row in chosen.items():
        estimates,variance=kalman(matrices['test'],final_cal,row['config'])
        for j,tag in enumerate(TAGS):
            available=np.isfinite(matrices['test'][:,:j+1]).any(axis=1)
            output[f'{family}_{tag}']=forecast(test,estimates[:,j],bounds,available)
            output[f'{family}_variance_{tag}']=variance[:,j]
    incumbent=con.execute('SELECT date,permno,auction_volume,tree_ridge_1530 FROM read_parquet(?) WHERE year(date)=? ORDER BY date,permno',
                          [str(ANCHORS/'matched_predictions.parquet'),year]).df()
    pd.testing.assert_frame_equal(output[['date','permno','auction_volume']],incumbent[['date','permno','auction_volume']],check_dtype=False)
    output['incumbent_1530']=incumbent.tree_ridge_1530.to_numpy()
    save_frame(con,output,complete)
    pd.DataFrame([{**r,'config':json.dumps(r.get('config'))} for r in meta_trials]).to_csv(OUT/f'meta_validation_{year}.csv',index=False)
    write_json(OUT/f'models/meta_{year}.json',{'year':year,'calibration':final_cal,'tuning_calibration':cal,'selected':chosen,
       'stack_models':stack_models,'bounds':bounds,'component_refit_end':f'{year-1}-06-30',
       'calibration_first':str(calibration.date.min()),'calibration_last':str(selection.date.max()),
       'test_first':str(test.date.min()),'component_models_frozen':True})
    con.close();progress(f'{year}: wrote all sequential forecasts')
    return verify_year(year)


def verify_year(year):
    con=duckdb.connect(config={'threads':4})
    path=OUT/f'predictions/forecasts_{year}.parquet';frame=con.execute('SELECT * FROM read_parquet(?) ORDER BY date,permno',[str(path)]).df()
    test=con.execute('SELECT * FROM read_parquet(?) WHERE year=? ORDER BY date,permno',[str(OUT/'panels/wide_panel.parquet'),year]).df()
    pd.testing.assert_frame_equal(frame[['date','permno','auction_volume']],test[['date','permno','auction_volume']],check_dtype=False)
    assert not frame.duplicated(['date','permno']).any()
    meta=json.loads((OUT/f'models/meta_{year}.json').read_text())
    assert pd.Timestamp(meta['component_refit_end'])<pd.Timestamp(meta['calibration_first'])
    assert pd.Timestamp(meta['calibration_last'])<pd.Timestamp(meta['test_first'])
    measurements=frame[[f'measurement_{t}' for t in TAGS]].to_numpy(float)
    metrics=[];diagnostics=[]
    for j,tag in enumerate(TAGS):
        for family in ['latest','multi']:
            if family=='multi' and j==0:
                np.testing.assert_array_equal(frame[f'multi_checkpoint_ridge_{tag}'],frame[f'latest_ridge_{tag}']);continue
            settings=json.loads((OUT/f'models/{family}_{tag}_{year}.json').read_text())
            assert set(settings['features']).isdisjoint({'auction_volume','log1p_auction_volume','daily_forecast','current_daily_volume','current_return'})
            assert not any(f.startswith(t+'_') for f in settings['features'] for t in TAGS[j+1:])
            prefix=[tag] if family=='latest' else TAGS[:j+1]
            valid=test[[f'available_{t}' for t in prefix]].any(axis=1).to_numpy()
            residual=predict_ridge(transform(test,settings['preprocessor']),np.array(settings['coefficient']))
            column=f'{family}_ridge_{tag}' if family=='latest' else f'multi_checkpoint_ridge_{tag}'
            np.testing.assert_allclose(forecast(test,residual,settings['bounds'],valid),frame[column],rtol=1e-9,atol=1e-7)
        valid=np.isfinite(measurements[:,:j+1]).any(axis=1)
        stack=meta['stack_models'][j]
        rebuilt=forecast(test,predict_ridge(stack_design(measurements[:,:j+1]),np.array(stack['coefficient'])),meta['bounds'],valid)
        np.testing.assert_allclose(rebuilt,frame[f'signal_stack_ridge_{tag}'],rtol=1e-9,atol=1e-7)
        for family in METHODS[:-1]:
            pred=frame[f'{family}_{tag}'].to_numpy()
            metrics.append({'year':year,'checkpoint':TIMES[j],'method':family,**score(frame.auction_volume,pred)})
            use_latest=family=='latest_ridge'
            available=test[f'available_{tag}'].to_numpy(bool) if use_latest else valid
            np.testing.assert_array_equal(pred[~available],test.loc[~available,'daily_forecast'])
    for family,row in meta['selected'].items():
        mean,variance=kalman(measurements,meta['calibration'],row['config'])
        assert (variance>0).all() and np.isfinite(variance).all()
        for j,tag in enumerate(TAGS):
            valid=np.isfinite(measurements[:,:j+1]).any(axis=1)
            pred=forecast(test,mean[:,j],meta['bounds'],valid)
            np.testing.assert_allclose(pred,frame[f'{family}_{tag}'],rtol=1e-9,atol=1e-7)
            # Future values and missingness must not affect a prefix forecast.
            if year==2022:
                altered=measurements[:128].copy();altered[:,j+1:]=999
                other,_=kalman(altered,meta['calibration'],row['config'])
                np.testing.assert_allclose(other[:,j],mean[:128,j],rtol=1e-12,atol=1e-12)
            error=np.log1p(frame.auction_volume.to_numpy())-np.log1p(pred)
            sd=np.sqrt(variance[:,j]);z=error/sd
            diagnostics.append({'year':year,'checkpoint':TIMES[j],'method':family,
                'coverage_80':float((np.abs(z)<=1.2815515655).mean()),'coverage_95':float((np.abs(z)<=1.9599639845).mean()),
                'mean_sd_log':float(sd.mean()),'standardized_rmse':float(np.sqrt(np.mean(z**2)))})
    metrics.append({'year':year,'checkpoint':'15:30','method':'incumbent_1530',**score(frame.auction_volume,frame.incumbent_1530)})
    pd.DataFrame(metrics).to_csv(OUT/f'annual_metrics_{year}.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/f'uncertainty_diagnostics_{year}.csv',index=False)
    con.close();progress(f'{year}: model reconstruction, causal prefix, fallback and finite-state checks passed')
    return pd.DataFrame(metrics)


def verify_math():
    from notebooks.modeling_utils import fit_ridge
    cal={'mu':0.,'p':1.,'h':[1.],'a':[0.],'R':[[1.]]}
    mean,p=kalman(np.array([[2.],[np.nan]]),cal,{'shrinkage':1.})
    np.testing.assert_allclose(mean[:,0],[1.,0.]);np.testing.assert_allclose(p[:,0],[.5,1.])
    cal={'mu':.2,'p':.7,'h':[.6,.9,1.1],'a':[.1,-.2,.05],
         'R':[[.4,.15,.1],[.15,.3,.12],[.1,.12,.5]]}
    obs=np.array([[.8,.9,1.2],[np.nan,.9,1.2]])
    mean,p=kalman(obs,cal,{'shrinkage':0.})
    for i in range(2):
        keep=np.isfinite(obs[i]);h=np.array(cal['h'])[keep];r=np.array(cal['R'])[np.ix_(keep,keep)]
        v=1/(1/cal['p']+h@np.linalg.solve(r,h))
        x=v*(cal['mu']/cal['p']+h@np.linalg.solve(r,obs[i,keep]-np.array(cal['a'])[keep]))
        np.testing.assert_allclose([mean[i,-1],p[i,-1]],[x,v],atol=1e-12)
    rng=np.random.default_rng(1);x=rng.normal(size=(100,4)).astype(np.float32);y=rng.normal(size=100)
    np.testing.assert_allclose(fit_grid(x,y)[GRID.index(.001)],fit_ridge(x,y,.001),atol=1e-12)
    return {'known_single_measurement':True,'missing_measurement_skipped':True,
            'correlated_filter_equals_batch_Gaussian_update':True,'ridge_system_matches_original_helper':True}


def summarize():
    from notebooks.evaluate_frozen_intraday import circular_weights,statistics
    annual=pd.concat([pd.read_csv(OUT/f'annual_metrics_{year}.csv') for year in YEARS],ignore_index=True)
    annual.to_csv(OUT/'annual_metrics.csv',index=False)
    summary=annual.assign(period=np.where(annual.year.eq(2025),'confirmation','development')).groupby(
        ['period','checkpoint','method'])[['rmse_log','mae_log','wape','aggregate_bias']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    uncertainty=pd.concat([pd.read_csv(OUT/f'uncertainty_diagnostics_{y}.csv') for y in YEARS],ignore_index=True)
    uncertainty.to_csv(OUT/'uncertainty_diagnostics.csv',index=False)
    con=duckdb.connect(config={'threads':4})
    paths=[str(OUT/f'predictions/forecasts_{year}.parquet') for year in YEARS]
    con.read_parquet(paths).create_view('predictions')
    candidates=['multi_checkpoint_ridge','signal_stack_ridge',*FAMILIES]
    refs=['latest_ridge','incumbent_1530']
    columns={m:(m if m=='incumbent_1530' else f'{m}_1530') for m in [*candidates,*refs]}
    expressions=['date','year','count(*) n','sum(auction_volume) actual']
    for method,column in columns.items():
        expressions.extend([f'sum(pow(ln(1+{column})-ln(1+auction_volume),2)) AS {method}_sse',
          f'sum(abs({column}-auction_volume)) AS {method}_ae',f'sum({column}) AS {method}_pred'])
    daily=con.sql('SELECT '+','.join(expressions)+' FROM predictions GROUP BY date,year ORDER BY year,date').df()
    daily.to_csv(OUT/'daily_sufficient_statistics.csv',index=False)
    weights={y:circular_weights(int(daily.year.eq(y).sum()),20,5000,SEED+y) for y in YEARS}
    intervals=[]
    for candidate in candidates:
        for reference in refs:
            for period,years in [('development',DEV),('confirmation',[2025])]:
                points=[];draws=[]
                for year in years:
                    d=daily[daily.year.eq(year)]
                    matrix=d[['n','actual',reference+'_sse',candidate+'_sse',reference+'_ae',candidate+'_ae',reference+'_pred',candidate+'_pred']].to_numpy(float)
                    points.append(statistics(matrix.sum(axis=0)));draws.append(statistics(weights[year]@matrix))
                point=np.mean(points,axis=0);sampled=np.mean(draws,axis=0)
                delta=point[1]-point[0];bounds=np.quantile(sampled[:,1]-sampled[:,0],[.025,.975],axis=0)
                for j,metric in enumerate(['rmse_log','wape','signed_bias','absolute_annual_bias']):
                    intervals.append({'candidate':candidate,'reference':reference,'period':period,'metric':metric,
                        'difference':delta[j],'lower':bounds[0,j],'upper':bounds[1,j],'block_dates':20,'replicates':5000})
    intervals=pd.DataFrame(intervals);intervals.to_csv(OUT/'paired_intervals.csv',index=False)
    late=annual[annual.checkpoint.eq('15:30')&annual.year.isin(DEV)]
    pivot=late.pivot(index='method',columns='year',values='rmse_log');wape=late.groupby('method').wape.mean()
    gates=[]
    for candidate in candidates:
        passes=[]
        for reference in refs:
            change=pivot.loc[candidate]-pivot.loc[reference]
            upper=intervals[(intervals.candidate==candidate)&(intervals.reference==reference)&(intervals.period=='development')&(intervals.metric=='rmse_log')].upper.item()
            passed=bool(change.mean()<=-.002 and change.max()<=.002 and wape[candidate]<=wape[reference] and upper<0)
            passes.append(passed)
            gates.append({'candidate':candidate,'reference':reference,'rmse_change':change.mean(),'worst_annual_change':change.max(),
                          'wape_change':wape[candidate]-wape[reference],'paired_rmse_upper':upper,'passes_reference':passed})
    gates=pd.DataFrame(gates);gates['passes_both']=gates.candidate.map(gates.groupby('candidate').passes_reference.all())
    gates.to_csv(OUT/'selection_gate.csv',index=False)
    passers=gates[gates.passes_both].candidate.unique()
    selected=min(passers,key=lambda m:pivot.loc[m].mean()) if len(passers) else 'incumbent_1530'
    # Availability sensitivity makes earlier-observation retention visible.
    complete=[]
    for year in YEARS:
        f=con.execute('SELECT * FROM predictions WHERE year=? ORDER BY date,permno',[year]).df()
        all_available=f[[f'available_{t}' for t in TAGS]].all(axis=1)
        for sample,mask in [('all_checkpoints_available',all_available),('latest_checkpoint_available',f.available_1530),('latest_checkpoint_missing',~f.available_1530)]:
            if mask.sum()==0:continue
            for method in METHODS:
                column=method if method=='incumbent_1530' else f'{method}_1530'
                complete.append({'year':year,'sample':sample,'method':method,**score(f.loc[mask,'auction_volume'],f.loc[mask,column])})
    pd.DataFrame(complete).to_csv(OUT/'availability_sensitivity.csv',index=False)
    # Predetermined group diagnostics; realized tails are explicitly ex post.
    segments=[]
    for year in YEARS:
        f=con.execute('SELECT * FROM predictions WHERE year=? ORDER BY date,permno',[year]).df()
        groups={'liquidity':np.where(f.liquidity_percentile.isna(),'missing',np.where(f.liquidity_percentile<1/3,'low',np.where(f.liquidity_percentile<2/3,'mid','high'))),
          'event':np.where(f.quarterly_options_expiry,'quarterly_expiry',np.where(f.monthly_options_expiry,'monthly_expiry',np.where(f.is_last_trading_day_month,'month_end','regular')))}
        for dimension,labels in groups.items():
            for label in sorted(set(labels)):
                mask=labels==label
                for method in METHODS:
                    column=method if method=='incumbent_1530' else method+'_1530'
                    segments.append({'year':year,'dimension':dimension,'segment':label,'method':method,
                       'dates':int(f.loc[mask,'date'].nunique()),**score(f.loc[mask,'auction_volume'],f.loc[mask,column])})
    pd.DataFrame(segments).to_csv(OUT/'annual_segment_metrics.csv',index=False)
    decision={'selected_method':selected,'decision_basis':'equal-year development 2022-2024; gate versus both references',
      'confirmation_previously_inspected':True,'selected_from':candidates,'math_checks':verify_math(),
      'annual_metric_rows':len(annual),'primary_gate_references':refs,
      'completed_utc':datetime.now(timezone.utc).isoformat()}
    write_json(OUT/'decision.json',decision);con.close();return summary


def run():
    prepare();verify_math()
    with threadpool_limits(limits=4):
        for year in YEARS:fit_year(year)
        summarize()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,choices=YEARS)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    prepare();verify_math()
    with threadpool_limits(limits=4):
        if args.summarize:summarize()
        elif args.year:fit_year(args.year)
        elif not args.prepare_only:
            for year in YEARS:fit_year(year)
            summarize()
