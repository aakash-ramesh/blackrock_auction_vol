"""One-covariate-at-a-time associations with auction volume in SHARES.
Uses the isolated panel built by build_panels.py. No model is selected or refitted.
"""
from pathlib import Path
import sys,json,hashlib,datetime
try:import duckdb
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import duckdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;T=OUT/'tables';T.mkdir(exist_ok=True)
con=duckdb.connect();con.execute("SET threads=4");con.execute("SET memory_limit='2GB'");con.execute("SET temp_directory='/tmp/blackrock_covariate_spill'")
P=OUT/'cache/analysis_panel.parquet';con.execute(f"CREATE VIEW panel AS SELECT * FROM read_parquet('{P}')")
catalog=pd.read_csv(ROOT/'outputs/eda/05_feature_engineering/enriched_feature_catalog.csv')
frozen=json.loads((ROOT/'outputs/eda/09_global_finalists/frozen_global_benchmark.json').read_text());chosen=set(frozen['features'])
metadata=json.loads((OUT/'panel_build_metadata.json').read_text())
booleans={'auction_spike_lag1','is_monday','is_friday','is_first_trading_day_month','is_last_trading_day_month','is_last_3_trading_days_month','is_last_5_trading_days_month','is_last_trading_day_quarter','is_last_3_trading_days_quarter','is_last_5_trading_days_quarter','is_last_trading_day_year','monthly_options_expiry','quarterly_options_expiry','after_long_market_gap','before_long_market_gap','is_early_close','early_close_month_end_window','expiry_month_end_window','lag1_dividend_flag','lag1_distribution_flag','lag1_price_adjustment_flag'}
categories={'month','day_of_week','ff12_code','size_decile','liquidity_decile'}
def field_units(c):
 if c=='auction_share_daily_volume' or 'volume_share' in c or 'share' in c and ('auction_share' in c or 'sector_volume' in c):return 'fraction'
 if c.endswith('shares_outstanding_thousands'):return 'thousands of shares'
 if 'return' in c or 'momentum' in c or 'volatility' in c or 'breadth' in c or 'dispersion' in c or 'spread' in c or 'hl_range' in c or 'ratio' in c or 'percentile' in c or 'cv20' in c or 'turnover' in c or 'amihud' in c:return 'native decimal / ratio'
 if 'dollar' in c or 'price_volume' in c or 'market_cap' in c:return 'USD'
 if any(v in c for v in ['volume','auction']) and c not in {'days_since_last_auction','prior_auction_observations'}:return 'shares'
 if any(c.endswith(x) for x in ['close','open','high','low','bid','ask','price','vwap','dividend']):return 'USD per share'
 return 'native units'
def intuition(c,group):
 if 'median' in c and 'auction' in c:return 'Typical recent auction liquidity; the median limits the influence of isolated historical spikes.'
 if ('avg' in c or c.startswith('lag')) and 'auction_volume' in c:return 'Recent auction demand and security scale can persist. The isolated association is not incremental predictive value.'
 if 'auction_share' in c or 'auction_to_adv' in c:return 'Historical auction participation relative to ordinary turnover can separate auction demand from general trading scale.'
 if c in {'auction_cv20','auction_short_long_ratio','auction_spike_lag1'}:return 'Instability or a recent surge may signal an unusual liquidity regime, or may revert. Direction is empirical.'
 if c in {'days_since_last_auction','prior_auction_observations'}:return 'History availability and freshness; may proxy seasoning, source coverage, or time rather than fundamental trading demand.'
 if 'options_expiry' in c or 'expiry_month_end' in c:return 'Scheduled expiry may coincide with synchronized positioning and benchmark-related flows. This is a mechanism hypothesis, not a causal estimate.'
 if group=='Calendar':return 'Scheduled trading opportunities and benchmark dates may concentrate or redistribute auction flow; overlapping flags and trends can confound individual contrasts.'
 if 'amihud' in c:return 'Absolute return per dollar traded proxies illiquidity. A relationship with share volume may reflect stock scale and correlated liquidity measures.'
 if 'quoted_spread' in c:return 'Wider quoted spreads may indicate thinner liquidity; auction demand may differ from continuous-market liquidity.'
 if 'turnover' in c:return 'Trading relative to shares outstanding measures activity intensity rather than sheer firm size.'
 if 'volume' in c or 'price_volume' in c:return 'More active trading can support a larger auction. Same-day total volume contains auction activity and has mechanical overlap with the target.'
 if 'return' in c or 'momentum' in c:return 'Price changes may accompany information and portfolio adjustment. Unsigned auction volume need not vary monotonically with signed returns.'
 if 'volatility' in c or 'range' in c or 'dispersion' in c:return 'Risk and disagreement may alter trading demand and liquidity provision. A univariate sign does not identify their separate channels.'
 if 'market_cap' in c or 'shares_outstanding' in c or group=='Size and history':return 'Firm and share-base scale help explain share counts, but may add little after historical auctions and trading liquidity are included.'
 if 'dividend' in c or 'distribution' in c or 'factor' in c:return 'Corporate actions can change shares, prices, or portfolio flows. Rare events and event timing require caution.'
 if 'price' in c or any(c.endswith(x) for x in ['close','open','high','low','bid','ask']):return 'Price level affects how a dollar order translates into shares. It is correlated with firm size and liquidity and need not have a positive share-volume relationship.'
 if group in {'Industry','Sector state'} or 'ff12' in c:return 'Industry composition and common sector conditions can proxy investor exposure and shared trading activity.'
 return 'Economic or contextual proxy; interpret jointly with availability, scale, and the family-level forecast tests.'
specs=[]
for row in catalog.to_dict('records'):
 c=row['variable'];group=row['group'];unit='date' if group in {'Calendar','Market state'} else 'sector_date' if group=='Sector state' else 'stock_date'
 specs.append(dict(feature=c,layer='engineered_daily',source_variable=c,group=group,kind='categorical' if c in categories else 'binary' if c in booleans else 'numeric',timing='known in advance' if group=='Calendar' else 'prior-close / point-in-time',daily_candidate=True,selected=c in chosen,native_unit=unit,unit=field_units(c),intuition=intuition(c,group)))
for group,fields in [('Stock raw',metadata['raw_numeric_fields']+metadata['raw_categorical_fields']),('Market state',metadata['raw_market_fields']),('Sector state',metadata['raw_sector_fields'])]:
 for raw in fields:
  for prior in [False,True]:
   name=('prior_raw_' if prior else 'raw_')+raw
   specs.append(dict(feature=name,layer='raw_prior_close' if prior else 'raw_same_day',source_variable=raw,group=group,kind='categorical' if raw in metadata['raw_categorical_fields'] else 'numeric',timing='previous available stock trading observation' if prior and group=='Stock raw' else 'previous trading day' if prior else 'same-day after close',daily_candidate=prior,selected=False,native_unit='date' if group=='Market state' else 'sector_date' if group=='Sector state' else 'stock_date',unit=field_units(raw),intuition=intuition(raw,group)))
for c in ['auction_price','auction_share_daily_volume']:
 specs.append(dict(feature=c,layer='outcome_diagnostic',source_variable=c,group='Auction outcomes',kind='numeric',timing='same-day outcome / target-derived',daily_candidate=False,selected=False,native_unit='stock_date',unit='fraction' if 'share' in c else 'USD per share',intuition='Outcome diagnostic. This contains closing information or is constructed from the target and must not enter the prior-close daily forecast.'))
# Intraday data were audited in the early stages, but are not daily-model predictors.
TRADE=str(ROOT/'blackrock_data/stock_intraday/trade_checkpoints/year=*/*.parquet')
con.execute(f"CREATE VIEW intraday AS SELECT t.date,t.permno,p.ff12_code,p.auction_volume,p.median20_auction_volume,t.checkpoint,t.cum_volume,t.cum_dollar_volume,t.cum_trade_count,t.recent_30m_volume,t.vwap,t.recent_30m_volume_share FROM read_parquet('{TRADE}',union_by_name=true) t JOIN panel p USING(date,permno) WHERE t.checkpoint_data_found AND NOT coalesce(t.checkpoint_after_market_close,false)")
for checkpoint in ['10:30','12:00','14:00','15:00','15:30']:
 for c in ['cum_volume','cum_dollar_volume','cum_trade_count','recent_30m_volume','vwap','recent_30m_volume_share']:
  specs.append(dict(feature=c+'__'+checkpoint.replace(':',''),layer='intraday_diagnostic',source_variable=c,group='Intraday',kind='numeric',timing='same day through '+checkpoint+' ET',daily_candidate=False,selected=False,native_unit='stock_date',unit=field_units(c),intuition='Same-day activity available only at this checkpoint. This was not an input to the existing daily model.',checkpoint=checkpoint))
for item in specs:
 if item['kind']=='binary':item['unit']='0/1 indicator'
 if item['kind']=='categorical':item['unit']='category label'
cat=pd.DataFrame(specs);cat.to_csv(T/'covariate_catalog.csv',index=False)
assert int(cat.selected.sum())==58
summaries=[];annuals=[];bins=[];nativebins=[];periods=[];missingrows=[];count=con.execute('SELECT count(*) FROM panel').fetchone()[0]
def corr(a,b):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return np.nan
 return float(np.corrcoef(a,b)[0,1])
def moments(a,b):
 if len(a)<3:return dict(pearson=np.nan,slope=np.nan,intercept=np.nan,r_squared=np.nan)
 ac=a-a.mean();bc=b-b.mean();den=float(ac@ac)
 slope=float(ac@bc/den) if den>0 else np.nan;r=corr(a,b)
 return dict(pearson=r,slope=slope,intercept=float(b.mean()-slope*a.mean()),r_squared=r*r)
def within(a,b,ids):
 ids=pd.factorize(ids,sort=False)[0];n=np.bincount(ids);xa=np.bincount(ids,weights=a)/n;ya=np.bincount(ids,weights=b)/n
 return moments(a-xa[ids],b-ya[ids])
def native(frame,kind):
 if kind=='date':return frame.groupby('date',as_index=False).agg(x=('x','first'),y=('y','mean'),permno=('permno','first'),rank_sample=('rank_sample','first'))
 if kind=='sector_date':return frame.groupby(['date','ff12_code'],as_index=False,dropna=False).agg(x=('x','first'),y=('y','mean'),permno=('permno','first'),rank_sample=('rank_sample','first'))
 return frame[['date','permno','x','y','rank_sample']]
def grouped(frame,spec,container,pop):
 if not len(frame):return
 x=frame.x;method='observed_levels'
 if spec['kind'] in {'categorical','binary'}:
  group=x.fillna('<MISSING>').astype(str)
 else:
  cuts=np.unique(np.quantile(x,np.linspace(0,1,11)))
  if x.nunique()<=10:
   group=x.astype(str);method='observed_numeric_values'
  elif len(cuts)<4:
   common=x.mode().iloc[0];other=x[x!=common];edges=np.unique(np.quantile(other,np.linspace(0,1,6)))
   group=pd.Series('common='+str(common),index=x.index);group.loc[other.index]=['other_Q'+str(i) for i in np.searchsorted(edges[1:-1],other,side='left')+1];method='common_value_separated_from_other_quantile_groups'
  else:
   group=pd.Series(np.searchsorted(cuts[1:-1],x,side='left')+1,index=x.index);method='full_population_quantiles_ties_kept'
 data=pd.DataFrame({'x':x,'y':frame.y,'bin':group});records=[]
 for label,g in data.groupby('bin',sort=True):
  xv=pd.to_numeric(g.x,errors='coerce')
  records.append(dict(feature=spec['feature'],population=pop,bin=str(label),grouping_method=method,observations=len(g),x_min=float(xv.min()) if xv.notna().any() else np.nan,x_median=float(xv.median()) if xv.notna().any() else np.nan,x_max=float(xv.max()) if xv.notna().any() else np.nan,mean_auction_shares=float(g.y.mean()),median_auction_shares=float(g.y.median()),p90_auction_shares=float(g.y.quantile(.9))))
 if spec['kind']=='numeric':records.sort(key=lambda r:r['x_median'])
 for i,r in enumerate(records):r['bin_order']=i+1;container.append(r)

def store():
 for name,data in [('feature_associations',summaries),('annual_associations',annuals),('panel_bin_curves',bins),('native_bin_curves',nativebins),('period_associations',periods),('missingness_contrasts',missingrows)]:pd.DataFrame(data).to_csv(T/(name+'.csv'),index=False)
for index,spec in enumerate(specs):
 c=spec['feature'];print(f'{index+1}/{len(specs)} {c}',flush=True)
 if spec['layer']=='intraday_diagnostic':
  view='intraday';source=spec['source_variable'];where=f"checkpoint='{spec['checkpoint']}'"
 else:view='panel';source=c;where='true'
 frame=con.execute(f'''SELECT date,permno,ff12_code,auction_volume AS y,median20_auction_volume, "{source}" AS x,hash(date,permno)%1000000<70000 AS rank_sample FROM {view} WHERE {where}''').fetchdf()
 raw_n=len(frame);missing=frame.x.isna()
 if spec['kind']!='categorical':
  frame['x']=pd.to_numeric(frame.x,errors='coerce').astype(float);missing=~np.isfinite(frame.x)
 missingrows.append(dict(feature=c,population_rows=raw_n,missing_or_nonfinite=int(missing.sum()),missing_pct=100*missing.mean(),mean_target_when_missing=frame.loc[missing,'y'].mean(),mean_target_when_present=frame.loc[~missing,'y'].mean()))
 valid=frame.loc[~missing].copy();primary=native(valid,spec['native_unit']);base=dict(feature=c,layer=spec['layer'],group=spec['group'],kind=spec['kind'],selected=spec['selected'],daily_candidate=spec['daily_candidate'],population_rows=raw_n,paired_stock_days=len(valid),missing_pct=100*missing.mean(),primary_observations=len(primary),primary_unit=spec['native_unit'],distinct_values=int(valid.x.nunique()),unit=spec['unit'])
 if spec['kind']!='categorical' and len(valid):
  a=valid.x.to_numpy(dtype=float);b=valid.y.to_numpy(dtype=float);pa=primary.x.to_numpy(dtype=float);pb=primary.y.to_numpy(dtype=float)
  stats=moments(a,b);ns=moments(pa,pb);ws=within(a,b,valid.permno)
  iqr=float(np.quantile(a,.75)-np.quantile(a,.25))
  sample=primary if spec['native_unit']!='stock_date' else primary[primary.rank_sample]
  rank=corr(sample.x.rank().to_numpy(),sample.y.rank().to_numpy())
  rankwithin=within(sample.x.rank().to_numpy(),sample.y.rank().to_numpy(),sample.permno)['pearson'] if spec['native_unit']=='stock_date' and len(sample) else np.nan
  base.update(pooled_pearson_raw=stats['pearson'],pooled_univariate_slope=stats['slope'],pooled_r_squared=stats['r_squared'],within_stock_pearson_raw=ws['pearson'],within_stock_slope=ws['slope'],primary_pearson_raw=ns['pearson'],primary_univariate_slope=ns['slope'],primary_r_squared=ns['r_squared'],x_iqr=iqr,pooled_associated_shares_per_iqr=stats['slope']*iqr if iqr>0 and spec['kind']=='numeric' else np.nan,primary_spearman=rank,rank_observations=len(sample),sample_within_stock_rank_corr=rankwithin)
  br=valid.y-valid.median20_auction_volume
  ok=np.isfinite(br);base['raw_baseline_residual_correlation']=corr(valid.loc[ok,'x'].to_numpy(),br[ok].to_numpy())
  for year,g in primary.groupby(primary.date.dt.year):
   st=moments(g.x.to_numpy(dtype=float),g.y.to_numpy(dtype=float));annuals.append(dict(feature=c,year=int(year),native_unit=spec['native_unit'],observations=len(g),**st))
  for label,mask in [('history_2015_2021',primary.date.dt.year<=2021),('development_2022_2024',primary.date.dt.year.between(2022,2024)),('confirmation_2025',primary.date.dt.year==2025)]:
   g=primary.loc[mask];periods.append(dict(feature=c,period=label,native_unit=spec['native_unit'],observations=len(g),**moments(g.x.to_numpy(dtype=float),g.y.to_numpy(dtype=float))))
 grouped(valid,spec,bins,'paired eligible stock-days');grouped(primary,spec,nativebins,'date mean shares per stock' if spec['native_unit']=='date' else 'sector-date mean shares per stock' if spec['native_unit']=='sector_date' else 'paired eligible stock-days')
 if spec['kind']=='binary' and valid.x.nunique()==2:
  g0=valid[valid.x==0];g1=valid[valid.x==1]
  if len(g0) and len(g1):base.update(binary_mean_difference_shares=g1.y.mean()-g0.y.mean(),binary_median_ratio=g1.y.median()/g0.y.median())
 if base['distinct_values']<2:base['limitation']='No within-sample covariate variation; isolated relationship is unidentified.'
 else:base['limitation']='Association only. No causal identification or incremental out-of-time test.'
 summaries.append(base)
 if (index+1)%10==0:store()
 del frame,valid,primary
store()
# Annual direction is descriptive, not a significance or stability guarantee.
summary=pd.DataFrame(summaries);annual=pd.DataFrame(annuals)
for c,g in annual.groupby('feature'):
 r=g.pearson.dropna();summary.loc[summary.feature==c,'positive_years']=int((r>0).sum());summary.loc[summary.feature==c,'negative_years']=int((r<0).sum());summary.loc[summary.feature==c,'years_with_variation']=len(r)
summary.to_csv(T/'feature_associations.csv',index=False)
(OUT/'analysis_metadata.json').write_text(json.dumps({'target':'official NYSE closing-auction volume in raw shares; no log transform in primary associations or bin outcomes','eligible_stock_days':count,'covariates_profiled':len(specs),'selected_features_covered':int(cat.selected.sum()),'rank_sample':'deterministic 7% hash(date,permno), shared across stock-date features; date/sector context ranks use all native observations','p_values':'not reported; no IID significance claims','causal_effects':'not identified','new_analysis_status':'retrospective economic diagnostics; not an original pre-selection test or new holdout','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('Association analysis complete',flush=True)
