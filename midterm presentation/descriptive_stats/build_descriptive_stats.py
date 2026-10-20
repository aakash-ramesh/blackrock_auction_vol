"""Profile the unmodified local source tables used through research stage 13.
Run with Python containing duckdb, pandas, numpy, matplotlib. Outputs stay beside this file.
"""
from pathlib import Path
import sys,json,html,base64,datetime,platform
try:
 import duckdb
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import duckdb
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TABLES=OUT/'tables';FIGS=OUT/'figures'
TABLES.mkdir(exist_ok=True);FIGS.mkdir(exist_ok=True)
SPECS={
 'crsp_daily':('stock_daily/crsp_daily/year=*/*.parquet',['date','permno'],'daily model source'),
 'auction_daily':('stock_daily/auction_daily/year=*/*.parquet',['date','permno'],'target and historical predictors'),
 'trade_checkpoints':('stock_intraday/trade_checkpoints/year=*/*.parquet',['date','permno','checkpoint'],'audited data, not a daily-model input'),
 'market_daily':('market/market_daily.parquet',['date'],'daily model source'),
 'sector_daily':('sector/sector_daily.parquet',['date','ff12_code'],'daily model source'),
 'event_calendar':('events/event_calendar.parquet',['date'],'daily model source'),
 'industry_history':('reference/industry_history.parquet',['permno','secinfostartdt','secinfoenddt'],'daily model classification source'),
 'source_anomalies':('quality/auction_source_anomalies.parquet',['date'],'eligibility control'),
 'security_master':('reference/security_master.parquet',['date','permno'],'upstream reference, not directly loaded by stages 01–13'),
 'trading_calendar':('reference/calendar.parquet',['date'],'upstream reference, not directly loaded by stages 01–13'),
 'taq_crsp_bridge':('reference/taq_crsp_bridge/year=*/*.parquet',['date','permno'],'upstream reference, already embedded in TAQ extracts'),
}
IDS={'permno','permco','ticker','ticker_daily','tradingsymbol','cusip','cusip_hist','sym_root','sym_suffix','taq_symbol_full','tr_seqnum','auction_tr_seqnum','tr_id'}
CODES={'year','month','quarter','day','day_of_week','bridge_year','match_lvl','siccd','ff12_code'}
TIME_FIELDS={'auction_time','time_m','time_m_nano'}
con=duckdb.connect()
con.execute("SET threads=4");con.execute("SET memory_limit='2GB'")
con.execute("SET temp_directory='/tmp/blackrock_midterm_duckdb'")
con.execute("SET TimeZone='America/New_York'")
def q(x):return '"'+x.replace('"','""')+'"'
def lit(x):return "'"+str(x).replace("'","''")+"'"
def df(sql):return con.execute(sql).fetchdf()
def save(name,data):
 if not isinstance(data,pd.DataFrame):data=pd.DataFrame(data)
 data.to_csv(TABLES/(name+'.csv'),index=False)
 return data

def units(name):
 if name in IDS or name in CODES:return 'identifier or category code'
 if 'thousands' in name:return 'USD thousands' if 'cap' in name else 'shares thousands'
 if 'bps' in name:return 'basis points'
 if 'dollar' in name or 'cap_usd' in name or 'price_volume' in name:return 'USD'
 if name=='auction_share_daily_volume' or 'volume_share' in name or 'percentile' in name or 'fraction' in name or 'breadth' in name:return 'fraction or ratio, native scale'
 if 'return' in name:return 'decimal return (dispersion is standard deviation)' if 'count' not in name else 'count'
 if 'price' in name or name in {'open','high','low','close','bid','ask','vwap'}:
  return 'dimensionless factor' if 'factor' in name else 'USD per share'
 if 'volume' in name or name=='recent_30m_volume':return 'shares'
 if 'minutes' in name:return 'minutes'
 if 'dividend' in name:return 'USD per share'
 if 'factor' in name or 'range' in name:return 'dimensionless, native scale'
 return 'count or native units (see field name)'

def numerical(dataset,view,fields,scope='full_history',where='true'):
 rows=[]
 for start in range(0,len(fields),5):
  batch=fields[start:start+5];expr=[]
  for i,c in enumerate(batch):
   v=f'CAST({q(c)} AS DOUBLE)';f=f'CASE WHEN isfinite({v}) THEN {v} END'
   expr.extend([f'count(*) AS n{i}',f'count({q(c)}) AS present{i}',f'count(*) FILTER(WHERE {q(c)} IS NOT NULL AND NOT isfinite({v})) AS nonfinite{i}',f'avg({f}) AS mean{i}',f'stddev_samp({f}) AS sd{i}',f'min({f}) AS min{i}',f'approx_quantile({f},[0.01,0.05,0.25,0.5,0.75,0.95,0.99]) AS quant{i}',f'max({f}) AS max{i}',f'count(*) FILTER(WHERE {v}=0) AS zero{i}',f'count(*) FILTER(WHERE {v}<0) AS neg{i}'])
  values=df(f'SELECT {",".join(expr)} FROM {view} WHERE {where}').iloc[0]
  for i,c in enumerate(batch):
   qu=values[f'quant{i}'];qu=[np.nan]*7 if qu is None or isinstance(qu,float) else list(qu)
   row=dict(dataset=dataset,scope=scope,variable=c,unit=units(c),rows=int(values[f'n{i}']),non_null=int(values[f'present{i}']),nonfinite=int(values[f'nonfinite{i}']),mean=values[f'mean{i}'],std_dev=values[f'sd{i}'],minimum=values[f'min{i}'],maximum=values[f'max{i}'],zero_count=int(values[f'zero{i}']),negative_count=int(values[f'neg{i}']))
   row.update(zip(['p01_approx','p05_approx','p25_approx','median_approx','p75_approx','p95_approx','p99_approx'],qu));row['missing_count']=row['rows']-row['non_null'];row['missing_pct']=100*row['missing_count']/row['rows'] if row['rows'] else np.nan
   rows.append(row)
 return rows

inventory=[];annual=[];schema_rows=[];field_rows=[];yearmissing=[];numeric=[];categories=[];keychecks=[];files_meta=[];schemas={};all_bool=[]
for name,(rel,keys,role) in SPECS.items():
 print('Profiling',name,flush=True)
 files=sorted((ROOT/'blackrock_data').glob(rel));path=str(ROOT/'blackrock_data'/rel)
 con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({lit(path)},union_by_name=true,hive_partitioning=false)')
 schema=[(r[0],r[1]) for r in con.execute(f'DESCRIBE {name}').fetchall()];schemas[name]=dict(schema)
 counts=df(f'SELECT count(*) AS rows'+(',count(DISTINCT date) AS dates,min(date) AS first_date,max(date) AS last_date' if 'date' in dict(schema) else ',NULL AS dates,min(secinfostartdt) AS first_date,max(secinfoenddt) AS last_date')+(',count(DISTINCT permno) AS securities' if 'permno' in dict(schema) else ',NULL AS securities')+f' FROM {name}').iloc[0].to_dict()
 counts.update(dataset=name,role=role,files=len(files),columns=len(schema),source_pattern='blackrock_data/'+rel,bytes=sum(p.stat().st_size for p in files));inventory.append(counts)
 file_presence={c:0 for c,t in schema};structural={c:0 for c,t in schema}
 for p in files:
  cols={r[0] for r in con.execute(f'DESCRIBE SELECT * FROM read_parquet({lit(p)},hive_partitioning=false)').fetchall()}
  nr=con.execute(f'SELECT count(*) FROM read_parquet({lit(p)})').fetchone()[0]
  files_meta.append(dict(dataset=name,file=str(p.relative_to(ROOT)),rows=nr,columns=len(cols),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
  for c in file_presence:
   file_presence[c]+=int(c in cols);structural[c]+=nr*int(c not in cols)
  schema_rows.append(dict(dataset=name,file=str(p.relative_to(ROOT)),columns=len(cols),fields=';'.join(sorted(cols))))
 keylist=','.join(q(c) for c in keys)
 kc=df(f'SELECT count(*) AS key_groups,coalesce(sum(n-1),0) AS duplicate_excess_rows,count(*) FILTER(WHERE n>1) AS duplicate_groups FROM (SELECT {keylist},count(*) n FROM {name} GROUP BY {keylist})').iloc[0].to_dict()
 kc.update(dataset=name,key=' × '.join(keys));keychecks.append(kc)
 if 'date' in dict(schema):
  y=df(f'SELECT year(date) AS year,count(*) AS rows,count(DISTINCT date) AS trading_dates'+(',count(DISTINCT permno) AS securities' if 'permno' in dict(schema) else '')+f' FROM {name} GROUP BY 1 ORDER BY 1');y.insert(0,'dataset',name);annual.extend(y.to_dict('records'))
 # Completeness is exact. Approximate distinct counts are explicitly labeled.
 for start in range(0,len(schema),12):
  batch=schema[start:start+12];expr=[]
  for i,(c,t) in enumerate(batch):expr.extend([f'count({q(c)}) AS nn{i}',f'approx_count_distinct({q(c)}) AS nd{i}'])
  r=df(f'SELECT {",".join(expr)} FROM {name}').iloc[0]
  for i,(c,t) in enumerate(batch):
   nr=int(counts['rows']);nulls=nr-int(r[f'nn{i}']);kind='identifier' if c in IDS else 'category_code' if c in CODES else 'boolean' if t=='BOOLEAN' else 'date/time' if t.startswith(('DATE','TIMESTAMP')) or c in TIME_FIELDS else 'continuous/count' if any(x in t for x in ['INT','DOUBLE','FLOAT','DECIMAL']) else 'category/text'
   field_rows.append(dict(dataset=name,variable=c,sql_type=t,role=kind,unit=units(c) if kind=='continuous/count' else kind,rows=nr,non_null=int(r[f'nn{i}']),missing_count=nulls,missing_pct=100*nulls/nr if nr else np.nan,distinct_approx=int(r[f'nd{i}']),files_with_field=file_presence[c],files_total=len(files),schema_absent_rows=structural[c],null_in_present_schema_rows=nulls-structural[c]))
  if 'date' in dict(schema):
   expr=[f'count(*)-count({q(c)}) AS {q(c)}' for c,t in batch]
   ym=df(f'SELECT year(date) AS year,count(*) AS rows,{",".join(expr)} FROM {name} GROUP BY 1')
   for row in ym.to_dict('records'):
    for c,t in batch:yearmissing.append(dict(dataset=name,year=row['year'],variable=c,rows=row['rows'],missing_count=row[c],missing_pct=100*row[c]/row['rows']))
 continuous=[c for c,t in schema if any(x in t for x in ['INT','DOUBLE','FLOAT','DECIMAL']) and c not in IDS|CODES]
 numeric.extend(numerical(name,name,continuous))
 if name in {'crsp_daily','market_daily','sector_daily'}:numeric.extend(numerical(name,name,continuous,'2015_2025',"date>=DATE '2015-01-01'"))
 for c,t in schema:
  if c in IDS|TIME_FIELDS or t.startswith(('DATE','TIMESTAMP')):continue
  if t=='BOOLEAN' or c in CODES or (t=='VARCHAR' and c not in {'reason','naics','icbindustry','uesindustry'}):
   cat=df(f"""SELECT CASE WHEN {q(c)} IS NULL THEN '<NULL>' WHEN CAST({q(c)} AS VARCHAR)='' THEN '<EMPTY STRING>' ELSE CAST({q(c)} AS VARCHAR) END AS value,count(*) AS count FROM {name} GROUP BY 1 ORDER BY count DESC LIMIT 30""");cat.insert(0,'variable',c);cat.insert(0,'dataset',name);cat['row_pct']=100*cat['count']/int(counts['rows']);categories.extend(cat.to_dict('records'))
  if t=='BOOLEAN' and 'date' in dict(schema):all_bool.append((name,c))
 # Incremental checkpoints make long runs reviewable.
 save('dataset_inventory',inventory);save('field_profile',field_rows);save('numeric_summary',numeric)

save('annual_inventory',annual);save('file_inventory',files_meta);save('file_schemas',schema_rows);save('missingness_by_year',yearmissing);save('categorical_frequencies_top30',categories);save('key_checks',keychecks)
flags=[]
for name,c in all_bool:
 r=df(f'SELECT year(date) AS year,count(*) AS rows,count(*) FILTER(WHERE {q(c)}) AS true_rows,count(*) FILTER(WHERE {q(c)} IS NULL) AS missing_rows FROM {name} GROUP BY 1 ORDER BY 1');r.insert(0,'variable',c);r.insert(0,'dataset',name);r['true_pct_all_rows']=100*r.true_rows/r.rows;flags.extend(r.to_dict('records'))
save('flag_rates_by_year',flags)

print('Computing target eligibility, distributions and comparisons',flush=True)
con.execute('''CREATE VIEW target_panel AS SELECT c.date,c.permno,c.daily_volume,c.market_cap_usd,c.return_total,c.close,
 a.auction_volume,a.auction_price,a.auction_share_daily_volume,coalesce(a.target_found,false) AS target_found,
 coalesce(an.exclude_from_model,false) AS exclude_source_anomaly,
 coalesce(a.target_found,false) AND a.auction_volume IS NOT NULL AND NOT coalesce(an.exclude_from_model,false) AS eligible,
 e.monthly_options_expiry,e.quarterly_options_expiry,e.is_last_trading_day_month,e.is_last_trading_day_quarter,e.is_early_close
 FROM crsp_daily c LEFT JOIN auction_daily a USING(date,permno)
 LEFT JOIN source_anomalies an USING(date) LEFT JOIN event_calendar e USING(date)
 WHERE c.date>=DATE '2015-01-01' ''')
coverage=save('annual_target_coverage',df('''SELECT year(date) AS year,count(*) AS stock_days,count(*) FILTER(WHERE target_found AND auction_volume IS NOT NULL) AS observed_labels,
 count(*) FILTER(WHERE eligible) AS eligible_labels,count(*) FILTER(WHERE exclude_source_anomaly) AS excluded_anomaly_stock_days,
 count(*) FILTER(WHERE auction_volume IS NULL OR NOT target_found) AS unobserved_labels,
 100.0*count(*) FILTER(WHERE target_found AND auction_volume IS NOT NULL)/count(*) AS observed_coverage_pct,
 100.0*count(*) FILTER(WHERE eligible)/count(*) AS eligible_coverage_pct FROM target_panel GROUP BY 1 ORDER BY 1'''))
numeric.extend(numerical('auction_daily','target_panel',['auction_volume','auction_price','auction_share_daily_volume'],'eligible_2015_2025','eligible'))
numeric.extend(numerical('trade_checkpoints','trade_checkpoints',['cum_volume','cum_dollar_volume','cum_trade_count','recent_30m_volume','vwap','recent_30m_volume_share'],'valid_observed_checkpoints','checkpoint_data_found AND NOT coalesce(checkpoint_after_market_close,false)'))
save('numeric_summary',numeric)
annualtarget=save('auction_by_year',df('''SELECT year(date) AS year,count(*) AS eligible_stock_days,count(DISTINCT date) AS dates,count(DISTINCT permno) AS securities,
 avg(auction_volume) AS mean_auction,median(auction_volume) AS median_auction,quantile_cont(auction_volume,0.25) AS p25_auction,
 quantile_cont(auction_volume,0.75) AS p75_auction,quantile_cont(auction_volume,0.95) AS p95_auction,quantile_cont(auction_volume,0.99) AS p99_auction,
 stddev_samp(auction_volume) AS sd_auction,sum(auction_volume) AS total_auction_shares,
 median(auction_share_daily_volume) AS median_auction_share,avg(auction_share_daily_volume) AS mean_auction_share,
 sum(auction_volume)/nullif(sum(daily_volume),0) AS volume_weighted_auction_share
 FROM target_panel WHERE eligible GROUP BY 1 ORDER BY 1'''))
# Stock-date-weighted calendar contrasts. Same-day auction/daily share is descriptive only.
calendar=[]
for c in ['monthly_options_expiry','quarterly_options_expiry','is_last_trading_day_month','is_last_trading_day_quarter','is_early_close']:
 r=df(f'''SELECT {q(c)} AS flag_value,count(*) AS stock_days,count(DISTINCT date) AS dates,
 avg(auction_volume) AS mean_auction,median(auction_volume) AS median_auction,
 median(auction_share_daily_volume) AS median_auction_share FROM target_panel WHERE eligible GROUP BY 1 ORDER BY 1''');r.insert(0,'event',c);calendar.extend(r.to_dict('records'))
save('calendar_target_descriptives',calendar)
size=save('contemporaneous_size_deciles',df('''WITH ranked AS (SELECT *,ntile(10) OVER(PARTITION BY date ORDER BY market_cap_usd,permno) AS size_decile FROM target_panel WHERE eligible AND market_cap_usd>0)
SELECT size_decile,count(*) AS stock_days,median(market_cap_usd) AS median_market_cap_usd,median(auction_volume) AS median_auction,
 avg(auction_volume) AS mean_auction,median(auction_share_daily_volume) AS median_auction_share FROM ranked GROUP BY 1 ORDER BY 1'''))
checkpoints=save('checkpoint_summary',df('''SELECT checkpoint,count(*) AS rows,count(*) FILTER(WHERE checkpoint_after_market_close) AS after_close_rows,
 count(*) FILTER(WHERE checkpoint_data_found AND NOT coalesce(checkpoint_after_market_close,false)) AS valid_observed_rows,
 count(*) FILTER(WHERE NOT coalesce(checkpoint_after_market_close,false)) AS within_session_rows,
 median(cum_volume) FILTER(WHERE checkpoint_data_found AND NOT coalesce(checkpoint_after_market_close,false)) AS median_cum_volume,
 median(vwap) FILTER(WHERE checkpoint_data_found AND NOT coalesce(checkpoint_after_market_close,false)) AS median_vwap,
 median(recent_30m_volume_share) FILTER(WHERE checkpoint_data_found AND NOT coalesce(checkpoint_after_market_close,false)) AS median_recent_volume_share
 FROM trade_checkpoints GROUP BY 1 ORDER BY 1'''))
# Full-population binned target distribution avoids plotting a convenience sample.
hist=save('auction_log_histogram',df('''SELECT floor(log10(1+auction_volume)*10)/10 AS bin_left_log10,count(*) AS stock_days
FROM target_panel WHERE eligible AND auction_volume>=0 GROUP BY 1 ORDER BY 1'''))
sector=save('sector_native_summary',df('''SELECT ff12_code,ff12_name,count(*) AS sector_days,median(sector_stock_count) AS median_stocks,
 avg(sector_return_vw) AS mean_daily_return,stddev_samp(sector_return_vw) AS sd_daily_return,
 median(sector_total_volume) AS median_daily_shares,median(sector_total_cap_usd) AS median_cap_usd FROM sector_daily
 WHERE date>=DATE '2015-01-01' GROUP BY 1,2 ORDER BY 1'''))
# Source consistency checks preserve raw values, including impossible or unusual ratios.
quality=save('domain_checks',df('''SELECT 'auction volume negative' AS check_name,count(*) AS flagged_rows FROM target_panel WHERE auction_volume<0
UNION ALL SELECT 'auction share above one',count(*) FROM target_panel WHERE eligible AND auction_share_daily_volume>1
UNION ALL SELECT 'auction shares above positive CRSP daily volume',count(*) FROM target_panel WHERE eligible AND daily_volume>0 AND auction_volume>daily_volume
UNION ALL SELECT 'auction raw ratio disagrees with volume/daily_volume (>1e-8)',count(*) FROM target_panel WHERE eligible AND daily_volume>0 AND abs(auction_share_daily_volume-auction_volume/daily_volume)>1e-8
UNION ALL SELECT 'CRSP high below low',count(*) FROM crsp_daily WHERE high<low
UNION ALL SELECT 'CRSP negative daily volume',count(*) FROM crsp_daily WHERE daily_volume<0
UNION ALL SELECT 'CRSP total return below -100 percent',count(*) FROM crsp_daily WHERE return_total < -1
UNION ALL SELECT 'checkpoint cumulative volume negative',count(*) FROM trade_checkpoints WHERE cum_volume<0
UNION ALL SELECT 'checkpoint recent volume share above one',count(*) FROM trade_checkpoints WHERE recent_30m_volume_share>1
UNION ALL SELECT 'checkpoint cumulative volume decreases within stock-day',count(*) FROM (SELECT cum_volume,lag(cum_volume) OVER(PARTITION BY date,permno ORDER BY checkpoint) prev FROM trade_checkpoints WHERE NOT coalesce(checkpoint_after_market_close,false)) WHERE cum_volume<prev'''))
# Exact quantiles for the principal report measures, using the stated population.
exact=[]
for dataset,view,fields,scope,where in [
 ('auction_daily','target_panel',['auction_volume','auction_price','auction_share_daily_volume'],'eligible_2015_2025','eligible'),
 ('crsp_daily','crsp_daily',['close','daily_volume','daily_price_volume','market_cap_usd','return_total','shares_outstanding_thousands'],'2015_2025',"date>=DATE '2015-01-01'"),
 ('market_daily','market_daily',['market_return_vw','market_total_volume','market_total_cap_usd'],'2015_2025',"date>=DATE '2015-01-01'")]:
 for c in fields:
  v=f'CASE WHEN isfinite(CAST({q(c)} AS DOUBLE)) THEN CAST({q(c)} AS DOUBLE) END'
  r=df(f"SELECT count(*) AS rows,count({v}) AS finite_rows,avg({v}) AS mean,stddev_samp({v}) AS std_dev,min({v}) AS minimum,quantile_cont({v},[0.01,0.05,0.25,0.5,0.75,0.95,0.99]) AS quantiles,max({v}) AS maximum FROM {view} WHERE {where}").iloc[0].to_dict()
  qu=r.pop('quantiles');r.update(zip(['p01','p05','p25','median','p75','p95','p99'],list(qu)));r.update(dataset=dataset,variable=c,scope=scope,unit=units(c));r['missing_or_nonfinite_pct']=100*(r['rows']-r['finite_rows'])/r['rows'];exact.append(r)
save('principal_exact_summary',exact)
# Native-frequency annual statistics for the main source measures.
coreannual=[]
for name,fields in [('crsp_daily',['close','daily_volume','market_cap_usd','return_total']),('market_daily',['market_return_vw','market_total_volume','market_total_cap_usd']),('sector_daily',['sector_return_vw','sector_total_volume','sector_total_cap_usd'])]:
 for c in fields:
  r=df(f'SELECT year(date) AS year,count(*) AS rows,count({q(c)}) AS observed,avg({q(c)}) AS mean,stddev_samp({q(c)}) AS std_dev,median({q(c)}) AS median,min({q(c)}) AS minimum,max({q(c)}) AS maximum FROM {name} GROUP BY 1 ORDER BY 1');r.insert(0,'variable',c);r.insert(0,'dataset',name);coreannual.extend(r.to_dict('records'))
save('principal_by_year',coreannual)
# The supplied industry reference is broader than this NYSE extract. Restrict its descriptive use.
con.execute("""CREATE VIEW eligible_industry AS SELECT t.date,t.permno,i.ff12_code,i.ff12_name
 FROM target_panel t LEFT JOIN industry_history i ON t.permno=i.permno AND t.date BETWEEN i.secinfostartdt AND i.secinfoenddt
 WHERE t.eligible QUALIFY row_number() OVER(PARTITION BY t.date,t.permno ORDER BY i.secinfostartdt DESC NULLS LAST,i.secinfoenddt DESC NULLS LAST)=1""")
save('industry_eligible_stockdays',df("""SELECT ff12_code,ff12_name,count(*) AS eligible_stock_days,count(DISTINCT permno) AS securities FROM eligible_industry GROUP BY 1,2 ORDER BY 1"""))

# Reconcile fresh calculations with existing small artifacts, without treating them as raw data.
reconciliation=[]
prior=pd.read_csv(ROOT/'outputs/eda/01_data_quality/dataset_inventory.csv')
for label,name in [('CRSP daily','crsp_daily'),('Auction daily','auction_daily'),('Trade checkpoints','trade_checkpoints'),('Market daily','market_daily'),('Sector daily','sector_daily')]:
 old=int(prior.loc[prior.dataset==label,'rows'].iloc[0]);new=int(next(r['rows'] for r in inventory if r['dataset']==name));reconciliation.append(dict(check='inventory rows: '+name,expected=old,actual=new,passed=old==new))
saved_sample=pd.read_csv(ROOT/'outputs/eda/01_data_quality/modeling_sample_by_split.csv')
for old_col,new_col in [('stock_days','stock_days'),('observed_targets','observed_labels'),('eligible_targets','eligible_labels')]:
 old=int(saved_sample[old_col].sum());new=int(coverage[new_col].sum());reconciliation.append(dict(check='sample counts: '+old_col,expected=old,actual=new,passed=old==new))
panel=ROOT/'outputs/eda/05_feature_engineering/enriched_daily_panel.parquet'
if panel.exists():
 old=con.execute(f'SELECT count(*) FROM read_parquet({lit(panel)}) WHERE eligible_target').fetchone()[0];new=int(coverage.eligible_labels.sum());reconciliation.append(dict(check='eligible labels versus canonical model panel',expected=old,actual=new,passed=old==new))
save('reconciliation_checks',reconciliation)
if not all(r['passed'] for r in reconciliation):raise RuntimeError('Reconciliation failed; inspect table before reporting.')
metadata={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'repository notebooks 01–13; options data excluded; expiry calendar flags retained','duckdb':duckdb.__version__,'pandas':pd.__version__,'python':platform.python_version(),'quantiles':'numeric_summary uses full-population approx_quantile; selected annual and checkpoint tables use exact medians/quantiles','source_files':len(files_meta),'mutations':'only this descriptive_stats directory; source tables never modified'}
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
print('Tables complete',flush=True)
# Plotting and narrative are separate functions so a report can be refreshed from saved tables.
import render_report
render_report.render(OUT)
print('Report complete:',OUT/'report.html',flush=True)
