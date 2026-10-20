"""Rebuild the existing daily feature SQL in an isolated analysis cache.
Only the panel_sql assignment is evaluated; no notebook or builder module is executed.
"""
from pathlib import Path
import ast,json,sys,hashlib
try:import duckdb
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import duckdb
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;CACHE=OUT/'cache';CACHE.mkdir(exist_ok=True)
con=duckdb.connect();con.execute("SET threads=4");con.execute("SET memory_limit='3GB'");con.execute("SET temp_directory='/tmp/blackrock_covariate_spill'")
paths={'AUCTION':'stock_daily/auction_daily/year=*/*.parquet','CRSP':'stock_daily/crsp_daily/year=*/*.parquet','MARKET':'market/market_daily.parquet','SECTOR':'sector/sector_daily.parquet','EVENTS':'events/event_calendar.parquet','INDUSTRY':'reference/industry_history.parquet','ANOMALIES':'quality/auction_source_anomalies.parquet'}
env={k:str(ROOT/'blackrock_data'/v) for k,v in paths.items()}
p=ROOT/'notebooks/05_feature_engineering_enriched_daily_panel.ipynb'
for cell in json.loads(p.read_text())['cells']:
 if cell['cell_type']!='code':continue
 source=''.join(cell['source']);tree=ast.parse(source)
 for node in tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='panel_sql' for t in node.targets):
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),env)
sql=env['panel_sql'];(OUT/'canonical_panel_query.sql').write_text(sql)
path=CACHE/'enriched_daily_panel.parquet'
if not path.exists():
 print('Building isolated canonical panel',flush=True)
 con.execute(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET,COMPRESSION ZSTD)")
print(con.execute(f"SELECT count(*) AS rows,count(*) FILTER(WHERE eligible_target) AS eligible,count(DISTINCT permno) AS securities FROM read_parquet('{path}')").fetchall(),flush=True)
assert con.execute(f"SELECT count(*) FROM read_parquet('{path}') WHERE eligible_target").fetchone()[0]==3499401
# Source-level economic variables used by the existing feature construction. All raw values remain untransformed.
raw_num=['close','open','high','low','bid','ask','daily_volume','daily_price_volume','return_total','market_cap_usd','shares_outstanding_thousands','ordinary_dividend','nonordinary_dividend','daily_price_factor']
raw_cat=['distribution_flag','security_type','security_subtype','share_type','us_incorporated','issuer_type']
select=','.join('"'+x+'"' for x in raw_num+raw_cat)
lag=','.join(f'lag("{x}") OVER(PARTITION BY permno ORDER BY date) AS "prior_{x}"' for x in raw_num+raw_cat)
rawpath=CACHE/'raw_stock_economic.parquet'
if not rawpath.exists():con.execute(f"COPY (SELECT date,permno,{select},{lag} FROM read_parquet('{env['CRSP']}',union_by_name=true)) TO '{rawpath}' (FORMAT PARQUET,COMPRESSION ZSTD)")
con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{path}') WHERE eligible_target")
con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{rawpath}')")
market_num=['market_return_vw','market_total_volume','market_total_dollar_volume','market_return_dispersion','market_breadth','market_median_hl_range']
sector_num=['sector_return_vw','sector_total_volume','sector_total_dollar_volume','sector_return_dispersion','sector_breadth']
for name,fields,source,part in [('m',market_num,env['MARKET'],'ORDER BY date'),('s',sector_num,env['SECTOR'],'PARTITION BY ff12_code ORDER BY date')]:
 cols=','.join('"'+c+'"' for c in fields);lags=','.join(f'lag("{c}") OVER({part}) AS "prior_{c}"' for c in fields)
 keys='date'+(',ff12_code' if name=='s' else '')
 con.execute(f"CREATE VIEW {name} AS SELECT {keys},{cols},{lags} FROM read_parquet('{source}')")
# Added current-auction price is an outcome diagnostic, never a forecast candidate.
extras=[]
for c in raw_num+raw_cat:
 extras += [f'r."{c}" AS "raw_{c}"',f'r."prior_{c}" AS "prior_raw_{c}"']
for prefix,fields in [('m',market_num),('s',sector_num)]:
 for c in fields:extras += [f'{prefix}."{c}" AS "raw_{c}"',f'{prefix}."prior_{c}" AS "prior_raw_{c}"']
# Keep eligible rows and source values together in a cache for projected single-feature reads.
analysis=CACHE/'analysis_panel.parquet'
if not analysis.exists():con.execute(f"COPY (SELECT p.*,{','.join(extras)} FROM p LEFT JOIN r USING(date,permno) LEFT JOIN m USING(date) LEFT JOIN s USING(date,ff12_code)) TO '{analysis}' (FORMAT PARQUET,COMPRESSION ZSTD)")
print('Analysis panel',analysis.stat().st_size/2**20,'MiB',flush=True)
summary=con.execute(f"SELECT count(*),count(DISTINCT (date,permno)),sum(auction_volume) FROM read_parquet('{analysis}')").fetchone();assert summary[0]==summary[1]==3499401
(OUT/'panel_build_metadata.json').write_text(json.dumps({'source_notebook':str(p.relative_to(ROOT)),'sql_sha256':hashlib.sha256(sql.encode()).hexdigest(),'rows':summary[0],'sum_auction_shares':str(summary[2]),'cache_git_ignored':True,'raw_numeric_fields':raw_num,'raw_categorical_fields':raw_cat,'raw_market_fields':market_num,'raw_sector_fields':sector_num},indent=2))
print('Panels complete',flush=True)
