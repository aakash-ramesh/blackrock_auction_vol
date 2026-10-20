"""Independent numerical and artifact checks for the covariate analysis.
Use --check-presentation to verify the deck hashes recorded for this discussion step.
"""
from pathlib import Path
from html.parser import HTMLParser
import sys,json,hashlib
try:import duckdb
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import duckdb
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent;T=OUT/'tables';ROOT=OUT.parents[1]
s=pd.read_csv(T/'feature_associations.csv');cat=pd.read_csv(T/'covariate_catalog.csv')
assert len(s)==s.feature.nunique()==193
assert set(s.feature)==set(cat.feature)
assert int(cat.selected.sum())==58
for file,total in [('panel_bin_curves.csv','paired_stock_days'),('native_bin_curves.csv','primary_observations'),('annual_associations.csv','primary_observations')]:
 data=pd.read_csv(T/file)
 for c,g in data.groupby('feature'):
  assert int(g.observations.sum())==int(s.loc[s.feature.eq(c),total].iloc[0]),(file,c)
assert cat[cat.layer.eq('raw_same_day')].daily_candidate.eq(False).all()
assert cat[cat.layer.eq('intraday_diagnostic')].daily_candidate.eq(False).all()
assert cat[cat.layer.eq('raw_prior_close')].daily_candidate.eq(True).all()
con=duckdb.connect();con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{OUT/'cache/analysis_panel.parquet'}')")
checks=[]
for c in ['median20_auction_volume','prior_raw_daily_volume','prior_raw_market_cap_usd','prior_raw_return_total']:
 r=s[s.feature.eq(c)].iloc[0]
 actual=con.execute(f'SELECT corr("{c}",auction_volume),regr_slope(auction_volume,"{c}") FROM p WHERE isfinite("{c}")').fetchone()
 for label,a,b in [('raw correlation',actual[0],r.pooled_pearson_raw),('univariate slope',actual[1],r.pooled_univariate_slope)]:checks.append(dict(check=f'Independent SQL {label}: {c}',expected=a,actual=b,passed=np.isclose(a,b,rtol=1e-10,atol=1e-10)))
 actual=con.execute(f'WITH z AS(SELECT permno,"{c}" x,auction_volume y FROM p WHERE isfinite("{c}")),w AS(SELECT x-avg(x) OVER(PARTITION BY permno) xw,y-avg(y) OVER(PARTITION BY permno) yw FROM z) SELECT corr(xw,yw) FROM w').fetchone()[0]
 checks.append(dict(check='Independent SQL within-stock correlation: '+c,expected=actual,actual=r.within_stock_pearson_raw,passed=np.isclose(actual,r.within_stock_pearson_raw,atol=1e-10)))
for left,right in [('lag1_daily_volume','prior_raw_daily_volume'),('lag1_price','prior_raw_close'),('lag1_market_return','prior_raw_market_return_vw'),('lag1_sector_return','prior_raw_sector_return_vw')]:
 n=con.execute(f'SELECT count(*) FROM p WHERE "{left}" IS DISTINCT FROM "{right}"').fetchone()[0];checks.append(dict(check='Equivalent source lag: '+left,expected=0,actual=n,passed=n==0))
assert all(c['passed'] for c in checks)
pd.DataFrame(checks).to_csv(T/'numerical_verification.csv',index=False)
class Atlas(HTMLParser):
 def __init__(self):super().__init__();self.entries=0;self.images=[]
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='details':self.entries+=1
  if tag=='img':self.images.append(attrs['src'])
a=Atlas();a.feed((OUT/'covariate_atlas.html').read_text());assert a.entries==193
for image in a.images:assert (OUT/image).is_file()
if '--check-presentation' in sys.argv:
 for name,sha in json.loads((OUT/'presentation_unchanged_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/'midterm presentation'/name).read_bytes()).hexdigest()==sha,name
print('Verified 193 specifications, 58 selected features, bin/native/annual totals, timing labels, 16 independent SQL checks and atlas figures.')
