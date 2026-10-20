"""Render the saved descriptive tables into a readable report and scientific charts."""
from pathlib import Path
import sys,html,base64,json,math
try:
 import matplotlib
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,FuncFormatter
import pandas as pd
import numpy as np

def render(out):
 out=Path(out);tables=out/'tables';figures=out/'figures';figures.mkdir(exist_ok=True)
 def read(n):return pd.read_csv(tables/(n+'.csv'))
 inv=read('dataset_inventory');coverage=read('annual_target_coverage');annual=read('auction_by_year');numeric=read('numeric_summary');exact=read('principal_exact_summary');profile=read('field_profile');cp=read('checkpoint_summary');flags=read('flag_rates_by_year');quality=read('domain_checks');keys=read('key_checks');recon=read('reconciliation_checks');industry=read('industry_eligible_stockdays')
 def val(ds,var,col='median',scope=None):
  x=exact[(exact.dataset==ds)&(exact.variable==var)]
  if scope:x=x[x.scope==scope]
  return float(x.iloc[0][col])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':140,'axes.labelcolor':'#243746','text.color':'#243746'})
 navy='#102C43';teal='#007F82'
 def finish(fig,name):fig.tight_layout();fig.savefig(figures/(name+'.png'),bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4.6));ax.plot(coverage.year,coverage.observed_coverage_pct,marker='o',color=teal,label='Observed official target');ax.plot(coverage.year,coverage.eligible_coverage_pct,marker='x',ls='--',color=navy,label='Eligible after source-date exclusion');ax.set(ylim=(90,100),ylabel='Stock-days with a target (%)',xlabel='Year',title='Auction-target coverage varies over time');ax.set_xticks(coverage.year);ax.legend(frameon=False);finish(fig,'target_coverage')
 hist=read('auction_log_histogram');fig,ax=plt.subplots(figsize=(10,4.6));ax.bar(hist.bin_left_log10,hist.stock_days,width=.098,align='edge',color=teal);ax.set(title='Distribution of eligible auction volume',xlabel='log10(1 + auction shares)',ylabel='Stock-days');finish(fig,'auction_distribution')
 fig,axes=plt.subplots(1,2,figsize=(11,4.4));axes[0].plot(annual.year,annual.median_auction/1000,marker='o',color=teal);axes[0].set(title='Median auction size',ylabel='Thousands of shares',xlabel='Year');axes[1].plot(annual.year,100*annual.median_auction_share,marker='o',color=teal,label='Median stock-day share');axes[1].plot(annual.year,100*annual.volume_weighted_auction_share,marker='s',color=navy,label='Ratio of aggregate share volumes');axes[1].set(title='Auction share of daily trading',ylabel='Percent',xlabel='Year');axes[1].legend(frameon=False,fontsize=9);finish(fig,'auction_size_and_share')
 fields=['open','close','high','low','return_total','return_price','bid','ask','daily_volume','market_cap_usd']
 missing=profile[(profile.dataset=='crsp_daily')&profile.variable.isin(fields)].sort_values('missing_pct')
 fig,ax=plt.subplots(figsize=(9,4.6));ax.barh(missing.variable,missing.missing_pct,color=teal);ax.set(title='Missingness in selected CRSP raw fields, 2011–2025',xlabel='Percent of source stock-days');finish(fig,'crsp_missingness')
 size=read('contemporaneous_size_deciles');fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(size.size_decile,size.median_auction,marker='o',color=teal);ax.set_yscale('log');ax.set(title='Auction size across contemporaneous market-cap deciles',xlabel='Daily size decile (1 = smallest)',ylabel='Median auction shares, log scale');ax.set_xticks(range(1,11));finish(fig,'auction_by_size')
 fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(cp.checkpoint,cp.median_cum_volume/1000,marker='o',color=teal);ax.set(title='Cumulative intraday trading at the five checkpoints',xlabel='Checkpoint, Eastern Time',ylabel='Median cumulative shares (thousands)');finish(fig,'checkpoint_volume')
 market=read('principal_by_year');market=market[(market.dataset=='market_daily')&(market.variable=='market_return_vw')&(market.year>=2015)]
 fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(market.year,100*market.std_dev,color=teal);ax.set(title='Daily market-return volatility by year',xlabel='Year',ylabel='Standard deviation of daily return (%)');ax.set_xticks(market.year);finish(fig,'market_return_volatility')
 fig,ax=plt.subplots(figsize=(10,4.8));labels=industry.ff12_name.fillna('Unclassified');ax.barh(labels,100*industry.eligible_stock_days/industry.eligible_stock_days.sum(),color=teal);ax.set(title='Industry composition of eligible stock-days',xlabel='Percent of eligible stock-days');finish(fig,'industry_composition')

 rows=int(coverage.stock_days.sum());observed=int(coverage.observed_labels.sum());eligible=int(coverage.eligible_labels.sum());low=coverage.loc[coverage.observed_coverage_pct.idxmin()];first=annual.iloc[0];last=annual.iloc[-1]
 med=val('auction_daily','auction_volume');mean=val('auction_daily','auction_volume','mean');p99=val('auction_daily','auction_volume','p99');sd=val('auction_daily','auction_volume','std_dev')
 direct=inv.iloc[:8];ref=inv.iloc[8:]
 def fmt(x):
  if pd.isna(x):return '—'
  if isinstance(x,(float,np.floating)):
   if abs(x)>=1000:return f'{x:,.2f}' if not x.is_integer() else f'{x:,.0f}'
   if x==0:return '0'
   if abs(x)<.001:return f'{x:.6g}'
   return f'{x:,.4f}'
  if isinstance(x,(int,np.integer)):return f'{x:,}'
  return str(x)
 def table(frame):return frame.to_html(index=False,escape=True,border=0,formatters={c:fmt for c in frame.columns})
 def mdtable(frame):
  cols=list(frame.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
  for row in frame.itertuples(index=False,name=None):lines.append('| '+' | '.join(fmt(v).replace('|','/') for v in row)+' |')
  return '\n'.join(lines)
 sections=[];markdown=[]
 def section(title,paragraphs=(),frame=None,chart=None):
  content=''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
  markdown.extend(['\n## '+title+'\n',*paragraphs])
  if frame is not None:content+='<div class="table-wrap">'+table(frame)+'</div>';markdown.append('\n'+mdtable(frame)+'\n')
  if chart:
   p=figures/(chart+'.png');content+='<img alt="'+html.escape(title)+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">';markdown.append(f'\n![{title}](figures/{chart}.png)\n')
  sections.append('<section><h2>'+html.escape(title)+'</h2>'+content+'</section>')
 section('Scope and populations',[
 'This report covers source data used through the first 13 research notebooks. A small selection of these results now appears in the presentation. Options-market data and FOMC-specific data are excluded. Monthly and quarterly options-expiry calendar flags remain included.',
 '“Raw data” here means the local, pre-modeling Parquet extracts and supplied context tables. The repository does not contain the original exchange tick feed used to create the extracts. Market and sector tables are already aggregated context inputs. No raw source files or research notebooks were changed.',
 'All source rows are profiled without imputation, winsorization, sampling, or replacing missing targets with zero. Full-history source summaries and the 2015–2025 modeling-period summaries are labeled separately. Daily-market and calendar statistics use trading days. Sector statistics use sector-days. Security statistics use stock-days. Intraday statistics use stock-day-checkpoints. These counts should not be added into one sample size.',
 'The full industry-history reference contains securities and intervals outside the study universe. Its raw inventory is reported separately from industry composition matched to eligible NYSE stock-days. The upstream security master, basic calendar, and symbol bridge are documented separately from the eight directly read inputs.'
 ])
 section('Source inventory: eight directly used inputs',frame=direct[['dataset','rows','dates','securities','first_date','last_date','columns','role']])
 section('Upstream reference inventory',frame=ref[['dataset','rows','dates','securities','first_date','last_date','role']])
 section('Target coverage and eligibility',[
 f'The 2015–2025 CRSP universe has {rows:,} stock-days. Of these, {observed:,} have an observed official auction label ({100*observed/rows:.2f}%). Excluding the documented source anomaly leaves {eligible:,} eligible labels ({100*eligible/rows:.2f}%).',
 f'The weakest observed-label coverage is {int(low.year)} at {low.observed_coverage_pct:.2f}%. The anomaly exclusion affects {int(coverage.excluded_anomaly_stock_days.sum()):,} stock-days on August 12, 2019, of which {observed-eligible:,} had an observed label. The 2019 eligible coverage therefore differs from observed-label coverage.',
 f"The CRSP universe has {rows-int(inv.loc[inv.dataset=='auction_daily','rows'].iloc[0]):,} more 2015–2025 stock-days than the auction extract. Those unmatched records remain missing targets. Coverage is measured relative to the CRSP universe, not just rows present in the auction file."
 ],coverage,chart='target_coverage')
 target=exact[exact.dataset=='auction_daily'][['variable','unit','finite_rows','mean','std_dev','minimum','p25','median','p75','p95','p99','maximum']]
 section('Auction outcomes: eligible stock-days',[
 f'Eligible auction volume has a mean of {mean:,.0f} shares, median of {med:,.0f}, standard deviation of {sd:,.0f}, and 99th percentile of {p99:,.0f}. The mean is {mean/med:.2f} times the median. Large auctions materially influence averages.',
 'These statistics use exact quantiles of the eligible population. The distribution chart counts every eligible observation in fixed log-volume bins. A log axis helps display the skew, without changing the raw share-volume target. Auction price and same-day auction/daily-volume share are outcomes or descriptive measures, not daily predictors.'
 ],target,chart='auction_distribution')
 section('Changes in auction size and share',[
 f'The median auction rises from {first.median_auction:,.0f} shares in 2015 to {last.median_auction:,.0f} in 2025. The median stock-day auction share rises from {100*first.median_auction_share:.2f}% to {100*last.median_auction_share:.2f}%. These are descriptive comparisons in a changing sample, not a fixed-stock or causal trend estimate.',
 'The aggregate-share series is sum(auction shares) / sum(CRSP daily shares) on eligible stock-days. It gives more weight to high-volume stocks and differs from the median stock-day share. It describes this extracted universe, not the whole NYSE. Annual sums are observed-sample totals and do not compensate for missing labels.'
 ],annual[['year','eligible_stock_days','securities','median_auction','mean_auction','p99_auction','median_auction_share','volume_weighted_auction_share']],chart='auction_size_and_share')
 crsp=exact[exact.dataset=='crsp_daily'][['variable','unit','finite_rows','mean','std_dev','minimum','p25','median','p75','p99','maximum','missing_or_nonfinite_pct']]
 section('CRSP raw stock characteristics, 2015–2025',[
 'Price, daily share volume, dollar volume, market capitalization, returns, and shares outstanding are summarized on all source stock-days in the modeling period, including stock-days lacking an auction target. Returns are decimal daily returns. Shares outstanding are stored in thousands. Market capitalization is in USD.',
 'Pooled price and market-cap means can be much larger than medians. A high price is not automatically an error. The all-field profile retains raw ranges and missingness so unusual values can be investigated without silently deleting them.'
 ],crsp)
 section('CRSP missingness',[
 'The chart uses the complete 2011–2025 source history. Tables also provide per-field missingness by year. Missingness in historical lags and rolling model features is a separate feature-engineering issue and is outside this raw-data report.'
 ],chart='crsp_missingness')
 sector=read('sector_native_summary')
 section('Market and sector context',[
 'Market statistics use one observation per trading day. Sector statistics use one observation per industry and trading day. They are not repeated once per stock. The full inputs contain 3,772 market days and 45,264 sector-days. The 2015–2025 subset contains 2,766 market days.',
 'The market-return chart shows the within-year standard deviation of daily returns, not an annualized return or a causal volatility effect. Supplied market/sector aggregates describe their construction universe; no external all-market coverage claim is made.'
 ],exact[exact.dataset=='market_daily'][['variable','unit','finite_rows','mean','std_dev','p25','median','p75','p99']],chart='market_return_volatility')
 section('Sector summaries, 2015–2025',frame=sector)
 eventflags=flags[(flags.dataset=='event_calendar')&(flags.year>=2015)]
 events=eventflags.groupby('variable',as_index=False)[['rows','true_rows','missing_rows']].sum();events['true_pct_dates']=100*events.true_rows/events.rows
 main=['monthly_options_expiry','quarterly_options_expiry','is_last_trading_day_month','is_last_trading_day_quarter','is_early_close','expiry_month_end_window','early_close_month_end_window']
 section('Calendar events, 2015–2025',[
 'Event frequencies below count distinct trading dates, not stock-days. Indicators overlap. For example, quarterly expiries are also monthly expiries, so event counts must not be summed as mutually exclusive groups. These flags are known in advance and require no options-market dataset.',
 'The separate calendar_target_descriptives.csv contrasts eligible auction outcomes by each event flag. Those contrasts are pooled associations. They do not control for stock composition, time trends, or overlapping calendar events.'
 ],events[events.variable.isin(main)])
 section('Stock size and industry composition',[
 'Size deciles are formed separately on each date from contemporaneous positive market capitalization, using eligible stock-days. This is a descriptive grouping. The forecasting pipeline uses lagged information for actual model inputs.',
 'Industry composition is matched by PERMNO and effective-date interval. If several records cover the same stock-date, the latest interval start/end is selected, matching the existing pipeline. Industry shares count eligible stock-days, so long-lived securities contribute more observations.'
 ],industry,chart='auction_by_size')
 section('Industry composition',chart='industry_composition')
 cp=cp.copy();cp['coverage_within_session_pct']=100*cp.valid_observed_rows/cp.within_session_rows
 section('Intraday trade checkpoints',[
 'These data were audited in the early research stages, but the daily models through stage 13 do not use same-day checkpoint inputs. The raw table has five rows for each represented stock-date.',
 'There are 29,540 post-close rows at each of the 14:00, 15:00, and 15:30 checkpoints. These are scheduled early-close exclusions, not ordinary data failures. Within-session coverage uses only scheduled-valid checkpoint rows in its denominator. Distribution summaries for usable checkpoints further require checkpoint_data_found=true.',
 'Median cumulative volume rises across checkpoints. The plotted points are medians of checkpoint cross-sections, not the path of one representative stock. Time labels are Eastern Time.'
 ],cp,chart='checkpoint_volume')
 structural=profile[profile.schema_absent_rows>0].copy()
 section('Schema differences and missing data',[
 f'The union of yearly auction schemas contains {int(inv.loc[inv.dataset=="auction_daily","columns"].iloc[0])} fields. {structural[structural.dataset=="auction_daily"].shape[0]} of those fields are absent from at least one annual file. Many are auxiliary fields or duplicated CRSP fields, rather than missing auction outcomes.',
 'field_profile.csv separates schema_absent_rows from null_in_present_schema_rows. The former counts rows in files where a column does not exist at all. Schema-absence counts come from each file’s actual schema and row count. Numeric summaries use only finite observations; nonfinite counts are reported separately.',
 'The comprehensive numeric_summary.csv includes every numeric measure except identifiers/category codes. Its quantiles are approximate full-population quantiles, explicitly named *_approx. principal_exact_summary.csv and the selected annual/checkpoint summaries use exact quantiles. No summary uses a convenience sample. Categorical frequencies retain up to 30 levels per field; identifier frequencies are omitted. Exact null counts, types, units, and approximate distinct counts are provided for every field.'
 ],structural[['dataset','variable','files_with_field','files_total','schema_absent_rows','null_in_present_schema_rows']])
 section('Keys, domain checks, and reconciliation',[
 f'All {len(keys)} profiled tables have zero duplicate excess rows for their stated keys. All {len(quality)} listed numeric-domain checks return zero flagged rows. This verifies the stated checks, not every possible source-data issue.',
 'Fresh source-row counts reconcile to the existing data inventory. Fresh eligible-label counts reconcile to the saved research sample-eligibility totals. The large engineered model panel is not required for this raw-data report. Source files are read without rewriting them. The run records file sizes and modification times for reproducibility.'
 ],quality)
 section('What these results support',[
 'The data support a discussion of high but time-varying target coverage, strongly skewed auction size, variation across size and industry, changing auction participation, and the distinction between daily predictors and intraday information.',
 'Before choosing slides, we should decide which few exhibits best explain the forecasting problem. Selected results now appear in two graph slides and one descriptive-summary slide. The full report remains available for reference. This report does not estimate causal effects, test model accuracy, or select the final descriptive exhibits.'
 ])
 links=[]
 for p in sorted(tables.glob('*.csv')):links.append(f'<li><a href="tables/{html.escape(p.name)}">{html.escape(p.name)}</a></li>')
 sections.append('<section><h2>Complete tables</h2><p>Use the CSV files for full precision and all variables. Every table is generated from the saved script.</p><ul>'+''.join(links)+'</ul></section>')
 title='Raw-data descriptive statistics: midterm scope'
 header=f'<header><h1>{title}</h1><p>Research through stage 13 · options-market data excluded · results kept separate from the presentation</p></header>'
 css='''body{font:16px/1.55 system-ui,sans-serif;color:#243746;background:#f4f6f7;margin:0}main{max-width:1250px;margin:auto;padding:36px}header{padding:8px 0 24px;border-bottom:3px solid #007f82}h1{font-size:34px;color:#102c43}h2{font-size:25px;color:#102c43}section{background:white;padding:28px;margin:24px 0}p{max-width:1050px}img{display:block;width:100%;max-width:1100px;margin:24px auto}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;border-bottom:1px solid #d9e2e6;padding:9px;white-space:nowrap}th{background:#eaf2f4;text-align:left;position:sticky;top:0}td:first-child{text-align:left}a{color:#007f82}#filter{padding:10px;width:320px;max-width:90%;font:inherit}@media print{body{background:white}main{padding:0}section{break-inside:avoid;padding:12px;margin:12px 0}.table-wrap{overflow:visible}table{font-size:9px}th,td{white-space:normal;padding:4px}#filter{display:none}}'''
 # A simple live section filter makes the long report easy to navigate without dependencies.
 script="document.getElementById('filter').addEventListener('input',e=>{const s=e.target.value.toLowerCase();document.querySelectorAll('section').forEach(x=>x.hidden=!x.innerText.toLowerCase().includes(s));});"
 (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style><main>'+header+'<p><label>Find a topic <input id="filter" placeholder="coverage, sector, missingness…"></label></p>'+''.join(sections)+'</main><script>'+script+'</script></html>')
 (out/'report.md').write_text('# '+title+'\n\n'+'\n\n'.join(markdown)+'\n\n## Complete tables\n\n'+''.join('- ['+p.name+'](tables/'+p.name+')\n' for p in sorted(tables.glob('*.csv'))))
 print(f'Rendered {len(sections)} report sections and {len(list(figures.glob("*.png")))} charts',flush=True)
if __name__=='__main__':render(Path(__file__).resolve().parent)
