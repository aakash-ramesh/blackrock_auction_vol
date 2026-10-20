"""Render the natural-unit association atlas and a compact findings note."""
from pathlib import Path
import sys,json,html,re
try:import matplotlib
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parent;T=OUT/'tables';F=OUT/'figures';F.mkdir(exist_ok=True)
summary=pd.read_csv(T/'feature_associations.csv');catalog=pd.read_csv(T/'covariate_catalog.csv');curves=pd.read_csv(T/'panel_bin_curves.csv');nativecurves=pd.read_csv(T/'native_bin_curves.csv')
NAVY='#102C43';TEAL='#007F82';GRAY='#8B9AA8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','text.color':NAVY,'axes.labelcolor':NAVY,'figure.dpi':120})
def compact(x,pos=None):
 if abs(x)>=1e6:return f'{x/1e6:g}m'
 if abs(x)>=1000:return f'{x/1000:g}k'
 return f'{x:g}'
def filename(c):return re.sub(r'[^A-Za-z0-9_-]','_',c)
def fmt(v):
 if pd.isna(v):return '—'
 if isinstance(v,(float,np.floating)):
  if abs(v)>=1000:return f'{v:,.1f}'
  if v==0:return '0'
  if abs(v)<.0001:return f'{v:.3g}'
  return f'{v:.4f}'
 if isinstance(v,(int,np.integer)):return f'{v:,}'
 return str(v)
def mdtable(d):return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v).replace('|','/') for v in r)+' |' for r in d.itertuples(index=False,name=None)])
def table(d):return d.to_html(index=False,escape=True,border=0,formatters={c:fmt for c in d.columns})
# Complete one-variable plots show only raw-share outcomes. Means and medians are not mixed with log targets.
sections=[]
for spec in catalog.to_dict('records'):
 c=spec['feature'];s=summary[summary.feature.eq(c)].iloc[0];d=curves[curves.feature.eq(c)].copy();d['order']=d.bin_order if 'bin_order' in d else pd.to_numeric(d.bin,errors='coerce');d=d.sort_values('order') if d.order.notna().all() else d.sort_values('bin')
 if not (F/(filename(c)+'.png')).exists() or '--refresh-figures' in sys.argv:
  fig,ax=plt.subplots(figsize=(6.6,3.7));xx=np.arange(len(d))
  if spec['kind'] in {'categorical','binary'}:
   ax.bar(xx,d.mean_auction_shares,color=TEAL,label='Mean');ax.plot(xx,d.median_auction_shares,'o',color=NAVY,label='Median')
   labels=d.bin.astype(str).tolist()
   if spec['kind']=='binary':labels=['No' if v in {'0','0.0'} else 'Yes' if v in {'1','1.0'} else v for v in labels]
   ax.set_xticks(xx,labels,rotation=45 if len(labels)>8 else 0);ax.set_xlabel('Observed category')
  else:
   ax.plot(xx,d.mean_auction_shares,'o-',color=TEAL,label='Mean');ax.plot(xx,d.median_auction_shares,'s--',color=NAVY,label='Median');ax.set_xticks(xx,d.bin.astype(str));ax.set_xlabel('Observed value / rare-value group' if d.grouping_method.str.contains('common|observed_numeric').any() else 'Covariate quantile bin (ties kept together)')
  ax.set_ylabel('Auction volume (shares)');ax.yaxis.set_major_formatter(FuncFormatter(compact));ax.set_title(c,fontsize=10);ax.legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(F/(filename(c)+'.png'),bbox_inches='tight');plt.close(fig)
 detail=s.to_frame('value').reset_index().rename(columns={'index':'statistic'})
 keep=['paired_stock_days','missing_pct','primary_observations','primary_unit','primary_pearson_raw','primary_spearman','rank_observations','within_stock_pearson_raw','pooled_univariate_slope','within_stock_slope','pooled_associated_shares_per_iqr','binary_mean_difference_shares','binary_median_ratio','positive_years','negative_years','distinct_values']
 detail=detail[detail.statistic.isin(keep)]
 layer=spec['layer'];badge='Selected daily feature' if spec['selected'] else 'Prior-close candidate' if spec['daily_candidate'] else 'Descriptive only / not a daily input'
 sections.append(f'''<details data-layer="{html.escape(layer)}"><summary>{html.escape(c)} <span>{html.escape(badge)}</span></summary>
<p><b>Source / unit:</b> {html.escape(str(spec['source_variable']))} / {html.escape(str(spec['unit']))}. <b>Timing:</b> {html.escape(spec['timing'])}.</p>
<p>{html.escape(spec['intuition'])}</p><p>{html.escape(str(s['limitation']))}</p>
<img src="figures/{filename(c)}.png" alt="Raw-share binned relationship for {html.escape(c)}"><div class="scroll">{table(detail)}</div>
<p>Binned outcomes above use paired eligible stock-days, even for shared date/sector context. Primary correlation uses the native observation unit stated in the table. Full native-unit bin curves are available in the CSV outputs. One-point category charts indicate no sample variation.</p><div class="scroll">{table(d.drop(columns='order'))}</div></details>''')
# Raw-source comparison is source-level, rather than duplicating 31 pairs without explanation.
rawrows=[]
for source,g in catalog[catalog.layer.isin(['raw_same_day','raw_prior_close'])].groupby('source_variable',sort=True):
 same=g[g.layer.eq('raw_same_day')].iloc[0];prior=g[g.layer.eq('raw_prior_close')].iloc[0];a=summary[summary.feature.eq(same.feature)].iloc[0];b=summary[summary.feature.eq(prior.feature)].iloc[0]
 rawrows.append(dict(source_covariate=source,group=prior['group'],native_unit=b.primary_unit,current_pearson_raw=a.get('primary_pearson_raw',np.nan),prior_pearson_raw=b.get('primary_pearson_raw',np.nan),prior_spearman=b.get('primary_spearman',np.nan),prior_within_stock_pearson_raw=b.get('within_stock_pearson_raw',np.nan),prior_stock_days=int(b.paired_stock_days),prior_missing_pct=b.missing_pct,distinct_prior_values=int(b.distinct_values),economic_hypothesis=prior.intuition))
rawtable=pd.DataFrame(rawrows);rawtable.to_csv(T/'raw_covariate_comparison.csv',index=False)
# Focus figures for discussion: each panel keeps the target in shares.
chosen_examples=['median20_auction_volume','prior_raw_daily_volume','prior_raw_market_cap_usd','prior_raw_return_total','lag1_quoted_spread','quarterly_options_expiry']
fig,axes=plt.subplots(2,3,figsize=(14,8))
for ax,c in zip(axes.flat,chosen_examples):
 d=curves[curves.feature.eq(c)].copy();d['order']=d.bin_order if 'bin_order' in d else pd.to_numeric(d.bin,errors='coerce');d=d.sort_values('order');x=np.arange(len(d));ax.plot(x,d.mean_auction_shares,'o-',color=TEAL,label='Mean');ax.plot(x,d.median_auction_shares,'s--',color=NAVY,label='Median');ax.set_title(c,fontsize=11);ax.set_xticks(x,d.bin.astype(str));ax.set_ylabel('Auction shares');ax.yaxis.set_major_formatter(FuncFormatter(compact));ax.set_xlabel('Quantile bin / binary category')
axes[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(F/'economic_examples.png',bbox_inches='tight');plt.close(fig)
comparison=pd.read_csv(T/'historical_baseline_comparison.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric,title in [(axes[0],'rmse_log','Log RMSE'),(axes[1],'wape','WAPE')]:
 factor=100 if metric=='wape' else 1;ax.plot(comparison.test_year,factor*comparison[metric+'_baseline'],'o--',color=GRAY,label='Median-20 fallback');ax.plot(comparison.test_year,factor*comparison[metric+'_selected'],'o-',color=TEAL,label='Selected 58-feature model');ax.set(title=title,xlabel='Forecast year',ylabel='Percent' if metric=='wape' else 'Log units');ax.set_xticks(comparison.test_year)
axes[0].legend(frameon=False,fontsize=9);fig.suptitle('Historical annual forward comparison (2025 is confirmation already inspected)');fig.tight_layout();fig.savefig(F/'historical_baseline_comparison.png',bbox_inches='tight');plt.close(fig)
family=pd.read_csv(T/'historical_feature_family_ablation.csv');family=family[family.specification.str.startswith('full_drop__')].copy();family['label']=family.specification.str.replace('full_drop__','',regex=False);family=family.sort_values('mean_rmse_delta_vs_full')
fig,ax=plt.subplots(figsize=(10,5));ax.barh(family.label,family.mean_rmse_delta_vs_full,color=TEAL);ax.axvline(0,color=GRAY);ax.set(title='Removing auction history or calendar causes the largest loss',xlabel='Change in mean log RMSE versus full 99-feature model');fig.text(.08,.01,'Exploratory family audit averaged across 2022–2025; not the later development-only selection.',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(F/'historical_family_ablation.png',bbox_inches='tight');plt.close(fig)
final=pd.read_csv(T/'historical_finalist_selection.csv').sort_values('development_rmse_log',ascending=False)
labels={'expanding_log_full':'Expanding, log level (99)','rolling5_log_full':'5-year, log level (99)','rolling3_residual_full':'3-year, residual (99)','rolling5_residual_auction_calendar':'5-year, auction + calendar (42)','rolling5_residual_auction_calendar_liquidity':'5-year, auction + calendar + liquidity (58)','rolling5_residual_full':'5-year, full residual (99)','rolling5_residual_recent_full':'5-year, full residual + recent weights (99)'}
fig,ax=plt.subplots(figsize=(11,5));colors=[TEAL if x else GRAY for x in final.selected_global_champion];ax.scatter(final.development_rmse_log,np.arange(len(final)),c=colors,s=65);ax.set_yticks(np.arange(len(final)),[labels[x] for x in final.candidate]);ax.axvline(final.development_rmse_log.min()+.002,color=NAVY,ls='--',label='Best + 0.002 tolerance');ax.set(xlabel='Mean annual log RMSE, 2022–2024',title='The selected model is the smaller practical tie');ax.legend(frameon=False,fontsize=9);fig.tight_layout();fig.savefig(F/'historical_finalist_selection.png',bbox_inches='tight');plt.close(fig)
# Table for a short, readable findings note.
examples=['median20_auction_volume','lag1_auction_volume','prior_raw_daily_volume','prior_raw_market_cap_usd','prior_raw_return_total','lag1_amihud','lag1_quoted_spread','quarterly_options_expiry','monthly_options_expiry','is_last_trading_day_month']
small=summary.set_index('feature').loc[examples].reset_index()[['feature','primary_unit','paired_stock_days','primary_observations','primary_pearson_raw','within_stock_pearson_raw','primary_spearman']]
small.to_csv(T/'discussion_associations.csv',index=False)
findings=f'''# Natural-unit covariate findings

## What was added

The new study covers **193 specifications**: all 99 engineered daily features, 31 economic source covariates measured contemporaneously and 31 prior-close counterparts, two closing-outcome diagnostics, and six intraday variables at each of five checkpoints. All 58 selected model features have individual results. Calendar and industry covariates are covered in the daily feature set. Identifiers, duplicated source representations, symbol-link metadata, and quality flags are not treated as economic predictors.

The primary target is always **official closing-auction volume in shares**. Every binned curve reports its mean and median in shares, without a log transformation. The old relationship work was useful but incomplete for this purpose: continuous relationships largely used a transformed target, while some binary and categorical summaries already reported raw medians.

## The main empirical distinctions

1. **Auction history is the strongest ordering signal.** The lagged 20-auction median has raw Pearson correlation about 0.521 and sampled Spearman correlation about 0.920 with the raw target. Its within-stock raw-share Pearson correlation is about 0.222. The rank relationship is much stronger than the raw linear relationship because extreme auction spikes contribute heavily to raw-share variation. A strong pooled relationship is not the same as a uniformly accurate forecast.
2. **Trading liquidity is informative, but same-day volume is not a permissible daily input.** Current daily volume has raw Pearson correlation about 0.527. The prior-close counterpart is about 0.393, with rank correlation about 0.830 and within-stock raw correlation about 0.138. Current-day total volume includes the auction itself. Its stronger contemporaneous association therefore includes mechanical overlap and unavailable closing information.
3. **Market cap describes stock scale more strongly than deviations within a stock.** Prior market cap has raw pooled Pearson correlation about 0.289 and rank correlation about 0.654, but within-stock raw Pearson correlation of only about 0.022. This is compatible with omitting a separate size family when auction and liquidity history already encode scale. It does not mean size is economically irrelevant.
4. **Signed prior returns do not show a strong monotone relationship.** Their raw Pearson correlation is about 0.001 and sampled rank correlation about 0.006. Annual raw correlations are positive in six years and negative in five. The raw-share bin curve is non-monotone. This does not rule out information in absolute returns, volatility, or interactions; absolute returns already enter a retained illiquidity proxy.
5. **Illiquidity proxies mostly separate the cross-section.** Lagged Amihud and quoted spread have negative rank associations (about −0.590 and −0.401), but within-stock raw correlations close to zero. In the spread bins, low-spread stocks generally have larger auctions, with some non-monotonicity at the tightest spreads. A large pooled rank effect does not prove incremental value for every liquidity feature.
6. **Calendar contrasts are large, but overlapping.** The quarterly-expiry raw median ratio is about 6.90 relative to non-quarterly dates; monthly expiry is about 1.92 and month-end about 1.82. Quarterly expiry has a date-level correlation about 0.727 with mean auction shares per stock, compared with stock-day pooled correlation about 0.180. The aggregation removes much cross-sectional scale noise, but does not identify a causal effect or remove overlapping events and trends.
7. **Some source fields cannot have an isolated estimated relationship in this universe.** Security type, subtype, share type, US incorporation, and issuer type are constant by the universe definition, both in current and prior representations. Their relationship is unidentified from within-sample variation. A one-category plot describes the sample and is not an effect estimate.
8. **Intraday activity can associate with the final auction without belonging in the existing daily model.** Cumulative volume at 15:30 has sampled rank correlation about 0.805. It is unavailable at the daily prediction cutoff and was not used by the selected daily model. Checkpoints after early closes are excluded. These diagnostics do not establish the improvement of an intraday forecast model.

{mdtable(small)}

The primary unit is a stock-date for stock-specific inputs, a date for shared calendar/market inputs, and a sector-date for sector inputs. For date and sector-date inputs, the primary raw target is the mean of eligible stock auction shares in that group. The within-stock column remains a separate stock-day-weighted association. Correlations across these different units should not be ranked as if they were one common experiment.

![Selected one-variable raw-share relationships](figures/economic_examples.png)

## Why the results support economic discussion rather than automatic feature selection

The retained 58 features span auction history, calendar, and liquidity. Large and liquid stocks generally have larger share auctions, but that observation does not prove that adding every size or volume proxy helps a model that already knows its auction history. The original forward family tests and the combined finalist comparisons supply incremental predictive evidence. The new one-variable curves supply economic interpretation and warn about scale, tails, timing, and redundancy.

A negative fitted model coefficient also need not contradict a positive marginal association. For example, the lagged 20-auction median is positively associated with auction size but has a negative conditional coefficient in the residual model. It is simultaneously the baseline offset and one of several highly correlated inputs. Preprocessing and regularization change coefficient units. That coefficient is not a causal statement that a bigger normal auction reduces future auction volume.

## How to read the complete outputs

- `covariate_atlas.html`: searchable individual graphs, timing labels, natural-unit slopes, group outcomes and economic hypotheses for every specification.
- `tables/raw_covariate_comparison.csv`: all 31 economic source covariates, comparing current and prior-close associations.
- `tables/selected_feature_dictionary.csv`: every selected feature, its family, timing, economic interpretation, new individual associations, and the original role/conditional-coefficient context.
- `tables/panel_bin_curves.csv`: exact group means, medians, 90th percentiles and observation counts for every feature, using paired eligible stock-days. Numeric bins use full-population quantile cutpoints with ties kept together. Low-cardinality numeric fields use observed values. For concentrated inputs, the common mass point is separated from quantile groups of the remaining values. Fewer than ten groups are possible; grouping_method and bin_order are recorded.
- `tables/native_bin_curves.csv`: corresponding date- or sector-date-weighted curves for shared inputs.
- `tables/annual_associations.csv` and `period_associations.csv`: annual and historical/development/confirmation-period raw associations.
- `tables/missingness_contrasts.csv`: missing/nonfinite counts and raw target means when an input is missing versus observed.

## Method and limits

The primary Pearson correlations, single-variable OLS slopes with an intercept, and binned outcomes use every finite pair in the eligible sample. Within-stock slopes/correlations demean both raw values on the same pairwise-complete stock sample. They remove time-invariant stock levels, not date shocks or time-varying confounding. The OLS coefficient is shares associated with one native unit of the input. The IQR-scaled association is a descriptive linear rescaling, not a causal predicted intervention. It is omitted for binary inputs or zero-IQR inputs; binary mean differences and median ratios are supplied instead.

The secondary stock-date rank correlations use a deterministic approximately 7% hash sample, roughly 245,000 rows per complete field, to keep exhaustive ranking computations manageable. Shared date/sector inputs use all their native observations for ranks. Exact rank counts are in the results. No target or covariate transformation is used for the primary raw-unit Pearson/OLS estimates. Rank measures are explicitly secondary ordering diagnostics and are invariant to monotone target transforms.

Missingness is retained, with pairwise finite rows used per input. No imputation or winsorization is applied. Primary results can therefore be sensitive to extreme observations; means, medians, ranks and annual directions should be read together. Categorical variables are treated as categories, not artificial numeric scales. Market/calendar primary units avoid pretending that the same common input constitutes millions of independent observations. No IID p-values or causal claims are made.

These are retrospective 2015–2025 diagnostics, not untouched validation or a substitute for a forward incremental-feature experiment. Current-day auction price, current-day daily volume, and auction share are explicitly marked as descriptive outcomes or unavailable daily inputs. Known options-expiry calendar flags do not use stock-specific options-market data.
'''
(OUT/'covariate_findings.md').write_text(findings)
intro='''<header><h1>Individual covariate associations with auction volume</h1><p>Primary target: closing-auction shares. Retrospective analysis of eligible 2015–2025 observations. Association does not establish causality or incremental forecast value.</p><p>Stock-date bin plots retain the target in shares. Shared market/calendar primary correlations use dates; sector context uses sector-dates. Secondary ranks use a deterministic 7% stock-date sample, or all native context observations. No IID p-values are reported.</p><p>Current-day outcomes and intraday inputs are marked separately from permissible prior-close information. All 58 selected model features have individual entries.</p></header>'''
css='''body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f3f6f7;color:#243746}main{max-width:1150px;margin:auto;padding:30px}h1{color:#102c43}header{border-bottom:3px solid #007f82;padding-bottom:15px}details{background:white;padding:18px;margin:12px 0}summary{cursor:pointer;font-weight:650;color:#102c43}summary span{font-size:12px;color:#637482;margin-left:12px}img{width:100%;max-width:750px}input,select{font:inherit;padding:8px;margin:6px}table{font-size:12px;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid #d9e2e6;text-align:left;white-space:nowrap}.scroll{overflow:auto}a{color:#007f82}#count{font-size:14px;color:#637482}'''
script="""const items=[...document.querySelectorAll('details')];function filter(){let n=0;const text=document.querySelector('#find').value.toLowerCase(),layer=document.querySelector('#layer').value;for(const d of items){const show=d.innerText.toLowerCase().includes(text)&&(!layer||d.dataset.layer===layer);d.hidden=!show;n+=show;}document.querySelector('#count').textContent=n+' specifications shown';}document.querySelector('#find').addEventListener('input',filter);document.querySelector('#layer').addEventListener('change',filter);filter();"""
select='<option value="">All specifications</option>'+''.join('<option value="'+x+'">'+x.replace('_',' ')+'</option>' for x in catalog.layer.unique())
(OUT/'covariate_atlas.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Raw-target covariate atlas</title><style>'+css+'</style><main>'+intro+'<p><input id="find" placeholder="Find a variable or economic idea"><select id="layer">'+select+'</select></p><p id="count"></p>'+''.join(sections)+'</main><script>'+script+'</script></html>')
print(f'Rendered {len(sections)} individual plots and the economic findings note.')
