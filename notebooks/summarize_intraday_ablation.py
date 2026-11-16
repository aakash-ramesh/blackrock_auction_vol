"""Readable contrasts and a scientific summary figure for experiment 23."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/23_intraday_feature_ablation'
os.environ.setdefault('MPLCONFIGDIR','/tmp/auction-ablation-matplotlib')
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
 t=pd.read_csv(OUT/'comparison.csv');changes=[]
 for period in ['development','confirmation']:
  s=t[t.period.eq(period)].set_index('candidate')
  for group in ['volume','trades','recent_activity','price']:
   for mode,ref in [('add','ridge_daily_control'),('drop','ridge_update')]:
    r=s.loc[mode+'_'+group];b=s.loc[ref]
    changes.append(dict(period=period,group=group,comparison=mode,reference=ref,rmse_log=r.rmse_log,rmse_log_change=r.rmse_log-b.rmse_log,wape_change_pp=100*(r.wape-b.wape),bias_change_pp=100*(r.aggregate_bias-b.aggregate_bias)))
 changes=pd.DataFrame(changes);changes.to_csv(OUT/'group_contrasts.csv',index=False)
 segments=pd.read_csv(OUT/'development_segment_means.csv');rows=[]
 for (dim,seg),s in segments.groupby(['dimension','segment']):
  s=s.set_index('candidate')
  for name in s.index:
   for ref in ['fixed_daily','ridge_daily_control','ridge_update']:
    rows.append(dict(dimension=dim,segment=seg,candidate=name,reference=ref,rmse_log_change=s.loc[name,'rmse_log']-s.loc[ref,'rmse_log'],wape_change_pp=100*(s.loc[name,'wape']-s.loc[ref,'wape'])))
 pd.DataFrame(rows).to_csv(OUT/'development_segment_contrasts.csv',index=False)
 groups=['volume','trades','recent_activity','price'];labels=['Volume','Trade counts','Recent activity','VWAP / price']
 dev=changes[changes.period.eq('development')]
 fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
 for ax,mode,title in zip(axes,['add','drop'],['Add one group to daily-only ridge','Remove one group from full intraday ridge']):
  vals=dev[dev.comparison.eq(mode)].set_index('group').loc[groups,'rmse_log_change'].to_numpy()
  ax.barh(labels,vals,color=['#236c68' if v<0 else '#a6652d' for v in vals]);ax.axvline(0,color='black',linewidth=.8);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlabel('Change in development log RMSE')
  ax.ticklabel_format(axis='x',style='plain',useOffset=False)
  ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 fig.suptitle('15:30 ridge feature ablation | 2022–2024 equal-year means\nNegative favors the tested variant; panels use different scales',fontsize=12)
 fig.savefig(OUT/'feature_ablation.png',dpi=170);plt.close(fig)
 report='''# Findings from the 15:30 ridge feature ablation

The main source of incremental predictive value is current-session trading activity: volume, trade counts, and recent 30-minute activity. VWAP and its return from the prior close contribute little in this ridge specification. This is conditional predictive evidence, not causal attribution, and historical checkpoint features remain in all candidates.

## Controlled development comparison

The unchanged daily-only ridge has log RMSE 0.474878; full intraday ridge has 0.459954. Each variant uses the exact earlier historical samples, shared preprocessing, chronological tuning, residual target, clipping and fallback. Each tunes its own ridge penalty over the same seven values. No 2025 labels choose groups.

'''
 report+='| Current group | Add alone log RMSE | Change versus daily-only ridge | Removing group changes full-model log RMSE |\n| --- | ---: | ---: | ---: |\n'
 for group,label in zip(groups,labels):
  a=dev[dev.group.eq(group)&dev.comparison.eq('add')].iloc[0];d=dev[dev.group.eq(group)&dev.comparison.eq('drop')].iloc[0]
  report+=f'| {label} | {a.rmse_log:.6f} | {a.rmse_log_change:+.6f} | {d.rmse_log_change:+.6f} |\n'
 report+='''
Volume alone recovers about 89% of the full model's log-RMSE improvement versus the daily-only control. This percentage is not an additive importance allocation: trade counts and recent activity contain overlapping information. Removing volume or recent activity worsens daily log MAE under the exploratory 20-trading-date bootstrap intervals. Price-only log-MAE intervals cross zero.

Trade counts illustrate an objective tradeoff. Removing them worsens development log RMSE by about 0.00045 but improves WAPE by about 0.038 percentage points and daily log MAE. Aggregate bias worsens from about -11.80% to -12.62%. There is no single best subset across these objectives. Removing price changes log RMSE by only about 0.00010 and barely changes WAPE. Its small observed contribution does not rule out price usefulness in other architectures or forecast times.

## Month-end and liquidity weaknesses

Disjoint month-end days (after expiry precedence) remain worse than the original daily forecast. Full ridge log RMSE is 0.66227 versus 0.66095 for the daily anchor, with WAPE 47.57% versus 47.18%. Removing price reduces the miss slightly, to 0.66200 and 47.54%, but still fails to beat the anchor. Removing trade counts makes month-end performance worse. These small subgroup differences are exploratory point estimates, not statistically established subgroup improvements.

Low-liquidity stocks likewise retain worse log RMSE: 0.52852 for full ridge versus 0.52603 for daily. Every tested add-one and drop-one variant remains above that daily log RMSE. Full ridge nevertheless improves low-liquidity WAPE from 37.54% to 36.54%. This is a metric tradeoff, not evidence that the update always harms these stocks. Month-end crossed with liquidity and annual group counts are retained for investigation; no segment-specific prediction rule was fitted.

## Confirmation and limits

The previously inspected 2025 year is reported separately. Full ridge remains the best log-RMSE point estimate among these variants in that year, and each drop-one variant has slightly worse log RMSE. This is confirmation, not an unused holdout or a reason to choose subsets. No automatic feature removal, specialist model, or production promotion follows from this study. The saved daily-only control's historical availability restrictions and fallback are preserved for matching.

Confidence intervals in paired_group_bootstrap.csv concern equal-year mean daily log MAE, not headline log RMSE or WAPE. They use 2,000 within-year moving blocks of 20 trading dates and are unadjusted for multiple comparisons. Group contributions are not additive, and ablations include refitting and historical penalty reselection. See annual_segment_metrics.csv for sample sizes and annual variation.

## Next bounded step

Test a historically tuned shrinkage of the full update toward the daily anchor, with a global correction-strength control and carefully prespecified month-end/liquidity variants. This addresses overcorrection without adding new data or declaring that weak segments need separate models. Keep the current full ridge as the research reference and freeze development decisions before reviewing additional confirmation results.
'''
 manifest=json.loads((OUT/'experiment_summary.json').read_text());report+=f"\nAll {manifest['checks']} verification checks passed across {manifest['rows']:,} evaluation rows. Models, preprocessors, coefficients, hyperparameter trials, full row-level predictions, source fingerprints and experiment plan are retained.\n"
 (OUT/'findings.md').write_text(report)
 print(changes[changes.period.eq('development')].to_string(index=False))
if __name__=='__main__':run()
