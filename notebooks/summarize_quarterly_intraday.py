"""Compact reader artifacts for the quarterly fixed-specification audit."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/26_quarterly_intraday_audit'
os.environ.setdefault('MPLCONFIGDIR','/tmp/auction-quarterly-matplotlib')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
 q=pd.read_csv(OUT/'quarter_metrics.csv');base=q[q.candidate.eq('fixed_daily')];full=q[q.candidate.eq('quarterly_ridge')];annual=q[q.candidate.eq('annual_ridge')]
 d=full.merge(base,on=['fold','year','quarter'],suffixes=('','_daily'),validate='one_to_one').merge(annual[['fold','rmse_log','wape']],on='fold',suffixes=('','_annual'),validate='one_to_one')
 d['rmse_change_vs_daily']=d.rmse_log-d.rmse_log_daily;d['wape_pp_change_vs_daily']=100*(d.wape-d.wape_daily);d['rmse_change_vs_annual']=d.rmse_log-d.rmse_log_annual
 d.to_csv(OUT/'quarterly_paired_changes.csv',index=False)
 m=json.loads((OUT/'experiment_summary.json').read_text());s=pd.read_csv(OUT/'comparison.csv');i=pd.read_csv(OUT/'metric_intervals.csv')
 report='# Quarterly walk-forward audit of the full 15:30 ridge\n\nThe quarterly procedure improves log RMSE versus the daily anchor in all 16 quarters. It has a small aggregate point-estimate advantage over the original annual intraday ridge, but their development paired log-RMSE and WAPE intervals include zero. Do not declare quarterly refitting superior or automatically replace the annual reference.\n\n```text\n'+s.to_string(index=False)+'\n```\n\n## Direct 20-date block intervals\n\n```text\n'+i[i.block_dates.eq(20)].to_string(index=False)+'\n```\n\n'
 report+='Each of 16 quarters uses the preceding complete quarter to choose a penalty, earlier 2021-onward history for tuning fits, and all available earlier history for final refitting. No shrinkage calibration stage is used. The daily LightGBM anchor remains annually fitted. New samples, fitting windows and bounds accompany the quarterly schedule, so the annual comparison does not isolate frequency alone. Actual sample keys and all fitted objects are saved.\n\nDevelopment years are 2022-2024; previously inspected 2025 is separate confirmation. This is retrospective evaluation of a fixed procedure, not a fresh holdout. Block intervals recompute annual RMSE/WAPE/bias and then equal-year means. They condition on fitted forecasts and exclude refitting and prior research-selection uncertainty. Quarter counts are descriptive, not independent significance tests. No new period or other checkpoint is scored.\n\n'
 report+=f"All {m['checks']} checks passed across {m['rows']:,} rows. The report covering all intraday work is in reports/intraday_progress_2026_11_16.\n"
 (OUT/'README.md').write_text(report)
 fig,axes=plt.subplots(2,1,figsize=(10,6.3),layout='constrained',sharex=True);colors=['#286b67' if y<2025 else '#677789' for y in d.year]
 for ax,column,title in zip(axes,['rmse_change_vs_daily','wape_pp_change_vs_daily'],['Log RMSE change','WAPE change in percentage points']):
  ax.bar(d.fold,d[column],color=colors);ax.axhline(0,color='black',linewidth=.8);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[1].tick_params(axis='x',rotation=55);fig.suptitle('Quarterly 15:30 ridge versus the fixed daily anchor\nNegative means lower error; gray denotes previously inspected 2025 confirmation',fontsize=12);fig.savefig(OUT/'quarterly_stability.png',dpi=160);plt.close(fig)
 print('Quarterly summaries saved')
if __name__=='__main__':run()
