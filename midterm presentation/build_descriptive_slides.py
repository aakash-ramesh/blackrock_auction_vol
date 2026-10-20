"""Build compact slide charts from the completed descriptive-statistics tables."""
from pathlib import Path
import sys
try:
 import matplotlib
except ImportError:
 sys.path.insert(0,'/tmp/blackrock_midterm_deps');import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parent
T=ROOT/'descriptive_stats/tables';OUT=ROOT/'assets';OUT.mkdir(exist_ok=True)
NAVY='#102C43';TEAL='#007F82';GRAY='#9AA9B4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.titlesize':18,'axes.titleweight':'bold','axes.labelsize':16,'xtick.labelsize':15,'ytick.labelsize':15,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.edgecolor':'#637482','savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42})
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
 fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
hist=pd.read_csv(T/'auction_log_histogram.csv');size=pd.read_csv(T/'contemporaneous_size_deciles.csv');exact=pd.read_csv(T/'principal_exact_summary.csv');target=exact[(exact.dataset=='auction_daily')&(exact.variable=='auction_volume')].iloc[0]
fig,axes=plt.subplots(1,2,figsize=(13.6,4.6),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
ax=axes[0]
ax.bar(hist.bin_left_log10,hist.stock_days/1000,width=.098,align='edge',color=TEAL)
ax.axvline(np.log10(1+target['median']),color=NAVY,lw=2,ls='--')
ax.set(title='Auction-volume distribution',xlabel='Auction shares, logarithmic scale',ylabel='Stock-days (thousands)')
ax.set_xticks([2,3,4,5,6,7,8],['100','1k','10k','100k','1m','10m','100m'])
ax.text(.03,.94,f"Median: {target['median']/1000:.1f}k\nMean: {target['mean']/1000:.1f}k\n99th percentile: {target['p99']/1e6:.2f}m",transform=ax.transAxes,ha='left',va='top',fontsize=16)
ax=axes[1]
ax.plot(size.size_decile,size.median_auction,marker='o',color=TEAL,lw=2.5)
ax.set_yscale('log');ax.set(title='Median auction size by stock size',xlabel='Market-cap decile (1 = smallest)',ylabel='Auction shares, logarithmic scale')
ax.set_xticks([1,2,4,6,8,10]);ax.set_yticks([1000,10000,100000,1000000],['1k','10k','100k','1m'])
ax.set_ylim(3000,650000)
ax.annotate('4.4k',xy=(1,size.iloc[0].median_auction),xytext=(1.2,6500),fontsize=16,color=NAVY)
ax.annotate('354k',xy=(10,size.iloc[-1].median_auction),xytext=(7.8,410000),fontsize=16,color=NAVY)
save(fig,'descriptive_scale')
annual=pd.read_csv(T/'auction_by_year.csv');events=pd.read_csv(T/'calendar_target_descriptives.csv')
fig,axes=plt.subplots(1,2,figsize=(13.6,4.6),layout='constrained',gridspec_kw={'width_ratios':[1,1.1]})
ax=axes[0];ax.plot(annual.year,100*annual.median_auction_share,marker='o',color=TEAL,lw=2.5)
ax.set(title='Median auction share over time',xlabel='Year',ylabel='Auction share of daily volume (%)',ylim=(0,17));ax.set_xticks([2015,2017,2019,2021,2023,2025])
ax.annotate('4.8%',xy=(2015,100*annual.iloc[0].median_auction_share),xytext=(2015.2,3),fontsize=16)
ax.annotate('14.0%',xy=(2025,100*annual.iloc[-1].median_auction_share),xytext=(2022.8,15.3),fontsize=16)
ax=axes[1];columns=['monthly_options_expiry','quarterly_options_expiry','is_last_trading_day_month'];labels=['Monthly\nexpiry','Quarterly\nexpiry','Month-end']
other=[];event=[]
for c in columns:
 sub=events[events.event==c]
 flags=sub.flag_value.astype(str).str.lower().eq('true')
 other.append(100*sub.loc[~flags,'median_auction_share'].iloc[0]);event.append(100*sub.loc[flags,'median_auction_share'].iloc[0])
x=np.arange(3);ax.bar(x-.18,other,width=.36,color=GRAY,label='Other dates');ax.bar(x+.18,event,width=.36,color=TEAL,label='Event dates')
ax.set(title='Calendar contrasts',ylabel='Median auction share of daily volume (%)',xticks=x,xticklabels=labels,ylim=(0,40))
for j,(a,b) in enumerate(zip(other,event)):
 ax.text(j-.18,a+.8,f'{a:.1f}',ha='center',fontsize=14);ax.text(j+.18,b+.8,f'{b:.1f}',ha='center',fontsize=14)
ax.legend(frameon=False,loc='upper left',fontsize=14,ncol=2)
save(fig,'descriptive_time_calendar')
print('Built two slide figures from saved descriptive tables.')
