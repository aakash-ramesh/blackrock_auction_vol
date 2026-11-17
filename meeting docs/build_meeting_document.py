"""Build the November 5 research presentation document from saved evidence.

Run with the Codex bundled Python. Figures use ReportLab graphics and the
bundled Sharp renderer. No models are fitted or research outputs modified.
"""
from pathlib import Path
import json
import subprocess
import textwrap
import pandas as pd
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, Color
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
NODE = '/Users/aramesh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
SHARP = '/Users/aramesh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'
NAVY, TEAL, GRAY, PALE = '#1E3A50', '#157F83', '#637381', '#EEF3F6'
T = ROOT / 'outputs/experiments/21_ridge_target_definitions'
I = ROOT / 'outputs/experiments/20_intraday_noon'
development = pd.read_csv(T / 'development_metrics.csv')
common = development.query("selection == 'log_rmse' and bounds_policy == 'common_bounds'").set_index('target')
annual = pd.read_csv(T / 'annual_metrics.csv')
times = pd.read_csv(I / 'checkpoint_time_comparison.csv')

LABELS = {
 'raw_volume':'Raw shares', 'sqrt_volume':'Square root shares', 'log_volume':'Log shares',
 'lag1_log_growth':'Lag one log growth', 'median20_log_residual':'Median 20 log residual',
 'median60_log_residual':'Median 60 log residual', 'mean20_log_residual':'Mean 20 log residual',
 'mean60_log_residual':'Mean 60 log residual', 'adv20_log_residual':'ADV 20 log residual',
 'median20_ratio':'Auction to median 20 ratio', 'adv20_ratio':'Auction to ADV 20 ratio',
 'median20_share_residual':'Median 20 share residual'}

def txt(d, x, y, s, size=13, color=NAVY, bold=False, anchor='start'):
    d.add(String(x, y, s, fontName='Helvetica-Bold' if bold else 'Helvetica',
                 fontSize=size, fillColor=HexColor(color), textAnchor=anchor))

def wrapped(d, x, y, s, width=34, size=13, color=NAVY, leading=18, bold=False):
    for j, line in enumerate(textwrap.wrap(s, width)):
        txt(d,x,y-leading*j,line,size,color,bold)

def figure(d, stem):
    svg = FIG / f'{stem}.svg'
    png = FIG / f'{stem}.png'
    renderSVG.drawToFile(d, str(svg))
    script = f"const sharp=require({json.dumps(SHARP)}); sharp(process.argv[1],{{density:210}}).png().toFile(process.argv[2]);"
    subprocess.run([NODE, '-e', script, str(svg), str(png)], check=True)
    return png

def flow_figure():
    d=Drawing(920,340)
    txt(d,0,316,'The panel preserves rows and separates label eligibility',22,bold=True)
    blocks=[(0,205,285,'CRSP stock day spine','3,594,404 rows','2015 to 2025'),
            (317,205,285,'Left join auction records','3,499,885 observed labels','94,519 labels remain missing'),
            (634,205,285,'Apply anomaly exclusion','3,499,401 eligible labels','484 observed labels excluded')]
    for x,y,w,a,b,c in blocks:
        d.add(Rect(x,y,w,83,fillColor=HexColor(PALE),strokeColor=HexColor('#C8D4DC'),rx=4))
        txt(d,x+13,y+59,a,15,bold=True);txt(d,x+13,y+35,b,16,TEAL,bold=True);txt(d,x+13,y+14,c,12)
    for x in [289,606]:
        d.add(Line(x,246,x+24,246,strokeColor=HexColor(TEAL),strokeWidth=2))
        d.add(Polygon([x+24,246,x+16,251,x+16,241],fillColor=HexColor(TEAL),strokeColor=None))
    txt(d,0,170,'The stored panel retains all 3,594,404 rows',18,bold=True)
    wrapped(d,0,143,'Auction histories, lagged stock data, lagged market and sector data, and known calendar fields produce 99 approved daily source features.',width=115,size=14,leading=20)
    d.add(Line(0,92,920,92,strokeColor=HexColor('#D9E1E6')))
    txt(d,0,67,'99 approved sources in the panel',17,bold=True)
    txt(d,0,42,'58 selected for the compact daily model',16,TEAL,bold=True)
    txt(d,0,17,'19 auction history + 23 calendar + 16 liquidity',13)
    txt(d,530,67,'Model preprocessing happens later',17,bold=True)
    txt(d,530,42,'Fit only on historical training rows',16,TEAL,bold=True)
    txt(d,530,17,'Panel missing values are preserved until fitting',13)
    return figure(d,'01_panel_pipeline')

def missing_figure():
    d=Drawing(920,255)
    txt(d,0,233,'Missing auction labels are concentrated in 2021 to 2023',21,bold=True)
    coverage=pd.read_csv(ROOT/'outputs/eda/01_data_quality/annual_target_coverage.csv')
    y0=43; h=146; maxv=8
    for tick in [0,2,4,6,8]:
        y=y0+h*tick/maxv
        d.add(Line(45,y,903,y,strokeColor=HexColor('#E1E7EB')))
        txt(d,35,y-4,f'{tick}%',11,GRAY,anchor='end')
    for j,r in enumerate(coverage.itertuples()):
        x=65+j*77; val=100*(1-r.target_coverage)
        d.add(Rect(x,y0,43,h*val/maxv,fillColor=HexColor(TEAL if r.year in [2021,2022,2023] else '#A4B6C3'),strokeColor=None))
        txt(d,x+21.5,y0+h*val/maxv+7,f'{val:.2f}%',11,anchor='middle')
        txt(d,x+21.5,23,str(r.year),11,anchor='middle')
    return figure(d,'02_missing_labels_by_year')

def targets_figure():
    d=Drawing(920,410)
    txt(d,0,386,'Recent auction log residuals lead the ridge comparison',22,bold=True)
    txt(d,0,361,'Development log RMSE after converting every forecast back to shares',14,GRAY)
    ranked=common.sort_values('rmse_log')
    x0=285; plot=565; ytop=330; step=24
    for tick in [0,.5,1,1.5,2]:
        x=x0+plot*tick/2
        d.add(Line(x,41,x,345,strokeColor=HexColor('#E1E7EB')))
        txt(d,x,20,f'{tick:.1f}',11,GRAY,anchor='middle')
    for j,(key,r) in enumerate(ranked.iterrows()):
        y=ytop-j*step
        txt(d,x0-12,y+1,LABELS[key],12,anchor='end',bold=(key=='median20_log_residual'))
        d.add(Rect(x0,y-3,plot*r.rmse_log/2,15,fillColor=HexColor(TEAL if key=='median20_log_residual' else '#829EAE'),strokeColor=None))
        txt(d,x0+plot*r.rmse_log/2+7,y-1,f'{r.rmse_log:.4f}',12,bold=(key=='median20_log_residual'))
    return figure(d,'03_ridge_target_comparison')

def intraday_figure():
    d=Drawing(920,350)
    txt(d,0,326,'Later information improves the forecast more',22,bold=True)
    txt(d,0,301,'Common regular session universe   Development averages for 2022 to 2024',14,GRAY)
    series=[('Daily anchor','fixed_daily',None),('Noon ridge','ridge_update','12:00'),('15:30 ridge','ridge_update','15:30'),('Noon LightGBM','lgbm_intraday_update','12:00'),('15:30 LightGBM','lgbm_intraday_update','15:30')]
    for x0,width,metric,label,lo,hi,fmt in [(202,285,'development_rmse_log','Log RMSE',.45,.48,'.4f'),(643,220,'development_wape','WAPE',.34,.36,'.2%')]:
        txt(d,x0,270,label,17,bold=True)
        for tick in ([.45,.46,.47,.48] if metric.endswith('log') else [.34,.35,.36]):
            x=x0+width*(tick-lo)/(hi-lo)
            d.add(Line(x,52,x,250,strokeColor=HexColor('#DFE6EB')))
            txt(d,x,31,format(tick,fmt if metric.endswith('wape') else '.2f'),11,GRAY,anchor='middle')
        for j,(name,key,cp) in enumerate(series):
            r=times[(times.candidate==key)&(times.checkpoint==(cp or '12:00'))].iloc[0]
            y=232-j*41
            if metric.endswith('log'): txt(d,184,y-4,name,13,anchor='end',bold=(name=='15:30 ridge'))
            v=float(r[metric]); x=x0+width*(v-lo)/(hi-lo)
            color=TEAL if name=='15:30 ridge' else ('#9DADB8' if key=='fixed_daily' else NAVY)
            d.add(Line(x0,y,x,y,strokeColor=HexColor(color),strokeWidth=3))
            d.add(Rect(x-4,y-4,8,8,fillColor=HexColor(color),strokeColor=None))
            txt(d,x+10,y-4,format(v,fmt),12,color,bold=True)
    txt(d,0,4,'Both horizontal axes are truncated for readability. Lower values are better.',12,GRAY)
    return figure(d,'04_intraday_accuracy')

def architecture_figure():
    d=Drawing(920,225)
    boxes=[(0,100,255,'Daily anchor','58 historical daily sources','Saved out of time forecast'),(330,100,260,'Checkpoint update','10 current session inputs','4 historical controls'),(665,100,255,'Closing auction forecast','Predict log correction','Reconstruct auction shares')]
    for x,y,w,a,b,c in boxes:
        d.add(Rect(x,y,w,100,fillColor=HexColor(PALE),strokeColor=HexColor('#CBD6DD'),rx=5))
        txt(d,x+14,y+72,a,16,bold=True);txt(d,x+14,y+45,b,14);txt(d,x+14,y+21,c,13)
    for x in [263,598]:
        d.add(Line(x,151,x+58,151,strokeColor=HexColor(TEAL),strokeWidth=2))
        d.add(Polygon([x+58,151,x+49,156,x+49,146],fillColor=HexColor(TEAL),strokeColor=None))
    txt(d,0,66,'Same architecture at noon and 15:30 ET',18,TEAL,bold=True)
    txt(d,0,38,'Missing or invalid checkpoint: keep the saved daily forecast exactly.',15)
    txt(d,0,12,'Full update: 73 source inputs   Daily only LightGBM control: 63 source inputs',13,GRAY)
    return figure(d,'05_intraday_architecture')

figures=[flow_figure(),missing_figure(),targets_figure(),intraday_figure(),architecture_figure()]

doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65)
sec.left_margin=Inches(.7);sec.right_margin=Inches(.7)
sec.header_distance=Inches(.2);sec.footer_distance=Inches(.28)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    st=doc.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
normal=doc.styles['Normal'];normal.font.size=Pt(11)
normal.paragraph_format.space_after=Pt(7);normal.paragraph_format.line_spacing=1.06
for name,size in [('Title',27),('Heading 1',21),('Heading 2',14),('Heading 3',12)]:
    st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True
    st.paragraph_format.space_before=Pt(10 if name!='Title' else 0)
    st.paragraph_format.space_after=Pt(7)
doc.styles['Caption'].font.size=Pt(9);doc.styles['Caption'].font.italic=False
doc.styles['Caption'].font.bold=False
doc.styles['Subtitle'].font.italic=False
doc.styles['Caption'].paragraph_format.space_after=Pt(9)
# Remove decorative borders inherited from the bundled Word template.
for st in doc.styles:
    for node in list(st.element.iter(qn('w:pBdr'))):
        node.getparent().remove(node)
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=foot.add_run('November 5 2026  |  ');r.font.size=Pt(9);r.font.color.rgb=RGBColor.from_string('637381')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)

def p(text='',bold_lead=None,style=None):
    q=doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        q.add_run(bold_lead).bold=True;q.add_run(text[len(bold_lead):])
    else:q.add_run(text)
    return q

def h(text,level=2): return doc.add_heading(text,level)
def page(kicker,title):
    doc.add_page_break()
    q=p(kicker);q.paragraph_format.space_after=Pt(4)
    for r in q.runs:r.font.size=Pt(10);r.font.color.rgb=RGBColor.from_string('637381');r.bold=True
    h(title,1)
def pic(path,caption,width=7.1):
    q=doc.add_paragraph();q.paragraph_format.space_after=Pt(3)
    run=q.add_run();inline=run.add_picture(str(path),width=Inches(width))
    inline._inline.docPr.set('descr',caption)
    p(caption,style='Caption')
def table(headers,rows,widths,font=10):
    t=doc.add_table(rows=1, cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for j,title in enumerate(headers):t.rows[0].cells[j].text=str(title)
    for row in rows:
        for j,value in enumerate(row):
            if j==0:cells=t.add_row().cells
            cells[j].text=str(value)
    for i,row in enumerate(t.rows):
        trPr=row._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit');trPr.append(cant)
        if i==0:
            repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
        for j,cell in enumerate(row.cells):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr()
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'1E3A50' if i==0 else ('F2F5F7' if i%2==0 else 'FFFFFF'));pr.append(shade)
            borders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
            pr.append(borders)
            mar=OxmlElement('w:tcMar')
            for side,v in [('top','70'),('bottom','70'),('left','85'),('right','85')]:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),v);el.set(qn('w:type'),'dxa');mar.append(el)
            pr.append(mar)
            for q in cell.paragraphs:
                q.paragraph_format.space_after=Pt(0);q.paragraph_format.space_before=Pt(0);q.paragraph_format.line_spacing=1.04
                q.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or len(str(cell.text))>35 else WD_ALIGN_PARAGRAPH.CENTER
                for r in q.runs:
                    r.font.size=Pt(font);r.bold=i==0;r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
    p('').paragraph_format.space_after=Pt(0)
    return t

# Page 1
doc.add_heading('Closing auction volume forecasting',0)
p('Data preparation target definitions and intraday updates',style='Subtitle')
p('November 5 2026  |  15 minute research presentation')
h('Executive overview')
p('We have a reproducible daily panel and working noon and 15:30 forecast updates for official NYSE closing auction shares. The evidence supports retaining the recent median log-residual target and taking the 15:30 ridge update forward for robustness testing.')
h('The three messages to present')
p('Data preparation. We preserve missing outcomes, exclude a documented source anomaly, and construct daily features only from prior information or known calendar dates. The panel contains 3,594,404 stock-days and 3,499,401 eligible targets. Missing predictors are handled inside historical fitting folds. [D1-D4]',bold_lead='Data preparation.')
p('Target definitions. Notebook 21 compares 12 ridge targets on identical rows and 58 source features. The 20-row median log residual has the lowest development log RMSE at 0.48811. The 60-row mean log residual wins share-based objectives by very small margins; these point rankings do not establish a material improvement. [T1-T3]',bold_lead='Target definitions.')
p('Intraday models. On matched regular-session rows, the daily anchor has log RMSE 0.46949; noon ridge reaches 0.46459 and 15:30 ridge reaches 0.45995. The 15:30 ridge WAPE is 34.59% versus 35.47% for the anchor. Noon captures about 51% of the later ridge log-RMSE gain. [I1-I4]',bold_lead='Intraday models.')
h('Suggested presentation route')
table(['Time','Topic','Pages'],[
 ('0 to 1 min','Opening and main conclusions','1'),
 ('1 to 5 min','Data preparation and treatment rules','2 to 3'),
 ('5 to 9 min','Target definitions and performance','5 to 7'),
 ('9 to 14 min','Noon and 15:30 models','8 to 10'),
 ('14 to 15 min','Recommendation and next decision','10')],[1.1,4.9,1.1],10.5)
p('Use the executive summaries and figures as the spoken route. Page 4 and the detailed tables support questions; page 11 lists the source evidence.',style='Caption')
p('Reading the scores. Development results are equal-year averages for 2022 to 2024. The previously inspected 2025 year is confirmation evidence. Notebook 21 uses the full eligible daily cohort; intraday studies exclude early closes. Compare models within the stated cohort.',bold_lead='Reading the scores.')

# Page 2
page('PART 1  |  EXECUTIVE SUMMARY','From source data to the daily panel')
p('We retain the CRSP stock-day spine, attach auction labels and historical context, and mark which rows can be used for supervision. Missing auction labels remain missing. Feature imputation happens only after historical training rows have been selected. [D1-D4]')
pic(figures[0],'Figure 1. Row accounting and feature flow. The 94,519 missing labels and 484 observed anomaly labels are disjoint exclusions from supervised use. [D2, D3]')
h('Source universe and identifiers')
p('The source universe is NYSE-primary, U.S.-incorporated ordinary common equity, across all market-cap sizes. PERMNO is the stable security key. CRSP supplies 4,927,078 stock-days over 2011 to 2025; the extra pre-2015 history supports daily lags before the first auction-label year. The final 2015 to 2025 panel spans 2,149 securities and 2,766 dates. [D1-D3]')
h('What defines the auction label')
p("The upstream TAQ extraction identifies the official NYSE auction with exchange N, correction code 00, and sale condition containing 6. Condition M is excluded because repeated or delayed reports make it unsafe to aggregate as the official auction. This extraction policy comes from the project handoff. [D1]")
h('How intraday trade aggregates are cleaned')
p('Checkpoint aggregates consolidate all venues, keep correction code 00 and regular-session trades, and exclude sale conditions O, Q, 6 and M. They contain cumulative volume, dollars, trade count, VWAP and recent activity. A usable checkpoint requires a source-found flag, positive cumulative volume and a checkpoint before the session close. Post-close checkpoint fields are missing by design. [D1, I1, I2]')
p('Presenter cue. The panel keeps all stock-days; eligibility determines which labels are scored. A missing auction record is not evidence of a zero-share auction.',bold_lead='Presenter cue.')

# Page 3
page('PART 1  |  CLEANING AND OMISSION','What is retained omitted or left missing')
table(['Issue','Implemented treatment'],[
 ('Security linking','Join TAQ on date and symbol root and suffix, then map to PERMNO. Ambiguous links remain missing.'),
 ('Missing auction label','94,519 panel rows remain unlabeled and are excluded from supervised fitting and evaluation. No zero filling.'),
 ('Source anomaly','Exclude August 12 2019 from supervised use. This removes 484 observed labels in the panel.'),
 ('Join multiplicity','Source key audits find zero duplicates. Overlapping industry records are resolved by latest applicable start and end dates.'),
 ('Unsafe daily inputs','Same-day final volume, closing returns, auction price and outcome fields are diagnostics rather than approved daily predictors.'),
 ('Ratios and spreads','Zero denominators become missing. Quoted spread is defined only when ask is at least bid and their sum is positive.'),
 ('Short histories','Retain otherwise eligible cold starts. Rolling summaries use available prior values; insufficient inputs can remain missing.')],[1.65,5.45],10.5)
pic(figures[1],'Figure 2. Missing-label rate on the CRSP daily spine before the anomaly exclusion. Coverage reaches its low of 93.02% in 2022. [D2]',7.1)
p('The upstream handoff records 99 unresolved links over the full CRSP bridge and 56 stock-days with ambiguous TAQ keys in 2024. These are missing-data treatments, not additional counts to subtract from the panel flow. Missing coverage in 2021 to 2023 is broad and persistent; the analysis does not impose an additional whole-year deletion. [D1, D2]')
p('Intraday omission is separate. Early-close sessions are excluded from both noon and 15:30 comparisons to keep their evaluation rows identical. Invalid regular-session checkpoints retain the daily forecast rather than dropping the row. [I1, I2]',bold_lead='Intraday omission is separate.')

# Page 4
page('PART 1  |  IMPUTATION AND VERIFICATION','How missing predictors enter the model')
p('The saved panel retains missing covariates. Each chronological fold fits its transformations, clipping limits, imputation values, scaling and encoding on historical training rows, then applies those saved parameters to validation or evaluation rows. The final refit relearns them using only dates before its forecast year. [D4]')
table(['Input','Fitted treatment'],[
 ('Continuous or ordered','For nonnegative inputs, apply log1p when the 99th percentile of absolute values divided by absolute median is at least 20. Clip at fitted 0.5th and 99.5th percentiles; fill missing with the clipped median; center by that median and divide by fitted standard deviation.'),
 ('Missing indicators','Add a flag when missingness occurs in fitting data. Reuse that flag structure; a newly missing evaluation value does not create a new column.'),
 ('Binary','Fill missing with zero and add a fitted missing flag when needed.'),
 ('Categorical','Map missing to MISSING; encode fitted levels with a reference category. Unseen levels map to OTHER.'),
 ('All missing or constant','All-missing continuous inputs use zero and unit scale. Near-zero fitted standard deviations use unit scale.'),
 ('Target reference','Missing or nonpositive auction-history reference uses the fitting median auction volume. ADV references use the median positive fitting ADV.')],[1.55,5.55],10.5)
h('How we keep histories behind the forecast date')
p('Stock rolling windows end one row before the forecast. A 20-row median uses nonmissing values inside the preceding 20 stock rows; it does not search for the last 20 nonmissing auctions. Market and sector inputs are lagged. Cross-sectional ranks use lagged stock inputs. Calendar variables are known in advance. [D3]')
h('Feature selection and evidence')
p('The panel has 99 approved source features across eight families. The compact model keeps 58: 19 auction-history, 23 calendar and 16 liquidity features. This feature reduction does not remove stock-day rows. One-hot encoding and missing indicators can create more than 58 matrix columns. [D3, T1]')
p('The reconstruction passed 23 recorded checks: annual eligible counts and all 99 feature null counts reconciled; keys were unique; eligible numeric inputs contained no nonfinite nonmissing values. Four history calculations were independently checked on 28,333 rows from 16 stocks. [D3]')
p('Data-treatment limitation. The anomaly is excluded as a supervised label, but its source values remain in later auction-history windows. This preserved behavior should be tested explicitly before changing the treatment. The legacy panel sample_split field also does not replace the annual folds used in these experiments. [D3]',bold_lead='Data-treatment limitation.')

# Page 5
page('PART 2  |  EXECUTIVE SUMMARY  |  NOTEBOOK 21','Which target definition should we keep')
p('We recommend retaining the 20-row median log-residual target for the log-error objective. Long-window auction log residuals are close competitors. Direct raw shares, square roots and unlogged ratios perform substantially worse under the common log-RMSE tuning rule. [T1, T2]')
pic(figures[2],'Figure 3. Twelve ridge targets with penalties selected by historical log RMSE and common forecast bounds. Bars show equal-year 2022 to 2024 means; lower is better. [T2]')
h('What the comparison controls')
p('All targets use the same 58 source features, shared preprocessing fitted on the original 99-source contract, identical historical samples and identical evaluation keys. Every prediction is reconstructed in auction shares before any reported metric is calculated. This is a ridge-only experiment; Notebook 21 does not compare tree model target definitions. [T1, T3]')
p('For each forecast year, tuning uses five historical years before the prior validation year; the final fit uses the five years immediately before evaluation. The sampled tuning, validation and refit caps are 250,000, 120,000 and 350,000 rows. Evaluation covers all 1,271,378 eligible stock-days in 2022 to 2025. [T1, T3]')
p('For example, the 2022 fold tunes on 2016 to 2020, validates on 2021 and refits on 2017 to 2021. A log residual compresses large share differences while its historical reference carries the stock\'s usual auction scale. Raw-share squared error places much more weight on large auctions; ratio errors also depend strongly on the size of their historical denominators. These properties help interpret the rankings, without proving the cause of every performance difference. [T3]')
p('Presenter cue. Changing the target changes what the linear model learns, while the business outcome remains the same official auction volume. The strongest targets describe departures from a stock\'s recent auction scale.',bold_lead='Presenter cue.')

# Page 6
page('PART 2  |  DEFINITIONS AND COMMON PERFORMANCE','Twelve ways to model the same auction')
p('Notation: A is current auction shares; L is the previous stock row auction; M20 and M60 are prior 20-row and 60-row auction medians; H20 and H60 are prior auction means; D20 is prior 20-row average daily volume. All references exclude the forecast day. [T1, T3]')
defs=[('median20_log_residual','log1p(A) minus log1p(M20)'),('median60_log_residual','log1p(A) minus log1p(M60)'),('mean20_log_residual','log1p(A) minus log1p(H20)'),('mean60_log_residual','log1p(A) minus log1p(H60)'),('lag1_log_growth','log1p(A) minus log1p(L)'),('log_volume','log1p(A)'),('adv20_log_residual','log1p(A) minus log1p(D20)'),('median20_ratio','A divided by M20'),('adv20_ratio','A divided by D20'),('sqrt_volume','Square root of A'),('median20_share_residual','A minus M20'),('raw_volume','A')]
table(['Target','Definition','Log RMSE','WAPE','Share R2'],[
 (LABELS[k],v,f'{common.loc[k,"rmse_log"]:.5f}',f'{common.loc[k,"wape"]:.2%}',f'{common.loc[k,"r2_raw"]:.3f}') for k,v in defs],[1.96,2.55,.92,.85,.82],9.5)
p('Table 1. Common historical log-RMSE penalty selection and common bounds. All values are development equal-year means; predictive share R2 is an annual out-of-time score averaged across years. [T2]',style='Caption')
h('How predictions return to shares')
p('For a log residual, add the predicted residual to log1p of the reference, then apply expm1. Direct log levels use expm1 alone. Ratios multiply by their reference. Share residuals add the reference. Square-root predictions are floored at zero before squaring. All reconstructions are kept nonnegative, then the common bound policy is applied. [T3]')
p('Example. If M20 is 100,000 shares and the predicted log residual is zero, the reconstructed forecast is 100,000 shares. A positive residual means a larger auction relative to that recent scale. Lag-one log growth is a log1p difference, rather than ordinary percentage growth.',bold_lead='Example.')
p('A ratio-minus-one target is omitted because an unpenalized intercept makes it equivalent to the ratio formulation. Realized same-day daily volume is never a target denominator. Raw and ratio targets are fitted without winsorizing the training target. [T1, T3]')

# Page 7
page('PART 2  |  OBJECTIVES AND ROBUSTNESS','The winner depends on the scoring objective')
p('Notebook 21 separately selects each target\'s ridge penalty by historical log RMSE, WAPE or share RMSE. The 14-value penalty grid spans 0.0000001 to 1,000,000, with an unpenalized intercept. These are three declared tuning strategies; evaluation-year labels do not select penalties. [T1, T3]')
table(['Historical tuning objective','Winning target','Winner score','Current target score'],[
 ('Log RMSE','Median 20 log residual','0.488111','0.488111'),
 ('WAPE','Mean 60 log residual','36.7110%','36.7310%'),
 ('Share RMSE','Mean 60 log residual','656,902','657,167')],[1.6,2.45,1.5,1.55],10.5)
p('Table 2. Development scores under common bounds. Current-target penalties are tuned for the same objective as the winner. The WAPE advantage is 0.020 percentage points, or 0.054% relative; the share-RMSE advantage is 265 shares, or 0.040% relative. [T2]',style='Caption')
h('Do the leading log targets behave similarly by year')
leaders=['median20_log_residual','median60_log_residual','mean60_log_residual']
rows=[]
for key in leaders:
    sub=annual.query("target == @key and selection == 'log_rmse' and bounds_policy == 'common_bounds'").set_index('year')
    rows.append((LABELS[key],*[f'{sub.loc[y,"rmse_log"]:.4f}' for y in [2022,2023,2024,2025]]))
table(['Target','2022','2023','2024','2025'],rows,[3.1,1,1,1,1],10.5)
p('Table 3. Annual log RMSE under the common log-RMSE tuning rule. The 2025 column is previously inspected confirmation evidence. [T2]',style='Caption')
h('Forecast bounds affect the scores')
p('The main comparison clips shares at the historical fitting sample\'s 0.1th and 99.9th percentiles. A second policy keeps only nonnegativity, using the same fitted models and selected penalties. For the current target, development log RMSE changes from 0.48811 to 0.48763 under its log-RMSE selection. Under WAPE selection, current-target WAPE changes from 36.7310% to 36.4900%; mean-60 remains slightly better at 36.4371%. This sensitivity is not separately optimized. [T2, T3]')
h('Interpretation and decision')
p('Under common log-RMSE tuning, the current target still underpredicts total shares by 14.50% on the equal-year development measure. Log inversion has no smearing correction. Reducing log error does not guarantee an unbiased total in a skewed share distribution. The tiny share-objective gains are point estimates without paired uncertainty tests; they do not justify replacing the current target on their own. [T1, T2]')
p('Notebook 21 is a new controlled ridge refit using fresh deterministic historical samples. Its 0.48811 reference is not the frozen ridge score from earlier experiments. All 317 recorded verification checks passed. [T1]')

# Page 8
page('PART 3  |  EXECUTIVE SUMMARY  |  NOTEBOOKS 19 AND 20','The later intraday update improves accuracy more')
p('We built a second forecasting layer that learns a correction around the saved daily LightGBM forecast. The 15:30 ridge update has the best aggregate point estimates among the tested update procedures. Noon ridge provides a smaller improvement several hours earlier. [I1-I4]')
pic(figures[3],'Figure 4. Daily anchor versus learned updates on identical regular-session evaluation rows. The daily anchor is the saved squared-error LightGBM daily model. [I3]')
table(['Procedure','Log RMSE','WAPE','Bias'],[
 ('Daily anchor','0.46949','35.47%','-14.47%'),
 ('Noon ridge','0.46459','34.85%','-13.02%'),
 ('15:30 ridge','0.45995','34.59%','-11.80%')],[3.05,1.35,1.35,1.35],11)
p('Table 4. Development equal-year averages for 2022 to 2024 on the common regular-session cohort. [I3]',style='Caption')
p('The 15:30 ridge gain is 0.00954 log-RMSE units, or 2.03% relative, and 0.876 WAPE percentage points. Noon ridge retains about 51% of that log-RMSE gain. Both continue to underpredict total shares. On 2025 confirmation, ridge log RMSE is 0.41677 at noon and 0.40916 at 15:30 versus 0.42901 for the anchor. [I3]')
p('Recommendation. Keep the daily anchor and advance 15:30 ridge to the next robustness study. Treat noon ridge as an optional earlier forecast. The 15:30 LightGBM update passes its incremental-information gate; noon LightGBM does not pass every criterion. [I1, I2, I4]',bold_lead='Recommendation.')

# Page 9
page('PART 3  |  MODEL DESIGN AND CHRONOLOGY','How the intraday correction is built')
pic(figures[4],'Figure 5. Both checkpoints estimate a log1p auction correction around the same saved daily anchor. Source-input counts precede encoding and missing flags. [I1, I2]')
p('The learned target is log1p of actual auction shares minus log1p of the fixed daily forecast. At prediction time we add the estimated correction to the daily log forecast, apply historical log-volume bounds and reconstruct shares. The daily anchor itself is preserved. [I1, I2]')
h('What is available at the checkpoint')
p('Ten current-session inputs include cumulative shares, dollars and trade count; recent 30-minute shares and their cumulative-volume fraction; VWAP; volume relative to prior daily ADV; volume and trade-count surprises versus prior checkpoint history; and VWAP return from the previous close. Four historical controls add average checkpoint shares and trade count, the historical auction-to-checkpoint ratio and prior price. The checkpoint averages and ratio use the preceding 20 stock rows. [I1, I2]')
p('The daily-only LightGBM control receives the 58 daily features, saved daily log forecast and four historical controls, for 63 sources. The full update adds ten checkpoint inputs, for 73. No candidate sees final daily volume, closing returns, later checkpoints, the current auction outcome or auction imbalance. [I1, I2]')
h('Why the stacking history is credible')
table(['Evaluation fold','Tuning and validation','Final update fit'],[
 ('2022','Fit January to September 2021; validate October to December 2021','Available 2021 out-of-time rows'),
 ('2023 to 2025','Fit earlier years from 2021; validate the immediately prior full year','All available rows from 2021 through the prior year')],[1.1,3.55,2.45],10.5)
p('Every historical anchor prediction is genuinely out of time. Fitting the correction on in-sample daily predictions would make the historical anchor unrealistically accurate. Update histories expand from 2021, while the base daily model uses rolling five-year fits. [I1, I2]')
p('Ridge searches seven penalties from 0.0000001 to 0.1. LightGBM searches 15, 31 and 63 leaves, with up to 2,000 trees and historical early stopping. Historical tune, validation and refit samples are capped at 250,000, 120,000 and 350,000 available-checkpoint rows. All fitted preprocessing and limits predate evaluation. [I1, I2]')

# Page 10
page('PART 3  |  PERFORMANCE AND DECISION','What the full intraday comparison establishes')
method_labels={'fixed_daily':'Daily anchor','checkpoint_ratio_rule':'Ratio rule','ridge_update':'Ridge update','lgbm_daily_control':'Daily only LightGBM','lgbm_intraday_update':'LightGBM update'}
rs=[]
for cp in ['12:00','15:30']:
    for method in ['fixed_daily','checkpoint_ratio_rule','ridge_update','lgbm_daily_control','lgbm_intraday_update']:
        if cp=='15:30' and method=='fixed_daily':continue
        r=times[(times.checkpoint==cp)&(times.candidate==method)].iloc[0]
        rs.append(('Both' if method=='fixed_daily' else cp,method_labels[method],f'{r.development_rmse_log:.5f}',f'{r.development_wape:.2%}',f'{r.development_bias:+.2%}',f'{r.confirmation_rmse_log:.5f}',f'{r.confirmation_wape:.2%}'))
table(['Time ET','Method','Dev log RMSE','Dev WAPE','Dev bias','2025 log RMSE','2025 WAPE'],rs,[.55,1.8,.95,.92,.92,1,.96],9)
p('Table 5. Equal-year 2022 to 2024 development metrics and 2025 confirmation. The simple rule multiplies current cumulative volume by a prior-only 20-row median auction-to-checkpoint ratio. [I3]',style='Caption')
h('Information gate and attribution')
p('The prespecified LightGBM gate requires at least 0.002 mean log-RMSE improvement versus both the daily anchor and daily-only learned control; no annual deterioration above 0.002; no mean WAPE worsening; and a paired log-MAE interval with its upper limit below zero against both. The 15:30 update passes. Noon improves log RMSE by only 0.000392 versus the anchor and worsens 2022 by 0.003503, so it fails the full gate. [I4]')
p('Ridge has the best point estimates, but no separate daily-only ridge control was fitted. Its gain over the anchor therefore measures the complete update procedure; it does not cleanly isolate current checkpoint information. The LightGBM control comparison supplies the direct current-information test. [I1, I2]')
h('Coverage and uncertainty')
p('Both studies score the same 1,260,168 regular-session stock-days in 2022 to 2025, excluding 11,210 eligible early-close rows. Missing checkpoints trigger exact daily fallback: 718 noon rows and 271 at 15:30, for coverage of 99.94% and 99.98%. Each study passed 124 recorded checks. Additional noon alignment and source-availability audits also passed. [I1, I2, I5]')
p('Paired time comparisons favor 15:30. For ridge, the development noon-minus-15:30 log-MAE difference is 0.00342 with a 95% interval of 0.00290 to 0.00393. This interval concerns log MAE rather than log RMSE. Whole-date resampling preserves same-date dependence, but not serial dependence or model-selection uncertainty. [I4]')
p('Closing message. We have established a usable research pipeline and a stronger late-day forecast candidate. The next decision is whether 15:30 ridge remains reliable on genuinely new dates, across liquidity and event segments, and under operational checkpoint availability. [I1, I2]',bold_lead='Closing message.')

# Page 11
page('DISCUSSION REFERENCE','Limits next work and source evidence')
h('What should happen next')
p('Next, test 15:30 ridge with a matched daily-only ridge control, event and year stability, tail errors, bound sensitivity, and checkpoint delays or outages. Evaluate on genuinely new dates before freezing a production choice. These are proposed tests. The completed studies establish forecast accuracy; an execution-cost study is needed to establish economic value.')
p('The minimum-25-tree refit rule increases some daily-only LightGBM controls above their early-stopped selections: 5 and 4 selected trees at 15:30, and 2 and 2 at noon, in 2023 and 2025. This can affect control comparisons. Checkpoint cutoffs rely on upstream aggregates; underlying trade timestamps are unavailable for an independent cutoff audit. [I1, I2]')
h('Metric definitions')
p('Log RMSE is root mean squared error between log1p actual and predicted shares. WAPE is total absolute share error divided by actual shares. Share RMSE emphasizes large absolute misses. Predictive share R2 compares squared errors with deviations around that year\'s actual mean and can be negative. Bias is predicted total divided by actual total minus one. Development averages annual scores with equal year weight. [T1, T3]')
h('Source evidence')
sources=[
 ('D1','blackrock_data/BlackRock Capstone Handoff — Predicting NYSE Closing Auction Volume.docx','Upstream universe, TAQ filters, linking and early closes'),
 ('D2','outputs/eda/01_data_quality/','dataset_inventory.csv, annual_target_coverage.csv and integrity_checks.csv; notebook 01'),
 ('D3','notebooks/05_feature_engineering_enriched_daily_panel.ipynb','Panel SQL and feature catalog; reconstruction README and checks in outputs/reproduction/01_daily_panel/'),
 ('D4','notebooks/modeling_utils.py','fit_preprocessor and apply_preprocessor; target references in compare_ridge_targets.py'),
 ('T1','outputs/experiments/21_ridge_target_definitions/README.md','Notebook 21 design and limits; experiment_plan.json; 317 verification checks'),
 ('T2','outputs/experiments/21_ridge_target_definitions/','development_metrics.csv, annual_metrics.csv and metric_winners.csv'),
 ('T3','notebooks/compare_ridge_targets.py','Target formulas, inverse transforms, fold dates, bounds and reference fallbacks'),
 ('I1','outputs/experiments/19_intraday_1530/README.md','Notebook 19 design and results; build_intraday_1530_update.py'),
 ('I2','outputs/experiments/20_intraday_noon/README.md','Notebook 20 design and results; build_intraday_noon_update.py'),
 ('I3','outputs/experiments/20_intraday_noon/checkpoint_time_comparison.csv','Matched noon and 15:30 development and confirmation metrics'),
 ('I4','outputs/experiments/20_intraday_noon/','paired_checkpoint_time_bootstrap.csv; intraday_information_gate.csv in experiments 19 and 20'),
 ('I5','outputs/experiments/20_intraday_noon/','coverage_by_year.csv, timing_alignment_checks.csv and full_checkpoint_availability_audit.csv; coverage in experiment 19')]
for key,path,description in sources:
    q=p(f'[{key}] {path}\n{description}')
    q.paragraph_format.space_after=Pt(4)
    for r in q.runs:r.font.size=Pt(8.5)
p('Paths are relative to the blackrock_auction_vol repository. Figures are generated from the cited saved tables. Panel row accounting was also checked directly against Notebook 21\'s daily_panel.parquet on November 5 2026.',style='Caption')

doc.core_properties.title='Closing auction volume forecasting'
doc.core_properties.subject='November 5 2026 presentation on data treatment target definitions and intraday updates'
doc.core_properties.author=''
doc.core_properties.keywords='NYSE closing auction data preparation ridge intraday noon 1530'
output=HERE/'Closing_Auction_Presentation_2026_11_05.docx'
doc.save(output)
print(output)
