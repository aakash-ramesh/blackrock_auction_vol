from pathlib import Path
import re
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
root=Path(__file__).resolve().parents[2]
out=root/'reports/project_progress_2026_11_04'
s=(out/'project_progress_report.md').read_text()
d=Document();sec=d.sections[0];sec.top_margin=Inches(.72);sec.bottom_margin=Inches(.68);sec.left_margin=Inches(.7);sec.right_margin=Inches(.7)
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
 st=d.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
normal=d.styles['Normal'];normal.font.size=Pt(10.5);normal.paragraph_format.space_after=Pt(7);normal.paragraph_format.line_spacing=1.12
for name,size in [('Title',26),('Heading 1',16),('Heading 2',12)]:
 st=d.styles[name];st.font.size=Pt(size);st.paragraph_format.keep_with_next=True
 st.paragraph_format.space_before=Pt(15 if name!='Title' else 0);st.paragraph_format.space_after=Pt(7)
d.styles['Title'].font.bold=True
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run('NYSE auction forecasting  |  ');r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string('666666')
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
lines=s.splitlines();i=0
while i<len(lines):
 line=lines[i]
 if not line.strip():i+=1;continue
 if line.startswith('| '):
  block=[]
  while i<len(lines) and lines[i].startswith('|'):
   row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r':?-+:?',v) for v in row):block.append(row)
   i+=1
  cols=len(block[0]);table=d.add_table(rows=1, cols=cols);table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
  if cols==6: widths=[2.28,.95,.95,.95,.95,.98]
  elif cols==5: widths=[2.4,1.1,1.15,1.1,1.31]
  elif cols==4:widths=[2.3,1.3,1.4,2.06]
  elif cols==3:widths=[1.5,2.8,2.76]
  else:widths=[1.8,5.26]
  if block[0][-1]=='Verification checks':widths=[1.9,4.15,1.01]
  if block[0][:2]==['Model','Target']:widths=[1.1,1.6,1.45,1.45,1.46]
  for j,w in enumerate(widths):table.columns[j].width=Inches(w)
  for ri,row in enumerate(block):
   cells=table.rows[0].cells if ri==0 else table.add_row().cells
   trpr=table.rows[ri]._tr.get_or_add_trPr()
   cant=OxmlElement('w:cantSplit');trpr.append(cant)
   if ri==0:
    repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
   for ci,(cell,value) in enumerate(zip(cells,row)):
    cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    tcpr=cell._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'253B4A' if ri==0 else ('F2F5F7' if ri%2==0 else 'FFFFFF'));tcpr.append(sh)
    borders=OxmlElement('w:tcBorders')
    for edge in ['top','left','bottom','right']:
     el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
    tcpr.append(borders)
    margins=OxmlElement('w:tcMar')
    for edge in ['top','left','bottom','right']:
     el=OxmlElement('w:'+edge);el.set(qn('w:w'),'85');el.set(qn('w:type'),'dxa');margins.append(el)
    tcpr.append(margins)
    p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.04
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 or cols<=3 else WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(value);r.font.size=Pt(9 if cols>=5 else 9.5);r.bold=ri==0
    if ri==0:r.font.color.rgb=RGBColor(255,255,255)
  d.add_paragraph().paragraph_format.space_after=Pt(0)
  continue
 if line.startswith('# '):d.add_paragraph(line[2:],'Title')
 elif line.startswith('### '):d.add_paragraph(line[4:],'Heading 2')
 elif line.startswith('## '):
  h=d.add_paragraph(line[3:],'Heading 1')
  if line.startswith('## Appendix'):h.paragraph_format.page_break_before=True
 elif line.startswith('Research completed'):d.add_paragraph(line,'Subtitle')
 else:
  p=d.add_paragraph(line)
  if line.startswith(('Source R','O1.','O2.','O3.','O4.','O5.','R1.','R2.','R3.','R4.','R5.','R6.','R7.','R8.')):
   for r in p.runs:r.font.size=Pt(9)
   p.paragraph_format.line_spacing=1.04
 i+=1
# Remove inherited template borders, including the default title rule.
for st in d.styles:
 for border in list(st.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
for para in d.paragraphs:
 for border in list(para._p.iter(qn('w:pBdr'))):border.getparent().remove(border)
for table in d.tables:
 if len(table.rows)>=3:
  for cell in table.rows[-2].cells:
   for para in cell.paragraphs:para.paragraph_format.keep_with_next=True
d.core_properties.title='NYSE Closing Auction Volume Forecasting Project Progress Report'
d.core_properties.subject='Research progression and new local experiments through November 4 2026'
d.core_properties.author='';d.core_properties.keywords='NYSE closing auction regression nonlinear LightGBM intraday'
d.save(out/'NYSE_Closing_Auction_Project_Progress_Report.docx')
print('Saved report; source words:',len(s.split()))
