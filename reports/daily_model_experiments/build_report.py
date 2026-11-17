"""Build the Word version of daily_model_experiments_report.md with python-docx.

Usage (from the repository root):
    python reports/daily_model_experiments/build_report.py

Supports the subset of Markdown the report uses: # title, ## and ### headings, paragraphs,
"- " bullets, **bold** runs, pipe tables and ![caption](relative/path.png) figures.
"""

from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "daily_model_experiments_report.md"
TARGET = HERE / "NYSE_Closing_Auction_Daily_Model_Experiments.docx"
TEXT_WIDTH = 7.1

doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin, section.bottom_margin = Inches(0.72), Inches(0.68)
section.left_margin = section.right_margin = Inches(0.7)
for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "List Bullet"]:
    style = doc.styles[name]
    style.font.name = "Calibri"
    style.font.color.rgb = RGBColor(0, 0, 0)
normal = doc.styles["Normal"]
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(7)
normal.paragraph_format.line_spacing = 1.12
for name, size in [("Title", 24), ("Heading 1", 15), ("Heading 2", 12)]:
    style = doc.styles[name]
    style.font.size = Pt(size)
    style.font.bold = True
    style.paragraph_format.keep_with_next = True
    style.paragraph_format.space_before = Pt(0 if name == "Title" else 14)
    style.paragraph_format.space_after = Pt(6)
doc.styles["List Bullet"].paragraph_format.space_after = Pt(3)

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
run = footer.add_run("NYSE auction forecasting  |  daily model experiments  |  ")
run.font.size = Pt(8)
run.font.color.rgb = RGBColor.from_string("666666")
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def add_runs(paragraph, text, size=None):
    """Add text to a paragraph, rendering **bold** spans."""
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        r = paragraph.add_run(part[2:-2] if bold else part)
        r.bold = bold
        if size:
            r.font.size = Pt(size)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tcpr.append(shading)
    borders = OxmlElement("w:tcBorders")
    for edge in ["top", "left", "bottom", "right"]:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "D9D9D9")
        borders.append(el)
    tcpr.append(borders)


def add_table(rows):
    cols = len(rows[0])
    first = min(2.6, TEXT_WIDTH * (0.5 if cols <= 3 else 0.3))
    widths = [first] + [(TEXT_WIDTH - first) / (cols - 1)] * (cols - 1)
    table = doc.add_table(rows=1, cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    numeric = [all(re.fullmatch(r"[−+\-—]?[\d.,%]*\d[\d.,%]*|—", r[c]) for r in rows[1:]) for c in range(cols)]
    for ri, values in enumerate(rows):
        cells = table.rows[0].cells if ri == 0 else table.add_row().cells
        trpr = table.rows[ri]._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if ri == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for ci, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Inches(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shade(cell, "253B4A" if ri == 0 else ("F2F5F7" if ri % 2 == 0 else "FFFFFF"))
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.04
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if (numeric[ci] and ci > 0 and ri > 0) else WD_ALIGN_PARAGRAPH.LEFT
            add_runs(p, value, size=8.5 if cols >= 6 else 9.5)
            if ri == 0:
                for r in p.runs:
                    r.bold = True
                    r.font.color.rgb = RGBColor(255, 255, 255)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


lines = SOURCE.read_text().splitlines()
figure = 0
i = 0
while i < len(lines):
    line = lines[i].rstrip()
    if not line.strip():
        i += 1
        continue
    if line.startswith("|"):
        block = []
        while i < len(lines) and lines[i].startswith("|"):
            row = [v.strip() for v in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", v) for v in row):
                block.append(row)
            i += 1
        add_table(block)
        continue
    image = re.fullmatch(r"!\[(.*)\]\((.*)\)", line)
    if image:
        figure += 1
        path = (HERE / image.group(2)).resolve()
        doc.add_picture(str(path), width=Inches(TEXT_WIDTH))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
        caption = doc.add_paragraph()
        add_runs(caption, f"Figure {figure}. {image.group(1)}", size=9)
        caption.runs[0].italic = True
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif line.startswith("# "):
        doc.add_paragraph(line[2:], "Title")
    elif line.startswith("### "):
        doc.add_paragraph(line[4:], "Heading 2")
    elif line.startswith("## "):
        heading = doc.add_paragraph(line[3:], "Heading 1")
        if line.startswith("## Appendix"):
            heading.paragraph_format.page_break_before = True
    elif line.startswith("Last updated"):
        doc.add_paragraph(line, "Subtitle")
    elif line.startswith("- "):
        add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
    elif line.startswith("MSE_blend ="):
        p = doc.add_paragraph()
        add_runs(p, line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
    else:
        p = doc.add_paragraph()
        add_runs(p, line, size=9 if line.startswith("Source:") else None)
    i += 1

# Remove inherited template borders, including the default title rule.
for style in doc.styles:
    for border in list(style.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
for paragraph in doc.paragraphs:
    for border in list(paragraph._p.iter(qn("w:pBdr"))):
        border.getparent().remove(border)

doc.core_properties.title = "NYSE Closing Auction Daily Model Experiments"
doc.core_properties.subject = "Daily model experiments (running report)"
doc.core_properties.author = ""
doc.core_properties.keywords = "NYSE closing auction LightGBM CatBoost HMM CNN LSTM blend"
doc.save(TARGET)
print(f"Saved {TARGET.name}; {figure} figures; source words: {len(SOURCE.read_text().split())}")
