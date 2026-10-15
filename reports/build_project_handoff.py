"""Create the capstone project handoff report as a polished DOCX."""

from pathlib import Path
from datetime import date
import json

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEST = OUT_DIR / "closing_auction_volume_project_handoff.docx"

NAVY = "17365D"
BLUE = "2F5597"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F3F5F7"
MID_GRAY = "667085"
LIGHT_BORDER = "D9D9D9"
WHITE = "FFFFFF"
BLACK = "000000"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color=LIGHT_BORDER, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.size = Pt(8.5)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(1.02)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.2)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.04

    title = styles["Title"]
    title.font.name = "Aptos Display"
    title.font.size = Pt(27)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(BLACK)
    title.paragraph_format.space_after = Pt(12)
    # Remove the bottom border carried by Word's built-in Title style.
    title_ppr = title.element.find(qn("w:pPr"))
    if title_ppr is None:
        title_ppr = OxmlElement("w:pPr")
        title.element.append(title_ppr)
    for border in list(title_ppr.findall(qn("w:pBdr"))):
        title_ppr.remove(border)

    for style_name, size, before, after in (
        ("Heading 1", 18, 14, 7),
        ("Heading 2", 13.5, 11, 5),
        ("Heading 3", 11.5, 8, 4),
    ):
        style = styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLACK)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    if "Caption Custom" not in styles:
        cap = styles.add_style("Caption Custom", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Caption Custom"]
    cap.font.name = "Aptos"
    cap.font.size = Pt(9)
    cap.font.italic = True
    cap.font.color.rgb = RGBColor.from_string(MID_GRAY)
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(9)
    cap.paragraph_format.keep_with_next = True

    if "Small Text" not in styles:
        small = styles.add_style("Small Text", WD_STYLE_TYPE.PARAGRAPH)
    else:
        small = styles["Small Text"]
    small.font.name = "Aptos"
    small.font.size = Pt(9)
    small.font.color.rgb = RGBColor.from_string(MID_GRAY)
    small.paragraph_format.space_after = Pt(4)

    for sec in doc.sections:
        header = sec.header.paragraphs[0]
        header.text = "NYSE Closing Auction Volume Prediction"
        header.alignment = WD_ALIGN_PARAGRAPH.LEFT
        header.runs[0].font.name = "Aptos"
        header.runs[0].font.size = Pt(8.5)
        header.runs[0].font.color.rgb = RGBColor.from_string(MID_GRAY)
        add_page_number(sec.footer.paragraphs[0])


def add_paragraph(doc, text="", bold_lead=None, style=None, keep=False):
    p = doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        lead.bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    p.paragraph_format.keep_together = keep
    return p


def add_bullets(doc, items, level=0):
    for item in items:
        p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        p.add_run(item)
        p.paragraph_format.space_after = Pt(3)


def add_numbered(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.add_run(item)
        p.paragraph_format.space_after = Pt(4)


def add_table(doc, headers, rows, widths=None, font_size=9, alignments=None, repeat=True):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header_cells = table.rows[0].cells
    for j, header in enumerate(headers):
        cell = header_cells[j]
        if widths:
            cell.width = Inches(widths[j])
        set_cell_shading(cell, NAVY)
        set_cell_border(cell)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(str(header))
        run.bold = True
        run.font.name = "Aptos"
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor.from_string(WHITE)
    if repeat:
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        tbl_header = OxmlElement("w:tblHeader")
        tbl_header.set(qn("w:val"), "true")
        tr_pr.append(tbl_header)

    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cell = cells[j]
            if widths:
                cell.width = Inches(widths[j])
            set_cell_border(cell)
            set_cell_margins(cell)
            set_cell_shading(cell, PALE_BLUE if i % 2 else WHITE)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = (alignments[j] if alignments else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(value))
            run.font.name = "Aptos"
            run.font.size = Pt(font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_figure(doc, image_path, caption, width=6.75):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    picture = p.add_run().add_picture(str(image_path), width=Inches(width))
    picture._inline.docPr.set("title", caption)
    picture._inline.docPr.set("descr", caption)
    cap = doc.add_paragraph(caption, style="Caption Custom")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def rel(path):
    return str(Path(path).relative_to(ROOT))


doc = Document()
configure_document(doc)

# Cover
p = doc.add_paragraph(style="Title")
p.add_run("Closing Auction Volume Prediction Project Handoff")
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
direct_ppr = p._p.get_or_add_pPr()
for border in list(direct_ppr.findall(qn("w:pBdr"))):
    direct_ppr.remove(border)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(18)
r = p.add_run("NYSE daily model research and current modeling baseline")
r.font.name = "Aptos Display"
r.font.size = Pt(15)
r.font.color.rgb = RGBColor.from_string(MID_GRAY)

meta = add_table(
    doc,
    ["Item", "Current status"],
    [
        ["Prepared for", "Capstone project team"],
        ["Updated through", "14 October 2026"],
        ["Data horizon", "Auction data 2015–2025 and CRSP data 2011–2025"],
        ["Primary model", "Rolling five year global ridge model with 58 daily safe features"],
        ["Current decision", "Retain the frozen global benchmark and do not adopt separate linear segment models"],
    ],
    widths=[1.35, 5.55],
    font_size=10,
)

add_paragraph(
    doc,
    "This report gives a teammate enough context to continue the work without repeating the data audit, target analysis, feature validation, or baseline model selection. It records what was tested, why the current benchmark was chosen, where it fails, and what should be done next.",
    keep=True,
)

doc.add_page_break()

# Executive handoff
doc.add_heading("Current Project Position", level=1)
add_paragraph(
    doc,
    "We have built and validated a daily NYSE closing auction volume forecasting pipeline. The target remains same day closing auction volume in shares. The frozen benchmark predicts a log residual relative to each stock's lagged 20 auction median and then reconstructs the forecast in shares. This formulation pools stocks with very different volume scales while preserving the original target.",
)
add_paragraph(
    doc,
    "The current primary benchmark uses a rolling five year training window and 58 features from auction history, calendar, and liquidity groups. It outperformed historical baselines and the original expanding ridge specification. Tests of size, liquidity, auction intensity, combined size and liquidity, event day, and history depth segmentation did not produce a material and statistically reliable replacement. The frozen global model should therefore remain the reference for the next phase.",
)
add_paragraph(
    doc,
    "The central remaining problem is not the normal daily level. Long horizon plots show that the model follows gradual changes in a stock's auction volume. Most large errors occur on isolated spikes, abrupt collapses, and structurally changing small or illiquid stocks. The next phase should diagnose those residuals, correct systematic retransformation bias, and test a nonlinear pooled model before creating additional hard segments.",
)

doc.add_page_break()
doc.add_heading("Immediate Pickup Instructions", level=2)
add_numbered(doc, [
    "Treat the frozen 58 feature model as the primary benchmark and the 99 feature recent weighted model as the secondary global challenger.",
    "Start with a universe wide audit of the largest daily residuals and classify them by known calendar events, price and volume shocks, corporate actions, earnings, and index events when those data become available.",
    "Test a training only retransformation or calibration correction because the current model systematically underpredicts aggregate raw volume.",
    "Test a nonlinear pooled model on the same folds and information set before revisiting separate models.",
    "Preserve the existing out of time evaluation convention and compare every new method with both frozen global finalists on identical stock days.",
])

doc.add_heading("Decisions That Should Not Be Reopened Without New Evidence", level=2)
add_bullets(doc, [
    "Do not redefine the business target. It is closing auction volume in shares.",
    "Do not use same day daily market variables in the daily model. Predictors must be available by the prior close, except for known calendar information.",
    "Do not adopt the size split solely because it had the lowest point estimate in Notebook 10. Its gain was below the practical threshold and its paired confidence interval crossed zero.",
    "Do not claim that 2025 is a pristine untouched holdout. It was inspected during model development and is confirmation evidence only.",
    "Do not interpret the absence of a successful linear split as proof that nonlinear regimes do not exist.",
])

# Scope and data
doc.add_heading("Project Objective and Scope", level=1)
add_paragraph(
    doc,
    "The capstone objective is to estimate the liquidity available in the closing auction. The broader deliverable calls for daily and intraday models across NYSE, Nasdaq, LSE, and potentially ATS or dark pool venues. Work completed so far covers the NYSE daily model using information available through the previous end of day. Intraday prediction and the other exchanges remain outside the current completed scope.",
)

doc.add_heading("Data Inventory", level=1)
add_table(
    doc,
    ["Dataset", "Period", "Scale", "Role"],
    [
        ["CRSP daily data", "2011–2025", "4,927,078 rows and 2,414 PERMNOs", "Price, return, volume, shares, size, security, and corporate action history"],
        ["NYSE auction data", "2015–2025", "3,594,398 source rows and 2,149 PERMNOs", "Closing auction target and auction history"],
        ["Enriched daily panel", "2015–2025", "3,594,404 rows and 120 columns", "Merged modeling panel with 99 approved daily safe features"],
        ["Eligible targets", "2015–2025", "3,499,401 stock days", "Rows available for supervised daily modeling"],
    ],
    widths=[1.45, 1.05, 2.15, 2.25],
    font_size=8.8,
)

doc.add_heading("Known Data Limitations", level=2)
add_bullets(doc, [
    "The source auction data contain 94,513 missing target values.",
    "12 August 2019 is a severe source anomaly and is excluded from modeling.",
    "Auction coverage in 2022 is approximately 93 percent and is weaker for small capitalization stocks.",
    "The present daily panel does not contain true order imbalance or order book measures.",
    "Earnings dates and index rebalance membership are not yet available.",
    "Intraday checkpoints exist, but they represent a different information set and should not be mixed into the daily model without a separate design.",
])

doc.add_heading("Target Definition and Information Timing", level=1)
add_paragraph(
    doc,
    "The business target is raw same day NYSE closing auction volume. The model is estimated on the difference between log one plus today's auction volume and log one plus the stock's lagged 20 auction median. The predicted residual is added back to the lagged baseline and transformed into shares. This is a modeling device rather than a new target.",
)
add_paragraph(
    doc,
    "The residual formulation was selected because it removes much of the persistent scale difference between stocks and asks the model to predict whether today's auction will be unusually high or low relative to the stock's recent level. It produced better out of time performance than modeling the log level directly. The lagged 20 auction median uses only information available before the prediction date.",
)
add_paragraph(
    doc,
    "All daily covariates are lagged or known calendar variables. Training preprocessing is estimated inside each historical fold. No future target information is used for imputation, clipping, scaling, feature selection, model fitting, or segment assignment.",
)

doc.add_page_break()

# Work completed
doc.add_heading("Work Completed", level=1)
add_paragraph(
    doc,
    "The notebooks form a sequential research record. Each notebook writes machine readable outputs under outputs/eda and has a corresponding builder script where applicable.",
)

notebook_rows = [
    ["01", "Data quality and coverage", "Audited source periods, identifiers, missing targets, anomalies, and usable modeling rows"],
    ["02", "Target exploratory analysis", "Studied distribution, persistence, calendar effects, industry, size, and the need for log scaling"],
    ["03", "Covariate quality and leakage", "Cataloged features, missingness, correlations, readiness, and information timing"],
    ["04", "Covariate target relationships", "Measured univariate, nonlinear, categorical, within stock, and annual relationship stability"],
    ["05", "Feature engineering", "Built the 120 column enriched panel with 99 approved daily safe predictors"],
    ["06", "Feature validation and roles", "Assigned predictor, segmentation, context, and caution roles without dropping potential segmentation variables"],
    ["07", "Temporal baselines", "Compared historical rules with a global ridge model on annual out of time folds"],
    ["08", "Global robustness", "Tested windows, penalty tuning, sample size, feature groups, objectives, signs, drift, tails, and a placebo"],
    ["09", "Global finalists", "Combined the strongest specifications and froze the primary and secondary global benchmarks"],
    ["10", "Segmentation tests", "Compared pooled interactions, hard splits, and targeted specialists with paired uncertainty tests"],
    ["11", "Train validation test backtest", "Reconstructed graphical fold performance for the frozen benchmark"],
    ["12", "Long horizon stock plots", "Generated daily walk forward predictions for six representative stocks from 2016 through 2025"],
    ["13", "Representative year daily plots", "Displayed unsmoothed daily errors and marked the five largest proportional misses per stock"],
]
add_table(doc, ["Notebook", "Purpose", "Completed result"], notebook_rows, widths=[0.7, 2.2, 4.0], font_size=8.4)

doc.add_page_break()
doc.add_heading("Feature Engineering Result", level=1)
add_paragraph(
    doc,
    "The enriched panel contains 99 approved daily safe features. Feature validation did not automatically delete variables with weak direct coefficients because some may define regimes or support later nonlinear interactions. The retained groups are listed below.",
)
add_table(
    doc,
    ["Feature group", "Examples and role"],
    [
        ["Auction history", "Lagged auction volumes, rolling means and medians, dispersion, participation ratios, history depth, and spike flags"],
        ["Liquidity", "Lagged daily volume, dollar volume, turnover, Amihud measures, quoted spread, and high low range"],
        ["Returns and price", "Lagged returns, momentum, volatility, price, and overnight gap"],
        ["Size and history", "Lagged market capitalization, shares outstanding, data age, and cross sectional percentiles"],
        ["Market state", "Lagged market return, volatility, volume, breadth, dispersion, and range"],
        ["Sector state", "Sector return, volatility, activity, dispersion, breadth, and stock relative measures"],
        ["Calendar", "Day, month, distance to period end, options expiry, market gaps, and early close indicators"],
        ["Security and corporate actions", "Industry code and lagged dividend, distribution, price adjustment, and security classification fields"],
    ],
    widths=[1.8, 5.1],
    font_size=8.8,
)

add_paragraph(
    doc,
    "Notebook 06 classified 70 variables as predictor plus segmentation features, 5 as direct predictors, 4 as segmentation only, 16 as context or interaction variables, and 4 as caution variables. Rare price adjustment and expiry window flags, plus unstable sector dispersion and dollar volume features, were retained but flagged for careful interpretation.",
)

doc.add_page_break()

# Model selection
doc.add_heading("Global Model Selection", level=1)
add_paragraph(
    doc,
    "Annual out of time folds use 2022 through 2025 as test years. The frozen selection uses 2022 through 2024 as development evaluation years and reports 2025 separately as confirmation. Every finalist tunes its ridge penalty using the immediately preceding historical year and then refits on a rolling or expanding history sample.",
)

add_table(
    doc,
    ["Candidate", "Development log RMSE", "Development WAPE", "2025 log RMSE", "2025 WAPE", "Features"],
    [
        ["Recent weighted residual full", "0.48677", "0.36734", "0.44614", "0.34142", "99"],
        ["Residual full", "0.48710", "0.36690", "0.45013", "0.34132", "99"],
        ["Frozen auction calendar liquidity", "0.48764", "0.36801", "0.44512", "0.34563", "58"],
        ["Auction calendar only", "0.49059", "0.36904", "0.44785", "0.34734", "42"],
        ["Original expanding log full", "0.49898", "0.37674", "0.45117", "0.35328", "99"],
    ],
    widths=[2.25, 1.05, 1.05, 0.95, 0.85, 0.65],
    font_size=8.2,
    alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.CENTER] * 5,
)

doc.add_heading("Frozen Benchmark Specification", level=2)
add_bullets(doc, [
    "Rolling five year estimation window",
    "Log residual target relative to the lagged 20 auction median",
    "Uniform observation weights",
    "58 source features from auction history, calendar, and liquidity",
    "Training only preprocessing with annual nested ridge penalty selection",
    "Daily forecasts reconstructed and evaluated in shares",
])
add_paragraph(
    doc,
    "The 58 feature candidate was selected because it was within 0.002 log RMSE of the best development model and used substantially fewer features. It also achieved the best 2025 log RMSE among the finalists. The 99 feature recent weighted residual model remains the secondary challenger because it was the best development point estimate and had strong raw volume performance.",
)

doc.add_heading("Robustness Evidence", level=2)
add_bullets(doc, [
    "Rolling three year and rolling five year windows performed similarly and both modestly outperformed expanding history.",
    "Results were stable across 100,000, 350,000, and 700,000 row training samples.",
    "Ridge penalties were small and validation performance was nearly flat across the best penalty range.",
    "Auction history and calendar features provided most of the linear predictive value. Dropping auction history increased log RMSE by about 0.163 and dropping calendar features increased it by about 0.074.",
    "A shuffled target placebo failed decisively with log RMSE around 1.887, WAPE around 0.898, and Spearman correlation around 0.103. This supports the integrity of the predictive pipeline.",
    "The model underpredicted the largest one percent of auctions by roughly 52 to 54 percent in the earlier robustness audit.",
])

doc.add_page_break()
doc.add_heading("Rolling Backtest Results", level=1)
add_figure(
    doc,
    ROOT / "outputs" / "eda" / "11_frozen_global_backtest" / "frozen_global_train_validation_test_backtest.png",
    "Figure 1  Frozen global model performance across rolling train validation and test folds",
    width=6.6,
)

add_table(
    doc,
    ["Test year", "Log RMSE", "WAPE", "Spearman", "Aggregate bias"],
    [
        ["2022", "0.53351", "38.11%", "0.94896", "−14.44%"],
        ["2023", "0.48366", "36.97%", "0.95623", "−14.16%"],
        ["2024", "0.44576", "35.32%", "0.95943", "−14.53%"],
        ["2025", "0.44512", "34.56%", "0.95880", "−9.62%"],
    ],
    widths=[1.1, 1.25, 1.25, 1.25, 1.4],
    font_size=9,
    alignments=[WD_ALIGN_PARAGRAPH.CENTER] * 5,
)
add_paragraph(
    doc,
    "Test accuracy improved over time and there is no classic widening train test gap. Training metrics appear worse because each training diagnostic pools an older five year distribution, while later validation and test years were easier under the chosen metrics. The persistent negative aggregate bias is the more actionable issue.",
)

doc.add_page_break()

# Segmentation
doc.add_heading("Segmentation and Specialist Model Tests", level=1)
add_paragraph(
    doc,
    "Notebook 10 tested whether one global ridge model should be replaced by separate models. The candidates included pooled interactions, hard splits by size, liquidity, auction intensity, a two by two size and liquidity grid, event specialists, and a limited history specialist. Segment assignments used only lagged features or known calendar information.",
)
add_figure(
    doc,
    ROOT / "outputs" / "eda" / "10_segmentation_tests" / "segmentation_architecture_comparison.png",
    "Figure 2  Development performance of segmented and pooled alternatives relative to the frozen global benchmark",
    width=6.6,
)

add_table(
    doc,
    ["Candidate", "Development log RMSE", "Change versus frozen", "Decision"],
    [
        ["Size split", "0.48648", "−0.00116", "Numerically best but below the 0.002 threshold and not significant"],
        ["Size and liquidity split", "0.48673", "−0.00091", "No material improvement"],
        ["Auction intensity split", "0.48722", "−0.00042", "No material improvement"],
        ["Liquidity split", "0.48767", "+0.00003", "No improvement"],
        ["History specialist", "0.48796", "+0.00032", "Sparse segment gains were uncertain"],
        ["Event specialist", "0.48907", "+0.00142", "Unstable and worse in development"],
    ],
    widths=[1.65, 1.2, 1.15, 2.9],
    font_size=8.4,
)
add_paragraph(
    doc,
    "No general architecture passed the practical improvement threshold, paired date block uncertainty test, annual stability requirement, and comparison with the secondary global challenger. No targeted specialist showed a significant development period improvement in its intended segment. Some cold start and expiry gains appeared in 2025, but those patterns were not stable in 2022 through 2024 and should not be treated as validated discoveries.",
)

doc.add_page_break()

# Diagnostics
doc.add_heading("Where the Frozen Model Performs Poorly", level=1)
add_table(
    doc,
    ["Regime", "Log RMSE", "WAPE", "Aggregate bias", "Interpretation"],
    [
        ["Cold start with 0 to 4 auctions", "1.405", "0.920", "−57%", "The lagged auction baseline is immature and the sample is very small"],
        ["Developing history with 5 to 19 auctions", "0.840", "0.507", "−5%", "History features remain unstable"],
        ["Quarterly options expiry", "0.789", "0.552", "−23%", "Closing flows differ materially on expiry dates"],
        ["Month end", "0.659", "0.491", "−20%", "Known institutional flow regime remains difficult"],
        ["Small stocks", "0.582", "0.439", "−24%", "Greater noise and weaker liquidity create persistent underprediction"],
        ["Top one percent of actual auctions", "1.271", "0.542", "−41%", "Extreme events remain strongly underpredicted"],
    ],
    widths=[1.6, 0.8, 0.75, 1.05, 2.7],
    font_size=8.3,
)

doc.add_heading("Long Horizon Stock Evidence", level=2)
add_figure(
    doc,
    ROOT / "outputs" / "eda" / "12_selected_stock_predictions" / "selected_stock_actual_vs_walk_forward_predictions.png",
    "Figure 3  Actual and predicted 20 trading day medians for representative stocks  Daily forecasts begin in 2016 and 2015 is a warm up year",
    width=6.55,
)
add_paragraph(
    doc,
    "The long horizon plots show that the global model follows gradual changes in the typical auction volume of large, mid sized, and small stocks. The smoothing in Figure 3 is only for visualization. The model generated one prediction per trading day, and all reported accuracy metrics use unsmoothed daily values.",
)

doc.add_page_break()
doc.add_heading("Daily Error Pattern", level=2)
add_figure(
    doc,
    ROOT / "outputs" / "eda" / "13_representative_stock_year_daily" / "representative_stock_year_daily_actual_vs_predicted.png",
    "Figure 4  Unsmoothed daily forecasts in mechanically selected representative years  Red crosses mark the five largest absolute log errors",
    width=6.35,
)
add_paragraph(
    doc,
    "The daily plots reveal the main failure mode. Normal days generally track the correct level, but the model often misses isolated jumps or collapses. Small and illiquid CATO also shows substantial negative calibration. These patterns are consistent with the universe wide tail diagnostics and indicate that better event and same day flow information is more likely to help than additional hard size or liquidity splits.",
)

doc.add_page_break()

# Recommendations
doc.add_heading("Recommended Next Phase", level=1)

doc.add_heading("Residual Event Study", level=2)
add_paragraph(
    doc,
    "Create a universe wide table of the largest absolute log errors and largest raw share errors. Join the miss dates to existing calendar indicators, lagged returns, abnormal volume, corporate action flags, security history, and later to earnings and index rebalance data. The purpose is to separate predictable events, missing information, source problems, and irreducible noise.",
)

doc.add_heading("Retransformation and Calibration", level=2)
add_paragraph(
    doc,
    "The log model systematically underpredicts aggregate volume after forecasts are transformed back into shares. Test a smearing correction estimated only on the training window. Compare global, prediction decile, and carefully pooled liquidity or size calibration factors. Report log RMSE, WAPE, aggregate bias, and tail bias together so calibration does not improve raw volume at the cost of proportional accuracy.",
)

doc.add_heading("Nonlinear Pooled Model", level=2)
add_paragraph(
    doc,
    "Test a gradient boosted tree or another nonlinear pooled learner on the same 99 feature panel. The model should retain the same target, folds, information timing, and benchmark comparisons. Nonlinear pooling can learn threshold and interaction effects without maintaining many independent models. Tune complexity inside the historical validation year and preserve full year test predictions.",
)

doc.add_heading("Hierarchical Stock Corrections", level=2)
add_paragraph(
    doc,
    "Test a shrunk stock specific residual correction on top of the global prediction. Stocks with long histories can receive stronger individual adjustments, while newer or sparse stocks remain close to the pooled estimate. This addresses persistent stock bias without fitting thousands of independent models.",
)

doc.add_heading("Spike and Uncertainty Models", level=2)
add_paragraph(
    doc,
    "After the residual study, consider a two stage model that estimates the probability of a spike and then its magnitude. Also consider prediction intervals or quantile forecasts. These extensions are most useful after same day imbalance data, earnings dates, and index rebalance membership are added because historical daily variables cannot fully anticipate idiosyncratic closing flows.",
)

doc.add_heading("Evaluation Rules for New Models", level=2)
add_bullets(doc, [
    "Use the same eligible target universe and annual walk forward folds.",
    "Fit preprocessing, calibration, feature selection, and hyperparameters on historical data only.",
    "Compare with both frozen global finalists on identical stock days.",
    "Report log RMSE, log MAE, WAPE, Spearman correlation, aggregate bias, top tail performance, and predeclared segment metrics.",
    "Use paired date block bootstrap intervals rather than comparing point estimates alone.",
    "Require stability across development years before interpreting a 2025 improvement as durable.",
])

doc.add_page_break()

# Reproducibility and artifacts
doc.add_heading("Reproducibility and Artifact Map", level=1)
add_paragraph(
    doc,
    "The project root is the Capstone directory. Notebook builders are stored under notebooks and outputs are stored under outputs/eda. The paths below are relative to the project root.",
)

artifact_rows = [
    ["Enriched modeling panel", "outputs/eda/05_feature_engineering/enriched_daily_panel.parquet", "3.59 million rows and 120 columns"],
    ["Approved feature catalog", "outputs/eda/05_feature_engineering/approved_enriched_daily_features.csv", "99 daily safe features and economic groups"],
    ["Feature validation roles", "outputs/eda/06_feature_validation/feature_validation_and_roles.csv", "Predictor segmentation context and caution roles"],
    ["Frozen benchmark specification", "outputs/eda/09_global_finalists/frozen_global_benchmark.json", "Authoritative primary model definition"],
    ["Finalist selection", "outputs/eda/09_global_finalists/global_finalist_selection.csv", "Development and 2025 comparison"],
    ["Global predictions", "outputs/eda/09_global_finalists/predictions/", "Full out of time predictions for 2022 through 2025"],
    ["Segmentation decision", "outputs/eda/10_segmentation_tests/segmentation_decision_table.csv", "Architecture adoption tests"],
    ["Backtest metrics", "outputs/eda/11_frozen_global_backtest/frozen_global_train_validation_test_metrics.csv", "Train validation and test fold metrics"],
    ["Selected stock predictions", "outputs/eda/12_selected_stock_predictions/selected_stock_daily_walk_forward_predictions.parquet", "Daily stock forecasts for 2015 through 2025 with 2015 warm up"],
    ["Largest representative misses", "outputs/eda/13_representative_stock_year_daily/representative_year_largest_daily_misses.csv", "Marked daily stock level errors"],
]
add_table(doc, ["Artifact", "Relative path", "Use"], artifact_rows, widths=[1.1, 2.8, 2.1], font_size=7.2)

doc.add_heading("Primary Notebooks", level=2)
add_bullets(doc, [
    "notebooks/09_global_finalists_and_benchmark_freeze.ipynb for the model selection decision",
    "notebooks/10_segmentation_and_specialist_model_tests.ipynb for global versus separate model evidence",
    "notebooks/11_frozen_global_train_validation_test_backtest.ipynb for fold diagnostics",
    "notebooks/12_selected_stock_walk_forward_predictions.ipynb for long horizon daily predictions",
    "notebooks/13_representative_stock_year_daily_predictions.ipynb for unsmoothed error examples",
])

doc.add_page_break()
doc.add_heading("Implementation Notes", level=2)
add_bullets(doc, [
    "Shared leakage safe preprocessing and ridge helpers are in notebooks/modeling_utils.py.",
    "Notebook construction scripts are named build_notebook_03.py through build_notebook_13.py, with the first two notebooks built by build_eda_notebooks.py.",
    "The master panel keeps all 99 approved features even though the frozen model uses 58. This is intentional because the remaining variables may support nonlinear models, diagnostics, or future segmentation tests.",
    "The 2015 stock plots contain actual values only. Leakage safe rolling forecasts begin in 2016 because the available auction history starts in 2015.",
])

doc.add_heading("Definition of Done for the Next Contributor", level=1)
add_paragraph(
    doc,
    "A new modeling experiment is complete only when it saves its specification, fold metrics, full out of time predictions, segment diagnostics, and paired uncertainty comparison. The notebook should explain the information set and any deviations from the frozen evaluation protocol. Any model proposed for adoption must improve the primary benchmark materially and consistently rather than win on one year, one subgroup, or one metric.",
)

doc.add_heading("Open Questions", level=2)
add_bullets(doc, [
    "Which known events explain the largest raw and proportional daily misses across the universe",
    "How much of the aggregate underprediction comes from log retransformation rather than missing predictors",
    "Whether a nonlinear pooled model can capture useful size liquidity and event interactions without hard splits",
    "Whether persistent stock bias remains after calibration and merits a hierarchical correction",
    "How much same day imbalance earnings and index rebalance data improve spike prediction",
])

doc.add_heading("First Files to Open", level=2)
add_bullets(doc, [
    "Open notebooks/09_global_finalists_and_benchmark_freeze.ipynb to understand the frozen benchmark decision.",
    "Open notebooks/10_segmentation_and_specialist_model_tests.ipynb to understand why separate linear models were not adopted.",
    "Read outputs/eda/09_global_finalists/frozen_global_benchmark.json before implementing a new comparison.",
    "Use outputs/eda/09_global_finalists/predictions as the reference prediction set for paired residual analysis.",
])

# Apply keep-with-next to table-adjacent labels and ensure metadata.
doc.core_properties.title = "Closing Auction Volume Prediction Project Handoff"
doc.core_properties.subject = "NYSE daily closing auction volume forecasting research handoff"
doc.core_properties.author = "Capstone project team"
doc.core_properties.keywords = "closing auction volume NYSE forecasting capstone handoff"

doc.save(DEST)
print(f"Wrote {DEST}")
