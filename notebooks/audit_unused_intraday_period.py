"""Read local source schemas and dates only. Never score or inspect forward volumes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import duckdb,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/25_frozen_intraday_evaluation'
CUTOFF='2025-12-31'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 assert json.loads((OUT/'evaluation_summary.json').read_text())['all_checks_passed']
 c=duckdb.connect();sources={
 'auction_labels':sorted((ROOT/'blackrock_data/stock_daily/auction_daily').rglob('*.parquet')),
 'crsp_daily':sorted((ROOT/'blackrock_data/stock_daily/crsp_daily').rglob('*.parquet')),
 'checkpoint_1530':sorted((ROOT/'blackrock_data/stock_intraday/trade_checkpoints').rglob('*.parquet')),
 'market_daily':[ROOT/'blackrock_data/market/market_daily.parquet'],
 'sector_daily':[ROOT/'blackrock_data/sector/sector_daily.parquet'],
 'calendar':[ROOT/'blackrock_data/reference/calendar.parquet'],
 'identifier_bridge':sorted((ROOT/'blackrock_data/reference/taq_crsp_bridge').rglob('*.parquet')),
 'industry_history':[ROOT/'blackrock_data/reference/industry_history.parquet'],
 'security_master':[ROOT/'blackrock_data/reference/security_master.parquet']}
 inventory=[]
 for family,files in sources.items():
  for p in files:
   if not p.exists():
    inventory.append(dict(family=family,path=str(p.relative_to(ROOT)),status='absent'));continue
   schema=c.sql(f"DESCRIBE SELECT * FROM read_parquet('{p}')").df();names=set(schema.column_name)
   where=" WHERE checkpoint='15:30'" if family=='checkpoint_1530' else ''
   # Project dates and counts only; do not read volume values, predictions, or error columns.
   if 'date' in names:
    query=f"SELECT count(*) AS row_count,min(date) first_date,max(date) last_date,count(*) FILTER(WHERE date>DATE '{CUTOFF}') forward_rows,count(DISTINCT date) FILTER(WHERE date>DATE '{CUTOFF}') forward_dates FROM read_parquet('{p}'){where}"
    r=c.sql(query).fetchone();inventory.append(dict(family=family,path=str(p.relative_to(ROOT)),status='date_metadata_read',bytes=p.stat().st_size,rows=r[0],first_date=str(r[1]),last_date=str(r[2]),forward_rows=r[3],forward_dates=r[4],accessed_columns='date, checkpoint' if where else 'date',query=query))
   else:
    inventory.append(dict(family=family,path=str(p.relative_to(ROOT)),status='schema_only_no_date_column',bytes=p.stat().st_size,columns=', '.join(schema.column_name),accessed_columns='schema only'))
 f=pd.DataFrame(inventory);f.to_csv(OUT/'forward_source_file_inventory.csv',index=False)
 summary=[]
 for family in sources:
  g=f[f.family.eq(family)];dated=g[g.status.eq('date_metadata_read')]
  summary.append(dict(family=family,files=len(g),dated_files=len(dated),first_date=dated.first_date.min() if len(dated) else None,last_date=dated.last_date.max() if len(dated) else None,forward_rows=int(dated.forward_rows.sum()) if len(dated) else None,has_post_2025_dates=bool(dated.forward_rows.sum()>0) if len(dated) else None,note='Range and counts do not certify continuity, label validity, point-in-time availability or prior non-use'))
 s=pd.DataFrame(summary);s.to_csv(OUT/'forward_source_coverage.csv',index=False)
 core=['auction_labels','crsp_daily','checkpoint_1530','market_daily','sector_daily','calendar']
 blockers=s[s.family.isin(core)&s.has_post_2025_dates.ne(True)].family.tolist()
 config=ROOT/'blackrock_data/config/project_config.json';cfg=json.loads(config.read_text())
 optional=[str(p.relative_to(ROOT)) for p in (ROOT/'blackrock_data/events').glob('*2026*')]
 audit={'audited_utc':datetime.now(timezone.utc).isoformat(),'scope':'Local workspace source inventory only; no remote source availability inferred','last_inspected_research_period_end':CUTOFF,'potential_unused_dates_must_be_after':CUTOFF,'forward_outcomes_scored':False,'forward_auction_volume_values_read':False,'source_date_metadata_only':True,'configured_source_end_date':cfg.get('end_date'),'missing_forward_core_components':blockers,'candidate_forward_period_locally_available':not bool(blockers),'prior_nonuse_verified':False,'holdout_ready':False,'prior_nonuse_note':'No attestation establishes that any future outcomes were unused elsewhere. 2015-2025 have appeared in training/descriptive/development/confirmation research. A newer date alone does not certify an untouched holdout.','optional_newer_files':optional,'optional_note':'An event file extending into 2026 cannot substitute for forward labels, daily features and checkpoint observations. These files were inventoried by name only.','script_sha256':sha(Path(__file__)),'config_sha256':sha(config)}
 (OUT/'unused_period_audit.json').write_text(json.dumps(audit,indent=2))
 report='# Audit of unused forward period availability\n\n'
 if blockers:report+='**No ready unused forward period was found in the audited local sources.** Missing post-2025 coverage in required components: '+', '.join(blockers)+'.\n\n'
 else:report+='Core files contain some post-2025 dates, but continuity, label completeness, point-in-time availability and prior non-use remain uncertified. Do not score a forward period yet.\n\n'
 report+='```text\n'+s[['family','files','first_date','last_date','forward_rows']].to_string(index=False)+'\n```\n\n'
 report+='The source configuration ends '+str(cfg.get('end_date'))+'. The FOMC event filename extends into 2026, but this is not auction/model coverage. Only schemas, date ranges, counts and filenames were read during this audit; no forward auction volume values, predictions or errors were inspected. The historical evaluation preceding this audit used only 2022–2025 saved forecasts.\n\n'
 report+='## Required before a new score\n\nObtain complete compatible forward auction records, CRSP inputs, 15:30 aggregates, market/sector context and calendar/identifier support. Verify date/key coverage, early-close handling, extraction cutoffs and label-validity flags without exploratory outcome analysis. Record provenance and prior-use attestation. Freeze a non-overlapping evaluation window and minimum duration, forecast delivery rules, annual training and historical tuning procedure, and economically justified acceptance criteria before scoring. Use only earlier out-of-time forecasts to fit every correction. Log forecasts before outcomes when feasible.\n\nA 2026 model would require the locked daily and intraday training procedures to be refitted using earlier data; a saved 2025 coefficient file must not silently be treated as that annual refit. Model coefficients and preprocessing, eligible universe and cutoff assumptions must be fingerprinted. Do not claim WAPE noninferiority without an independently justified margin. Score once at the fixed end, without optional stopping or retrospective subgroup changes.\n\nNo new forward period was scored. Existing 2025 data cannot be reused as a fresh holdout. Data stored outside this local workspace were not audited.\n'
 (OUT/'unused_period_audit.md').write_text(report)
 print(s[['family','files','last_date','forward_rows']].to_string(index=False));print('Forward period scored: False. Holdout ready: False.')
if __name__=='__main__':run()
