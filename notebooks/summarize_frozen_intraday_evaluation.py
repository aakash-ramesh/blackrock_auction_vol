"""Write the evaluation findings only after historical inference and source audit."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/experiments/25_frozen_intraday_evaluation'
def run():
 m=json.loads((OUT/'evaluation_summary.json').read_text());audit=json.loads((OUT/'unused_period_audit.json').read_text());assert m['all_checks_passed'] and not audit['forward_outcomes_scored']
 i=pd.read_csv(OUT/'metric_intervals.csv');p=pd.read_csv(OUT/'period_metric_differences.csv');a=pd.read_csv(OUT/'annual_subgroup_metrics.csv');primary=i[i.dimension.eq('overall')&i.block_dates.eq(20)]
 text='''# Frozen daily versus 15 30 evaluation findings

The frozen 15:30 ridge improves historical log RMSE, WAPE and absolute annual aggregate bias versus the frozen daily LightGBM anchor. Direct metric-matched date-block intervals remain favorable in development and previously inspected 2025 confirmation. **This is retrospective evidence conditional on the fitted forecasts, not a fresh holdout assessment or production certification. No new period was scored.**

## Primary 20 trading date block results

All differences below are intraday minus daily. Negative is favorable for error measures and absolute bias. Signed bias is a directional quantity: positive indicates an upward shift, not automatically an improvement. Here both forecasts underpredict, so the upward shift accompanies reduced absolute annual bias. WAPE and bias differences and intervals are expressed in percentage points.

| Period | Metric | Daily | Intraday | Difference | 95 percent interval for difference |
| --- | --- | ---: | ---: | ---: | --- |
'''
 names={'rmse_log':'Log RMSE','wape':'WAPE pp','signed_bias':'Signed bias pp','absolute_annual_bias':'Absolute annual bias pp'}
 for row in primary.itertuples():
  q=p[p.period.eq(row.period)&p.dimension.eq('overall')&p.metric.eq(row.metric)].iloc[0];scale=1 if row.metric=='rmse_log' else 100
  text+=f'| {row.period} | {names[row.metric]} | {q.baseline*scale:.5f} | {q.candidate*scale:.5f} | {row.difference*scale:+.5f} | [{row.ci_lower*scale:+.5f}, {row.ci_upper*scale:+.5f}] |\n'
 text+='''
Development statistics are equal-year averages of annual metrics for 2022–2024. Confirmation is 2025 alone. They are not pooled metrics or averages of date-level percentage errors. For absolute annual bias, take the absolute bias within each year and then average, rather than taking the absolute value of the mean signed bias.

## What changed statistically

Earlier intraday intervals addressed log MAE, while the headline metric was log RMSE. This module recomputes log RMSE itself in every bootstrap replicate. WAPE and bias also recompute share-total ratios using each replicate's actual-volume denominator. Stock-days within each date stay together, and the same dates and weights are used for both forecasts and all metrics. Resampling runs separately within years before equal-year aggregation.

There are 5000 circular moving-block replicates at each length. The prespecified primary length is 20 trading dates; 5 and 60 are sensitivity analyses. Overall log-RMSE and WAPE intervals favor intraday at all three lengths in both periods. No favorable block length was selected after scoring.

Circular wrapping makes date inclusion uniform but introduces artificial adjacency between each year's end and start. Inference relies on within-year stability and limited dependence; it does not guarantee coverage under structural changes. These conditional intervals omit parameter refitting, tuning and repeated research-selection uncertainty. They must not be presented as a new independent test after the many earlier experiments. Secondary and subgroup intervals are pointwise, unadjusted for multiplicity. No p-values, noninferiority or production promotion are declared.

## Subgroup reporting

Annual counts, distinct dates, actual share totals and metrics are retained for predefined size thirds, liquidity thirds, history depth, disjoint calendar events, and month-end crossed with liquidity. Explicit missing groups are included, even when empty. Period summaries require all constituent years; no year is silently dropped. Actual-volume tails are excluded from forecast-time subgroup definitions.

Intervals are suppressed if any year has fewer than 30 subgroup dates, fewer than 500 observations, or an invalid share denominator. The full trading calendar is resampled for eligible subgroups, including dates with zero subgroup observations. An invalid replicate suppresses the interval rather than being discarded. Expiry and month-end groups can have thousands of stocks but too few dates for defensible subgroup intervals; their point estimates and sample counts remain visible. Read metric_intervals.csv status/reason fields and annual_subgroup_metrics.csv before interpreting a subgroup.

All subgroup intervals are exploratory, not evidence for a new routing or specialist rule. The main frozen contrast is full intraday ridge versus daily; shrinkage and feature variants are not reopened here.

## Unused period availability

'''
 if not audit['candidate_forward_period_locally_available']:
  text+='The local source audit found **no ready forward period beyond December 31 2025**. All six core date-bearing components—auction records, CRSP inputs, 15:30 aggregates, market context, sector context and calendar—end on that date. The 2026 FOMC filename does not supply auction-model coverage. '
 else:text+='Some source dates extend beyond 2025, but readiness and prior non-use have not been certified. '
 text+='The audit inspected source schemas, dates, counts and filenames, without reading forward auction volumes or generating/scoring forward forecasts. Local presence alone cannot prove non-use elsewhere; an attestation and provenance record are required. See unused_period_audit.md and the exact per-file date queries in forward_source_file_inventory.csv.\n\n'
 text+='''## Next action before any new score

Acquire compatible post-2025 source data or begin prospective collection. Before examining outcomes, freeze a fixed evaluation window and duration, the annual training and inner historical tuning procedures, eligibility and checkpoint-delivery rules, the primary metric and an independently justified WAPE acceptance/noninferiority margin if needed. Fit every layer only on earlier information and retain out-of-time inputs for stacked models. Record prior-use attestation and lock forecast snapshots. Score once at the fixed end without optional stopping. A chronological historical reanalysis cannot make already inspected years untouched again.

## Reproduction and artifacts

From the repository root, run:

```sh
python notebooks/evaluate_frozen_intraday.py
python notebooks/audit_unused_intraday_period.py
python notebooks/summarize_frozen_intraday_evaluation.py
```

The existing local project environment is sufficient. No model fitting or new external dependency is needed. Large historical row-level forecasts under experiment 19 must be present or regenerated using that experiment's documented pipeline. The reader notebook displays saved results.

- evaluation_protocol.md defines estimands, bootstrap, subgroup rules and forward scoring restrictions.
- annual_subgroup_metrics.csv and period_metric_differences.csv retain counts and matched point estimates.
- metric_intervals.csv contains direct metric intervals, primary/sensitivity labels and sparse-group suppressions.
- daily_sufficient_statistics.csv and bootstrap_weight_fingerprints.csv preserve the computation and resampling provenance.
- unused_period_audit.json, unused_period_audit.md and forward_source_coverage.csv record local availability and the unresolved provenance requirement.
- validation_checks.csv and input_fingerprints.csv retain verification evidence.
'''
 text+=f"\nAll **{m['checks']} checks passed** across **{m['rows']:,} stock-days**. Checks include unequal-denominator synthetic formulas, root and year weighting, paired resampling arithmetic and determinism, source hashes, complete/unique keys, prediction validity, exact fallback, annual reconciliation, exhaustive date-level partitions, finite bootstrap metrics and the unchanged protocol fingerprint. The source audit inventoried **{len(pd.read_csv(OUT/'forward_source_file_inventory.csv'))} local Parquet files**; no forward period was scored.\n"
 (OUT/'README.md').write_text(text)
 artifacts=[p for p in OUT.glob('*') if p.is_file() and p.name!='artifact_fingerprints.json']
 (OUT/'artifact_fingerprints.json').write_text(json.dumps([{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(artifacts)],indent=2))
 print('Findings and artifact fingerprints saved')
if __name__=='__main__':run()
