"""Assemble the intraday research narrative from saved numerical artifacts."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
E=ROOT/'outputs/experiments'
def records(path):
 with path.open() as f:return list(csv.DictReader(f))
def table(rows,columns):
 lines=['| '+' | '.join(label for key,label,fmt in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
 for r in rows:
  values=[]
  for key,label,fmt in columns:
   value=r[key]
   values.append(fmt(value) if callable(fmt) else str(value))
  lines.append('| '+' | '.join(values)+' |')
 return '\n'.join(lines)
def num(x):return f'{float(x):.5f}'
def pct(x):return f'{100*float(x):.3f}%'
def comparison(folder,mapping):
 r=[x for x in records(E/folder/'comparison.csv') if x['period']=='development']
 d={x['candidate']:x for x in r}
 return table([dict(d[k],label=v) for k,v in mapping.items()],[('label','Forecast',None),('rmse_log','Log RMSE',num),('wape','WAPE',pct),('aggregate_bias','Bias',pct)])
def main():
 q=E/'26_quarterly_intraday_audit';m=json.loads((q/'experiment_summary.json').read_text());assert m['all_checks_passed']
 annual=records(q/'annual_metrics.csv');quarters=records(q/'quarter_metrics.csv');devq=[r for r in quarters if int(r['year'])<=2024];by={(r['fold'],r['candidate']):r for r in quarters};wins={}
 for period,years in [('development',[2022,2023,2024]),('confirmation',[2025])]:
  ks=sorted(set(r['fold'] for r in quarters if int(r['year']) in years))
  wins[period]={metric:sum(float(by[(k,'quarterly_ridge')][metric])<float(by[(k,'fixed_daily')][metric]) for k in ks) for metric in ['rmse_log','wape']};wins[period]['quarters']=len(ks)
 allsummary=records(q/'comparison.csv')
 qtable=table([dict(r,label={'fixed_daily':'Daily anchor','annual_ridge':'Annual intraday ridge','quarterly_ridge':'Quarterly intraday ridge'}[r['candidate']]) for r in allsummary],[('period','Period',None),('label','Forecast',None),('rmse_log','Log RMSE',num),('wape','WAPE',pct),('aggregate_bias','Bias',pct)])
 intervals=[r for r in records(q/'metric_intervals.csv') if r['reference']=='fixed_daily' and r['block_dates']=='20' and r['metric'] in ['rmse_log','wape']]
 for r in intervals:
  scale=100 if r['metric']=='wape' else 1;r['effect']=f"{scale*float(r['difference']):+.5f}";r['interval']=f"[{scale*float(r['ci_lower']):+.5f}, {scale*float(r['ci_upper']):+.5f}]";r['metric_label']='WAPE percentage points' if scale==100 else 'Log RMSE'
 itable=table(intervals,[('period','Period',None),('metric_label','Metric',None),('effect','Change versus daily',None),('interval','95 percent interval',None)])
 text='''# NYSE Closing Auction Intraday Forecasting Research Report

Research completed through November 16 2026

## Executive overview

We have built and evaluated an intraday forecasting layer that updates a prior-close daily forecast of official NYSE closing-auction share volume. Work now includes 15:30 and noon forecasts, matched controls, feature-group ablation, correction shrinkage, direct metric-matched uncertainty estimates, a forward-data availability audit, and a quarterly walk-forward procedure audit. This report explains the sequence from its starting point, what each stage added, and what the evidence does and does not establish.

The daily anchor is nonlinear LightGBM. The leading initial intraday update was ridge regression around that anchor. That combination is deliberate: the intraday task is to predict an incremental correction to an already nonlinear daily forecast. A simpler correction can outperform a more flexible one without contradicting the success of the nonlinear daily model. The unchanged annual update remains a research reference; the new quarterly audit evaluates a more frequently refitted procedure rather than automatically replacing it.

Every historical evaluation year has been inspected. Chronological fitting and independent reconstruction reduce leakage and implementation risks, but cannot erase repeated research selection. The results are retrospective forecasting evidence, not a pristine holdout test, causal explanation, or quantified execution benefit. Compatible local auction and feature data stop at the end of 2025. No 2026 period has been scored. [19–26]

## 1 Starting point and the business question

The supplied repository already contained the daily research, quality audits, enriched feature panel, regression benchmark, segmentation experiments, stock examples, options and FOMC studies, written reports, and a nonlinear challenger specification. It also contained prepared intraday checkpoint aggregates. The earlier written reports identified intraday forecasting as an important unbuilt extension. Our contribution was to turn those prepared inputs into a validated update pipeline and investigate its behavior.

The target is official NYSE closing-auction volume in shares for an eligible security and trading date. We ask whether information observed by a specified time during that session improves the forecast made using prior-close information. This differs from predicting which stocks generally trade more: much of the pooled ranking signal comes from persistent differences in size and liquidity. The harder question is whether an update captures deviations from a stock's normal auction volume.

Before building the intraday layer, we locally reconstructed the daily panel, reconciled the frozen ridge forecasts, and reran compact nonlinear challengers against verified row-level references. The selected daily anchor was squared-error LightGBM using 58 daily source features and a median-relative log target. It improved mean development log RMSE from 0.48764 for the historical ridge to 0.46941 on the full eligible daily universe. Direct log levels and one-day log growth did not justify changing the target; four historical calibration corrections did not pass the joint accuracy and bias gate. These earlier stages supplied the fixed daily forecasts used by the intraday experiments. [Daily foundation]

## 2 Data and the forecast-time contract

The daily unit is date and PERMNO. The raw intraday unit adds checkpoint time. Tickers are descriptive rather than a safe permanent identifier. The prepared checkpoints are 10:30, noon, 14:00, 15:00 and 15:30 ET, although the implemented update models so far are noon and 15:30 only.

The original extraction handoff identifies the auction from NYSE uncorrected trades with condition 6. It describes cumulative consolidated regular-session trade aggregates excluding conditions O, Q, 6 and M. We reused that prepared extraction contract rather than constructing a new tick feed. The stored aggregates do not contain individual trade timestamps, so our checks cannot independently certify every extraction cutoff. Point-in-time symbol mapping, corrections, corporate-action conventions, publication delays and revised source records remain important upstream limitations.

The daily model cannot see same-day final volume, closing returns, auction prices or current-session trading. The intraday update can see current trading only through its checkpoint. It cannot see final auction shares or subsequent checkpoints. Calendar inputs may be known in advance; lagged stock inputs and historical checkpoint normalizations end before the forecast date. A FOMC decision released at 14:00 is conceptually available to later updates but has not been added to these models.

Missing auction labels remain missing and are never filled with zero. The documented August 12 2019 anomaly is excluded as a supervised label, while the original panel's history treatment was preserved during reproduction. Intraday training starts in 2021 because that is the earliest retained out-of-time daily-anchor forecast history available to this update pipeline.

The common intraday evaluation contains 1,260,168 regular-session stock-days over 2022–2025. Early-close days are excluded because 15:30 is after those auctions. Noon occurs before an early close but uses the same exclusion for the timing comparison. Missing or invalid regular-session checkpoints fall back exactly to the daily anchor. Full daily tables elsewhere include early closes and therefore have a different universe. [19,20]

## 3 The update model from scratch

Let A be actual auction shares and D the saved daily forecast. The learned update target is log1p(A) minus log1p(D). Ridge or LightGBM predicts that residual. We add the predicted correction to log1p(D), apply the historical update-fitting forecast bounds, then transform back with expm1 and enforce nonnegativity. Missing checkpoints bypass the correction and retain D exactly.

All learned models start with the original 58 daily source features, the daily log forecast and four prior-only controls: historical checkpoint average volume, historical checkpoint average trade count, median auction-to-checkpoint ratio and prior price. This gives 63 control sources. Ten current-session sources expand the full update to 73 sources. Encodings and missing indicators mean source counts differ from matrix-column counts.

| Current group | Source variables |
| --- | --- |
| Volume | Cumulative shares, cumulative dollars, cumulative volume relative to prior daily ADV, cumulative volume surprise relative to prior checkpoints |
| Trade counts | Cumulative trades and trade-count surprise relative to prior checkpoints |
| Recent activity | Prior 30-minute volume and its share of cumulative volume |
| VWAP and price | Checkpoint VWAP and VWAP return from the prior close |

Historical normalizations use preceding stock rows, excluding the current row. The 20-row window does not search backward until it collects 20 nonmissing observations. Market and stock source fields are retained according to the existing timing contract.

Missing covariates remain explicit until fitting. Historical preprocessing can log-transform highly skewed nonnegative variables, clip continuous inputs at fitted 0.5 and 99.5 percentiles, fill with historical medians, center and scale, and add missing flags when observed during fitting. Binary missing values use zero plus an indicator where needed. Categorical levels and unknown-level handling are learned historically. Forecast bounds use historical 0.1 and 99.9 percentiles of log auction shares, a separate operation from feature clipping. No evaluation-year distribution determines these parameters. [19,22]

## 4 Initial annual training and validation

An intraday model must not train on in-sample daily forecasts that are more accurate than the daily predictions it will receive in use. We used annually out-of-time daily forecasts beginning in 2021, including an additional historical 2021 daily model fitted through 2020. Each update-year forecast uses only earlier update history.

For the original 2022 update fold, tuning fits January–September 2021 and validates October–December 2021. The final update refits on 2021. Later annual folds tune with the previous full year as validation and earlier years from 2021 as fitting history, then refit through the preceding year. The update history expands from 2021, unlike the daily model's rolling five-year fitting window.

Ridge tests seven strengths from 1e-7 through 1e-1. LightGBM tests 15, 31 and 63 leaves, with historical early stopping and a capped tree budget. Shared historical samples cap tuning, validation and refit rows at 250000, 120000 and 350000. Evaluation years are scored in full. Shared preprocessing and bounds make the learned controls comparable. The original minimum-25-tree final-refit convention can override very small validation-selected tree counts; that caveat affects the learned tree control, not the fixed daily reference or ridge.

The initial comparison includes the fixed daily anchor, a cumulative-volume ratio rule, ridge update, daily-only tree control and full intraday tree update. The simple ratio multiplies current cumulative volume by a prior historical auction/checkpoint ratio. Its poor results show that observing current volume is not sufficient to justify a naive scaling rule. [19]

## 5 Initial 15 30 results

The following development metrics average 2022–2024 annual results equally. Bias is forecast share total divided by actual total minus one. Negative bias indicates underprediction.

'''
 r=records(E/'19_intraday_1530/intraday_comparison.csv');mapping={'fixed_daily':'Daily anchor','checkpoint_ratio_rule':'Historical ratio rule','ridge_update':'Ridge update','lgbm_daily_control':'Daily-only tree control','lgbm_intraday_update':'Intraday LightGBM'}
 text+=table([dict(x,label=mapping[x['candidate']]) for x in r],[('label','Forecast',None),('development_rmse_log','Log RMSE',num),('development_wape','WAPE',pct),('development_bias','Bias',pct)])
 text+='''

Ridge has the strongest aggregate point estimates. The 15:30 LightGBM update passed the original information gate against both the fixed daily anchor and its matching daily-only tree control. That gate was defined for the tree model; favorable ridge results should not be relabeled as a predeclared ridge promotion decision. The daily-only ridge comparison was implemented later to address that gap.

The 15:30 coverage rate is approximately 99.98%. All 124 checks passed, including exact keys and labels, reference preservation, missing-checkpoint fallback, recalculated metrics, saved-model reconstruction and independent historical-window calculations on 16228 rows from 16 stocks. We saved coefficients, tree models, preprocessing, feature order, realized samples and input fingerprints. [19]

## 6 Noon and the timing comparison

We repeated the same design with noon inputs and prior noon histories. The training sample keys and order originate from the 15:30 study, restricted where noon observations are unavailable; current features are rebuilt from noon data. Evaluation keys, targets and daily anchors match exactly. Noon coverage is approximately 99.94%.

Noon ridge reaches development log RMSE 0.46459 and WAPE 34.85%, compared with 0.45995 and 34.59% at 15:30. Noon captures approximately 51% of the 15:30 ridge gain over the daily anchor in log RMSE. Noon LightGBM does not pass the full improvement gate against the fixed daily anchor. Paired date-based comparisons favor 15:30 for both update families in development and 2025.

The first noon build failed an independent history check. Materializing the source join before rolling calculations corrected 137 false missing-checkpoint flags. Every noon model was rerun. Full checkpoint-availability audits then matched source data for both noon and the unchanged 15:30 panel. This issue and correction are retained in the research record. Both final checkpoint studies passed 124 checks. [20]

## 7 Robustness and the matched ridge control

The frozen-forecast robustness audit found that annual 15:30 ridge improves log RMSE in all four years and in 32 of 36 development months. Monthly WAPE improves in 31 of 36 development months. In 2025 it improves log RMSE in all twelve months and WAPE in eleven. Five- and twenty-date block log-MAE intervals remain favorable. These are conditional retrospective comparisons, not independent monthly hypothesis tests.

Subgroup results are uneven. Month-end and low-liquidity development log accuracy can be worse than the daily anchor, even when overall metrics improve. Large actual auctions remain substantially underpredicted. Actual-volume tail membership is an after-the-fact diagnostic, never an input or a forecast-time routing definition. [21]

We then fitted a daily-only ridge control with the same exact historical rows, target, preprocessing, tuning budget, clipping and fallback as the full ridge, but without the ten current-session sources. Historical checkpoint controls remain in both models. Each version selects its own penalty on earlier validation data.

'''
 text+=comparison('22_intraday_ridge_control',{'fixed_daily':'Daily anchor','ridge_daily_control':'Daily-only ridge','ridge_update':'Full intraday ridge'})
 text+='''

The full update improves log RMSE and WAPE against both references in every year from 2022 through 2025. Its advantage over the matching control remains favorable in twenty-date block log-MAE intervals. This supports incremental predictive value from current trading within the matched design, rather than attributing the gain to another regression layer alone. It does not identify a causal market mechanism. All 60 checks passed; the original full-ridge tuning choices, coefficients and forecasts were reproduced without alteration. [22]

## 8 Feature group ablation

We added each of the four current-session groups individually to the daily-only control and removed it individually from full ridge. All eight variants use the same historical samples, source-wise preprocessing, target and bounds; each tunes its own penalty. Source features and their missing indicators move together. No group choice uses 2025 outcomes.

'''
 contrasts=[r for r in records(E/'23_intraday_feature_ablation/group_contrasts.csv') if r['period']=='development'];bygroup={(r['group'],r['comparison']):r for r in contrasts};rows=[]
 for group,label in [('volume','Volume'),('trades','Trade counts'),('recent_activity','Recent activity'),('price','VWAP and price')]:
  rows.append(dict(label=label,alone=bygroup[group,'add']['rmse_log'],drop=bygroup[group,'drop']['rmse_log_change']))
 text+=table(rows,[('label','Group',None),('alone','Added alone log RMSE',num),('drop','Removal changes full log RMSE',lambda x:f'{float(x):+.6f}')])
 text+='''

Volume alone recovers about 89% of the full ridge's log-RMSE improvement versus the daily-only control. Trade counts and recent activity also help alone, with recent activity having the largest removal effect on log RMSE. These shares cannot be summed: correlated groups substitute for one another. Price-only log-MAE intervals cross zero, and removing price barely changes the headline results in this ridge specification.

Removing trade counts illustrates a metric tradeoff. It slightly worsens development log RMSE but improves WAPE by about 0.038 percentage points and improves daily log MAE; bias becomes more negative. It is therefore not evidence that the group is universally unnecessary. Every tested removal leaves the month-end or low-liquidity weaknesses unresolved relative to the daily anchor on the problematic metrics. All 179 checks passed across the common universe. [23]

## 9 Correction shrinkage

We next tested whether partial use of the update reduces overcorrection. Strength zero retains the daily forecast; one retains full intraday ridge; intermediate strengths interpolate their log1p forecasts. Exact endpoints are preserved and no extra clipping is applied. Four variants shrink globally, on disjoint month-end days, for low-liquidity stocks, or in month-end/liquidity cells. Non-targeted rows retain full ridge.

Strengths on a 0 to 1 grid in 0.1 steps are selected from historical validation only. A cell needs at least eight dates and 500 rows; otherwise it falls back to global strength. Q4 2021 contains only three month-end dates, so the first fold does not estimate a separate month-end coefficient. The same validation period selected the original penalty and strength; this is joint historical tuning, not an independent calibration split.

'''
 text+=comparison('24_intraday_shrinkage',{'ridge_update':'Full ridge','global_shrink':'Global shrinkage','month_end_shrink':'Month-end only','low_liquidity_shrink':'Low liquidity only','month_end_liquidity_shrink':'Combined cells'})
 text+='''

The development-only adoption decision was saved before scoring 2025. No variant passed the predefined joint gate. Global shrinkage's log-RMSE gain was below 0.002 and its paired log-MAE interval crossed zero. Low-liquidity shrinkage improved development log RMSE but worsened WAPE by only about 0.0061 percentage points. That is a very small tradeoff, but the predeclared rule required no worsening. We did not relax it afterward. The combined variant also failed that requirement.

In 2025 every shrinkage variant had worse log RMSE than full ridge, although global shrinkage improved WAPE. Low-liquidity correction strength therefore appears time-dependent rather than a permanent subgroup fix. The study retained full ridge as its reference, saved all variants, and passed 152 checks. No economically justified WAPE noninferiority margin has been agreed. The zero-worsening research gate should not be mistaken for a demonstrated business threshold. [24]

## 10 Metric matched statistical evaluation

Earlier intervals measured daily log MAE while headline selection emphasized log RMSE. We corrected that mismatch with a dedicated evaluation module for the frozen daily-versus-full-15:30 contrast. Each replicate recomputes annual RMSE from squared log errors, WAPE from absolute error and actual-share totals, and signed aggregate bias from share totals. Annual metrics are then averaged equally. Absolute annual bias is taken within each year before averaging.

Using 5000 circular moving-block replicates with a prespecified twenty-trading-date primary length, the development intraday-minus-daily log-RMSE difference is -0.00954 with interval [-0.01135, -0.00781]. WAPE improves by 0.876 percentage points with interval 0.717 to 1.035 points. Absolute annual aggregate bias improves by 2.668 points with interval 2.081 to 3.280 points. The five- and sixty-date sensitivity results also favor intraday on overall errors. [25]

Every sampled date carries both forecasts and all stocks together. This preserves cross-sectional dependence within dates and some serial dependence. Circular wrapping introduces an artificial year-end/start adjacency; stationarity and weak-dependence assumptions remain limitations. These intervals condition on fitted forecasts and omit refitting and repeated model-selection uncertainty. They are approximate descriptive intervals, not proof of future performance.

Predefined size, liquidity, history and disjoint-event groups retain annual counts and explicit missing groups. Subgroup intervals are suppressed if any year has fewer than thirty dates, fewer than 500 stock-days or invalid denominators. Thousands of stocks on a handful of expiry dates do not create thousands of independent event observations. Secondary and subgroup intervals are unadjusted for multiplicity and exploratory. All 94 checks passed, including synthetic unequal-denominator tests, annual reconciliation and resampling arithmetic. [25]

## 11 Why a new forward test is unavailable locally

A metadata-only audit examined 57 local Parquet files without reading new auction-volume outcomes. Auction records, CRSP, 15:30 checkpoints, market context, sector context and calendar all end on December 31 2025. A FOMC filename extending into 2026 does not supply the required target and feature coverage. No new forward period was generated or scored.

If new data become available, continuity, label validity, extraction cutoffs, prior-use history and publication delays must be checked before scoring. A provenance attestation is needed because local file presence cannot prove that nobody used outcomes elsewhere. The test window, training schedule, primary metric, operational acceptance criteria and stopping rule must be fixed before the test. Historical unused data can be useful if non-use is established; prospective locked forecasts provide stronger protection. Already inspected 2025 cannot become fresh by renaming a split. [25]

## 12 Quarterly walk forward procedure audit

Because new forward data are unavailable, the next bounded step tests stability of the fitting procedure within existing history. We implemented sixteen outer quarters from 2022Q1 through 2025Q4, keeping the full 73-source 15:30 ridge architecture fixed. No shrinkage, feature selection or new external signal enters this audit.

For each evaluation quarter, the immediately preceding complete quarter selects one of the seven ridge penalties. Tuning fits all earlier update history from 2021; final refit then uses available history through that validation quarter. For 2022Q1, this means fit January–September 2021, validate Q4 2021, refit on 2021 and predict Q1 2022. The optional third calibration window discussed earlier is not needed because no shrinkage is fitted here. Every subsequent quarter advances the same rule.

The saved daily anchor remains annually fitted and out of time; this is quarterly retraining of the intraday correction, not quarterly retraining of the daily LightGBM. Daily lags and prior-only checkpoint histories continue updating during evaluation. All preprocessing, encoding, clipping and coefficients are relearned inside each historical fit. Evaluation-quarter labels never tune that quarter's model.

Samples retain the original 250000, 120000 and 350000 caps. Ordered, single-thread reservoir sampling has fixed fold seeds, and realized date/PERMNO keys are saved in fitting order against a fingerprinted source panel. Complete forecasts, 16 coefficient files, tuning and final preprocessors, bounds, fold windows and every validation trial are retained. The audit reconstructs each saved model and checks exact daily anchors, eligible keys, checkpoint fallback and written forecasts.

'''
 text+=qtable+'\n\n'+itable
 text+=f'''\n\nThe quarterly update improves log RMSE versus daily in **{wins['development']['rmse_log']} of {wins['development']['quarters']} development quarters** and **{wins['confirmation']['rmse_log']} of {wins['confirmation']['quarters']} confirmation quarters**. WAPE improves in **{wins['development']['wape']} of {wins['development']['quarters']}** and **{wins['confirmation']['wape']} of {wins['confirmation']['quarters']}** respectively. Quarter counts summarize stability; quarters are not treated as independent replications for a significance test. Annual metrics concatenate the full quarterly predictions before calculating each year's score. [26]\n\n'''
 text+='''The annual intraday reference is retained on exactly the same test rows. Comparing quarterly with annual updates changes historical samples, fitting windows and forecast bounds as well as refit frequency, so the contrast describes complete procedures rather than a pure frequency experiment. Secondary quarterly-versus-annual intervals are reported separately, not used for automatic promotion. In development the quarterly-minus-annual log-RMSE difference is -0.001069 with a 95 percent interval of [-0.002129, +0.000052]; the WAPE difference is -0.0660 percentage points with interval [-0.1776, +0.0445]. Both include zero. The 2025 log-RMSE difference is -0.000238 with interval [-0.000923, +0.000441], also including zero. The small aggregate point-estimate advantage therefore does not establish that quarterly refitting is superior to the annual intraday procedure.

The predefined subgroup diagnostics leave the earlier weaknesses unresolved. Quarterly development month-end log RMSE is 0.66217 and WAPE 47.70%, compared with 0.66095 and 47.18% for daily. In the low-liquidity group, quarterly log RMSE is 0.52813 versus daily 0.52603, while WAPE improves to 36.32% from 37.54%. These are equal-year exploratory group means, not independent subgroup significance tests.

Quarterly intervals reuse the direct metric-matched date-block calculation with twenty dates primary and five and sixty as sensitivity checks versus daily. They condition on the newly fitted quarterly forecasts and do not fully quantify the uncertainty of repeatedly rerunning the training procedure. Nested chronological tuning supports cleaner procedure assessment, but model architecture, feature choices and evaluation years were already inspected. This audit cannot undo that research-selection history.

'''
 text+=f"All **{m['checks']} checks passed** across **{m['rows']:,} evaluation rows** in sixteen non-overlapping quarters. The frozen design hash was verified unchanged. [26]\n\n"
 text+='''## 13 Current conclusion and practical next steps

The most defensible result is that existing current-session trading information adds predictive value over the daily anchor in chronological historical comparisons. The matched ridge control supports that conclusion, and activity-related inputs account for most of the observed gain. The strongest weaknesses remain metric tradeoffs, large-auction underprediction, sparse event evidence, publication-time assumptions and stability across periods.

The quarterly audit adds evidence about the fixed procedure without supplying a fresh holdout. It should not be used to restart an unrestricted search over features, boundaries or quarter lengths. The next decision is to specify the intended operational objective and acceptable WAPE/bias tradeoffs, then either obtain a documented unused forward period or collect locked prospective forecasts. Other prepared checkpoints remain possible future experiments, not completed results in this report.

Neither the annual nor quarterly result establishes execution-cost savings, optimal participation decisions, causality or production readiness. Report typical proportional accuracy, raw-share error and total-volume bias together. Keep the original annual model and all negative experiments intact so later claims remain traceable.

## Appendix Source and artifact map

All paths are relative to the project root. The source CSVs and JSON manifests, rather than rounded prose, are the authoritative numerical record.

Daily foundation. outputs/reproduction/01_daily_panel, 02_benchmark_reconciliation and 03_compact_nonlinear; outputs/experiments/17_target_formulations and 18_error_diagnosis_calibration. Original data provenance is in blackrock_data/BlackRock Capstone Handoff — Predicting NYSE Closing Auction Volume.docx. Original notebooks and reports were preserved.

19. outputs/experiments/19_intraday_1530. Annual 15:30 design, comparisons, checkpoints, models, preprocessing, samples, coverage, bootstrap and 124 validation checks. Entry point notebooks/build_intraday_1530_update.py.

20. outputs/experiments/20_intraday_noon. Noon comparisons, source-availability correction record, matched timing checks and 124 validation checks. Entry points notebooks/build_intraday_noon_update.py and compare_intraday_times.py.

21. outputs/experiments/21_intraday_robustness. Annual/monthly and group diagnostics, 5/20-date log-MAE sensitivity, and frozen-forecast reconciliation. Entry point notebooks/audit_intraday_robustness.py.

22. outputs/experiments/22_intraday_ridge_control. Exact matched daily-only ridge experiment and 60 checks. Entry point notebooks/fit_intraday_ridge_control.py.

23. outputs/experiments/23_intraday_feature_ablation. Group definitions, eight variants, annual and segment results, paired intervals and 179 checks. Entry points notebooks/ablate_intraday_ridge_features.py and summarize_intraday_ablation.py.

24. outputs/experiments/24_intraday_shrinkage. Frozen variant plan and development decision, historical strengths, sparse-cell fallback, annual and segment results, and 152 checks. Entry points notebooks/shrink_intraday_ridge_update.py and summarize_intraday_shrinkage.py.

25. outputs/experiments/25_frozen_intraday_evaluation. Written evaluation protocol, direct RMSE/WAPE/bias intervals, 94 checks, subgroup suppressions and metadata-only unused-period audit. Entry points notebooks/evaluate_frozen_intraday.py, audit_unused_intraday_period.py and summarize_frozen_intraday_evaluation.py.

26. outputs/experiments/26_quarterly_intraday_audit. Frozen quarterly plan, fold manifest, all penalty trials, realized sample keys, fitted preprocessors and coefficients, full quarter predictions, annual/quarter/segment metrics, direct metric intervals and verification summary. Entry point notebooks/quarterly_intraday_walkforward.py. The companion reader notebook displays saved results without retraining.
'''
 (OUT/'intraday_research_report.md').write_text(text)
 print('Report source written:',len(text.split()),'words')
if __name__=='__main__':main()
