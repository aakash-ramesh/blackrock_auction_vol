"""Assemble the historical evidence trail and the new, separately labeled diagnostics."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;T=OUT/'tables';EDA=ROOT/'outputs/eda'
def read(p):return pd.read_csv(EDA/p)
def md(d):
 def f(v):
  if pd.isna(v):return '—'
  if isinstance(v,(float,np.floating)):return f'{v:,.4f}'
  if isinstance(v,(int,np.integer)):return f'{v:,}'
  return str(v).replace('|','/')
 lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
 return '\n'.join(lines+['| '+' | '.join(f(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])
b=read('07_temporal_baselines/overall_baseline_metrics.csv');b=b[b.model.eq('median20_fallback')][['test_year','observations','rmse_log','wape','aggregate_bias']]
f=read('09_global_finalists/global_finalist_metrics.csv');chosen='rolling5_residual_auction_calendar_liquidity';f=f[f.candidate.eq(chosen)][['test_year','observations','rmse_log','wape','aggregate_bias']]
comparison=b.merge(f,on='test_year',suffixes=('_baseline','_selected'));assert comparison.observations_baseline.eq(comparison.observations_selected).all()
comparison['relative_rmse_improvement_pct']=100*(1-comparison.rmse_log_selected/comparison.rmse_log_baseline);comparison['relative_wape_improvement_pct']=100*(1-comparison.wape_selected/comparison.wape_baseline)
comparison.to_csv(T/'historical_baseline_comparison.csv',index=False)
finalists=read('09_global_finalists/global_finalist_selection.csv');finalists.to_csv(T/'historical_finalist_selection.csv',index=False)
segments=read('10_segmentation_tests/segmentation_decision_table.csv');segments.to_csv(T/'historical_segmentation_decision.csv',index=False)
family=read('08_global_robustness/economic_feature_group_summary.csv');family.to_csv(T/'historical_feature_family_ablation.csv',index=False)
chosenrow=finalists[finalists.selected_global_champion].iloc[0];best=finalists.loc[finalists.development_rmse_log.idxmin()];dev=comparison[comparison.test_year.between(2022,2024)];con=comparison[comparison.test_year.eq(2025)].iloc[0]
raw=pd.read_csv(T/'feature_associations.csv');cat=pd.read_csv(T/'covariate_catalog.csv');coeff=read('09_global_finalists/champion_coefficient_stability.csv')
roles=read('06_feature_validation/feature_validation_and_roles.csv')[['variable','recommended_role','quality_caution','annual_sign_consistency']]
bridge=cat[cat.selected].merge(raw[['feature','paired_stock_days','primary_pearson_raw','within_stock_pearson_raw','primary_spearman','binary_median_ratio','positive_years','negative_years']],on='feature').merge(roles,left_on='feature',right_on='variable',how='left').merge(coeff,left_on='feature',right_on='source_feature',how='left')
bridge['coefficient_warning']='Historical coefficients predict a log residual after training-fold preprocessing. They are not univariate share-volume effects. Source-level category coefficient averages are not standalone directions.'
bridge.to_csv(T/'selected_feature_dictionary.csv',index=False)
evidence=[
(1,'Reliable sample','2015–2025 auction labels; missing outcomes retained as missing; one source-anomaly date excluded.','Prevents a model from learning artificial zero auctions.','01_data_quality/annual_target_coverage.csv'),
(2,'Target economics','Skew, auction growth, calendar differences, stock scale and persistence.','Supports robust historical baselines, transformed estimation, and multiple accuracy metrics.','02_target_eda/headline_target_findings.csv'),
(3,'Information boundary','Only prior-close data or known calendar information approved. Correlated predictors identified.','Restricts deployable daily inputs and motivates shrinkage rather than naive coefficient interpretation.','03_covariate_audit/feature_catalog.csv'),
(4,'Initial relationship analysis','Marginal, within-stock, annual, and joint associations for initial approved predictors; transformed continuous target.','Separates stock scale from changes within a stock; not a forecast-selection test.','04_relationships/numeric_relationship_summary.csv'),
(5,'Feature construction','99 point-in-time-safe derived features across seven families.','Builds a common master panel; preserves variables for later empirical tests.','05_feature_engineering/enriched_feature_catalog.csv'),
(6,'Feature role audit','Predictor, context, caution and segmentation roles; no feature deletion.','Avoids selecting only on pooled correlations or optimistic IID p-values.','06_feature_validation/feature_validation_and_roles.csv'),
(7,'Forward baseline comparison','Historical forecasts and global ridge on the same eligible annual evaluation universe.','Shows incremental predictive value over a deployable trailing-median baseline.','07_temporal_baselines/overall_baseline_metrics.csv'),
(8,'Specification and economic-family tests','History windows, penalties, sample sizes, targets/weights, ablation, placebo, drift and tail errors.','Identifies persistent auction/calendar signal and useful target normalization; exposes tradeoffs.','08_global_robustness/economic_feature_group_summary.csv'),
(9,'Global model selection','58-feature five-year rolling log-residual ridge selected within a 0.002 development-RMSE tolerance.','Chooses parsimony among practically tied candidates, rather than claiming lowest error on every metric.','09_global_finalists/global_finalist_selection.csv'),
(10,'Architecture challenge','Interactions, hard splits and targeted hybrids compared against the frozen model.','No tested replacement clears the practical improvement and paired-uncertainty requirements.','10_segmentation_tests/segmentation_decision_table.csv'),
(11,'Reconstructed backtest','Train, validation and test performance for the frozen model.','Confirms forward performance and clarifies that train/test periods have different regimes.','11_frozen_global_backtest/frozen_global_train_validation_test_metrics.csv'),
(12,'Stock-level behavior','Six selected securities with annually refitted forward predictions; 2015 warm-up only.','Checks practical behavior and reveals unequal performance, without establishing universe-wide representativeness.','12_selected_stock_predictions/selected_stock_walk_forward_metrics.csv'),
(13,'Daily prediction errors','Unsmoothed representative stock-years, selected near each stock’s median annual log RMSE.','Exposes missed spikes that smoothed charts can hide; does not explain their causes by itself.','13_representative_stock_year_daily/representative_stock_years.csv')]
mapdf=pd.DataFrame(evidence,columns=['notebook','question','analysis_conducted','decision_role','saved_evidence']);mapdf.to_csv(T/'notebook_evidence_map.csv',index=False)
audit=[]
for n in range(1,14):
 p=next((ROOT/'notebooks').glob(f'{n:02}_*.ipynb'));nb=json.loads(p.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];errors=[o.get('ename','') for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error'];audit.append(dict(notebook=n,file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),code_cells=len(code),cells_with_execution_count=sum(c.get('execution_count') is not None for c in code),saved_error_outputs=';'.join(errors)))
pd.DataFrame(audit).to_csv(T/'notebook_review_manifest.csv',index=False)
# Existing analysis gap is specific: several calendar/industry summaries already use raw medians.
gap=pd.DataFrame([
['02','Raw target distribution, persistence, size/industry and calendar summaries','Not an exhaustive one-input-at-a-time raw-source study.'],
['04','Sixteen continuous/ordered predictors; pooled/within and annual relationships, mostly transformed. Binary/category summaries include raw target medians.','Continuous Pearson/within and binned means mainly use log target and selected transformed predictors, not all raw inputs in shares.'],
['06','All 99 daily-safe features audited using ranks, log/residual diagnostics and grouped separation.','Broad coverage, but primarily engineered features and transformed/ranked outcomes, not a complete natural-unit source-covariate analysis.'],
['New','193 diagnostic specifications: 99 engineered daily features, 31 raw same-day inputs, 31 prior-close raw counterparts, 2 auction-outcome diagnostics, and 30 checkpoint-specific associations.','Retrospective, descriptive extension. It does not select or refit a model or create a new holdout.']
],columns=['stage','already_available_or_added','remaining_interpretation_limit']);gap.to_csv(T/'relationship_analysis_gap_audit.csv',index=False)
# Family-level bridge intentionally distinguishes economic intuition from predictive evidence.
intuition=pd.DataFrame([
['Auction history','Previous auction volume and previous auction/daily-volume share','19','Typical auction liquidity persists. Different horizons describe level, change and instability.','Highest rank associations; removing this family causes the largest log-error deterioration.','Use the median-20 offset plus regularized corrections.'],
['Calendar','Trading dates, expiry schedule, session length, month/quarter/year boundaries','23','Benchmark-related and expiry-related flows may concentrate near known dates.','Calendar ablation has a large deterioration. Binary associations overlap and are not causal effects.','Keep known calendar inputs.'],
['Liquidity','Daily volume, price-volume measure, shares outstanding, absolute returns, bid/ask, high/low/close','16','Trading capacity, turnover and illiquidity help distinguish normal auction scale and conditions.','Strong isolated volume relationships; liquidity’s extra gain is modest and metric-dependent. The 58-feature finalist beats the 42-feature finalist on the primary criterion.','Keep this family, but do not claim every member was independently validated as necessary.'],
['Size/history (separate family)','Market cap, shares, coverage/age, cross-sectional size ranks','0','Scale matters economically, but auction and liquidity history can already encode it.','Strong pooled size association weakens substantially within stock. Extra family adds little in the tested conditional models.','No separate family in the selected specification. Shares still enter turnover.'],
['Returns/price (separate family)','Returns, closing prices, momentum and volatility','0','Information and risk may influence activity, but signed returns need not align monotonically with unsigned auction volume.','Prior signed raw return is weak and changes annual sign. Family adds little in the tested linear specification.','Omit the separate family. Price and absolute-return information still enter liquidity proxies.'],
['Market/sector state','Aggregate returns, volume, dispersion, breadth and ranges','0','Common conditions may affect market-wide or industry-wide demand.','Some univariate associations exist, but tested incremental gains beyond the core are small.','Omitted from the parsimonious selected model, not declared economically irrelevant.'],
['Security/corporate actions','Security classifications, dividends, distributions, price adjustment factors','0','Eligibility and corporate events may affect the share base or trading needs.','Several raw security fields are constant by construction; rare-event associations are not reliable proof of incremental forecast value.','Keep source context; separate model family is omitted.']
],columns=['economic_family','raw_input_bridge','selected_features','economic_hypothesis','evidence_and_caution','model_decision']);intuition.to_csv(T/'economic_family_bridge.csv',index=False)

story=f'''# Analysis story leading to the current model

The coherent story is: **establish a trustworthy prior-close forecasting problem, learn a stock’s normal auction scale, add calendar and liquidity corrections, and retain one regularized global model because extra complexity has not earned its place.**

This is a synthesis of the work through the first 13 notebooks. The new raw-covariate analysis is a separate retrospective diagnostic. It was not part of the original selection sequence and should not be presented as though it selected the existing model. No presentation slides or research notebooks were changed.

## 1. The sample and target come before the model

The outcome is official closing-auction share volume for a stock and trading date. Usable labels begin in 2015; earlier CRSP history supplies stock-market lags. Missing labels are not zero auctions. Excluding the source anomaly leaves 3,499,401 eligible stock-days. Coverage varies across years and stock sizes, so reporting only one aggregate score would hide an important source of variation.

The outcome is strongly right-skewed: its median is 55,226 shares, its mean is about 190,445, and its 99th percentile is 2,075,293. A log estimation scale reduces the domination of unusually large auctions, while raw-share metrics preserve economic scale. This is an estimation choice, not a change to the business target.

## 2. Persistence suggests forecasting normal auction size first

The initial target analysis finds a pooled lag-1 correlation of 0.894 on the log scale, but a median stock-level lag-1 correlation of only 0.459. Much of the apparent persistence distinguishes high-volume stocks from low-volume stocks. There is nevertheless useful within-stock history: the initial relationship study reports a within-stock correlation of 0.682 for the transformed 20-auction median.

The resulting intuition is to start from a stock’s recent typical auction and predict a correction. It also makes a trailing-median forecast a demanding baseline. We cannot claim useful modeling progress merely by ranking big stocks above small stocks.

## 3. Feature auditing defines what can be predicted honestly

Realized current-day CRSP outcomes and intraday checkpoints cannot enter the daily forecast made using prior-close information. Calendar flags known in advance can. The engineered panel retains 99 daily-safe features; the feature-role audit annotates all of them and does not delete features on the basis of a p-value.

The relationship work distinguishes pooled scale, within-stock variation, annual stability, and association with historical-baseline errors. Important predictors are also highly correlated: the reported rank correlation between the five-auction mean and 20-auction median is about 0.954, and that between lagged daily volume and its 20-day average is about 0.953. This makes ridge a sensible transparent benchmark. The first 13 notebooks do not establish that ridge dominates all nonlinear model classes.

## 4. The first predictive question is whether a model beats recent history

The evaluation uses chronological annual forecasts, training-only preprocessing, historical tuning, and a fallback for stocks lacking enough auction history. The deployable median-20 reference therefore scores the same eligible universe rather than quietly dropping cold starts.

The table below joins the saved reference and selected-model results. This is a comparison assembled now, not a claim that the final model already existed at the first baseline stage. Counts agree in every year.

{md(comparison[['test_year','observations_selected','rmse_log_baseline','rmse_log_selected','wape_baseline','wape_selected','aggregate_bias_selected']])}

Across the 2022–2024 development years, the mean annual log RMSE falls from {dev.rmse_log_baseline.mean():.4f} to {dev.rmse_log_selected.mean():.4f}, about {100*(1-dev.rmse_log_selected.mean()/dev.rmse_log_baseline.mean()):.1f}% relative improvement. Mean annual WAPE falls from {100*dev.wape_baseline.mean():.2f}% to {100*dev.wape_selected.mean():.2f}%, about {100*(1-dev.wape_selected.mean()/dev.wape_baseline.mean()):.1f}% relative improvement. These are averages of annual metrics, not pooled error estimates. WAPE is aggregate absolute error divided by aggregate actual volume; it is not the average stock-day percentage error.

## 5. Robustness and economic-family tests narrow the specification

The specification audit tests expanding versus rolling histories, penalty selection, training samples and seeds, feature families, target formulations, weighting, a shuffled-target placebo, calibration, tails and drift.

The largest family evidence favors auction history and calendar information. In the saved exploratory four-year summary, full-model log RMSE is 0.4870, removing calendar raises it to 0.5606, and removing auction history raises it to 0.6496. An auction-history-plus-calendar model reaches 0.4890, close to the full 99-feature model. These particular family results average 2022–2025, so use them as exploratory context rather than the later development-only selection criterion.

Liquidity’s increment is smaller and depends on the metric: removing it mildly worsens log RMSE in that audit while improving WAPE. We should not describe every liquidity variable as individually proven necessary. The combined finalist tests are what support retaining the liquidity family.

A median-20 log-residual formulation outperforms the direct log-level formulation in the objective audit. A five-year rolling history is competitive with expanding and three-year histories; the differences are modest. More recent data and a stock-specific offset fit the observed changes in auction participation, but the comparison does not identify a unique economically optimal memory length.

The shuffled-target placebo loses most useful ranking ability (Spearman about 0.103 and log RMSE about 1.887 in its 2025 sample), supporting that real feature-target alignment matters. This negative control does not by itself certify every aspect of the pipeline.

## 6. The chosen model is the parsimonious practical tie

The selected specification is one global ridge model using a rolling five-year training window, uniform stock-day weights, and 58 source features: 19 auction-history, 23 calendar, and 16 liquidity features. Source-feature count differs from the expanded design-matrix width because preprocessing adds dummy variables and missingness indicators.

The selection criterion is mean annual log RMSE over 2022–2024, with a tolerance of 0.002. The best 99-feature finalist scores {best.development_rmse_log:.6f}; the selected 58-feature model scores {chosenrow.development_rmse_log:.6f}. Their difference is only {chosenrow.development_rmse_log-best.development_rmse_log:.6f}, so the smaller model wins the pre-stated practical-tie rule. The 42-feature auction/calendar model scores 0.490588 and lies outside that tolerance. The selected model does not minimize every reported metric.

For baseline B(i,t), the stock’s lagged 20-auction median with a historical-training-median fallback, the estimated outcome is:

`r(i,t) = log(1 + A(i,t)) − log(1 + B(i,t))`

The ridge predicts this residual using fold-specific transformations, clipping, imputation, scaling, categorical encoding, and shrinkage fitted only on historical data. Prediction returns to shares using:

`A_hat(i,t) = max(0, exp(log(1 + B(i,t)) + r_hat(i,t)) − 1)`

The implementation also clips the final log forecast to the training log-target 0.1% and 99.9% quantiles before transforming back. This is a log1p residual, not exactly `log(A/B)`. Exponentiating a log-space point forecast does not generally recover the conditional arithmetic mean in shares, which is relevant when discussing underprediction of large auctions.

## 7. Strong heterogeneity does not automatically justify separate models

The architecture challenge compares pooled interactions, independent splits by size/liquidity/auction intensity, a size–liquidity grid, and event/cold-start specialists against the frozen global benchmark.

The best size split lowers development log RMSE by approximately 0.001162, below the 0.002 practical threshold. Its paired log-MAE difference has a date-block interval from about −0.000711 to +0.000129, which includes zero. That interval concerns log MAE, not the RMSE threshold. No tested replacement clears all the decision requirements. Keeping one model is a supported complexity decision, not proof that heterogeneity or nonlinear interactions do not exist.

The event specialist performs better in 2025 on some metrics but worse in the development aggregate. Choosing it from that confirmation result would reverse the stated selection discipline.

## 8. Backtests and daily case studies define the limits of the choice

The reconstructed backtest confirms the frozen model’s forward results. Train, validation and test cover different periods and samples, so lower test error than training error cannot by itself be interpreted as an absence of overfitting. The historical regime and the sampling differ.

The six selected-stock examples make the behavior tangible. Their 2015 observations are warm-up actuals, with predictions beginning in 2016. For subsequent years the model is refitted on earlier dates; daily lagged inputs update during each prediction year. The 2016 penalty is a documented fallback because there is insufficient earlier history for the nested tuning used later. These long histories visualize a specification chosen retrospectively in the later research, not a claim it was selected prospectively in 2015.

The unsmoothed plots choose each stock’s representative year mechanically near its median annual log RMSE and mark its largest proportional misses. They reveal daily spikes hidden by smoothing. They do not explain a miss as an earnings or rebalance effect without additional event evidence, and the six continuously observed stocks are not a representative estimate of all cold starts and delistings.

The selected model still underpredicts large and difficult auctions. The saved pooled 2022–2025 diagnostic reports approximately −41.1% aggregate bias in the largest 1% of actual auctions, versus −5.7% in the lower 99%. Small-stock and short-history errors are also larger. Overall development aggregate bias averages about −14.4%; the 2025 confirmation-year bias is about −9.6%. A good global ranking score therefore does not establish adequate calibration for every execution use.

## What the economic narrative can honestly claim

{md(intuition)}

A particularly important distinction is that dropping the separate returns/price family does **not** remove all price or return information. The retained Amihud proxy uses absolute return and dollar volume; quoted spread uses bid and ask; high–low range uses high, low and close. Turnover uses shares outstanding. Similarly, historical auction and trading activity encode stock scale even without a standalone market-cap predictor.

Individual association, economic plausibility, incremental predictive information, and model-selection evidence answer different questions. The final model was chosen by combined forward comparisons, not by collecting every variable with a large marginal correlation.

## Existing relationship-analysis coverage and the new gap fill

{md(gap)}

The new primary outcome is always auction volume in shares. No logarithm is used in raw Pearson correlations, OLS slopes, within-stock share-level associations, or binned mean/median outcomes. Secondary rank correlations describe ordering and are explicitly labeled. Market/calendar variables use date-level mean auction shares per stock as their primary association unit; sector context uses sector-date mean shares. Stock-date-weighted results are also supplied for comparison.

The analysis is observational, pairwise complete-case, and one variable at a time. It does not identify causal effects or show that a variable adds value after the remaining predictors. No IID p-values or unsupported significance claims are supplied. See `covariate_findings.md`, the per-covariate HTML atlas, and the CSV tables for results and definitions.

## How to tell this story later in the presentation

A compact sequence would be: normal auction history as a strong baseline; calendar and liquidity corrections; practical-tie selection of the smaller global model; architecture checks and remaining tail failures. The raw-covariate diagnostics can supply one economic-intuition exhibit, chosen after discussion. Nothing has been added to the presentation in this step.

## Evidence trail through all thirteen notebooks

{md(mapdf)}

All saved artifact paths in the last column are relative to `outputs/eda/`. The review manifest records notebook hashes and saved execution/error metadata. Historical metrics are read from existing outputs; the new analysis rebuilds the existing feature SQL in its own Git-ignored cache and leaves the research outputs unchanged. The 2025 year had been inspected during model development and is confirmation evidence, not a pristine untouched holdout.
'''
(OUT/'model_choice_story.md').write_text(story)
print('Model story and evidence tables written.')
