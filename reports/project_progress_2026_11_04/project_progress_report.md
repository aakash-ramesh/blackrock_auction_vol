# NYSE Closing Auction Volume Forecasting Project Progress Report

Research completed through November 4 2026

## Executive overview

We have extended the supplied closing auction research from a regularized regression benchmark into a reproducible comparison of nonlinear daily forecasts, alternative estimation targets, historical calibration methods, and intraday updates. The business outcome throughout remains official NYSE closing auction volume in shares. The project now has a stronger daily forecasting candidate and working noon and 15:30 updates, with saved predictions and model reconstruction checks.

The most important new finding is that nonlinear modeling adds measurable value to the daily forecast. On the full eligible universe, the rebuilt 58 feature LightGBM model reduces mean annual development log RMSE from 0.48764 for the frozen ridge benchmark to 0.46941, a relative improvement of about 3.74%. WAPE falls from 36.80% to 35.45%, or 1.35 percentage points. These are 2022–2024 averages, giving each year equal weight. The gain does not eliminate systematic underprediction or large auction misses. [R3]

The intraday work adds a second layer around that daily LightGBM forecast. On a common regular session universe, the 15:30 ridge correction has the strongest aggregate point estimates among the tested update methods: development log RMSE of 0.45995 and WAPE of 34.59%, compared with 0.46949 and 35.47% for the fixed daily anchor on those same rows. Noon also helps, but less: noon ridge reaches 0.46459 and 34.85%. Noon captures about 51% of the 15:30 ridge improvement in development log RMSE. [R6, R7]

We also completed useful negative experiments. One day log growth did not justify replacing the median relative target. None of four historical calibration corrections met the combined accuracy and bias criteria. A simple intraday volume ratio rule performed much worse than the learned forecasts. These results narrow the implementation choices rather than leaving every plausible extension open. [R4–R7]

This remains research evidence. The 2025 period was inspected in earlier work and is confirmation evidence, not an untouched holdout. The next planned robustness study of the 15:30 ridge update has not yet been performed. No production deployment, economic execution test, or newly sourced external dataset is included in the completed work.

## 1 Original research and how the new work builds on it

### 1.1 What was already supplied

The repository already contained the source data, daily feature engineering, data quality and leakage audits, descriptive research, a sequence of regression experiments, segmentation tests, individual stock backtests, options and FOMC studies, written reports, and presentation material. Notebooks 01–15 contain this foundation. Notebook 16 already contained a nonlinear challenger design and saved summary results. The new work therefore did not invent the nonlinear idea or its original experimental settings; it repaired and strengthened the evidence behind that comparison and then extended it. [O1–O5]

The two supplied October reports describe the state through the daily regression work and the options and FOMC studies. They emphasize recent auction history, calendar effects, weak daily spike prediction, systematic underforecasting, and the intraday opportunity. Their statements that nonlinear and intraday models had not been built describe that earlier research state. They should not be read as the current status after the new experiments. [O1]

The original notebook sequence established the following progression:

| Existing work | Main contribution |
| --- | --- |
| Notebooks 01 and 02 | Source quality, target eligibility, coverage, skewness, persistence, calendar and size effects |
| Notebooks 03 through 06 | Prior close information policy, enriched panel, 99 approved features, feature roles and missingness |
| Notebooks 07 through 09 | Temporal baselines, global ridge, specification tests, compact 58 feature benchmark freeze |
| Notebook 10 | Pooled interactions, hard splits and specialists tested against the frozen ridge |
| Notebooks 11 through 13 | Train and forward backtests, selected stock walk forward forecasts and unsmoothed daily examples |
| Notebooks 14 and 15 | Options and FOMC research with explicit information time restrictions |
| Notebook 16 | Original nonlinear residual challenger specification and historical summary results |

The first major modeling result was that a stock’s recent auction scale is a demanding baseline. The supplied analysis reports a target median of 55,226 shares, a mean near 190,445 shares and a 99th percentile near 2.08 million shares. Pooled log lag one correlation is about 0.894, while the median stock level lag one correlation is only about 0.459. A model can rank large stocks well without accurately predicting unusual movement within a stock. This distinction is central to interpreting both the original and new results. [O1, O3]

### 1.2 What we added during the local continuation

We first rebuilt the panel locally and reconciled saved benchmark forecasts. We then reran the compact nonlinear candidates against the verified regression reference, retaining complete forecasts, fitted models, preprocessors, realized training samples and fingerprints. After establishing that reproducible base, we conducted the target formulation study, residual diagnosis and calibration study, and two intraday experiments. [R1–R7]

| New stage | Main deliverable | Verification checks |
| --- | --- | ---: |
| Panel reconstruction | Isolated rebuilt daily panel and source checks | 23 |
| Benchmark reconciliation | Metrics and row alignment for saved regression forecasts | 81 |
| Compact nonlinear reproduction | L2 and Huber LightGBM plus fixed blends | 60 |
| Target formulation experiment | Controlled ridge and LightGBM comparison across three targets | 92 |
| Error diagnosis and calibration | Residual audit, 2021 historical forecasts and four corrections | 87 |
| 15:30 update | Rule, ridge update, daily only control and LightGBM update | 124 |
| Noon update | Corresponding noon models and matched timing comparison | 124 |

These counts describe checks within each stage, not independent statistical tests or a guarantee that every source feature has been independently validated. Additional timing alignment and full checkpoint availability audits accompany the noon comparison.

## 2 Data treatment and imputation

### 2.1 Sources and observation units

The daily unit is one date and PERMNO security identifier. Intraday observations add a checkpoint time. Auction labels span 2015–2025, while CRSP daily history begins in 2011 to support lag construction before the first labeled year. Calendar, market and sector inputs already in the project support the approved daily feature panel. Trade checkpoint aggregates provide cumulative trading information during the session. [O2, R1]

The reconstructed daily panel has 3,594,404 rows across 2,149 securities and 2,766 trading dates. There are 3,499,401 eligible auction targets. These panel counts should not be confused with raw source record counts, the supervised sample, or the later evaluation subsets. [R1]

### 2.2 Label treatment

A label is observed only when the source says the target was found and auction volume is nonmissing. Missing auctions remain missing; they are never imputed as zero. The documented August 12, 2019 source anomaly is excluded from supervised labels. No minimum history or liquidity filter removes otherwise eligible cold start observations from the daily forecast evaluation. [O2, R1]

Reproduction deliberately preserved an important existing behavior: the anomaly date is excluded as a training label, but its observed source values remain available when later auction history features are calculated. Masking those values from subsequent histories would be a separate data treatment experiment. It has not been implemented or tested as part of the completed comparisons. [R1]

### 2.3 Feature timing and rolling histories

The daily model uses information available by the previous close or known in advance from the calendar. Current day final volume, closing returns, auction prices, auction outcomes and intraday observations are excluded from daily model inputs. Ratios with zero denominators become missing rather than infinite. Market and sector variables are lagged. Cross sectional percentiles use lagged stock information. [O2]

Stock rolling histories end before the prediction date. The prior 20 row auction median takes the median of observed auction values inside the preceding 20 stock rows. It does not search backward until it accumulates 20 nonmissing auctions. The same distinction applies when interpreting the new checkpoint history windows. Keeping the original definition fixed was necessary to attribute later gains to modeling changes rather than a silent change in history construction. [R1, R6, R7]

### 2.4 Preprocessing inside each historical fold

Missing covariates remain missing in the feature panel. The model fitting pipeline learns how to treat them using historical fitting data only. The shared preprocessing helper applies these rules:

| Variable type | Historical fitting treatment | Later evaluation treatment |
| --- | --- | --- |
| Continuous and ordered | Optionally log1p transform nonnegative highly skewed variables; clip at 0.5th and 99.5th percentiles; fill with clipped median; center at median and divide by fitted standard deviation | Apply saved transform, limits, median and scale |
| Continuous missingness | Add a missing flag if missing values occur during fitting | Reuse that flag structure |
| Binary | Fill missing with zero; add a missing flag when needed | Reuse fitted structure |
| Categorical | Treat missing as MISSING; encode observed levels with a reference level and an OTHER column | Unseen levels map to OTHER |
| Entirely missing continuous input | Use a zero fallback and unit scale | Retain the same fitted treatment |

The log decision is a deterministic historical skew rule rather than an arbitrary transform chosen after viewing test errors. A missing indicator is created only if missingness occurs in the fitting data, so the pipeline does not guarantee a separate indicator for every newly missing evaluation input. These details are in the saved preprocessing records. [O2, R8]

When the stock specific median or prior auction needed for a target reference is missing, the reference uses the fitting sample’s global median auction shares. Eligible rows are retained and reference missingness is recorded. Final daily share forecasts are reconstructed from logs, constrained by historical 0.1st and 99.9th percentile log target bounds, and kept nonnegative. Feature clipping and final forecast clipping are different operations. [R3, R4]

### 2.5 What the reconstruction established

All 99 approved feature null counts and the feature catalog matched the saved feature engineering artifacts. Annual eligible counts matched all 11 years, panel keys were unique, and eligible numeric inputs had no nonfinite nonmissing values. An independent calculation reproduced four history features on 28,333 rows from 16 stocks. The original full panel was unavailable, so this is not a claim of all cell identity against a retained historical panel or a complete independent audit of every feature formula. [R1]

## 3 From historical rules to the regression benchmark

### 3.1 Why regression was a reasonable starting point

The original research first asked whether any model could beat yesterday’s auction or a trailing median on the same eligible universe. Auction history and liquidity inputs are highly redundant: the supplied analysis reports rank correlations near 0.954 between the five day auction mean and 20 day median, and near 0.953 between lagged daily volume and its 20 day average. Ridge regression shrinks unstable coefficients while providing a transparent first global model. [O3]

The ridge model pools stocks rather than estimating a separate regression for every security. In its selected formulation, it predicts a log correction around the stock’s recent median auction scale. This lets the model focus on departures from normal while the offset carries much of the persistent size difference across stocks.

### 3.2 How the original benchmark was selected

The original specification audit compared expanding, rolling five year and rolling three year histories; ridge penalties; sampling choices; feature families; target formulations; weighting objectives; calibration diagnostics; drift; and a shuffled target negative control. Notebook 09 then combined promising choices into finalists. Development selection used 2022–2024, with a practical tie tolerance of 0.002 mean annual log RMSE and a preference for fewer source features among tied models. [O2, O3]

The frozen benchmark uses a rolling five year history, uniform stock day weights, a median relative log target and 58 source features: 19 auction history, 23 calendar and 16 liquidity features. The 58 source count is not the number of encoded matrix columns, because missing flags and categorical encodings can expand the matrix. The master panel retains all 99 approved features. [O2, R3]

The original family tests show why auction history and calendar remain central, while liquidity provides a smaller and metric dependent increment. The exploratory auction plus calendar result should not be interpreted as proving that a 42 feature model replaced the final 58 feature benchmark. Family ablations and finalist selection used different comparisons. Likewise, a weak individual association does not establish that a variable can never help a nonlinear interaction. [O3]

### 3.3 How much regression improved the baseline

Using the saved reference comparison, the deployable trailing median baseline has mean annual development log RMSE near 0.6211 and WAPE near 43.47%. The selected ridge reaches 0.48764 and 36.80%, approximately 21.5% lower log RMSE and 15.3% lower WAPE relative to that median baseline. These are equal year development averages, not a pooled score. [O3, R2]

The original segmentation study then tested pooled interactions, separate models by size, liquidity and intensity, a size by liquidity grid, and event or limited history specialists. None of the tested alternatives justified replacing the pooled ridge under the stated criteria. This rejects those tested linear architectures; it does not establish that nonlinear interactions or all specialist designs are useless. [O2, O3]

## 4 Reconciliation before accepting nonlinear improvements

Before extending the model, we recomputed the saved Notebook 09 metrics from its forecast files. All 196 annual metrics, covering seven models, four years and seven metrics, reproduced within tolerance. Independent SQL calculations agreed with the project metric helper. The 1,271,378 prediction rows for 2022–2025 matched exactly the rebuilt panel’s eligible keys and target context. [R2]

That check exposed an evidence problem in the supplied Notebook 16 artifacts. Its reported ridge metrics differed in all four years even though the current code reads the Notebook 09 forecasts rather than refitting ridge. The original nonlinear row level forecast files were absent. Therefore its summary improvement and paired tests could not be verified from the available checkout. A stale output or different input snapshot is possible, but the exact historical cause is unknown. [R2]

We did not repair this by editing a summary CSV or substituting a new ridge number into an old paired test. Instead, we preserved the original artifacts and reran the compact nonlinear specification in isolation against the verified saved ridge forecasts. This is one of the main contributions of the local continuation: the nonlinear result now rests on aligned, retained forecasts rather than an unreconciled summary. [R3]

## 5 The transition to nonlinear daily models

### 5.1 What changed and what stayed fixed

Ridge estimates an additive linear correction in the transformed feature matrix. Gradient boosted trees can learn thresholds and interactions without predefining every cross product. That flexibility is relevant when an expiry effect varies with liquidity or recent auction behavior, but the completed comparison does not identify a particular economic mechanism as the cause of the gain.

We reran squared error LightGBM, Huber LightGBM and two fixed 50 50 blends with ridge. Every candidate used the compact 58 feature source set and median relative estimation target. The original five year rolling chronology, historical validation, sampling convention, seeds, preprocessing and final bounds were retained. Shared preprocessing fits the 99 approved source features before applying the compact mask, preserving the original convention. The blend averages forecasts in log1p space, not raw shares. [R3]

Squared error loss penalizes large residuals strongly; Huber loss reduces the influence of extreme residuals beyond its transition point. Testing both asks whether robustness improves general forecasting quality rather than assuming that the most flexible or most outlier resistant loss must win.

The tree search considered 15, 31 and 63 leaves, a learning rate of 0.03, at most 2,000 trees and early stopping patience of 100. Historical tuning, validation and final refit samples were capped at 250,000, 120,000 and 350,000 rows. The actual samples are saved. Test years were scored in full, and no test labels selected hyperparameters. Some selected fits approached the tree limit; no larger search was added after observing the results. [R3]

### 5.2 Verified daily results

| Daily candidate | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen ridge | 0.48764 | 36.80% | −14.38% | 0.44512 | 34.56% |
| LightGBM squared error | 0.46941 | 35.45% | −14.40% | 0.43065 | 32.94% |
| LightGBM Huber | 0.47208 | 35.29% | −15.69% | 0.43209 | 32.94% |
| Equal log blend of ridge and squared error trees | 0.46963 | 35.50% | −14.84% | 0.43040 | 33.11% |
| Equal log blend of ridge and Huber trees | 0.47139 | 35.45% | −15.43% | 0.43087 | 33.08% |

Source R3. Development means 2022–2024 with equal year weight. These rows use the full eligible daily universe, including early closes.

All four challengers passed the original development replacement gate: at least 0.002 mean log RMSE improvement, no annual development deterioration above 0.002 and a paired date level log MAE interval with an upper bound below zero. Squared error LightGBM had the best development log RMSE and became the fixed daily anchor for later experiments. Huber had slightly better development WAPE but deeper aggregate underprediction. The blend did not deliver a decisive all metric advantage. [R3]

The pure squared error model improved annual log RMSE from 0.53351 to 0.51010 in 2022, 0.48366 to 0.46093 in 2023 and 0.44576 to 0.43720 in 2024. Its 2025 log RMSE and WAPE also improved. However, its 2025 paired log MAE interval crosses zero. Improved confirmation point estimates should therefore not be described as established significance for that particular paired metric. [R3]

### 5.3 What nonlinear modeling did not solve

In pooled development diagnostics, squared error trees improve log RMSE and WAPE for small stocks, the top actual volume tail and displayed event groups, but top tail and expiry bias worsen. Cold start log RMSE worsens even as cold start WAPE improves. Only 758 development rows have fewer than five prior observed auctions, which limits broad claims from that group. Actual top tail membership is an after the fact diagnostic and never a model input. [R3]

The correct conclusion is that nonlinear daily modeling earns a place in the research pipeline. It does not establish a universally superior model for every objective, segment or operational use.

## 6 Alternative targets and the growth rate experiment

### 6.1 Three ways to estimate the same business outcome

Let A be current auction shares, L the previous stock row’s auction volume, and M the median auction volume in the preceding 20 stock rows. The controlled experiment considered these estimation targets:

| Estimation target | Definition | Forecast reconstruction before bounds |
| --- | --- | --- |
| Direct log level | log1p(A) | expm1(predicted log level) |
| One day log growth | log1p(A) − log1p(L) | expm1(predicted growth + log1p(L)) |
| Median relative log residual | log1p(A) − log1p(M) | expm1(predicted residual + log1p(M)) |

The growth definition is a log1p difference, not ordinary percentage growth. The prior reference is the previous stock row, not necessarily the most recent nonmissing auction. Missing L or M uses the historical fitting median. All candidates reconstruct shares and score the same eligible observations. [R4]

The comparison crossed these three targets with ridge and squared error LightGBM. It reused the exact saved historical samples from the nonlinear reproduction, feature mask, preprocessing convention, search budgets and forecast bounds. The median relative LightGBM forecasts were reused exactly. No new calibration, weighting or anomaly history rule was bundled into the target test. [R4]

### 6.2 Results and decision

| Model | Target | Development log RMSE | Development WAPE | Development bias |
| --- | --- | ---: | ---: | ---: |
| Ridge refit | Direct log level | 0.49650 | 37.06% | −15.37% |
| Ridge refit | One day log growth | 0.49063 | 37.42% | −14.85% |
| Ridge refit | Median relative | 0.48778 | 36.74% | −14.52% |
| LightGBM | Direct log level | 0.47068 | 35.69% | −16.05% |
| LightGBM | One day log growth | 0.47029 | 35.34% | −16.06% |
| LightGBM | Median relative | 0.46941 | 35.45% | −14.40% |

Source R4. The controlled ridge refits use the shared nonlinear samples and are distinct from the externally frozen historical ridge. Their small metric differences are expected and are not the earlier artifact discrepancy.

Neither alternative target passed the target change gate within its model family. Growth gave LightGBM slightly lower WAPE, but worse log RMSE and more negative aggregate bias than the median relative formulation. Its slightly better 2025 point estimates did not determine selection. We retained the median relative target and did not pursue growth as an assumed solution to structural auction expansion. This test addresses one day log growth; it does not rule out all trend or adaptive history designs. [R4]

## 7 Error diagnosis and the calibration tradeoff

### 7.1 Where the error is concentrated

The largest 1% of absolute share error rows account for 35.65% of development absolute share error. The top 1% of actual auction volumes account for 31.82%. By comparison, observations with fewer than 20 prior auctions account for only about 0.29%. Cold starts have large relative errors, but are not the dominant source of universe wide share error. [R5]

The diagnostic tables separate persistent security bias from common date misses, compare size and history groups, inspect forecast size deciles and actual tails, and identify large share errors separately from large log errors. Calendar and lagged market annotations provide context, not a causal explanation of each miss. [R5]

The upper forecast bound mechanically removes shares equal to about 4.22% of the net development volume shortfall. That finding does not support blaming most underprediction on clipping. We reconstructed unclipped forecasts only for diagnosis; no unclipped model was promoted. Structural growth, log estimation, omitted event information and model misspecification remain possible contributors that this diagnostic does not causally separate. [R5]

### 7.2 Why we built an additional 2021 model

To evaluate a correction in 2022, its parameters need genuinely earlier forecast errors. We generated 2021 daily predictions from a model tuned on 2020 and fitted through 2020, preserving historical chronology. This added 321,564 out of time forecast rows, saved models and historical samples. Each later calibration uses the full previous year’s out of time errors, never its own evaluation year’s labels. Those 2021 forecasts later also supplied historical training evidence for intraday stacking. [R5]

### 7.3 Four corrections and why none was adopted

We tested global share total scaling, a mean log residual shift, exponential residual smearing and forecast size decile ratio corrections shrunk toward the global factor. The decile method uses historical thresholds and a fixed n divided by n plus 10,000 shrinkage rule. No correction parameters were fitted on the year they evaluated. [R5]

| Method | Development log RMSE | Development WAPE | Development bias | Decision |
| --- | ---: | ---: | ---: | --- |
| Uncalibrated daily LightGBM | 0.46941 | 35.45% | −14.40% | Retained |
| Global share ratio | 0.49292 | 39.07% | +1.16% | Failed |
| Global log shift | 0.46921 | 35.53% | −13.37% | Failed |
| Global smearing | 0.55687 | 47.63% | +16.60% | Failed |
| Shrunk decile ratio | 0.50474 | 39.08% | +1.16% | Failed |

Source R5. The gate required improved absolute aggregate bias without worsening mean WAPE, limited log RMSE deterioration, annual WAPE stability and a favorable paired daily WAPE interval.

A forecast can have nearly zero average log residual and still underpredict total shares because a typical log outcome and arithmetic expected volume are different targets in a skewed distribution. Share scaling nearly removes aggregate bias but materially raises absolute error. The supplied reports’ suggestion that underforecasting might be easily fixed by calibration is therefore a hypothesis we tested, not a conclusion we retained. The uncalibrated daily forecast remains the anchor. [O1, R5]

## 8 Building the first intraday forecasting layer

### 8.1 Design of the update

The 15:30 experiment is new implementation beyond the supplied daily work. It predicts a correction around the saved daily LightGBM forecast rather than rebuilding the daily model. Its estimation target is log1p(actual auction shares) minus log1p(fixed daily forecast). The update is reconstructed in shares after adding the predicted correction to the anchor’s log forecast. [R6]

The current checkpoint contributes ten inputs: cumulative volume, dollar volume, trade count, recent 30 minute volume, VWAP, recent volume share, cumulative volume relative to prior daily ADV, cumulative volume relative to prior checkpoint volume, trade count relative to prior checkpoint trades, and VWAP return from the prior close. Four prior only control inputs supplement the original 58 daily features and daily anchor: prior checkpoint average volume, prior checkpoint average trades, historical auction to checkpoint volume ratio and prior price. This produces 63 control sources and 73 full update sources. [R6]

The study compares five methods: unchanged daily forecast; current cumulative volume times a historical auction to checkpoint ratio; ridge residual update; daily only LightGBM control; and LightGBM intraday update. The learned daily only control has no current session checkpoint data. It uses the same history, sample rows, preprocessing, target and bounds as the intraday tree model, helping distinguish extra current information from merely adding another fitting layer. There is no separately fitted daily only ridge control, so a ridge gain over the fixed daily anchor is not by itself a clean isolation of the current checkpoint’s contribution. [R6]

### 8.2 Honest historical stacking and coverage

Training the update on fitted daily predictions would make its historical inputs unrealistically accurate. Instead, the update uses saved daily forecasts that were out of time in 2021–2025. For the 2022 fold, update tuning fits January through September 2021 and validates October through December 2021. Later folds validate on the prior full year, fit tuning models on earlier available history from 2021, and refit through the preceding year. This expanding update history differs from the five year daily model history. [R6]

The main intraday evaluation excludes early close days because 15:30 is after those auctions. Missing or invalid regular session checkpoints fall back exactly to the unchanged daily forecast for every candidate. The study includes 1,260,168 regular session rows over 2022–2025; 15:30 checkpoint coverage is about 99.98%. A separate available checkpoint summary avoids conflating missing data handling with observed checkpoint performance. [R6]

The checkpoint cutoff contract relies on the upstream aggregate files and their after market close flags. The aggregates do not contain the individual trade timestamps needed for an independent audit of every cutoff. All learned update models and the ratio rule share historical update fitting log volume bounds; the fixed daily forecast retains its original saved bounds. [R6]

### 8.3 Results at 15 30

| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed daily anchor | 0.46949 | 35.47% | −14.47% | 0.42901 | 32.83% |
| Checkpoint ratio rule | 0.69153 | 49.26% | −15.63% | 0.61422 | 46.69% |
| Ridge update | 0.45995 | 34.59% | −11.80% | 0.40916 | 31.78% |
| Daily only tree control | 0.47871 | 35.95% | −13.39% | 0.43034 | 33.23% |
| LightGBM update | 0.46215 | 34.77% | −11.93% | 0.41559 | 32.12% |

Source R6. These are regular session results. Their daily anchor differs slightly from the full universe daily table because early close observations are excluded.

The 15:30 LightGBM update passed its predefined information gate against both the fixed anchor and learned daily only control. The gate requires at least 0.002 mean development log RMSE improvement, no annual deterioration beyond 0.002, no mean WAPE worsening and a paired development log MAE upper confidence bound below zero against each reference. Ridge had the best aggregate point estimates among the tested methods. That does not contradict the success of nonlinear daily modeling: predicting a full daily residual and estimating an incremental correction around an already nonlinear forecast are different tasks. [R6]

The intraday tree gate was specified for LightGBM. Ridge’s favorable estimates and paired intervals motivate further robustness work, but we should not rewrite the gate retrospectively as if it had been a predeclared final ridge promotion rule. The ratio rule’s poor performance also shows that observed cumulative volume alone, multiplied by a short historical ratio, is not an adequate replacement. [R6]

## 9 Noon and the value of waiting until 15 30

### 9.1 A matched timing comparison

The noon experiment repeats the same five method design using noon observations and prior noon histories only. Historical sample keys and ordering originate from the 15:30 study; rows without valid noon inputs are removed before fitting. Noon features are rebuilt rather than copied from the later checkpoint. Evaluation keys, auction labels and fixed daily forecasts match exactly across times. [R7]

Noon precedes early close auctions, but early close days remain excluded solely to preserve the common comparison cohort. The full regular session universe remains scored even when noon data are missing, through the same exact daily fallback. Noon coverage is about 99.94%. This compares complete forecast procedures at two times; training availability and checkpoint specific historical control features are not perfectly identical. [R7]

### 9.2 Results at noon

| Method | Development log RMSE | Development WAPE | Development bias | 2025 log RMSE | 2025 WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed daily anchor | 0.46949 | 35.47% | −14.47% | 0.42901 | 32.83% |
| Checkpoint ratio rule | 0.76257 | 54.10% | −11.86% | 0.68849 | 52.32% |
| Ridge update | 0.46459 | 34.85% | −13.02% | 0.41677 | 32.25% |
| Daily only tree control | 0.47902 | 35.96% | −13.34% | 0.42988 | 33.19% |
| LightGBM update | 0.46910 | 35.24% | −13.10% | 0.42109 | 32.44% |

Source R7. Noon ridge captures roughly half the development log RMSE gain available from the 15:30 ridge update. Noon LightGBM beats its learned daily only control but fails the full gate against the fixed daily anchor: its mean log RMSE gain is only about 0.00039 and one development year worsens by about 0.00350. A lower average WAPE alone is insufficient under the prespecified criteria.

Paired comparisons favor 15:30 for both model families in development and 2025. For ridge, the development mean date level log MAE difference, noon minus 15:30, is +0.00342 with a 95% whole date bootstrap interval of approximately +0.00290 to +0.00393. In 2025 it is +0.00722 with an interval of +0.00647 to +0.00796. Positive means lower error at 15:30. These intervals concern log absolute error, not confidence intervals on the headline log RMSE difference. [R7]

### 9.3 A validation issue found and corrected

The initial noon build failed the independent checkpoint history check. Explicitly materializing the source join before rolling calculations corrected 137 false missing checkpoint flags. All noon models were rerun on the corrected panel. Full source availability audits then found zero mismatches for both noon and the unchanged 15:30 panel. This is a concrete example of why saved predictions and independent feature reconstruction matter beyond checking that a training routine completed. [R7]

## 10 Reading the evidence correctly

Log RMSE measures squared error on log1p shares and gives greater weight to large proportional misses. Log MAE measures absolute log error. WAPE is total absolute share error divided by total actual shares; it is not the average stock day percentage error. Aggregate bias is total forecast shares divided by actual shares minus one. Spearman measures ranking and can be high when between stock size dominates, even with weak daily movement tracking. No single metric establishes all forms of forecasting quality. [R8]

Headline development metrics average the 2022, 2023 and 2024 annual metrics equally. Segment diagnostics are often pooled across those years. Paired uncertainty calculations first aggregate errors within each date and then resample whole dates 2,000 times. This preserves same date cross sectional dependence but not serial dependence across dates. It is not a moving multi day block bootstrap, and the reported intervals are not adjusted for multiple model comparisons. [R3–R7]

The daily and intraday studies also have different universes. Full daily evaluation contains 1,271,378 observations; the common regular session intraday evaluation contains 1,260,168. A direct chain of percentages across those two headline tables would mix cohorts. Comparisons within each table are aligned; claims spanning daily and intraday stages should state the relevant universe.

A further implementation convention matters for the learned tree control. Final refits retain a minimum of 25 trees even when historical early stopping chooses fewer. At 15:30, the daily only control selected five trees in 2023 and four in 2025; at noon it selected two in both years. All four were refitted with 25. This can affect the learned control comparison, although the fixed daily reference is unchanged. No sensitivity study of that rule has been completed. The noon narrative inherited an older five and four statement; the numbers here follow the actual saved hyperparameter records. [R6, R7]

## 11 Existing external studies and what remains outside the new implementation

The supplied options study examines open interest, expiring interest, option volume, implied volatility, put call ratios and approximate gamma measures, with different lag rules for different fields. Its existing forecast test found negligible incremental value from the broad options block, while coverage ends in August 2025. The strong descriptive expiry effect largely overlaps the existing calendar signal. Small cap or expiry specific options extensions remain hypotheses; we have not newly integrated or revalidated them against the rebuilt daily LightGBM anchor. [O1, O4]

The supplied FOMC study correctly works at the date level because all stocks experience the same announcement. It describes about 72 announcements inside its 2017–2025 window. Scheduled meeting dates can be known before the session; the decision itself arrives at 14:00 and cannot enter the daily or noon forecast. Decisions could be considered in a later afternoon extension, but the completed 15:30 implementation does not include them. A statistically detectable descriptive announcement effect is not proof of incremental forecast improvement, and policy surprise is not measured by the existing decision file. [O1, O4]

The new experiments use the existing data. They do not add order imbalance, earnings schedules, rebalance membership, policy expectations, VIX, rates, CPI or payroll release features. The 10:30, 14:00 and 15:00 checkpoint models have not been implemented in this continuation. No new segmentation architecture, adaptive anomaly masking, clipping policy change or production monitoring system has been completed.

## 12 Current position and next steps

The frozen ridge remains an intact, verified historical benchmark. The rebuilt squared error LightGBM is the leading daily research anchor under the primary development criterion. The median relative target remains preferred over the tested direct and growth alternatives. The uncalibrated forecast remains preferred over the four tested corrections. At intraday times, ridge gives the best aggregate point estimates, and 15:30 is stronger than noon. These choices describe the current evidence rather than a production certification. [R2–R7]

The next bounded study should test whether the 15:30 ridge gains persist across annual and monthly periods, size, liquidity, history depth, expiry and month end groups, and the largest actual auctions. It should report group sample sizes, share error contribution and bias alongside log error. It should also examine a matched daily only ridge control, sensitivity to historical update windows and fallback assumptions, and the minimum tree refit convention where relevant. These are proposed checks, not completed results.

Before final model selection, the project should answer which forecast origin is required, whether the operational objective is typical proportional accuracy or aggregate capacity accuracy, what early close behavior is needed, and how much checkpoint delay or missingness is acceptable. A genuinely unused forward period is needed for a stronger assessment after the specification is fixed. Execution costs, participation decisions and economic benefits require a separate evaluation; the current accuracy gains do not directly quantify savings.

## Appendix A New artifacts and reproducibility map

All paths below are relative to the repository root. Original notebooks, raw inputs and historical research outputs were preserved. New work is isolated under reproduction and experiment directories.

| Stage | Entry point | Main output directory |
| --- | --- | --- |
| Daily panel reconstruction | notebooks/reproduce_daily_panel.py | outputs/reproduction/01_daily_panel |
| Saved benchmark reconciliation | notebooks/reconcile_saved_benchmarks.py | outputs/reproduction/02_benchmark_reconciliation |
| Compact nonlinear rerun | notebooks/reproduce_compact_nonlinear.py | outputs/reproduction/03_compact_nonlinear |
| Target formulation comparison | notebooks/compare_target_formulations.py | outputs/experiments/17_target_formulations |
| Error diagnosis and calibration | notebooks/diagnose_and_calibrate.py | outputs/experiments/18_error_diagnosis_calibration |
| 15:30 update | notebooks/build_intraday_1530_update.py | outputs/experiments/19_intraday_1530 |
| Noon update | notebooks/build_intraday_noon_update.py | outputs/experiments/20_intraday_noon |
| Matched checkpoint comparison | notebooks/compare_intraday_times.py | outputs/experiments/20_intraday_noon |

The new experiment reader notebooks are 17_target_formulation_comparison.ipynb, 18_error_diagnosis_calibration.ipynb, 19_intraday_1530_update.ipynb and 20_intraday_noon_update.ipynb. Executed versions live in their output directories. Rebuilt feature and nonlinear notebooks are retained in the reproduction folders. Models, preprocessors, reconstruction settings, actual samples and forecast files make later verification possible without relying on notebook display state.

## Appendix B Source ledger

O1. Supplied reports: reports/Closing_Auction_Feature_Register.pdf and reports/Closing_Auction_Volume_Trends_and_Forecasting_Priorities.pdf. These describe the earlier daily regression, options and FOMC research state. Their causal interpretations and absolute ranking claims are not treated as newly verified results.

O2. Original notebooks 01–13 and their saved outputs under outputs/eda. In particular, notebooks 05–09 define the feature contract, preprocessing, evaluation and frozen benchmark; notebook 10 tests segmented alternatives; notebooks 11–13 provide backtest and stock examples.

O3. midterm presentation/data_analysis/model_choice_story.md, the related presentation analysis material and outputs/eda/09_global_finalists/frozen_global_benchmark.json. Used for the historical modeling progression and baseline comparison, not as proof of new nonlinear results.

O4. notebooks/14_options_and_auction_volume.ipynb and notebooks/15_fomc_and_auction_volume.ipynb. These remain existing external data studies, not new integrations in the intraday models.

O5. notebooks/16_nonlinear_model_challengers.ipynb and outputs/eda/16_nonlinear_challengers. Historical design and summaries; the new reconciliation records why their saved result could not be accepted without reproduction.

R1. outputs/reproduction/01_daily_panel/README.md, validation_summary.json, validation_checks.csv, annual_reconciliation.csv, missingness_reconciliation.csv and independent_history_checks.csv.

R2. outputs/reproduction/02_benchmark_reconciliation/README.md, reported_metric_reconciliation.csv, ridge_reference_discrepancy.csv and verified_benchmark_manifest.json.

R3. outputs/reproduction/03_compact_nonlinear/README.md, reproduction_summary.json, nonlinear_candidate_metrics.csv, nonlinear_candidate_selection.csv, nonlinear_date_block_bootstrap.csv and nonlinear_segment_diagnostics.csv.

R4. outputs/experiments/17_target_formulations/README.md, experiment_plan.json, target_comparison_summary.csv, annual_metrics.csv and paired_target_bootstrap.csv.

R5. outputs/experiments/18_error_diagnosis_calibration/README.md, experiment_summary.json, calibration_comparison.csv, error_concentration.csv, clipping_diagnostics.csv, paired_bootstrap.csv and historical_2021 artifacts.

R6. outputs/experiments/19_intraday_1530/README.md, experiment_summary.json, intraday_comparison.csv, selected_hyperparameters.csv, intraday_information_gate.csv, paired_bootstrap.csv and coverage_by_year.csv.

R7. outputs/experiments/20_intraday_noon/README.md, experiment_summary.json, intraday_comparison.csv, selected_hyperparameters.csv, checkpoint_time_comparison.csv, paired_checkpoint_time_bootstrap.csv, timing_alignment_checks.csv and full_checkpoint_availability_audit.csv.

R8. notebooks/modeling_utils.py. Source definitions for transformation, imputation, encoding, ridge fitting and forecast metrics. New scripts retain additional stage specific chronological, prediction and reconstruction checks.
