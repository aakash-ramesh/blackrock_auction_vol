# Technical reference notes from the earlier draft

Retained for preparation. This is not the recording script. Use the current `speaker_notes.md` in the presentation folder for read-aloud narration. The timing labels below describe the earlier outline.

# Speaker notes for the midterm presentation

Twenty-three main slides followed by one backtest appendix slide and one literature reference slide. Delivery target: about six and a half minutes for project background and data, about two minutes for descriptive statistics, eight and a quarter minutes for analysis, and two and a half minutes for challenges and next steps. The numbered narration totals 19 minutes 10 seconds, leaving 50 seconds for transitions and recording buffer within the 20-minute group presentation.

## 1. Forecasting Closing Auction Volume — 0:20

Our project forecasts how many shares will trade in a stock's closing auction. We focus here on daily NYSE forecasts using information available before the prediction day. We will explain the problem and data, then discuss the forecasting work so far, including a ridge benchmark.

## 2. The closing auction — 0:50

The closing auction brings buy and sell interest together at a single clearing event. Its price serves as the official closing benchmark. Investors who track an index care about executing near that benchmark, which helps explain why auction participation matters. Our outcome is the number of shares executed in that auction. An individual trader's available fill also depends on their order and market conditions.

The 4:00 p.m. time describes a normal NYSE session. The process begins then, and the auction need not print at exactly 4:00:00. Early-close sessions use a shorter schedule.

Source: [L1].

## 3. Business objective and project scope — 0:55

The project's strategic aim is to improve execution planning at the close. A volume forecast gives a trader an estimate of the auction's scale when deciding how much of an order to allocate there. We have not measured execution-cost savings, so forecasting accuracy should be presented as an intermediate objective.

The project objectives cover daily models and intraday updates for several venues. Our completed modeling concerns daily NYSE forecasts, using historical auction activity, liquidity, and known calendar events. Intraday trade data are already prepared, but that does not establish that an intraday forecasting model is complete. Nasdaq, LSE, and ATS/dark-pool forecasts are broader deliverables.


## 4. Forecast target and information set — 0:55

The target is the number of auction shares for a stock and date. The vector x represents all predictors available when the daily forecast is made, at the previous trading day's close. It can contain many economic families, rather than restricting the model to two types of variables.

Auction history, stock characteristics, liquidity, market and sector conditions, and scheduled events are examples. Additional families can be tested if their values are available at the forecast origin. Yesterday's final trading volume qualifies, and tomorrow's scheduled expiry can be known already. Tomorrow's realized trading volume cannot enter this prior-close forecast.

The restriction concerns timing. It does not impose an exhaustive list of predictor categories. We test whether these inputs improve forecasts over simple historical rules, including the recent auction median.

### If asked about the notation and extensions

The superscript tau identifies the forecast origin, and the vector can contain stock-specific and common variables or information about the prediction day that is already known. In an intraday model, the origin moves to a checkpoint and additional observations become eligible. Feature eligibility does not guarantee incremental predictive value.

The broad notation does not claim that every proposed input has been sourced or tested. VIX/rates and macro-release indicators remain planned extensions. A log-level or log-residual target transformation is a separate modeling choice; forecasts still return to auction shares.


## 5. Background literature: closing auctions — 1:00

Bogousslavsky and Muravyev study US closing auctions and their connection to passive investment. Their findings motivate studying auction volume separately from general daily trading. Their evidence on lower price impact supports the relevance of the close, without implying that every auction is cheaper for every order.

Kandel, Rindi, and Bosetti study Milan and Paris. Their results show that auction design can change trading near the close. This is a reason to begin with one clearly defined venue and be cautious when extending a model elsewhere.

These studies provide background motivation. Our forecasting analysis addresses a different empirical question.

Sources: [L1], [L2].

## 6. Background literature: volume forecasting — 0:55

Białkowski, Darolles, and Le Fol model intraday volume for VWAP execution using common and stock-specific components. Their setting motivates the use of market context and individual trading history. Our target differs because it is the official closing auction.

Our project asks whether a pooled forecast improves on a stock's recent auction history, whether its performance is stable across stocks, and whether known calendar events and market context add information beyond recent auction history.

Sources: [L3].

## 7. Data sources: auction targets and stock history — 0:55

Our auction labels come from TAQ extracts obtained via WRDS. CRSP supplies daily stock history, security characteristics, and permanent identifiers. The two layers have different date ranges and different counts because they serve different roles. Earlier CRSP observations support lagged features before the auction sample begins.

These counts describe our extracted universe, not the full NYSE or each year's stock count. Source auction rows include missing outcomes and are not a count of valid training labels. The universe can change over time. The project uses dated links rather than assuming a ticker always represents the same security.

### If asked about sample construction

Dated links connect TAQ symbols to PERMNO, and joins use security and trading date. A valid target requires an identified official auction record with reported share volume. Missing labels remain missing. Annual auction schemas are standardized, and the documented source anomaly on August 12, 2019 is excluded. The auction modeling sample begins in 2015, with earlier CRSP history supporting stock-level lags. Rolling predictors exclude the current forecast day, as defined in the information set.

## 8. Data sources: context and intraday trades — 0:45

The dataset also provides market context, industry/sector measures, event flags, and a trading calendar. These enrich the daily panel, with realized market information lagged appropriately.

TAQ trades have been aggregated at five checkpoints for an intraday model. Their source row count is roughly five times the auction row count because each stock-date has five checkpoint records. Some checkpoint records fall after an early close and must be masked. Trade checkpoints are distinct from an auction imbalance feed. The original handoff reports that the imbalance feed was unavailable under the project's WRDS entitlement.


## 9. Auction-volume distribution and stock-size comparison — 0:40

Before modeling, we examine the outcome's distribution and scale. The histogram shows a long right tail: median auction volume is 55,226 shares, versus a mean near 190,445. Large auctions pull up the average.

The right chart groups stock-days by market capitalization. Median auction volume differs by about 81 times between the smallest and largest size deciles. This uses contemporaneous size for description, while forecast inputs are lagged.

These patterns motivate stock-specific scale adjustments and both proportional and share-scale error measures. They do not demonstrate forecasting performance.

## 10. Auction participation: time and calendar comparisons — 0:40

We next describe participation over time and around calendar events. Median auction share of daily volume rises from 4.8% in 2015 to 14.0% in 2025 in this changing sample.

The event comparisons show higher median shares on expiry and month-end dates. These groups overlap, so the bars do not isolate causal event effects. The patterns motivate chronological evaluation and testing known calendar information beyond recent auction history.

Same-day auction share is a descriptive outcome. Its denominator is unavailable to a prior-close forecast. Expiry flags are known calendar dates and require no stock-specific options data.

## 11. Descriptive statistics: summary — 0:30

We have about 3.50 million eligible labels, with annual observed-target coverage around 93% to 99%. Missing labels remain missing.

The target is skewed, differs sharply across stock sizes, and varies over time and around calendar events. These facts motivate several error measures and chronological comparisons. We now examine individual covariates in auction shares before asking whether they add value together in a forecasting model.

## 12. Individual covariate relationships with auction volume — 0:45

Both plots use actual auction shares as the target. We sort observations into ten groups by one input at a time. Recent auction medians and previous-day trading volume both increase strongly with auction size, supporting their role in describing normal auction scale.

The mean lies above the median because large auctions pull up the average. These relationships combine differences across stocks with changes within a stock. They provide economic intuition, but they do not establish how much each input adds when the other predictors are present. Forward model comparisons address that question.

### If asked about the full individual-variable analysis

The broader diagnostic review covers 193 specifications: 99 engineered daily inputs, 31 same-day raw fields, their 31 prior-close counterparts, two auction-outcome diagnostics, and 30 checkpoint associations. Each examines auction shares directly. Those counts include derived variables and diagnostics, so they do not represent 193 independent raw inputs. This retrospective review helps interpret the ridge benchmark. It did not determine the ridge reference originally.

- Auction history: recent auction levels and shares describe persistent stock scale, changes in participation, and instability. Several summaries overlap strongly.
- Trading volume and liquidity: lagged daily volume has a strong pooled relationship. Turnover, dollar volume, spread, high–low range, and Amihud measures describe trading activity and liquidity conditions. An isolated relationship does not prove a unique contribution from every proxy.
- Stock size: market capitalization has a much stronger pooled relationship than a relationship within stocks. Large stocks tend to have larger auctions, but size alone explains little of their day-to-day variation.
- Returns and prices: prior signed returns have a weak raw relationship that changes direction across years. Auction volume is unsigned, so both positive and negative information shocks could raise activity. Absolute-return information can still enter liquidity proxies.
- Market and sector context: common conditions may affect auction activity. Date-level and sector-date summaries help avoid treating repeated common inputs as independent stock-level evidence.
- Calendar: the event contrasts already shown support testing predictable timing effects. Overlapping flags prevent interpreting those contrasts as isolated causal effects.
- Security and corporate-action fields: some classifications are constant under the sample filters. Rare corporate events and share adjustments need careful interpretation, and the current tests do not establish a useful separate feature family.
- Same-day raw outcomes and intraday checkpoints: these describe contemporaneous auction activity and potential intraday updates. They are unavailable to the prior-close daily forecast. Same-day total volume also contains the auction mechanically.

The plots use the pairwise-complete 2015–2025 sample and are retrospective descriptive evidence. Missing input values can change the population across individual comparisons. No causal or statistical-significance claim follows from these curves.

## 13. Stock size and liquidity: individual relationships — 0:35

Here we extend the individual analysis beyond auction history. The left input is prior-day market capitalization. The right is the prior-day relative bid–ask spread. Each curve shows actual auction shares within input deciles.

Larger stocks generally have larger auctions, and wider-spread stocks generally have smaller auctions. The spread pattern is not perfectly monotonic at the tightest end. These pooled relationships largely distinguish stocks of different scales and liquidity. They do not establish that a change in a stock's size or spread causes its next auction to change.

### If asked about definitions and incremental value

Market capitalization is a prior-close raw stock characteristic. Quoted spread is a lagged liquidity measure derived from bid/ask quotes, without using auction volume. The target remains auction shares, not a logarithm or a normalized auction share. Each panel uses pairwise-complete eligible stock-days, ten quantile groups with ties kept together, and its own vertical range.

For market cap, pooled raw Pearson correlation is about 0.289, compared with about 0.022 within stocks. For spread it is about −0.050 pooled and −0.0005 within stocks. The stronger pooled ordering largely describes the cross-section. The new curves use the full retrospective 2015–2025 sample, rather than an unused model-evaluation period.

The economic hypothesis is that larger and more liquid securities support larger share auctions. The curves support discussing scale, but they do not prove that market cap or each liquidity proxy adds information once auction history and other inputs are present. The later family-removal and combined ridge comparisons address that different question. The absence of a separate size family from the ridge reference does not make stock size economically irrelevant.

## 14. Stock returns and auction volume — 0:35

These plots examine the previous day's signed return and absolute return. Signed-return deciles run from the largest losses to the largest gains. Absolute-return deciles run from small to large price moves.

Neither pooled curve shows a strong monotonic relationship with today's auction shares. A directional price move therefore does not translate into a simple rising auction-volume pattern here. That does not rule out effects conditional on stock size, calendar events, or other inputs. It motivates testing more flexible relationships without assuming that they will improve forecasts.

### If asked about the result and economic intuition

Both measures come from prior-day stock returns and contain no auction-volume input. The signed return uses the raw total-return field. Absolute return is its magnitude. Unlike a directional return, auction volume is unsigned, so an economic hypothesis could involve movement size or interactions rather than a one-direction response.

Mean auction shares vary roughly from 180k to 201k across signed-return deciles, while medians vary around 52k–60k. Absolute-return curves are similarly fairly flat and non-monotonic. The zero-based vertical ranges help show that these differences are modest relative to auction levels. A large apparent effect should not be created by narrowly cropping the axes.

Raw Pearson correlation is about 0.0012 for signed returns and −0.0031 for absolute returns. The within-stock absolute-return correlation is about 0.019. These are descriptive linear summaries, not evidence that all nonlinear return effects are absent. Absolute returns also enter the retained Amihud liquidity proxy, so omitting a separate returns family does not remove every use of return information.

## 15. Market and sector activity: individual relationships — 0:35

These inputs are common across many stocks, so we summarize the target at the same level. The left uses each day's average auction shares per eligible stock. The right uses each sector-day's average.

Higher prior-day market trading volume generally accompanies higher average auction volume. The sector curve rises and falls across volume groups, without a comparable simple increasing pattern. These are associations across dates or sector-days, including changes in composition and regimes. The forward family tests are still needed to establish whether either context measure helps beyond stock history.

### If asked about aggregation and chart interpretation

The left has 2,765 paired dates, and the right has 33,180 paired sector-days. The inputs are the saved prior-close market and sector total-volume measures. Each group outcome is the average of the eligible stocks' raw auction shares in that date or sector-date. Within each input decile, the navy line is the mean of those group outcomes and the teal line is their median. It is not the median individual-stock auction shown in the earlier stock-day charts.

Dates receive equal weight on the left, and sector-days receive equal weight on the right. This avoids pretending that a common market input supplies millions of independent stock observations. The market mean group outcome rises from about 142k shares in the lowest input decile to 267k in the highest. Sector means vary non-monotonically, roughly between 176k and 269k. No sector fixed effects or time-trend controls are applied here.

The economic hypothesis is that common trading conditions may influence auction participation across stocks. The curves do not establish causality or incremental forecast value. The historical market and sector proxies can reflect changing constituents, sector composition, and trends. Their native-unit correlations should not be ranked directly against stock-day correlations as though they shared a common analysis unit.

## 16. Feature-family removal tests — 1:00

We now move from one input at a time to a controlled forecast comparison. Starting from the full ridge model, we remove one economic family and measure the change in mean annual log RMSE over 2022 through 2024. A positive bar means forecasts become worse.

Removing auction history raises error by about 0.177, and removing calendar information raises it by about 0.078. Those are the dominant changes. Removing liquidity raises log RMSE by only about 0.00045. The remaining family changes are also small. The economic story is that recent auction history establishes normal scale, while known dates help explain systematic deviations around that scale. Liquidity may provide a smaller correction.

This supports prioritizing auction history and calendar information. It does not show that every variable in a retained family is necessary, or that an omitted family is economically irrelevant. The next comparisons test the target formulation and the ridge specifications.

### If asked about the audit design and economic interpretation

This earlier audit uses direct log-level ridge with an expanding training history. The feature masks reuse the full model's selected penalty within each annual fold. Each family is removed separately while holding the other audit settings fixed. The chart averages only the three development years and does not include 2025. It is a controlled sensitivity check rather than independent optimization of every possible reduced model.

The full model's development log RMSE is 0.49898. Removing liquidity yields 0.49942, but mean annual WAPE improves from 37.67% to 37.51%. Its increment is therefore small and metric-dependent. The later residual-ridge specification comparison supports retaining the liquidity family under the primary log-RMSE criterion.

Omitting the separate returns/price family still leaves absolute return in the Amihud measure, bid and ask in quoted spread, and high, low, and close in the trading-range measure. Shares outstanding enter turnover. Historical auction and trading activity also encode stock scale even without a separate market-cap predictor.

## 17. Target formulation: deviation from the 20-day median — 1:15

Our business target remains auction volume in shares. The modeling question is whether to predict its log level directly or predict a correction around the stock's recent typical auction.

Let B be the trailing 20-day auction median, calculated entirely from earlier observations. We subtract log one plus B from log one plus actual auction volume and train the model to predict that residual. A positive residual means an auction above the recent baseline, and a negative residual means one below it. We then add the predicted correction back to the baseline and convert the result to shares.

The comparison holds the five-year training history, 99 inputs, and uniform observation weights fixed. Mean annual development log RMSE falls from 0.49609 for direct log levels to 0.48710 for the median residual. That supports learning deviations around recent stock-specific scale. It also gives calendar and liquidity information a clear role: explaining why tomorrow might differ from the recent norm.

### If asked about the exact transformation

This is a difference of log1p volumes, rather than the raw difference in shares. It approximates a log ratio when both volumes are large. The baseline window covers the preceding 20 stock trading observations, excludes today, and ignores missing auction labels in the median. If it is unavailable, the fallback is the global auction median from historical training data. It does not use future labels or the median of the forecast year.

The displayed inverse explains the forecast's economic scale. The implementation also clips the final log forecast to the historical training target's 0.1% and 99.9% bounds before conversion. Exponentiating a log-space point forecast does not generally produce the conditional arithmetic mean of auction shares. That distinction matters when interpreting systematic underprediction in the largest auctions.

## 18. Ridge benchmark: specification comparison — 1:15

Ridge is one modeling approach we have built so far. This table compares specifications within ridge, rather than selecting a final project model.

The weights here apply to training observations. Equal weights give each stock-day the same influence on fitting error. Recent-data weights give more influence to recent observations, with a two-year half-life. They are different from the coefficients that ridge learns for its predictors.

The 99-input, recent-weighted ridge scores 0.48677. The 58-input, equally weighted ridge scores 0.48764, about 0.00088 worse. Within these trials, the 0.002 tolerance allows the smaller specification to serve as a consistent reference. The 42-input version falls outside that tolerance.

The reference uses a five-year history and auction, calendar, and liquidity inputs. It gives us a reproducible comparison point, while further model development remains open.

### If asked about training weights

Weights multiply each training row's squared residual in the fitting objective. Uniform weights are all equal. Recent-data weights are proportional to `(1/2)^(age_in_years / 2)`, where age is measured from the start of the prediction year. A row two years older has half the relative weight, and one four years older has one quarter. This favors newer observations within the same historical window. It does not add more recent observations or give the model future information.

These are observation weights, not predictor coefficients, portfolio weights, or the weights implicit in an evaluation metric. Equal training-row weights do not imply equal influence per stock if stocks have different numbers of training observations. The WAPE definition remains total absolute share error divided by total actual shares regardless of this training choice. The comparison also changes the feature count between the displayed rows, so their score difference cannot be attributed only to weighting.

### If asked about selection, tuning, or architecture

The source-feature count precedes categorical encoding and missingness indicators. The fitted design matrix has more columns. Preprocessing is fitted using historical training data, and the penalty is selected using earlier validation data. A rolling window uses up to five prior years as history becomes available.

The model with all 99 inputs and uniform weights scores 0.48710. The 58-input ridge reference therefore reflects the tolerance and parsimony rule within these ridge experiments, rather than a claim that fewer inputs produce the absolute lowest error. It also does not win every reported metric. The work supports ridge as a transparent regularized benchmark, without establishing superiority over every nonlinear model class.

The best size split improves development log RMSE by about 0.00116, below the 0.002 practical improvement threshold. A separate date-block uncertainty interval for its paired log-MAE difference is approximately −0.000711 to +0.000129, including zero. The uncertainty interval concerns log MAE, while the practical threshold concerns log RMSE. No tested replacement clears all requirements. These results do not rule out useful heterogeneity or future specialist models.

## 19. Annual forecast errors: ridge versus recent median — 1:00

This slide asks a simple question: does the ridge benchmark improve on predicting a stock's recent typical auction?

The dark bars use a simple rule: the forecast itself equals the stock's trailing 20-day auction median, using only earlier observations. The teal bars use ridge. Each pair scores the same stock-days in the displayed year, and lower bars mean less error. We show WAPE, which divides total absolute share errors by total actual auction shares.

Across the development years, mean annual WAPE falls from 43.47% for the median rule to 36.80% for ridge. For intuition, 36.8% WAPE means absolute errors totaling 36,800 shares for every 100,000 actual shares. That is a material improvement over this simple rule, but substantial error remains.

The separate 2025 confirmation result uses a year already inspected during research. It is not a new untouched holdout.

### If asked about the median-rule baseline

Yes: if the stock's recent 20-day auction median is 50,000 shares, this rule predicts 50,000 shares for the next auction. It has no fitted calendar or liquidity adjustment. The window covers the previous 20 stock trading observations and ignores missing auction labels. If no median is available, use the historical training-median fallback, so both methods retain the same eligible evaluation rows.

The ridge residual model starts from this baseline but learns an adjustment in log1p space. Separately, the later rolling-median history slide smooths actual and ridge forecast series for display. That display smoother includes the plotted date and is not the simple prior-only forecast baseline.

### If asked about aggregation and remaining errors

The headline figures are equal-weight averages of annual WAPEs, rather than one pooled statistic across all three years. The reduction is 6.67 percentage points, or about 15.3% relative to the median rule. Within each year, WAPE aggregates errors and actual shares across eligible stock-days. Log RMSE, retained in the appendix, is root mean squared error in log one plus shares. It is not a percentage and cannot be read as WAPE. The saved annual development mean log RMSE falls from 0.6211 for the median rule to 0.4876 for the ridge reference.

Mean annual aggregate volume bias is about −14.4% across the development years. The 2025 confirmation bias is about −9.6%, with log RMSE 0.4451 and WAPE 34.56%. A separate pooled 2022–2025 diagnostic reports roughly −41.1% aggregate bias among the largest 1% of actual auctions, versus −5.7% among the lower 99%. This tail statistic includes confirmation evidence and is not a development-only model-selection result. Small stocks and short-history stocks also remain difficult.

Daily stock examples reveal missed spikes that smoothed plots can hide. The analysis does not identify those misses as earnings or rebalance effects without additional event evidence. Six continuously observed case-study stocks do not establish performance for every new listing or delisting.

## 20. Actual versus predicted volume: 20-day rolling medians — 0:30

These curves show the trailing 20-day medians of actual and predicted auction shares for the same two stocks over the full history. They help us see changes in typical activity across years. The shaded 2015 period contains actuals only, with forecasts beginning in 2016. Smoothing makes the broad movements easier to see, but it can hide large daily misses. The next slide returns to unsmoothed forecasts.

### If asked about smoothing and the historical reconstruction

Each line is the trailing median of its own series over 20 stock trading observations, requiring at least five non-missing values. The displayed median includes the plotted date and up to 19 earlier observations. The 2015 forecast series remains missing, and the first forecast-median point requires five available forecasts in 2016. The series continue across year boundaries.

This is a display smoother for actuals and previously generated daily forecasts. It differs from the model's median offset, which excludes the prediction day. We are not adding the simple median baseline as a third line or fitting a model to smoothed targets. The unsmoothed predictions and error metrics remain unchanged.

The historical specification was selected retrospectively. Annual model fitting uses earlier dates, and daily lagged predictors update during the forecast year. In 2016, insufficient earlier history prevents the later nested tuning procedure, so the historical reconstruction uses the documented modal-penalty fallback of 0.0001. Later folds tune on earlier years. These plots do not establish that the specification was chosen prospectively in 2015.

JPM and CATO are illustrative continuously observed examples, with separate logarithmic axis ranges. The lines show broad activity and do not replace the annual common-row benchmark comparison or the unsmoothed daily diagnostics. The previously inspected 2025 period is part of this retrospective history, not a new holdout.

## 21. Daily actual versus predicted auction volume — 0:45

These unsmoothed forecasts show the remaining error directly. Ridge WAPE is 30.8% for JPM in 2021 and 46.4% for CATO in 2017. Those are substantial errors. On the same rows, the recent-median rule scores 37.9% and 56.6%, respectively.

Ridge improves these examples, but the improvement does not establish sufficient accuracy for execution decisions. Some actual spikes remain far above the forecasts. The crosses mark the largest absolute log errors, which can include overprediction and underprediction. Different years and axis ranges make these illustrative cases rather than a controlled comparison of stock size.

### If asked about selection and information timing

The saved selection calculates each stock's annual log RMSE over 2016–2025 and picks the year nearest its median, breaking ties by year. We use two of the existing six case-study stocks. JPM's example has 252 observations, log RMSE 0.35770, and WAPE 30.82%. CATO's has 251 observations, log RMSE 0.55116, and WAPE 46.36%. These percentages describe different stocks and years and do not estimate a controlled size effect.

### If asked whether WAPE is too large

Yes, the remaining error is economically substantial. WAPE 46.4% means the sum of absolute share errors is 46.4% of the sum of actual shares over the scored year. It does not mean 46.4% error on every day, and it is not a prediction-accuracy percentage. Overpredictions and underpredictions do not cancel in its numerator.

The same-row median-rule WAPEs are 37.8750% for JPM and 56.6098% for CATO, compared with ridge at 30.8215% and 46.3621%. Every example row has an observed trailing-median baseline, so no fallback or row-dropping is needed for this comparison. These results support improvement over a simple rule, not a claim that the residual error is acceptable. No execution-use tolerance or cost-benefit test has yet established that. Inspect bias and large-auction misses alongside WAPE.

### If asked about plot markers and timing

The logarithmic vertical axes display auction shares without smoothing the series. Markers use the largest absolute differences in log1p actual and predicted shares, rather than the largest share errors. The stocks are illustrative continuously observed examples, not a representative sample of the full universe.

The forecasting specification was selected retrospectively in later research. Within these historical walk-forward calculations, fitting uses earlier dates and daily predictors obey the prior-close boundary. Example-year selection uses the full saved performance history and is an ex-post presentation choice, not an untouched evaluation design. No new fitting or tuning was performed for this exhibit.

## 22. Remaining challenges — 1:00

The ridge benchmark improves on a simple historical rule, but the aggregate improvement leaves several practical challenges.

First, large auctions remain underpredicted, and the daily case studies show sharp spikes that the forecast misses. We need to investigate those errors before attributing them to a particular event. A missed spike alone cannot tell us whether a macro announcement, an index event, or another source of trading demand caused it.

Second, small stocks and stocks with limited histories have higher errors. We need to distinguish limited data coverage from the model's ability to forecast these securities and assess how the historical fallback behaves.

Third, a forecast made from the previous close cannot respond to information arriving during the session. This creates a clear motivation for testing intraday updates while retaining the daily model as the starting forecast.

### If asked about the evidence and interpretation

The tail and segment findings come from the saved forward diagnostics. They establish differences in forecast performance, rather than a causal explanation for those differences. Coverage and limited history are plausible contributors that require further analysis. The existing six-stock examples do not represent every small stock, new listing, or delisting.

Auction history is already a major input and supplies the 20-day median offset. Further history work should examine adaptation to changes in stock activity, sensitivity to the historical window, and fallback behavior. The proposal does not assume that longer history or faster adaptation will necessarily improve forecasts.

## 23. Roadmap and next steps — 1:30

We propose four directions tied to the remaining challenges.

First, diagnose large errors, coverage, and limited-history fallbacks, and refine how auction history responds to changing stock activity.

Second, test nonlinear models, such as boosted trees, alongside ridge. These could capture thresholds and interactions, for example whether a volume signal behaves differently on event days. This is a hypothesis, not an established improvement. Compare models using the same inputs and chronological folds to isolate the modeling contribution.

Third, test additional daily context: lagged VIX and interest rates, plus scheduled CPI and nonfarm-payroll indicators. We still need to source and validate those inputs. Daily features must be available at the previous close or known in advance.

Fourth, develop intraday updates from the prepared checkpoints to test whether observed trading activity improves the daily forecast.

Compare every extension against the fixed ridge reference, inspecting share-scale error and calibration as well as log errors. Seek additional periods unused in model development and require evidence of incremental value before adding complexity.

### If asked about nonlinear effects

The current ridge specification is linear in its transformed design columns. Transformed inputs, categorical indicators, and prior interaction experiments already introduce some structure, so nonlinear-model work is not a claim that every existing relationship is linear in raw economic units. More flexible models could learn thresholds or interactions without specifying each one manually. This does not guarantee better future forecasts.

Start with the same eligible inputs, target formulation, chronological folds, and evaluation rows as the ridge reference. Fit transformations and tune complexity only on earlier data. Compare log error, WAPE, calibration, tails, and stock groups, and assess whether gains are stable across years. This separates a model-class change from adding new information. The proposed nonlinear models have not been fitted or shown to improve performance in this presentation update.

### If asked about data availability and evaluation

VIX, interest-rate series, and the macro-release calendar are proposed additions. No completed ingestion or improvement from these inputs is claimed here. Specify which rate measure to test and verify historical availability timestamps before building features. A scheduled-release flag must use the schedule known at the forecast origin. A surprise measure requires a contemporaneously available expectation and release value and can enter only after both are available.

The intraday checkpoints already exist at 10:30, 12:00, 14:00, 15:00, and 15:30 Eastern Time. Each update can use only the observations available by that checkpoint. Exclude checkpoints after an early close, and compare daily and updated forecasts on a common eligible sample. Prepared cumulative trade data do not imply that an auction-imbalance feed or an intraday model is available.

Hold the ridge reference fixed during controlled comparisons while continuing to develop other models. Proposed tests should examine both log-space error and share-scale WAPE, volume bias, and segment/tail performance. Tune extensions on historical folds, and reserve additional unexamined periods if obtainable. The previously inspected 2025 year remains confirmation evidence rather than a newly untouched test set. Additional data availability and predictive gains remain open questions.

## Appendix: ridge diagnostic, training and evaluation stages

Backup only, outside the planned narration time.

Each horizontal position identifies a forecast year. The three lines describe in-sample training performance, historical validation performance used for tuning, and the subsequent annual forward evaluation. Both log RMSE and WAPE are lower-is-better metrics. Unlike the main annual-error slide, this appendix compares one ridge specification across stages, rather than two forecasting methods. The forward-evaluation line reproduces the ridge benchmark's annual results shown in the main performance comparison.

Training uses 350,000 sampled rows from a rolling historical window. The historical validation sample has 120,000 rows, with a prior-year validation period. The annual forward evaluation uses all eligible rows in the forecast year. For example, the 2022 fold reports final training over 2017–2021, historical validation in 2021, and forward evaluation in 2022. Historical validation tunes the penalty before the final refit, whose historical window can include that validation year.

These lines therefore compare different periods, samples, and fitted stages. Forward errors below training errors do not establish that the model cannot overfit. Regime changes and sample composition also affect the comparison. The main annual-error slide compares the median rule against ridge on common forward rows. This appendix addresses a separate fitting diagnostic. It adds training and historical-validation errors, and its forward-evaluation line is the same ridge result as the main slide. Preserve the distinction between 2022–2024 development evaluation and 2025 confirmation that had already been inspected during research.

## Literature references

Keep this as a backup slide and do not narrate each citation during the recording.

## Suggested overall timing

- Project definition, literature, and sources: about 6.5 minutes.
- Three descriptive-statistics slides: about 2 minutes.
- Covariate analysis and model findings: about 8.25 minutes.
- Challenges, future roadmap, and next steps: about 2.5 minutes.
- Recording transitions and buffer: 50 seconds.

The deck covers project understanding and background literature, data and descriptive statistics, completed analysis, and challenges with a future roadmap. The “If asked” material supports questions and is outside the planned narration time. The numbered narration targets total 19 minutes 10 seconds, leaving 50 seconds for recording transitions and buffer.
