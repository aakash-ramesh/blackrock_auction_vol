# Recording script

Read the paragraphs below aloud. The numbered headings identify the slides and are not spoken. The main script has approximately 2,600 words. At about 135 words per minute, allow roughly 19 minutes of speech, with the remaining time for pauses and slide changes. Slides 1–24 form the main recording. The three appendix scripts at the end are separate from the main recording.

## 1. Forecasting Closing Auction Volume

Hello everyone. We are Aakash Ramesh, Arjo Bhattacharya, Alisa Priyadarsini, and Ivan Sysoev. Our project focuses on forecasting the number of shares traded in a stock's closing auction. We'll introduce the problem and data, discuss the patterns we have found, and examine a ridge benchmark. We'll close with the remaining challenges and next steps.

## 2. The closing auction

Let's begin with the closing auction itself. This is a clearing event where buy and sell orders come together to determine an official closing price. Under the exchange's rules, the process seeks to execute as much matched volume as possible.

On a normal New York Stock Exchange trading day, the closing-auction process begins at four p.m. Eastern Time. Early-close days follow a shorter schedule.

The closing price matters for portfolio valuation and index tracking. Investors seeking that benchmark can therefore concentrate their trading near the close. Our focus is the volume of shares executed in the auction. Predicting that volume gives us a starting point for understanding the liquidity available at the end of the day.

## 3. Business objective and project scope

The practical objective is to support decisions about how much of an order to allocate to the close. A volume forecast can help estimate the auction's scale, although a trader's actual fill also depends on their order and market conditions.

The broader project includes daily forecasts and intraday updates across several exchanges and trading venues. For this midterm, our completed modeling focuses on daily forecasts for New York Stock Exchange primary common stocks.

We have prepared intraday trading data, but we are not presenting a completed intraday model yet. Execution-cost savings and broader trading applications remain to be evaluated.

## 4. Forecast target and economic inputs

Our target is the official closing-auction volume in shares for each stock and date. The equation allows any predictors available at the previous trading day's close.

The table organizes our economic hypotheses. Recent auction and trading activity may tell us about persistent demand. Stock size and liquidity may explain differences in trading scale and participation. Market and sector activity may capture common changes in demand. Known calendar information may capture scheduled changes in participation or session length.

These are reasons to investigate the variables, rather than claims that their effects are already established. Other predictor families remain possible.

The restriction is availability. Yesterday's volume and tomorrow's known calendar flags are eligible; tomorrow's realized volume is not. We will examine individual relationships, test what families add together, and evaluate forecasts against recent-history rules.

## 5. Background literature

Three studies provide the background for our project.

Bogousslavsky and Muravyev connect U.S. closing-auction activity with indexing and exchange-traded funds. They report lower price impact than continuous trading in their setting, which does not imply that every order is cheaper to execute there.

Kandel and coauthors study Milan and Paris. They show that closing auctions affect trading near the close and that auction design matters for participation.

Białkowski and coauthors develop an intraday volume model for volume-weighted average price strategies. Their model separates common market activity from stock-specific dynamics and updates forecasts during the day.

These papers establish context. They do not directly validate our closing-auction forecasts.

## 6. Research questions

We organize our analysis around four questions.

First, how much can recent auction history explain? We compare predicting the log level directly with predicting a correction to the stock's trailing twenty-day auction median.

Second, what information do other inputs add? We examine liquidity, known calendar information, stock characteristics, and market or sector conditions. Individual relationships provide economic intuition; removing input families tests their contribution within a model.

Third, can forecasting models beat simple historical rules in chronological forward tests? We examine both proportional accuracy and errors measured in shares.

Finally, where does performance weaken? We compare stock sizes and history lengths, examine missed daily spikes, and check whether forecast volumes are systematically too low.

All daily predictors must be available at the previous close or known in advance. These questions guide the data and results that follow.

## 7. Data sources: auction targets and stock history

We combine two main data layers. Auction records come from Trade and Quote extracts obtained through Wharton Research Data Services. CRSP supplies daily stock information, including prices, returns, trading volume, shares outstanding, and corporate actions.

Both rows show our common 2015 through 2025 analysis window, with about three point six million rows in each source. Earlier source history is available for constructing lags.

We connect records using the trading date and a permanent security identifier, called PERMNO.

Source rows can contain missing outcomes. Our analytical sample has about three and a half million eligible labels, with annual observed-target coverage between roughly ninety-three and ninety-nine percent.

## 8. Data sources: context and intraday trades

We also have daily market and sector measures, calendar information, and security reference data. Market and sector outcomes enter the daily analysis with appropriate lags, while scheduled events can be known ahead of time.

The intraday data summarize trading at five checkpoints, from ten-thirty in the morning through three-thirty in the afternoon, Eastern Time. They could support updates to the daily forecast as the session progresses.

Checkpoints after an early close must be excluded.

With the sources established, we can first describe the outcome and then examine how the candidate inputs relate to it.

## 9. Auction-volume distribution and stock-size comparison

This slide describes the distribution and scale of auction volume before modeling.

On the left, the median auction is about fifty-five thousand shares, while the mean is about one hundred ninety thousand. Large auctions pull the average upward.

On the right, median auction volume rises sharply across stock-size groups. The largest-size group's median is about eighty-one times the smallest group's median.

These patterns motivate accounting for stock scale and examining both proportional errors and errors measured in shares.

## 10. Auction participation: time and calendar comparisons

Here, we examine the auction's share of total daily trading volume.

The left chart shows that the median share rises from about five percent in 2015 to fourteen percent in 2025 in our changing sample.

The right chart shows higher participation around expiry and month-end dates. These event groups overlap, so the comparisons do not identify separate causal effects.

Same-day auction share is used here for description; it is not available to the prior-close daily forecast.

Together, these patterns motivate accounting for each stock's typical auction size and evaluating forecasts across stocks and over time. We now examine which historical inputs help explain auction volume.

## 11. Individual covariate relationships with auction volume

We begin the predictor analysis by examining each input separately. This provides economic intuition before we test inputs together.

The left chart uses the stock's recent auction median. The right uses previous-day total trading volume, which includes trading beyond the auction itself.

For each input, we sort stock-days into ten groups, from lowest to highest. Navy shows today's mean auction shares within each group, and teal shows the median.

Both relationships rise strongly: larger recent auctions and higher trading activity generally accompany larger auctions today. These curves combine differences across stocks with changes within a stock. They show associations; the later model tests address added forecast value.

## 12. Stock size and liquidity: individual relationships

We use the same approach here for prior-day market capitalization and quoted spread.

Larger stocks generally have larger auctions. Stocks with tighter bid–ask spreads also generally have larger auctions, although that pattern is not perfectly smooth.

Much of this relationship reflects differences across stocks. The association becomes much weaker when we examine changes within the same stock.

This provides economic context for the inputs. That does not establish a causal effect, or prove that every size and liquidity measure adds value once recent auction history is already known.

## 13. Stock returns and auction volume

These plots consider the previous day's signed return and absolute return.

On the left, the groups run from the largest losses to the largest gains. On the right, they run from small price movements to large movements, regardless of direction.

Neither curve shows a strong, consistent increasing or decreasing relationship with auction shares.

This is useful because a plausible economic story does not always produce a strong marginal relationship. Returns might still matter through thresholds, interactions, or particular market conditions. These plots motivate further testing, rather than a conclusion that return information can never help.

## 14. Market and sector activity: individual relationships

Market and sector inputs are shared across stocks, so we summarize the target at the same level.

The left chart uses each day's average auction shares per eligible stock. The right uses each sector-day's average. We group these outcomes by previous-day market or sector trading volume.

Higher market activity generally accompanies larger average auctions. The sector relationship rises and falls across groups instead of following the same simple pattern.

Composition and market conditions remain mixed together here. We have now examined individual associations across several families. Next, we ask which families add information when other inputs are already present.

## 15. Feature-family removal tests

Starting from a full ridge specification, we remove one economic family at a time while holding the other settings fixed. This addresses incremental information within that model, beyond the separate relationships we just saw.

The chart measures the change in root mean squared error in log auction shares. A positive bar means that removing the family makes forecasts worse.

Auction history and calendar information produce the largest changes. Liquidity's contribution is much smaller and depends on the error measure.

This earlier audit uses direct log levels and expanding training history. Its results support those families within that setup. They do not prove that every retained variable is necessary. We also need to test how the target is formulated.

## 16. Target formulation: deviation from the 20-day median

We can predict the log level directly, or start from each stock's trailing twenty-day auction median and predict a correction.

The residual is the log of one plus actual shares, minus the log of one plus that historical median. Positive values mean above the baseline; negative values mean below it. We add the predicted correction back in log space and convert to shares.

Log RMSE was our development criterion. It compares errors across stock scales and penalizes large log misses without letting the largest share volumes dominate.

Holding the inputs, history window, and weights fixed, mean annual log RMSE falls from about zero point four nine six to zero point four eight seven. The relative-error checks agree: MAPE falls from thirty-seven point five to thirty-four point seven percent, and MSPE from two point seven seven to one point four eight.

MAPE averages absolute errors divided by actual shares; MSPE squares those relative errors, using unscaled ratios. All actuals here are positive. These metrics are especially sensitive to small actual auctions.

## 17. Ridge benchmark: specification comparison

Ridge is one approach we have built so far and remains a preliminary benchmark.

Training weights control each observation's influence. Equal weights treat stock-days equally; recent-data weights emphasize newer observations with a two-year half-life. They differ from fitted predictor coefficients.

We prefer fewer inputs within zero point zero zero two of the best development log RMSE. The fifty-eight-input version meets that tolerance and becomes our reference, using five years of history.

The extra columns show why the metric choice matters. This reference has the lowest MAPE among these three specifications, at about thirty-four point six percent. The larger, recently weighted model has the lowest log RMSE and MSPE.

The comparison supports a practical reference within ridge, while further model development and evaluation remain open.

## 18. Annual forecast errors: ridge versus recent median

We now evaluate forecasts against a simple historical rule.

The rule predicts the stock's trailing twenty-day auction median. A median of fifty thousand shares gives a forecast of fifty thousand shares. A historical training-median fallback handles unavailable stock medians.

Dark bars show that rule; teal bars show ridge. Both use earlier fitting information and identical eligible evaluation stock-days.

WAPE means total absolute share error divided by total actual shares. Mean annual development WAPE falls from about forty-three point five to thirty-six point eight percent. For a sample, thirty-six point eight percent means absolute errors totaling thirty-six point eight shares per hundred actual shares.

Substantial error remains. The 2025 confirmation year had already been inspected, so it is not an untouched holdout. We next examine the direction of errors.

## 19. Forecast calibration in share volume

Calibration asks whether forecast volumes match actual volumes on average.

We sort the 2022 through 2024 forecasts into ten equally sized groups by predicted auction volume. On the left, each point compares mean forecast shares with mean actual shares. Points above the dashed equality line indicate underprediction. Both axes use logarithmic scales.

The right chart shows forecast shares divided by actual shares within each group. Every point is below one. Across the whole sample, predicted shares total about eighty-six percent of actual shares, a fourteen point four percent shortfall.

This concerns average shares; it does not mean every auction is underpredicted. Fitting in log space makes average-share calibration a separate question. Any correction must be learned using earlier data and evaluated forward.

We now look at how forecasts track activity over time.

## 20. Actual versus predicted volume: 20-day rolling medians

These twenty-day rolling medians show what the forecasts follow over time for two example stocks.

Forecasts follow broad changes in typical activity. The shaded 2015 period contains actuals only; forecasts start in 2016.

The display smoothing includes the current observation and differs from the prior-only median used as the forecast baseline. Each forecast uses earlier fitting data, although the specification was chosen retrospectively.

Smoothing can hide difficult daily departures. The next slide removes that smoothing.

## 21. Daily actual versus predicted auction volume

These unsmoothed examples show JPM in 2021 and CATO in 2017, using years near each stock's median annual log error.

Ridge WAPE is about thirty-one percent for JPM and forty-six percent for CATO, versus thirty-eight and fifty-seven percent for the median rule. Substantial errors remain.

Several spikes rise above the forecast. Their economic causes need investigation. Crosses mark the largest absolute log errors, including overprediction and underprediction.

Different years and vertical ranges make these illustrative cases. Next, we compare both methods across stock groups over one common period.

## 22. Forecast errors across stock groups

Both panels use identical stock-days for ridge and the median rule from 2022 through 2024. WAPE pools observations within each group; the earlier headline averaged annual WAPEs. Counts below the bars are stock-days.

Ridge improves every displayed group. For small-stock observations, WAPE falls from about fifty-three to forty-six percent; for large-stock observations, from forty-one to thirty-four percent.

Cases with fewer than five prior auctions remain difficult: WAPE falls from about one hundred forty-three to ninety-two percent. WAPE can exceed one hundred percent when absolute errors exceed total actual volume.

All cases with fewer than twenty prior auctions account for only one-third of one percent of evaluated stock-days. They deserve focused fallback checks, while overlapping size and history characteristics prevent attributing these differences to a single cause.

## 23. Current and remaining challenges

We have four main challenges to address.

First, sharp auction spikes remain difficult, and total forecast shares are about fourteen point four percent below actual shares in our development sample.

Second, small stocks and stocks with short histories still have high errors despite improvements over the median rule. Short-history cases are rare, but their fallback forecasts need attention.

Third, our daily information cutoff prevents the forecast from responding to news or trading activity arriving during the session.

Finally, ridge remains a preliminary benchmark. The 2025 confirmation year was already inspected, so we need additional evaluation periods unused in development.

These limitations determine our next steps.

## 24. Roadmap and next steps

We will investigate large misses and coverage, test calibration using earlier data, and refine history features and limited-history fallbacks.

We will compare nonlinear models with ridge using the same inputs and time splits. Additional context, including lagged VIX, rates, and scheduled economic releases, still needs sourcing and testing. Prepared trading checkpoints provide a basis for intraday updates.

New stock-specific options and index-rebalancing studies remain post-midterm work. All extensions need comparison with the fixed ridge reference and further evaluation on unused periods.

We have a data foundation, interpretable relationships, and a preliminary benchmark that improves on recent history. The remaining challenges give us a clear agenda for the next phase. Thank you.

---

# Appendix scripts — outside the main recording

## 25. Ridge diagnostic: training and evaluation stages

This additional slide examines one ridge specification across training, historical validation, and forward evaluation. It answers a different question from the annual bar chart, which compares ridge against the recent-median rule.

Here, the training line measures error on the historical sample used for the refit. The validation line uses an earlier validation year, and the forward-evaluation line measures subsequent-year predictions. The latter is the same ridge result used in the main annual comparison.

The charts use log-scale root mean squared error and WAPE. Lower is better for both. However, the lines cover different periods and samples, so their ordering is not a simple test for overfitting. Lower forward error than training error does not by itself establish that overfitting is absent.

The 2025 result remains confirmation evidence from a year already inspected during research.

## 26. Covariates and deviations from the recent median

This supporting analysis asks which inputs are associated with departures from a stock's recent typical auction.

The outcome is the log of one plus actual shares, minus the log of one plus the prior twenty-day auction median. Zero means equality to the baseline. Positive values mean an auction above it; negative values mean one below it.

Each panel groups stock-days by a prior-close input. Navy shows the mean log deviation and teal the median. Trading-volume groups show increasing departures. The market-capitalization relationship is much flatter than its relationship with raw auction shares.

On the same paired observations, market capitalization's rank correlation falls from about zero point six three with auction shares to about zero point zero one with this adjusted outcome. This suggests its strong level association mainly describes scale in this sample.

These remain descriptive associations, without controls for other predictors. They do not establish causality or incremental forecast value. The analysis uses 2022 through 2024 and excludes cases without a stock-specific recent median. Additional spread and absolute-return comparisons are available in the supporting analysis.

## 27. References: background literature

This slide lists the three background studies discussed earlier. They motivate our interest in closing-auction activity, venue design, and the use of common and stock-specific information in volume forecasting. They provide context for the project; the performance results shown today come from our own data analysis and forecasting comparisons.

