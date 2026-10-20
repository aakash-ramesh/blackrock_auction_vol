# Recording script

Read the paragraphs below aloud. The numbered headings identify the slides and are not spoken. The main script has approximately 2,500 words. At 130–135 words per minute, allow about 18–19 minutes of speech, with the remaining time for pauses and slide changes. Slides 1–24 form the main recording. The three appendix scripts at the end are separate from the main recording.

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

## 5. Background literature: closing auctions

Two studies help motivate our focus on the close.

Bogousslavsky and Muravyev connect U.S. closing-auction activity with indexing and exchange-traded funds. They also report lower price impact than in continuous trading in their studied setting. This supports the relevance of auction participation, without implying that every order is cheaper to execute there.

Kandel and coauthors examine closing auctions in Milan and Paris. Their findings show that auction design can affect trading near the close.

Together, these studies motivate treating auction volume as a distinct forecasting problem and being careful when extending results across venues. They provide background for our project rather than direct evidence about our own forecast accuracy.

## 6. Background literature: volume forecasting

A related volume-forecasting study is the paper by Białkowski and coauthors on improving volume-weighted average price strategies, often called VWAP strategies.

Their approach separates common market activity from stock-specific dynamics and updates volume forecasts during the day. Our target is different because we focus on the official closing auction. However, the distinction between common conditions and individual trading history is useful motivation.

This leads to our empirical questions. How much can recent stock history explain? Do known calendar events add information? Does a model that pools stocks perform consistently across different sizes and liquidity levels? Our data analysis and benchmark comparisons address these questions.

## 7. Data sources: auction targets and stock history

We combine two main data layers. Auction records come from Trade and Quote extracts obtained through Wharton Research Data Services. CRSP supplies daily stock information, including prices, returns, trading volume, shares outstanding, and corporate actions.

The auction extracts cover 2015 through 2025 and contain about three point six million source rows. CRSP history starts in 2011, allowing us to construct earlier stock-level lags.

We connect records using the trading date and a permanent security identifier, called PERMNO.

These counts describe our extracted sample. Source rows can contain missing outcomes, so they are not the same as the number of usable forecasting labels.

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

The patterns motivate evaluating forecasts forward in time and testing known calendar inputs. Same-day auction share is used here for description; it is not available to the prior-close daily forecast.

## 11. Descriptive statistics: summary

To bring those observations together, we have about three and a half million eligible labels, with annual observed-target coverage between roughly ninety-three and ninety-nine percent.

Auction volume has a long right tail, differs greatly across stocks, and varies over time and around calendar events. These facts explain why one average or one aggregate accuracy measure is insufficient.

We now move from describing the outcome to examining individual predictors.

## 12. Individual covariate relationships with auction volume

We begin the predictor analysis by examining each input separately. This provides economic intuition before we test inputs together.

The left chart uses the stock's recent auction median. The right uses previous-day total trading volume, which includes trading beyond the auction itself.

For each input, we sort stock-days into ten groups, from lowest to highest. Navy shows today's mean auction shares within each group, and teal shows the median.

Both relationships rise strongly: larger recent auctions and higher trading activity generally accompany larger auctions today. These curves combine differences across stocks with changes within a stock. They show associations; the later model tests address added forecast value.

## 13. Stock size and liquidity: individual relationships

We use the same approach here for prior-day market capitalization and quoted spread.

Larger stocks generally have larger auctions. Stocks with tighter bid–ask spreads also generally have larger auctions, although that pattern is not perfectly smooth.

Much of this relationship reflects differences across stocks. The association becomes much weaker when we examine changes within the same stock.

This provides economic context for the inputs. That does not establish a causal effect, or prove that every size and liquidity measure adds value once recent auction history is already known.

## 14. Stock returns and auction volume

These plots consider the previous day's signed return and absolute return.

On the left, the groups run from the largest losses to the largest gains. On the right, they run from small price movements to large movements, regardless of direction.

Neither curve shows a strong, consistent increasing or decreasing relationship with auction shares.

This is useful because a plausible economic story does not always produce a strong marginal relationship. Returns might still matter through thresholds, interactions, or particular market conditions. These plots motivate further testing, rather than a conclusion that return information can never help.

## 15. Market and sector activity: individual relationships

Market and sector inputs are shared across stocks, so we summarize the target at the same level.

The left chart uses each day's average auction shares per eligible stock. The right uses each sector-day's average. We group these outcomes by previous-day market or sector trading volume.

Higher market activity generally accompanies larger average auctions. The sector relationship rises and falls across groups instead of following the same simple pattern.

Composition and market conditions remain mixed together here. We have now examined individual associations across several families. Next, we ask which families add information when other inputs are already present.

## 16. Feature-family removal tests

Starting from a full ridge specification, we remove one economic family at a time while holding the other settings fixed. This addresses incremental information within that model, beyond the separate relationships we just saw.

The chart measures the change in root mean squared error in log auction shares. A positive bar means that removing the family makes forecasts worse.

Auction history and calendar information produce the largest changes. Liquidity's contribution is much smaller and depends on the error measure.

This earlier audit uses direct log levels and expanding training history. Its results support those families within that setup. They do not prove that every retained variable is necessary. We also need to test how the target is formulated.

## 17. Target formulation: deviation from the 20-day median

Another question is how to formulate the prediction problem.

We can predict the log level of auction volume directly, or start from the stock's recent typical auction and predict an adjustment.

Here, the starting point is its trailing twenty-day auction median. The residual is the log of one plus actual auction volume, minus the log of one plus that median. Positive values mean an auction above the recent baseline; negative values mean one below it.

We predict that adjustment, add it back to the baseline in log space, and convert the forecast to shares.

Holding the history window, inputs, and observation weights fixed, development log error falls from about zero point four nine six to zero point four eight seven. This supports the residual formulation within these ridge trials.

## 18. Ridge benchmark: specification comparison

Ridge is one approach we have built so far, and serves as a reference for continued work.

The weights apply to training observations. Equal weights give each stock-day equal influence. Recent-data weights emphasize newer observations with a two-year half-life. These differ from ridge's fitted predictor coefficients.

The full, recently weighted model scores about zero point four eight six eight; the fifty-eight-input model scores zero point four eight seven six. That difference falls within our tolerance of zero point zero zero two.

We use the smaller version as our ridge reference, with five years of history and equal observation weights. Further model comparisons remain open.

## 19. Annual forecast errors: ridge versus recent median

We now evaluate forecasts against a simple historical rule.

The rule predicts the stock's trailing twenty-day auction median. A median of fifty thousand shares gives a forecast of fifty thousand shares. A historical training-median fallback handles unavailable stock medians.

Dark bars show that rule; teal bars show ridge. Both use earlier fitting information and identical eligible evaluation stock-days.

WAPE means total absolute share error divided by total actual shares. Mean annual development WAPE falls from about forty-three point five to thirty-six point eight percent. For a sample, thirty-six point eight percent means absolute errors totaling thirty-six point eight shares per hundred actual shares.

Substantial error remains. The 2025 confirmation year had already been inspected, so it is not an untouched holdout. We next examine the direction of errors.

## 20. Forecast calibration in share volume

Calibration asks whether forecast volumes match actual volumes on average.

We sort the 2022 through 2024 forecasts into ten equally sized groups by predicted auction volume. On the left, each point compares mean forecast shares with mean actual shares. Points above the dashed equality line indicate underprediction. Both axes use logarithmic scales.

The right chart shows forecast shares divided by actual shares within each group. Every point is below one. Across the whole sample, predicted shares total about eighty-six percent of actual shares, a fourteen point four percent shortfall.

This concerns average shares; it does not mean every auction is underpredicted. Fitting in log space makes average-share calibration a separate question. Any correction must be learned using earlier data and evaluated forward.

We now look at how forecasts track activity over time.

## 21. Actual versus predicted volume: 20-day rolling medians

These twenty-day rolling medians show what the forecasts follow over time for two example stocks.

Forecasts follow broad changes in typical activity. The shaded 2015 period contains actuals only; forecasts start in 2016.

The display smoothing includes the current observation and differs from the prior-only median used as the forecast baseline. Each forecast uses earlier fitting data, although the specification was chosen retrospectively.

Smoothing can hide difficult daily departures. The next slide removes that smoothing.

## 22. Daily actual versus predicted auction volume

These unsmoothed examples show JPM in 2021 and CATO in 2017, using years near each stock's median annual log error.

Ridge WAPE is about thirty-one percent for JPM and forty-six percent for CATO, versus thirty-eight and fifty-seven percent for the median rule. Substantial errors remain.

Several spikes rise above the forecast. Their economic causes need investigation. Crosses mark the largest absolute log errors, including overprediction and underprediction.

Different years and vertical ranges make these illustrative cases. Next, we compare both methods across stock groups over one common period.

## 23. Forecast errors across stock groups

Both panels use identical stock-days for ridge and the median rule from 2022 through 2024. WAPE pools observations within each group; the earlier headline averaged annual WAPEs. Counts below the bars are stock-days.

Ridge improves every displayed group. For small-stock observations, WAPE falls from about fifty-three to forty-six percent; for large-stock observations, from forty-one to thirty-four percent.

Cases with fewer than five prior auctions remain difficult: WAPE falls from about one hundred forty-three to ninety-two percent. WAPE can exceed one hundred percent when absolute errors exceed total actual volume.

All cases with fewer than twenty prior auctions account for only one-third of one percent of evaluated stock-days. They deserve focused fallback checks, while overlapping size and history characteristics prevent attributing these differences to a single cause.

## 24. Roadmap and next steps

Our next steps follow from these weaknesses. We will examine large misses, coverage, and limited-history fallbacks. We will test calibration using earlier data and refine how history features adapt to changing activity.

We will compare ridge with nonlinear models, such as boosted trees, beginning with the same inputs and time splits.

Additional daily context could include lagged VIX, interest rates, and scheduled inflation or employment releases. These still need sourcing and testing. Our prepared intraday checkpoints also provide a basis for updates as new information arrives.

New stock-specific options and index-rebalancing studies remain post-midterm work. Extensions will be compared against the fixed ridge reference, seeking periods unused in development.

We have established the data foundation, interpreted predictor relationships, and built a preliminary benchmark that improves on recent history. The remaining errors give us a concrete agenda for the next phase. Thank you.

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

