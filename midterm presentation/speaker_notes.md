# Recording script

Read the paragraphs below aloud. The numbered headings identify the slides and are not spoken. The main script has approximately 2,400 words. At 130–135 words per minute, allow about 18–19 minutes of speech, with the remaining time for pauses and slide changes. Slides 1–23 form the main recording. The two appendix scripts at the end are separate from the main recording.

## 1. Forecasting Closing Auction Volume

Hello everyone. Our project focuses on forecasting the number of shares traded in a stock's closing auction. We'll introduce the problem and data, discuss the patterns we have found, and examine a ridge benchmark. We'll close with the remaining challenges and next steps.

## 2. The closing auction

Let's begin with the closing auction itself. This is a clearing event where buy and sell orders come together to determine an official closing price. Under the exchange's rules, the process seeks to execute as much matched volume as possible.

On a normal New York Stock Exchange trading day, the closing-auction process begins at four p.m. Eastern Time. Early-close days follow a shorter schedule.

The closing price matters for portfolio valuation and index tracking. Investors seeking that benchmark can therefore concentrate their trading near the close. Our focus is the volume of shares executed in the auction. Predicting that volume gives us a starting point for understanding the liquidity available at the end of the day.

## 3. Business objective and project scope

The practical objective is to support decisions about how much of an order to allocate to the close. A volume forecast can help estimate the auction's scale, although a trader's actual fill also depends on their order and market conditions.

The broader project includes daily forecasts and intraday updates across several exchanges and trading venues. For this midterm, our completed modeling focuses on daily forecasts for New York Stock Exchange primary common stocks.

We have prepared intraday trading data, but we are not presenting a completed intraday model yet. Execution-cost savings and broader trading applications remain to be evaluated.

## 4. Forecast target and information set

Our target is straightforward: the official closing-auction volume, measured in shares, for a particular stock and date.

The predictor vector in the equation represents whatever information is available when the forecast is made. For the daily model, that cutoff is the previous trading day's close.

This allows many possible feature families, including auction history, liquidity, stock characteristics, market conditions, and events known in advance. It does not limit us to a fixed set of categories.

For example, yesterday's trading volume is available, and we may already know that tomorrow is an expiry date. Tomorrow's realized trading volume is not available. The key restriction is timing. We then ask whether eligible inputs improve forecasts over simple rules based on recent auction history.

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

These plots examine two inputs separately: the stock's recent auction median and its previous day's total trading volume.

For each input, we sort stock-days into ten groups, from the lowest values to the highest. The navy line shows today's mean auction volume within each group, and the teal line shows its median.

Both relationships rise strongly. Stocks with larger recent auctions or higher trading activity generally have larger auctions today.

This supports their economic relevance, but it does not tell us how much each input adds after other predictors are included. The curves also combine differences across stocks with changes within a stock.

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

The left chart uses each day's average auction shares per eligible stock. The right uses each sector-day's average. We then group those outcomes by the previous day's market or sector trading volume.

Higher market activity generally accompanies larger average auctions. The sector relationship rises and falls across groups instead of showing the same simple increasing pattern.

These comparisons still mix changes in composition and market conditions. They help interpret common activity, but forward forecasting tests are needed to establish incremental value.

## 16. Feature-family removal tests

We now move from individual relationships to inputs considered together.

Starting from a full ridge specification, we remove one economic family at a time and examine the change in forecasting error. The measure is root mean squared error after transforming auction shares into log space. Lower error is better, so a positive bar means that removing the family makes performance worse.

Auction history and calendar information produce the largest changes. Removing liquidity has a much smaller effect, and its contribution depends on the error measure.

This earlier audit uses direct log levels and an expanding training history. It is a controlled comparison within that setup. It does not prove that every retained variable is necessary, or that omitted families are economically irrelevant.

## 17. Target formulation: deviation from the 20-day median

Another question is how to formulate the prediction problem.

We can predict the log level of auction volume directly, or start from the stock's recent typical auction and predict an adjustment.

Here, the starting point is its trailing twenty-day auction median. The residual is the log of one plus actual auction volume, minus the log of one plus that median. Positive values mean an auction above the recent baseline; negative values mean one below it.

We predict that adjustment, add it back to the baseline in log space, and convert the forecast to shares.

Holding the history window, inputs, and observation weights fixed, development log error falls from about zero point four nine six to zero point four eight seven. This supports the residual formulation within these ridge trials.

## 18. Ridge benchmark: specification comparison

Ridge is one modeling approach we have built so far. We use it as a reference for continued work, rather than presenting it as the final project model.

The weights in this table apply to training observations. Equal weights give each stock-day equal influence. Recent-data weights emphasize newer observations, with a two-year half-life. These are different from the predictor coefficients learned by ridge.

The full, recently weighted specification scores about zero point four eight six eight. The fifty-eight-input version scores about zero point four eight seven six. That difference is within our tolerance of zero point zero zero two for these ridge comparisons.

We therefore use the smaller specification as a consistent reference. It uses a five-year history, equal observation weights, and auction-history, calendar, and liquidity inputs. Further model comparisons remain open.

## 19. Annual forecast errors: ridge versus recent median

The next question is whether ridge improves on a simple historical forecast.

The recent-median rule predicts exactly the stock's trailing twenty-day auction median. If that median is fifty thousand shares, the forecast is fifty thousand shares. A historical training-median fallback handles cases without a usable stock median.

The dark bars show that rule, and the teal bars show ridge. Each year's forecasts use earlier fitting data and are evaluated on the same eligible stock-days. Lower bars mean less error.

Our measure is weighted absolute percentage error, or WAPE. We calculate it as total absolute share error divided by total actual shares. Mean annual development WAPE falls from about forty-three point five percent to thirty-six point eight percent. For a given sample, thirty-six point eight percent means absolute errors totaling thirty-six thousand eight hundred shares per hundred thousand actual shares.

The improvement is useful, but the remaining error is substantial. The 2025 confirmation year had already been inspected during research, so it is not an untouched holdout.

## 20. Actual versus predicted volume: 20-day rolling medians

These curves show twenty-day rolling medians of actual and predicted auction shares for two example stocks over the full history.

The forecast follows broad changes in activity. The shaded 2015 period contains actuals only, with forecasts beginning in 2016.

This smoothing is for display. It is different from the prior-only median used as the simple forecast baseline. The model specification was also chosen retrospectively, although each historical forecast uses earlier fitting data.

Smoothing can hide daily misses, so let's look at those next.

## 21. Daily actual versus predicted auction volume

These are unsmoothed forecasts for JPM in 2021 and CATO in 2017. Each example year is selected near that stock's median annual error.

Ridge WAPE is about thirty-one percent for JPM and forty-six percent for CATO. Those are substantial errors. The corresponding median-rule errors are about thirty-eight and fifty-seven percent.

Ridge improves both examples, but that does not establish sufficient accuracy for execution decisions. Several auction spikes still rise well above the forecast.

The crosses mark the largest absolute log errors, which can include overprediction and underprediction. Different years and vertical ranges make these illustrative cases, rather than a controlled comparison of stock size.

## 22. Remaining challenges

The results leave three practical challenges.

First, large auctions and sharp spikes remain difficult. We need to investigate these misses before attributing them to particular economic events.

Second, errors are higher for small stocks and stocks with limited histories. Coverage, fallback behavior, and adaptation to changing activity all deserve further examination.

Third, the prior-close forecast cannot respond to information arriving during the trading day. This creates a clear reason to test intraday updates.

These challenges motivate better calibration and further modeling work.

## 23. Roadmap and next steps

Our next steps follow directly from those challenges.

We will first examine large errors, coverage, and limited-history cases, and refine how historical features respond to changing stock activity.

We also plan to compare ridge with nonlinear models, such as boosted trees. These could capture thresholds or interactions that the current specification misses. We will begin with the same information set and time splits, so we can distinguish a model change from adding new data.

Another direction is additional daily context, including lagged VIX and interest rates and scheduled inflation or employment releases. These inputs still need to be sourced and tested. We also plan to develop forecast updates using the prepared intraday checkpoints.

For every extension, we will compare accuracy and calibration against the fixed ridge reference, examine difficult stocks and large auctions, and seek additional periods unused in model development.

At this stage, we have established the data foundation, identified useful patterns, and built a ridge benchmark that improves on recent history while leaving important errors to address. That is the basis for the next phase of the project. Thank you.

---

# Appendix scripts — outside the main recording

## 24. Ridge diagnostic: training and evaluation stages

This additional slide examines one ridge specification across training, historical validation, and forward evaluation. It answers a different question from the annual bar chart, which compares ridge against the recent-median rule.

Here, the training line measures error on the historical sample used for the refit. The validation line uses an earlier validation year, and the forward-evaluation line measures subsequent-year predictions. The latter is the same ridge result used in the main annual comparison.

The charts use log-scale root mean squared error and WAPE. Lower is better for both. However, the lines cover different periods and samples, so their ordering is not a simple test for overfitting. Lower forward error than training error does not by itself establish that overfitting is absent.

The 2025 result remains confirmation evidence from a year already inspected during research.

## 25. References: background literature

This slide lists the three background studies discussed earlier. They motivate our interest in closing-auction activity, venue design, and the use of common and stock-specific information in volume forecasting. They provide context for the project; the performance results shown today come from our own data analysis and forecasting comparisons.
