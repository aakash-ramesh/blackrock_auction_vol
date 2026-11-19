# Literature review and proposed closing-auction control formulation

Research date: 18 November 2026. This is formulation research and a local data audit, not a new execution backtest or implementation of notebook 12.

**Recommendation:** start with a close-benchmarked stochastic execution model using the tuned auction-volume forecasts, historical joint price/liquidity scenarios, and explicit cost sensitivities. A directional price-prediction model is optional. The available data support coarse interval-price and auction-relative risk estimation, but not identification of the hypothetical order's queue fills or causal market impact.

## 1. Relevant primary literature

### Almgren and Chriss — Optimal Execution of Portfolio Transactions (2001; December 2000 manuscript)

[Original paper, PDF mirror](https://smallake.kr/wp-content/uploads/2016/03/optliq.pdf).

Foundational treatment of expected execution cost versus price uncertainty, with temporary and permanent impact. Its stochastic-price setup makes clear that execution optimization can begin without a directional price forecast. Its arrival-price risk formulation must be adapted to a close benchmark. This is a theoretical foundation, rather than a closing-auction calibration recipe.

### Frei and Westray — Optimal Execution in Hong Kong given a Market-On-Close Benchmark (2018)

[Author manuscript](https://www.math.ualberta.ca/~cfrei/CFNW2018.pdf); [publisher record](https://doi.org/10.1080/14697688.2017.1334126).

Studies execution relative to the specific historical Hong Kong closing benchmark and separates benchmark-related and execution-related risk. The transferable lesson is to define the benchmark's actual construction before optimizing slippage. Its historical median-price mechanism and parameters should not be transplanted into NYSE auction modelling. The indexed abstract and introduction were available; the author PDF repeatedly timed out during this review, so detailed implementation claims are not inferred from it.

### Frei and Yan — An Explicit Optimal Strategy for Flow Trades at NASDAQ around its Close

[Author manuscript](https://www.math.ualberta.ca/~cfrei/CFCY2018.pdf).

The most directly relevant analytic template found: split a fixed order between continuous trading and a closing auction, with stochastic regular-market price changes, a terminal auction-price shock, imbalance-related price effects and an explicit close-relative objective. Its stylized assumptions include independence restrictions and a Nasdaq mechanism. We can borrow the state/action structure while treating imbalance and impact coefficients as unidentified with our present inputs.

### Cartea and Jaimungal — A Closed-Form Execution Strategy to Target Volume Weighted Average Price (2016)

[Published-paper institutional copy](https://ora.ox.ac.uk/objects/uuid%3Ab68272ca-a941-4d48-a87a-d06963482db9/files/m728f9239cf2fd998ac4d6246359dbb7d); [author's research page](https://utstat.toronto.edu/sjaimung/research.htm).

Provides a stochastic-volume execution framework in which expected remaining activity and order flow influence trading. Its benchmark is VWAP, not the close; the schedule cannot be adopted unchanged. It supports making future liquidity conditional on information currently available. The indexed original-paper abstract and author's description were reviewed; the institutional full-text endpoint returned an access error.

### Goyal, Jegadeesh and Wu — Price Impact in Closing Auctions, Opening Auctions, and Continuous Markets (2026)

[Published article](https://doi.org/10.1017/S0022109026102592).

Empirically compares auction and continuous-market price impact and finds a square-root specification useful in its sample. It motivates separate auction/continuous cost curves and stock heterogeneity. Their auction imbalance and market-price inputs are richer than ours, so reported coefficients are external sensitivity references, not estimates of our simulated order's causal impact.

### Salek, Challet and Muni Toke — Price impact in equity auctions: zero, then linear (2023 manuscript)

[Primary preprint](https://arxiv.org/abs/2301.05677v2).

Finds distinct impact regions in Paris auction books, including tick-related zero impact, a linear region and nonlinear behavior for larger perturbations. It warns against treating one linear or square-root cost curve as a universal structural law. Venue, data and impact definitions differ from the preceding paper; the two studies should motivate sensitivity analysis rather than a forced choice based on title alone.

### Challet and Gourianov — Dynamical regularities of US equities opening and closing auctions (2018 manuscript)

[Primary preprint](https://arxiv.org/abs/1802.01921v2).

Studies indicative prices, imbalances and auction dynamics. It motivates separating total auction liquidity from signed price pressure and explains why indicative-price/imbalance data would enrich a future observation model. We do not have those fields in the audited checkpoint schemas.

### Graf and Mastrolia — Learning Optimal Liquidation with Closing Auctions (2026, version 3)

[Primary preprint](https://arxiv.org/abs/2601.17247v3); [full text](https://arxiv.org/html/2601.17247v3).

A recent two-phase continuous-market/auction control example. Historical inputs provide midprice paths, while order flow and auction mechanics remain simulated. This supports an explicitly labelled hybrid historical/simulation experiment, not a claim that historical prices validate simulated fills. The added simulator assumptions and learning machinery exceed what is needed for our first small dynamic program.

### Exchange evidence and current mechanics

[NYSE auction timelines](https://www.nyse.com/trade/auctions) and [official order-type fact sheet](https://www.nyse.com/publicdocs/nyse/markets/nyse/NYSE_Opening_and_Closing_Auctions_Fact_Sheet.pdf) establish order-specific restrictions. The current general NYSE MOC/LOC cutoff is 15:50 ET, with exceptions and restricted changes afterward. These are current mechanics, not an assertion that every historical period had identical rules. Our last observation at 15:30 is an observation limit, not the exchange deadline.

[NYSE's August 2023 impact research](https://www.nyse.com/data-insights/closing-auction-immediate-market-impact-price-drift-and-transaction-cost-of-trading) uses imbalance messages and reference-price changes to measure impact and subsequent drift. It illustrates the identification gap between our aggregate tape and an order-level impact study.

## 2. What historical price information is actually available locally?

Audited schemas:

- `blackrock_data/stock_daily/auction_daily/year=2025/auction_daily_2025.parquet`: auction price, auction shares, auction timestamp, target-found flags, CRSP daily OHLC/bid/ask and adjustment fields, plus auction-versus-close diagnostics.
- `blackrock_data/stock_daily/crsp_daily/year=2025/crsp_daily_2025.parquet`: daily OHLC/bid/ask, returns, price/share adjustments and security context.
- `blackrock_data/stock_intraday/trade_checkpoints/year=2025/trade_checkpoints_2025.parquet`: cumulative dollar/share volume, cumulative VWAP, recent volume, quality flags and times 10:30, 12:00, 14:00, 15:00 and 15:30.

A schema presence check is not a complete quality certification. A targeted 2024–2025 audit found:

| Year | Auction/daily rows | Positive matched auction price and volume | Positive daily close |
| --- | ---: | ---: | ---: |
| 2024 | 321,631 | 316,212 | 319,880 |
| 2025 | 311,210 | 307,853 | 309,901 |

These are raw-file counts, not the final regular-session simulation universe. Early-close, anomaly, checkpoint and price-quality filters still apply. The audited files end in December 2025; all evaluated forecasting years have already been inspected.

A continuous-window historical VWAP can be reconstructed from increasing cumulative fields:

\[
\bar P_{(t_k,t_{k+1}]}=
\frac{\mathrm{CumDollar}_{t_{k+1}}-\mathrm{CumDollar}_{t_k}}
{\mathrm{CumVolume}_{t_{k+1}}-\mathrm{CumVolume}_{t_k}}.
\]

It is an outcome for that completed interval, unknown at its starting decision. Cumulative checkpoint VWAP is an average since the source session's start, not the instantaneous midprice or executable bid/ask at the checkpoint. Daily bid/ask fields do not supply timestamped intraday quotes. We must check corporate actions, stale/nonpositive values, identifier alignment and aggregation conventions before forming returns.

The checked schemas do not provide an intraday auction indicative-price series, signed imbalance, full order books, our order's queue position or execution records, or a measured continuous window after 15:30. The gap from the 15:30 proxy to auction price therefore combines unobserved continuous price movement and auction-specific price formation. It cannot be identified as a pure auction jump.

## 3. Price uncertainty can be modelled before price direction

We need a conditional distribution of prices and liquidity. It is not necessary to begin with a model claiming to predict which way prices will move.

A useful hierarchy is:

1. **No-alpha reference:** assume zero conditional directional return; estimate historical interval variation and close-relative covariance.
2. **Historical joint scenarios:** resample or weight whole prior-day vectors of window volumes, interval prices, final auction volume and auction price. Preserve their dependence. Conditioning uses only current observables and prior history.
3. **Optional price signal:** test a simple historically validated predictor of future auction-relative returns. Set its mean to zero when evidence does not justify the additional signal.

In a scale-invariant, no-alpha model, the absolute price level may cancel from notional-normalised decisions. Price affects allocation through relative expected prices, volatility, covariance, trading costs and constraints. One should not claim that merely observing a stock price creates timing alpha.

Forecast auction volume is not signed auction imbalance. A strong volume model does not automatically predict auction-price direction or own-order impact.

## 4. Define the benchmark and completion mandate before solving

Let Q be a parent buy quantity, x_k the continuous fills, and a the auction quantity. For a completed order, sum(x_k)+a=Q. Let B^0 be the unaffected closing-price reference, approximated in replay by the historical auction print absent our additional simulated order. Let P_k^0 be continuous-window reference prices.

A transparent economic close-relative cost is

\[
C^\pi=\sum_k x_k(P_k^0-B^0)
+\sum_k c_k(x_k,V_k)+c_A(a,V_A).
\]

Auction reference cash cancels against the benchmark for shares executed at B^0; auction participation still has explicit fees/impact cost. Normalise dollars to basis points with the known arrival notional Q P_ref, rather than an unknown future denominator.

This baseline treats historical prices as exogenous reference paths and costs as additional modelled costs. If we instead let our auction order change the clearing price, it also changes the actual official-close benchmark. We must then distinguish an unaffected economic reference from endogenous close tracking. Adding impact to paid price while silently retaining the old benchmark and calling it actual-close tracking is inconsistent. Benchmark slippage can also hide economic harm to the rest of a fund's holdings when the trader moves the close.

For the initial mandate, use full completion as an explicit assumption with an eligible market-on-close terminal order and model its cost. Actual execution is not guaranteed solely by our available data. Partial completion is a distinct order/mandate scenario requiring an explicit terminal mark/penalty or genuine LOC/rejection model.

### Revise the old participation/fill convention

The earlier prototype used fill=min(request, rho*historical auction volume). That is a stylized capacity convention. A fixed 5% fraction is not an exchange allocation rule and aggregate total volume is not a queue-fill model.

Prefer a self-imposed participation guardrail, an impact cost, or the optional chance condition

\[
\Pr(a/V_A>\rho\mid\mathcal F_t)\le\epsilon.
\]

For a quantity chosen at t and a continuous positive volume distribution, this corresponds to a lower-tail limit a <= rho F^{-1}_{V_A|F_t}(epsilon). It is not a guarantee of completion or of exchange fills. A strict completion mandate and an inflexible participation limit can be infeasible together; define slack or fallback explicitly.

## 5. Why the close benchmark changes timing risk

This is our derivation under a stylized baseline, not an empirical claim about the dataset. Suppose dS_t=sigma_t dW_t, B^0=S_T+J, J is independent and centred with variance tau_A^2, and X_t is cumulative shares already executed continuously. Assume full completion, predictable finite-variation trading and no impact for this calculation.

The price part of close-relative cost is

\[
L_{price}=\int_0^T(S_t-B^0)\,dX_t
=-\int_0^T X_t\,dS_t-X_TJ.
\]

Under these assumptions,

\[
\operatorname{Var}(L_{price})
=\mathbb E\!\left[\int_0^T\sigma_t^2X_t^2\,dt\right]
+\tau_A^2\mathbb E[X_T^2].
\]

Thus already-executed shares create exposure to later movements of the close benchmark. An arrival-price problem instead exposes remaining inventory. This is why automatically copying an arrival-price q_t^2 risk penalty can reverse the timing incentive. With drift, correlated auction shocks/liquidity, different benchmarks or price-dependent fills, derive or simulate the full expression rather than assuming this variance formula remains exact.

Our checkpoint data can estimate coarse close-relative price covariance. They cannot separately identify sigma for the final unobserved half-hour and tau_A from a last-preauction quote. A direct empirical joint model avoids pretending that decomposition is observed.

## 6. Objective and solution method

The main recommendation is a small scenario-based dynamic program. State variables include remaining inventory, current price proxy, cumulative executed quantity, latest volume forecast, volatility context and the conditional scenario distribution. Observations are revealed only when their checkpoint arrives. A scenario tree must enforce identical actions for paths sharing the same observed history; it must not reveal later historical prices or forecasts at noon.

A clean time-consistent risk-sensitive objective, on basis-point total cost L, is

\[
\min_\pi\ \rho_\gamma(L^\pi),\qquad
\rho_\gamma(L)=\frac1\gamma\log\mathbb E[e^{\gamma L}],\quad\gamma>0.
\]

On a fixed Gaussian-cost schedule this equals expected cost plus gamma/2 times its variance. Under dynamic policies or non-Gaussian liquidity it must be evaluated as defined. It requires finite exponential moments; finite empirical scenarios do not prove that the real tail distribution satisfies this assumption, so include tail stresses and avoid extrapolating the utility guarantee.

For a simpler first solver, an additive quadratic close-tracking risk penalty is transparent and compatible with the earlier DP. Label it a risk surrogate outside the assumptions in Section 5. Static mean-variance is an interpretable one-decision benchmark, but re-solving unconditional mean-variance at every checkpoint is not automatically an ordinary Bellman problem; state whether decisions are precommitted, use an augmented formulation, or use the time-consistent risk-sensitive objective above.

### One-decision diagnostic

Buy u shares continuously at reference price p and reserve Q-u for the auction. For deterministic cost curves and uncertain B^0 with conditional mean gap m=E[B^0-p|F_t] and variance s^2,

\[
J(u)=c_c(u)+c_A(Q-u)-u m+\lambda u^2s^2.
\]

This diagnostic shows the competing forces: positive expected auction premium favours buying earlier; close-relative uncertainty penalises doing so; expensive auction participation favours continuous trading. Start with m=0 and estimate s^2. With uncertain auction costs or volume-price correlation, use the full joint scenarios rather than this reduced expression.

## 7. Connect our forecast models to joint scenarios

Use the selected tuned daily/15:30 ridge forecasts as historical information, with checkpoint timing and actual fitting windows preserved. Do not carry Kalman state variance into the controller as if it were a calibrated auction-volume distribution.

One possible volume scenario uses an out-of-time residual e_A:

\[
V_A^{(s)}=\max\{0,\exp[\log(1+\widehat V_{A,t})+e_A^{(s)}]-1\}.
\]

Sample e_A jointly with continuous-window liquidity and relative-price errors from the same historical session. This is a candidate scenario construction, not an established calibrated model. Validate lower-tail coverage, large-volume behavior and conditional dependence. Avoid blindly multiplying a current prediction by an unconditional historical volume ratio.

At noon, later forecasts must be scenario outcomes, not today's realised future outputs. At 15:30 a forecast can affect auction submission, budget/limit choice or a future continuous window if one is explicitly modelled. Under a rule that sends all remaining shares to auction and has no later continuous action, a refreshed volume prediction may have no remaining allocation decision to change. The current exchange cutoff does not create missing post-15:30 observations in our dataset.

## 8. What should be estimated, assumed, or postponed?

| Component | Initial treatment |
| --- | --- |
| Auction-volume location | Existing out-of-time forecasting models |
| Auction-volume forecast errors | Historical residual calibration; validate lower tail |
| Window reference prices and close basis | Reconstruct from historical cumulative fields; prior-only joint scenario calibration |
| Directional price drift | Zero baseline; optional simple predictor with genuine incremental validation |
| Dependence of price and volume | Preserve in whole-day scenarios; diagnose by liquidity/event group |
| Spreads and causal impact | Sensitivity ranges and external empirical guidance; not identified from aggregate fields |
| Continuous fills | Participation-limited proxy with clearly labelled execution assumptions |
| Auction order allocation | Explicit order-type model/mandate; do not infer from total auction volume alone |
| Indicative prices/imbalance | Obtain new feed if a realistic auction-clearing model is required |
| Post-15:30 continuous trading | Additional compatible data, or explicitly simulated sensitivity |

Compare linear, square-root and, if useful, piecewise auction-impact costs. Separate coefficients by continuous/auction mechanism, with plausible stock heterogeneity. Published impact estimates have different denominators, venues and measurement definitions; do not paste them into a/V_A formulas without conversion and assumptions.

## 9. Proposed sequence before notebook 12 implementation

1. Freeze benchmark, side, completion mandate and allowed order types.
2. Build a price/volume quality and timing audit, including interval VWAPs and close-basis distributions.
3. Solve the one-decision diagnostic with zero drift, historical price risk and volume uncertainty.
4. Restore the old historical optimizer as a reference, explicitly labelling its stylized fill rule.
5. Build a small matched scenario DP, adding volume forecasting and then price uncertainty as separate controlled changes.
6. Test zero-alpha and optional price-signal variants, impact curves, risk aversion and participation constraints.
7. Report economic cost, close tracking, allocations, completion, and tail outcomes separately, with paired date-block comparisons and sensitivity to assumptions.

No price predictor, queue model, new control policy or new forward period was trained/scored for this literature review.
