"""Market-wide regime inputs for the HMM work (Notebook 16c onward).

One row per trading day. Every input is first computed "as of the close of day d"
using data up to and including d, then shifted forward one trading day, so the
value stored on date t uses only information known by the end of t-1 and can be
joined to the daily forecasting panel on date without leakage.

Inputs (see Notebook 16c for the reasoning):
    1. activity            log(market share volume / median of previous 60 regular days), 5-regular-day mean
    2. realized_vol        log of annualized 20-day std of the value-weighted market return
    3. auction_share_dev   log(20-regular-day mean market auction share / its median over the previous 250 regular days)
    4. dispersion          log of the 20-day mean cross-sectional return dispersion
    5. fear_premium        log(median stock 30-day implied vol) - log(realized vol); missing after 2025-08-29
    6. drawdown            value-weighted market index / its 252-day high - 1

"Regular" days exclude month-end, quarter-end, monthly/quarterly options expiry and early-close
days, so calendar spikes don't leak into the slow-moving baselines. Those event days are
already handled by the calendar features in the forecasting model.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

INPUTS = ["activity", "realized_vol", "auction_share_dev", "dispersion", "fear_premium", "drawdown"]
CORE_INPUTS = ["activity", "realized_vol", "auction_share_dev"]
LABELS = {
    "activity": "Market activity (log vol vs 60d median)",
    "realized_vol": "Realized volatility (log, annualized)",
    "auction_share_dev": "Auction share vs 1-year median (log)",
    "dispersion": "Return dispersion (log, 20d mean)",
    "fear_premium": "Fear premium (log IV - log RV)",
    "drawdown": "Drawdown from 1-year high",
}
# Well-known stress episodes, for shading plots only (never used as model inputs).
EPISODES = [
    ("2015-08-17", "2015-09-30", "Aug 2015 selloff"),
    ("2018-10-01", "2018-12-31", "Q4 2018"),
    ("2020-02-20", "2020-06-30", "COVID"),
    ("2022-01-03", "2022-10-31", "2022 bear market"),
    ("2023-03-08", "2023-03-31", "Mar 2023 banks"),
    ("2025-04-02", "2025-04-30", "Apr 2025 tariffs"),
]


def _project_root():
    cwd = Path.cwd().resolve()
    root = next((p for p in [cwd, *cwd.parents] if (p / "blackrock_data").exists()), None)
    if root is None:
        raise FileNotFoundError("Could not locate blackrock_data.")
    return root


def load_daily_sources(con=None, root=None):
    """Raw market-level daily series (one row per trading day), before any transformation."""
    root = Path(root or _project_root())
    con = con or duckdb.connect()
    data = root / "blackrock_data"
    return con.execute(f"""
        WITH auction AS (
            SELECT date, sum(auction_volume) / nullif(sum(daily_volume), 0) AS auction_share,
                   sum(auction_volume) AS total_auction_volume
            FROM read_parquet('{data}/stock_daily/auction_daily/*/*.parquet', union_by_name=true)
            WHERE target_found AND daily_volume > 0
            GROUP BY date
        ), options AS (
            SELECT date, median(iv_atm_30d) AS median_iv
            FROM read_parquet('{data}/stock_daily/options_daily/*/*.parquet', union_by_name=true)
            WHERE iv_atm_30d > 0
            GROUP BY date
        )
        SELECT m.date, m.market_total_volume, m.market_return_vw, m.market_return_dispersion,
               a.auction_share, a.total_auction_volume, o.median_iv,
               (e.is_last_trading_day_month OR e.is_last_trading_day_quarter
                OR e.monthly_options_expiry OR e.quarterly_options_expiry OR e.is_early_close) AS event_day
        FROM read_parquet('{data}/market/market_daily.parquet') m
        JOIN read_parquet('{data}/events/event_calendar.parquet') e USING (date)
        LEFT JOIN auction a USING (date)
        LEFT JOIN options o USING (date)
        ORDER BY m.date
    """).df()


def _regular_rolling(series, regular, window, stat, min_periods):
    """Rolling statistic over the last `window` regular days, ending at (and including) each date."""
    values = series[regular]
    rolled = getattr(values.rolling(window, min_periods=min_periods), stat)()
    return rolled.reindex(series.index).ffill()


def build_regime_inputs(con=None, root=None):
    """Return a DataFrame indexed by date with the six inputs available for prediction on that date."""
    raw = load_daily_sources(con, root).set_index("date")
    regular = ~raw["event_day"].astype(bool)
    asof = pd.DataFrame(index=raw.index)

    # 1. Activity: today's volume vs the median of the previous 60 regular days, smoothed over 5 regular days.
    volume = raw["market_total_volume"].astype(float)
    baseline = _regular_rolling(volume, regular, 60, "median", 40).shift(1)
    activity_daily = np.log(volume / baseline)
    asof["activity"] = _regular_rolling(activity_daily, regular, 5, "mean", 3)

    # 2. Realized volatility of the value-weighted market return (all days count for returns).
    returns = raw["market_return_vw"].astype(float)
    realized = returns.rolling(20, min_periods=15).std() * np.sqrt(252)
    asof["realized_vol"] = np.log(realized)

    # 3. Auction share: 20-regular-day mean vs its median over the previous 250 regular days.
    share = raw["auction_share"].astype(float)
    share20 = _regular_rolling(share, regular & share.notna(), 20, "mean", 15)
    share_base = _regular_rolling(share20, regular & share20.notna(), 250, "median", 120).shift(1)
    asof["auction_share_dev"] = np.log(share20 / share_base)

    # 4. Cross-sectional return dispersion, 20-day mean.
    asof["dispersion"] = np.log(raw["market_return_dispersion"].astype(float).rolling(20, min_periods=15).mean())

    # 5. Fear premium: implied over realized volatility (both annualized).
    asof["fear_premium"] = np.log(raw["median_iv"].astype(float)) - np.log(realized)

    # 6. Drawdown of the value-weighted market index from its 252-day high.
    index = (1 + returns.fillna(0)).cumprod()
    asof["drawdown"] = index / index.rolling(252, min_periods=120).max() - 1

    # Shift: the value computed at the close of day d becomes available for predicting day d+1.
    available = asof.shift(1)
    available["event_day"] = raw["event_day"].astype(bool)
    available["auction_share_today"] = share            # same-day diagnostics, never model inputs
    available["log_total_auction_today"] = np.log(raw["total_auction_volume"].astype(float))
    available["market_return_today"] = returns
    return available


def tercile_edges(frame, inputs, start="2015-01-01", end="2021-12-31"):
    """Tercile cut points per input, fitted on the 2015-2021 training era only."""
    era = frame.loc[start:end, inputs]
    return {col: era[col].quantile([1 / 3, 2 / 3]).tolist() for col in inputs}


def assign_terciles(frame, edges):
    out = pd.DataFrame(index=frame.index)
    for col, (lo, hi) in edges.items():
        out[col] = pd.cut(frame[col], [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])
    return out


def shade_episodes(ax, alpha=0.12):
    for start, end, _ in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=alpha, lw=0)


# ---------------------------------------------------------------------------
# Hidden Markov model: fit on a training window, filter forward without look-ahead
# ---------------------------------------------------------------------------

# Chosen in Notebook 16c: fear premium is redundant with realized volatility and
# dispersion drifts out of its historical range, so both are left out.
HMM_INPUTS = ["activity", "realized_vol", "auction_share_dev", "drawdown"]
HMM_FEATURES = [
    "hmm_p_calm", "hmm_p_stress", "hmm_state_rank", "hmm_entropy", "hmm_regime_age",
    "hmm_exp_activity", "hmm_exp_realized_vol", "hmm_exp_auction_share_dev", "hmm_exp_drawdown",
]


def hmm_parameter_count(k, d):
    """Free parameters of a full-covariance Gaussian HMM: start probs, transitions, means, covariances."""
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def fit_hmm(train, k, restarts=10, seed=0):
    """Best-of-`restarts` Gaussian HMM on standardized training rows (a contiguous daily sequence)."""
    from hmmlearn.hmm import GaussianHMM

    best, best_ll = None, -np.inf
    for r in range(restarts):
        model = GaussianHMM(n_components=k, covariance_type="full", n_iter=500, tol=1e-4,
                            random_state=seed + r, min_covar=1e-3)
        try:
            model.fit(train)
            ll = model.score(train)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = model, ll
    return best, best_ll


def select_hmm(train, max_states=6, restarts=10, seed=0, states=None):
    """Fit 1..max_states states; return (best-BIC model, BIC table). BIC = -2 logL + p ln(T).

    If `states` is given, only that number of states is fitted (a fixed-k sensitivity check).
    """
    rows, models = [], {}
    n, d = train.shape
    for k in ([states] if states else range(1, max_states + 1)):
        model, ll = fit_hmm(train, k, restarts, seed)
        if model is None:
            continue
        p = hmm_parameter_count(k, d)
        rows.append({"states": k, "log_likelihood": ll, "parameters": p, "bic": -2 * ll + p * np.log(n)})
        models[k] = model
    table = pd.DataFrame(rows)
    chosen = int(table.loc[table.bic.idxmin(), "states"])
    table["chosen"] = table.states == chosen
    return models[chosen], table


def _emission_loglik(model, x):
    from scipy.stats import multivariate_normal

    return np.column_stack([
        multivariate_normal(model.means_[s], model.covars_[s], allow_singular=True).logpdf(x)
        for s in range(model.n_components)
    ])


def filter_hmm(model, obs):
    """Forward-filtered state probabilities P(state_t | obs_1..t), one row per date.

    Unlike hmmlearn's predict_proba (forward-backward smoothing), this never looks at later rows.
    Because each observation row t already uses only data up to t-1, the output on date t is
    usable for predicting date t. Rows with missing inputs skip the update and carry the prediction.
    """
    k = model.n_components
    obs = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs).all(axis=1)
    loglik = np.full((len(obs), k), np.nan)
    if valid.any():
        loglik[valid] = _emission_loglik(model, obs[valid])
    probs = np.zeros((len(obs), k))
    prior = model.startprob_.copy()
    for t in range(len(obs)):
        if valid[t]:
            log_post = np.log(np.clip(prior, 1e-300, None)) + loglik[t]
            log_post -= log_post.max()
            post = np.exp(log_post)
            post /= post.sum()
        else:
            post = prior
        probs[t] = post
        prior = post @ model.transmat_
    return probs


def hmm_regime_features(inputs, train_start, train_end, max_states=6, restarts=10, seed=0, states=None):
    """Fit an HMM on [train_start, train_end] only and return k-independent regime features for all dates.

    The number of states is chosen by BIC over 1..max_states, unless `states` fixes it.

    Returns (features DataFrame indexed by date, BIC table, state profile table, fitted model, scaler).
    States are ordered by mean realized volatility, so state 0 is the calmest and the last is the most stressed.
    """
    raw = inputs[HMM_INPUTS]
    train_rows = raw.loc[train_start:train_end].dropna()
    mean, std = train_rows.mean(), train_rows.std()
    z_all = (raw - mean) / std
    model, bic = select_hmm(((train_rows - mean) / std).to_numpy(), max_states, restarts, seed, states)

    order = np.argsort(model.means_[:, HMM_INPUTS.index("realized_vol")])
    probs = filter_hmm(model, z_all.to_numpy())[:, order]
    means = model.means_[order]
    k = model.n_components

    features = pd.DataFrame(index=raw.index)
    features["hmm_p_calm"] = probs[:, 0]
    features["hmm_p_stress"] = probs[:, -1]
    top = probs.argmax(axis=1)
    features["hmm_state_rank"] = top / max(k - 1, 1)
    features["hmm_entropy"] = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    changed = np.r_[True, top[1:] != top[:-1]]
    features["hmm_regime_age"] = pd.Series(np.arange(len(top)), index=raw.index).where(changed).ffill().rsub(np.arange(len(top))).to_numpy()
    for j, col in enumerate(HMM_INPUTS):
        features[f"hmm_exp_{col}"] = probs @ means[:, j]
    features.loc[raw.index < pd.Timestamp(train_start)] = np.nan  # no features before the HMM's own window

    profile = pd.DataFrame(means * std.to_numpy() + mean.to_numpy(), columns=HMM_INPUTS)
    profile.index.name = "state (0 = calmest)"
    train_top = pd.Series(top, index=raw.index).loc[train_rows.index]
    profile["share_of_training_days"] = train_top.value_counts(normalize=True).reindex(range(k)).fillna(0).to_numpy()
    profile["expected_duration_days"] = 1 / (1 - np.diag(model.transmat_)[order])
    return features, bic, profile, model, (mean, std)
