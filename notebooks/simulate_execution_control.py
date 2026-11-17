"""Replay a close-targeted parent order on historical NYSE stock-days."""
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/24_execution_control"
CONT_CAP = 0.10
CONT_SPREAD = 2.0
CONT_IMPACT = 60.0
AUCTION_IMPACT = 100.0
UNFILLED_COST = 100.0
RISK_AVERSION = 0.001
POLICIES = ("Continuous only", "Fixed 50% auction", "Trailing history", "Checkpoint adaptive")
ORDER_SIZES = (0.01, 0.05)
AUCTION_CAPS = (0.025, 0.05, 0.10)


def load_tape():
    data = ROOT / "blackrock_data"
    auction = data / "stock_daily/auction_daily/year=202[45]/auction_daily_*.parquet"
    tape = data / "stock_intraday/trade_checkpoints/year=202[45]/trade_checkpoints_*.parquet"
    with duckdb.connect() as con:
        rows = con.execute("""
            WITH checkpoints AS (
                SELECT date, permno, max(sym_root) AS ticker,
                    max(cum_volume) FILTER (WHERE checkpoint='12:00') AS morning,
                    max(cum_volume) FILTER (WHERE checkpoint='14:00') AS v14,
                    max(cum_volume) FILTER (WHERE checkpoint='15:30') AS v1530,
                    max(cum_dollar_volume) FILTER (WHERE checkpoint='12:00') AS d12,
                    max(cum_dollar_volume) FILTER (WHERE checkpoint='14:00') AS d14,
                    max(cum_dollar_volume) FILTER (WHERE checkpoint='15:30') AS d1530
                FROM read_parquet(?)
                WHERE checkpoint IN ('12:00','14:00','15:30')
                    AND checkpoint_data_found
                    AND NOT checkpoint_after_market_close
                    AND NOT coalesce(bridge_collision, true)
                    AND cum_volume > 0 AND cum_dollar_volume > 0
                GROUP BY date, permno
                HAVING count(*)=3 AND count(DISTINCT checkpoint)=3
            )
            SELECT date, permno, c.ticker, morning,
                v14-morning AS first_volume, v1530-v14 AS second_volume,
                (d14-d12)/(v14-morning) AS first_vwap,
                (d1530-d14)/(v1530-v14) AS second_vwap,
                a.auction_volume, a.auction_price, a.year
            FROM checkpoints c
            JOIN read_parquet(?, union_by_name=true) a USING(date,permno)
            WHERE a.target_found AND a.auction_volume > 0 AND a.auction_price > 0
                AND v14 > morning AND v1530 > v14 AND d14 > d12 AND d1530 > d14
                AND (a.year=2024 OR a.primary_exchange='N')
            ORDER BY permno, date
        """, [str(tape), str(auction)]).df()
    assert not rows.duplicated(["date", "permno"]).any()
    number_of_valid_dates = rows.groupby("year").date.nunique().to_dict()
    rows["window_volume"] = rows.first_volume + rows.second_volume + rows.auction_volume
    prior = rows[rows.year == 2024]
    cohorts = prior.groupby("permno").agg(
        ticker=("ticker", "last"), valid_days_2024=("date", "size"),
        median_window_volume_2024=("window_volume", "median"),
    )
    cohorts = cohorts[cohorts.valid_days_2024 >= 220].sort_values(
        ["median_window_volume_2024", "ticker"]
    )
    assert len(cohorts) >= 24
    cohort = cohorts.iloc[((np.arange(24) + 0.5) * len(cohorts) / 24).astype(int)]
    return rows[rows.permno.isin(cohort.index)], cohort, number_of_valid_dates


def interval_risk(q, fill, risk_rate, hours):
    executed = 1.0 - q
    return risk_rate * hours * (executed**2 + executed * fill + fill**2 / 3)


def interval_cost(q, fill, volume_per_order, risk_rate, hours):
    impact = CONT_IMPACT * fill / volume_per_order
    return (CONT_SPREAD + impact) * fill + interval_risk(q, fill, risk_rate, hours)


def auction_cost(q, volume_per_order, auction_cap, risk_rate):
    fill = np.minimum(q, auction_cap * volume_per_order)
    impact = AUCTION_IMPACT * fill**2 / volume_per_order
    return impact + UNFILLED_COST * (q - fill) + 0.5 * risk_rate * (1.0 - q)**2


def solve_second(inventory, scenarios, auction_cap, risk_rate, grid):
    q = np.asarray(inventory).reshape(-1, 1, 1)
    request = q * grid.reshape(1, -1, 1)
    second_volume = scenarios[:, 1]
    auction_volume = scenarios[:, 2]
    fill = np.minimum(request, CONT_CAP * second_volume)
    value = interval_cost(q, fill, second_volume, risk_rate, 1.5)
    value += auction_cost(q - fill, auction_volume, auction_cap, risk_rate)
    expected = value.mean(axis=2)
    optimal = expected.argmin(axis=1)
    return expected[np.arange(len(q)), optimal], q[:, 0, 0] * grid[optimal]


def solve_first(scenarios, regimes, auction_cap, risk_rate, grid):
    # Each scenario contributes its own period-one volume and price-risk exposure.
    volume = scenarios[:, 0]
    fill = np.minimum(grid[:, None], CONT_CAP * volume)
    continuation = np.empty_like(fill)
    for regime in np.unique(regimes):
        in_regime = regimes == regime
        values, _ = solve_second(grid, scenarios[in_regime], auction_cap, risk_rate, grid)
        continuation[:, in_regime] = np.interp(1.0 - fill[:, in_regime], grid, values)
    value = interval_cost(1.0, fill, volume, risk_rate, 2.0) + continuation
    return grid[np.argmin(value.mean(axis=1))]


def replay(request1, policy, scenario, order, auction_cap, risk_rate, grid,
           second_scenarios=None):
    v1, v2, va = scenario / order
    fill1 = min(request1, CONT_CAP * v1)
    q = 1.0 - fill1
    if policy == "Continuous only":
        request2 = q
    elif policy == "Fixed 50% auction":
        request2 = max(0.0, q - 0.5)
    else:
        _, request = solve_second(q, second_scenarios, auction_cap, risk_rate, grid)
        request2 = request[0]
    fill2 = min(request2, CONT_CAP * v2)
    q_before_auction = q - fill2
    auction_request = 0.0 if policy == "Continuous only" else q_before_auction
    auction_fill = min(auction_request, auction_cap * va)
    unfilled = max(0.0, q_before_auction - auction_fill)
    friction = (CONT_SPREAD * (fill1 + fill2)
                + CONT_IMPACT * (fill1**2 / v1 + fill2**2 / v2)
                + AUCTION_IMPACT * auction_fill**2 / va)
    penalty = UNFILLED_COST * unfilled
    risk = (
        interval_risk(1.0, fill1, risk_rate, 2.0)
        + interval_risk(q, fill2, risk_rate, 1.5)
        + 0.5 * risk_rate * (1.0 - q_before_auction)**2
    )
    assert min(fill1, fill2, auction_fill, unfilled) >= 0
    assert np.isclose(fill1 + fill2 + auction_fill + unfilled, 1.0)
    assert fill1 <= CONT_CAP * v1 and fill2 <= CONT_CAP * v2
    assert auction_fill <= auction_cap * va
    return {
        "request_noon": request1, "request_1400": request2,
        "filled_noon": fill1, "filled_1400": fill2, "auction_request": auction_request,
        "auction_fill": auction_fill, "unfilled": unfilled, "friction_bps": friction,
        "penalty_bps": penalty, "risk_bps": risk,
        "objective_bps": friction + penalty + risk,
    }


def run(rows, grid, order_sizes=ORDER_SIZES, auction_caps=AUCTION_CAPS):
    records = []
    columns = ["first_volume", "second_volume", "auction_volume"]
    for permno, stock in rows.groupby("permno", sort=False):
        stock = stock.reset_index(drop=True)
        vols = stock[columns].to_numpy(dtype=float)
        morning = stock.morning.to_numpy(dtype=float)
        vwap_change = 1e4 * np.log(stock.second_vwap / stock.first_vwap).to_numpy()
        for j in np.flatnonzero(stock.year.to_numpy() == 2025):
            if j < 60:
                continue
            current = stock.iloc[j]
            history = slice(j - 60, j)
            prior = vols[history]
            assert stock.date.iloc[j - 1] < current.date
            risk_rate = RISK_AVERSION * np.var(vwap_change[history], ddof=1) / 1.75
            nearby = np.argsort(
                np.abs(np.log(morning[history] / morning[j])), kind="stable",
            )[:36]
            conditioned = prior[nearby]
            # The regime is learned at 14:00, once the first interval closes.
            cutoffs = np.quantile(conditioned[:, 0], [1 / 3, 2 / 3])
            regimes = np.searchsorted(cutoffs, conditioned[:, 0], side="right")
            current_regime = np.searchsorted(cutoffs, vols[j, 0], side="right")
            if current_regime not in regimes:
                present = np.unique(regimes)
                current_regime = present[np.abs(present - current_regime).argmin()]
            for size in order_sizes:
                order = max(1, round(size * np.median(prior.sum(axis=1))))
                daily = prior / order
                intraday = conditioned / order
                second_adaptive = intraday[regimes == current_regime]
                for cap in auction_caps:
                    first_daily = solve_first(daily, np.zeros(len(daily)), cap, risk_rate, grid)
                    first_adaptive = solve_first(intraday, regimes, cap, risk_rate, grid)
                    for policy, request, next_scenarios in zip(
                        POLICIES, (0.5, 0.25, first_daily, first_adaptive),
                        (None, None, daily, second_adaptive),
                    ):
                        result = replay(
                            request, policy, vols[j], order, cap, risk_rate, grid, next_scenarios,
                        )
                        timing = 1e4 * (
                            result["filled_noon"] * (current.first_vwap / current.auction_price - 1)
                            + result["filled_1400"] * (current.second_vwap / current.auction_price - 1)
                        )
                        records.append({
                            "date": current.date, "permno": permno, "ticker": current.ticker,
                            "order_fraction": size, "auction_cap": cap, "policy": policy,
                            "order_shares": order, "morning": morning[j],
                            "first_volume": vols[j, 0], "second_volume": vols[j, 1],
                            "auction_volume": vols[j, 2], "buy_timing_bps": timing, **result,
                        })
    return pd.DataFrame.from_records(records)


def summarize(results):
    scored = results.assign(
        cost_bps=results.friction_bps + results.penalty_bps,
        has_unfilled=results.unfilled > 1e-9, timing_squared=results.buy_timing_bps**2,
    )
    groups = scored.groupby(["order_fraction", "auction_cap", "policy"], sort=False)
    summary = groups.agg(
        stock_days=("date", "size"), median_order_shares=("order_shares", "median"),
        cost_bps=("cost_bps", "mean"), friction_bps=("friction_bps", "mean"),
        risk_bps=("risk_bps", "mean"), objective_bps=("objective_bps", "mean"),
        auction_request_pct=("auction_request", "mean"),
        auction_fill_pct=("auction_fill", "mean"), unfilled_pct=("unfilled", "mean"),
        days_unfilled_pct=("has_unfilled", "mean"), timing_rms_bps=("timing_squared", "mean"),
    ).reset_index()
    summary["timing_rms_bps"] = np.sqrt(summary.timing_rms_bps)
    for col in summary.filter(regex="_pct$"):
        summary[col] *= 100
    return summary


def paired_differences(results):
    base = results.loc[results.auction_cap == 0.05].copy()
    base["cost_bps"] = base.friction_bps + base.penalty_bps
    random = np.random.default_rng(824)
    rows = []
    for size, group in base.groupby("order_fraction"):
        for metric in ("cost_bps", "objective_bps", "unfilled"):
            paired = group.pivot(index=["date", "permno"], columns="policy", values=metric)
            delta = paired["Checkpoint adaptive"] - paired["Trailing history"]
            if metric == "unfilled":
                delta *= 100
            by_date = delta.groupby(level="date").agg(["sum", "count"])
            draws = random.integers(0, len(by_date), size=(2000, len(by_date)))
            stats = by_date.to_numpy()[draws].sum(axis=1)
            lo, hi = np.quantile(stats[:, 0] / stats[:, 1], [0.025, 0.975])
            rows.append((size, metric, delta.mean(), lo, hi))
    return pd.DataFrame(rows, columns=["order_fraction", "metric", "delta", "lower", "upper"])


def plot_summary(summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    base = summary[summary.auction_cap == 0.05]
    score_limit = base.objective_bps.max()
    palette = ["#a0a9ae", "#a0a9ae", "#53738d", "#007a6d"]
    for ax, size in zip(axes, ORDER_SIZES):
        data = base[base.order_fraction == size].set_index("policy").loc[list(POLICIES)]
        labels = ["Continuous only", "Fixed 50% auction", "Trailing history", "Checkpoint adaptive"]
        heights = np.arange(len(data))
        ax.barh(heights, data.cost_bps, color=palette, height=0.58)
        ax.barh(heights, data.risk_bps, left=data.cost_bps, color=palette, alpha=0.32, height=0.58)
        ax.set_yticks(heights, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, score_limit * 1.18)
        for height, score in zip(heights, data.objective_bps):
            ax.text(score + score_limit * 0.025, height, f"{score:.2f}", va="center")
        ax.set_title(f"Order size: {size:.0%} of trailing window volume", loc="left", pad=10)
        ax.set_xlabel("Mean objective (bps per parent order)")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Close-targeted execution on 2025 NYSE stock-days", x=0.03, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.03, 0.85, "Solid = modeled trading friction and unfilled penalty    Light = close-tracking risk penalty")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.70, bottom=0.15, wspace=0.57)
    fig.savefig(OUT / "execution_results.png", dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def verify_solver():
    # At zero inventory every action has zero fill. Accrued tracking risk
    # remains because the entire parent order was completed before the close.
    tape = np.array([[10.0, 10.0, 10.0]])
    grid = np.linspace(0, 1, 41)
    values, requests = solve_second(0.0, tape, 0.05, 2.0, grid)
    assert np.allclose(values, 4.0) and np.allclose(requests, 0.0)
    # With friction on both venues, abundant auction liquidity and no tracking
    # risk, all shares go to the auction.
    tape[:, 2] = 1e6
    assert solve_first(tape, np.zeros(1), 0.05, 0.0, grid) == 0
    assert solve_second(1.0, tape, 0.05, 0.0, grid)[1][0] == 0
    # With zero risk and binding auction capacity, the linear unfilled cost
    # makes taking the full available continuous fill optimal.
    tape = np.array([[10.0, 2.0, 1.0]])
    assert np.isclose(solve_second(1.0, tape, 0.05, 0.0, grid)[1][0], 0.20)


def main():
    verify_solver()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cohort, valid_dates = load_tape()
    results = run(rows, np.linspace(0, 1, 21))
    assert np.isfinite(results.select_dtypes(include="number")).all().all()
    counts = results.groupby(["date", "permno"]).size()
    assert (counts == len(ORDER_SIZES) * len(AUCTION_CAPS) * len(POLICIES)).all()
    summary = summarize(results)
    results.to_csv(OUT / "stock_day_replay.csv.gz", index=False, compression={
        "method": "gzip", "mtime": 0,
    })
    summary.to_csv(OUT / "summary.csv", index=False)
    paired_differences(results).to_csv(OUT / "paired_differences.csv", index=False)
    cohort.to_csv(OUT / "cohort.csv")
    plot_summary(summary)
    print(f"Usable market dates: {valid_dates}; replay dates {results.date.nunique()}; "
          f"stock-days {len(counts)} across {results.permno.nunique()} stocks", flush=True)
    print(summary.to_string(index=False), flush=True)

    # Run both grids on identical historical days for a numerical check.
    sample_ids = cohort.index[::4]
    sample = rows[rows.permno.isin(sample_ids)]
    fine = run(sample, np.linspace(0, 1, 41), auction_caps=(0.05,))
    coarse = results[results.permno.isin(sample_ids) & (results.auction_cap == 0.05)]
    keys = ["order_fraction", "auction_cap", "policy"]
    coarse_summary = summarize(coarse).set_index(keys)
    fine_summary = summarize(fine).set_index(keys)
    delta = fine_summary.select_dtypes(include="number") - coarse_summary.select_dtypes(include="number")
    delta.reset_index().to_csv(OUT / "grid_sensitivity.csv", index=False)
    print("Fine-minus-coarse objective (bps) on six stocks:", flush=True)
    print(delta.objective_bps.to_string(), flush=True)


if __name__ == "__main__":
    main()
