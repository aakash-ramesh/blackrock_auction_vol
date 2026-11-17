"""Sequence neural-network challengers for the daily model (Notebook 16e).

Adapts a CNN + inception + LSTM closing-auction architecture to the daily problem:
for each stock-day (t), the network reads that stock's last L trading days of raw daily
history (days t-L .. t-1, all known by the close of t-1) and predicts the same residual
target as every other challenger: log1p(auction_volume_t) - log1p(median20_auction_volume_t).

Protocol, sampled rows, seeds and evaluation are identical to the LightGBM "large" tier in
Notebook 16a (challenger_utils), so results can be compared row for row with 16a predictions.
"""

import gc
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import challenger_utils as cu
from challenger_utils import fit_preprocessor, apply_preprocessor, forecast_metrics

SEQ_LEN = 60  # longest rolling window in the panel
EVENT_FLAGS = [
    "is_last_trading_day_month", "is_last_trading_day_quarter", "monthly_options_expiry",
    "quarterly_options_expiry", "is_early_close", "is_friday",
]
CHANNELS = (["rel_log_auction", "auction_missing", "rel_log_daily_volume", "auction_share",
             "return_x10", "abs_return_x10"] + [f"past_{f}" for f in EVENT_FLAGS] + ["row_present"])
TARGET_DAY_CALENDAR = [f for f in cu.BOOLEAN + cu.CATEGORICAL if f != "auction_spike_lag1"]
assert "avg20_daily_volume" in cu.BASE_FEATURES

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TRAIN = {"batch": 1024, "lr": 1e-3, "weight_decay": 1e-4, "max_epochs": 8, "patience": 2}
if cu.SMOKE:
    TRAIN.update(max_epochs=2, patience=1)


# ---------------------------------------------------------------------------
# History store: every panel row, sorted by (permno, date), raw daily values.
# ---------------------------------------------------------------------------

def build_history_store():
    flags = ", ".join(f"coalesce({f}, false)::FLOAT AS {f}" for f in EVENT_FLAGS)
    frame = cu.sql(f"""
        SELECT permno, date,
               CASE WHEN auction_volume IS NULL OR coalesce(exclude_source_anomaly, false) THEN NULL
                    ELSE ln(1 + auction_volume) END AS log_auction,
               ln(1 + current_daily_volume) AS log_daily_volume,
               auction_share_daily_volume AS auction_share,
               current_return, {flags}
        FROM read_parquet('{cu.PANEL}')
        ORDER BY permno, date
    """)
    store = {
        "permno": frame["permno"].to_numpy(np.int64),
        "date": frame["date"].to_numpy("datetime64[ns]"),
        "log_auction": frame["log_auction"].to_numpy(np.float32),
        "log_daily_volume": frame["log_daily_volume"].to_numpy(np.float32),
        "auction_share": frame["auction_share"].to_numpy(np.float32),
        "return": frame["current_return"].to_numpy(np.float32),
        "flags": frame[EVENT_FLAGS].to_numpy(np.float32),
    }
    store["keys"] = pd.DataFrame({"permno": store["permno"], "date": store["date"],
                                  "position": np.arange(len(frame), dtype=np.int64)})
    del frame
    gc.collect()
    return store


def sequence_index(store, frame):
    """Arrays needed to gather each row's history window (position, reference levels)."""
    keyed = frame[["permno", "date"]].assign(date=pd.to_datetime(frame["date"]).astype("datetime64[ns]"))
    merged = keyed.merge(store["keys"], on=["permno", "date"], how="left", validate="one_to_one")
    if merged["position"].isna().any():
        raise AssertionError("Some rows are missing from the history store.")
    fallback = float(np.nanmedian(frame["median20_auction_volume"]))
    return {
        "position": merged["position"].to_numpy(np.int64),
        "ref_auction": np.log1p(frame["median20_auction_volume"].fillna(fallback).to_numpy(np.float64)).astype(np.float32),
        "ref_daily_volume": np.log1p(frame["avg20_daily_volume"].fillna(
            float(np.nanmedian(frame["avg20_daily_volume"]))).to_numpy(np.float64)).astype(np.float32),
    }


def gather_windows(store, index, rows):
    """History windows for the selected rows: float32 array (n, SEQ_LEN, len(CHANNELS)).

    Window step k holds store position (p - SEQ_LEN + k), i.e. strictly earlier rows of the
    same stock; the last step is the stock's previous trading day. Positions belonging to
    another stock (not enough history) are zero with row_present = 0.
    """
    position = index["position"][rows]
    offsets = np.arange(SEQ_LEN, 0, -1)
    idx = position[:, None] - offsets[None, :]
    safe = np.clip(idx, 0, None)
    present = (idx >= 0) & (store["permno"][safe] == store["permno"][position][:, None])

    log_auction = store["log_auction"][safe]
    auction_ok = present & np.isfinite(log_auction)
    rel_auction = np.where(auction_ok, np.clip(log_auction - index["ref_auction"][rows, None], -6, 6), 0)
    log_dv = store["log_daily_volume"][safe]
    rel_dv = np.where(present & np.isfinite(log_dv),
                      np.clip(log_dv - index["ref_daily_volume"][rows, None], -6, 6), 0)
    share = np.where(present, np.nan_to_num(np.clip(store["auction_share"][safe], 0, 1)), 0)
    ret = np.where(present, np.nan_to_num(np.clip(store["return"][safe] * 10, -3, 3)), 0)
    flags = store["flags"][safe] * present[..., None]

    windows = np.concatenate([
        np.stack([rel_auction, (present & ~auction_ok).astype(np.float32), rel_dv, share, ret, np.abs(ret)], -1),
        flags, present[..., None].astype(np.float32),
    ], axis=-1)
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                                 nn.BatchNorm1d(c_out), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Inception(nn.Module):
    """Parallel convolutions with different kernel sizes (multi-scale), plus a pooling branch."""

    def __init__(self, c_in, branch=24, kernels=(3, 7, 15)):
        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, branch, 1)
        self.branches = nn.ModuleList(nn.Conv1d(branch, branch, k, padding=k // 2) for k in kernels)
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(c_in, branch, 1))
        self.out_channels = branch * (len(kernels) + 1)
        self.norm = nn.Sequential(nn.BatchNorm1d(self.out_channels), nn.ReLU())

    def forward(self, x):
        z = self.bottleneck(x)
        return self.norm(torch.cat([b(z) for b in self.branches] + [self.pool(x)], dim=1))


class SequenceEncoder(nn.Module):
    """Conv stack (increasing depth, same length) -> inception -> LSTM -> last hidden state."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.convs = nn.Sequential(ConvBlock(channels, 32, 5), ConvBlock(32, 64, 5))
        self.inception = Inception(64)
        self.lstm = nn.LSTM(self.inception.out_channels, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, seq):                      # seq: (B, L, C)
        z = self.inception(self.convs(seq.transpose(1, 2)))
        _, (h, _) = self.lstm(z.transpose(1, 2))
        return h[-1]


class DailyNet(nn.Module):
    """mode: 'tabular' (MLP only), 'sequence' (sequence + target-day calendar), 'hybrid' (both)."""

    def __init__(self, mode, tab_dim, channels=len(CHANNELS)):
        super().__init__()
        self.mode = mode
        self.seq = SequenceEncoder(channels) if mode in ("sequence", "hybrid") else None
        self.tab = nn.Sequential(nn.Linear(tab_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU())
        joint = 64 + (self.seq.out_dim if self.seq is not None else 0)
        self.head = nn.Sequential(nn.Linear(joint, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, tab, seq=None):
        parts = [self.tab(tab)]
        if self.seq is not None:
            parts.append(self.seq(seq))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _batches(n, batch, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch):
        yield order[start:start + batch]


def predict(model, x_tab, store, index, batch=8192):
    model.eval()
    out = np.empty(len(x_tab), dtype=np.float32)
    with torch.no_grad():
        for rows in _batches(len(x_tab), batch):
            tab = torch.from_numpy(x_tab[rows]).to(DEVICE)
            seq = torch.from_numpy(gather_windows(store, index, rows)).to(DEVICE) if model.seq is not None else None
            out[rows] = model(tab, seq).float().cpu().numpy()
    return out


def train(mode, x_tab, y, store, index, seed, epochs, validation=None, log=print):
    """Train for up to `epochs`; with `validation`, stop early and return the best epoch.

    validation = dict(x_tab, index, base, y_log, train_log) evaluated in log space exactly like
    the challenger protocol (residual prediction + base, clipped to the training range).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DailyNet(mode, x_tab.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    y = y.astype(np.float32)
    history, best_state, best_rmse, best_epoch, waited = [], None, np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, count = time.time(), 0.0, 0
        for rows in _batches(len(x_tab), TRAIN["batch"], rng):
            tab = torch.from_numpy(x_tab[rows]).to(DEVICE)
            seq = torch.from_numpy(gather_windows(store, index, rows)).to(DEVICE) if model.seq is not None else None
            target = torch.from_numpy(y[rows]).to(DEVICE)
            loss = torch.mean((model(tab, seq) - target) ** 2)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * len(rows)
            count += len(rows)
        row = {"epoch": epoch, "train_mse": total / count, "seconds": time.time() - started}
        if validation is not None:
            residual = predict(model, validation["x_tab"], store, validation["index"])
            prediction = cu.final_log_prediction(residual, validation["base"], validation["train_log"])
            row["validation_rmse_log"] = float(np.sqrt(np.mean((validation["y_log"] - prediction) ** 2)))
            if row["validation_rmse_log"] < best_rmse - 1e-5:
                best_rmse, best_epoch, waited = row["validation_rmse_log"], epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
        history.append(row)
        log(f"      epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))
        if validation is not None and waited >= TRAIN["patience"]:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, (best_epoch if validation is not None else epochs), history


# ---------------------------------------------------------------------------
# Fold data and experiment loop
# ---------------------------------------------------------------------------

def prepare_fold(test_year, store, tier="large"):
    """Same sampled rows and preprocessing as challenger_utils.prepare_tier, plus sequence indexes."""
    rows = cu.TIERS[tier]
    validation_year = test_year - 1
    tune = cu.load_period(validation_year - 5, validation_year, rows["tune"], cu.SEED + test_year)
    validation = cu.load_period(validation_year, test_year, rows["validation"], cu.SEED + 100 + test_year)
    refit = cu.load_period(test_year - 5, test_year, rows["refit"], cu.SEED + 200 + test_year)
    test = cu.load_period(test_year, test_year + 1)

    x_tune, prep_tune = fit_preprocessor(tune, cu.CONTINUOUS, cu.BOOLEAN, cu.CATEGORICAL)
    x_validation = apply_preprocessor(validation, prep_tune, cu.CONTINUOUS, cu.BOOLEAN, cu.CATEGORICAL)
    y_tune, validation_base = cu.residual_target_and_base(tune, validation)
    x_refit, prep = fit_preprocessor(refit, cu.CONTINUOUS, cu.BOOLEAN, cu.CATEGORICAL)
    x_test = apply_preprocessor(test, prep, cu.CONTINUOUS, cu.BOOLEAN, cu.CATEGORICAL)
    y_refit, test_base = cu.residual_target_and_base(refit, test)
    data = {
        "x_tune": x_tune.astype(np.float32), "y_tune": y_tune, "prep_tune": prep_tune,
        "index_tune": sequence_index(store, tune), "tune_log": tune["log1p_auction_volume"].to_numpy(float),
        "x_validation": x_validation.astype(np.float32), "index_validation": sequence_index(store, validation),
        "validation_base": validation_base, "y_validation_log": validation["log1p_auction_volume"].to_numpy(float),
        "x_refit": x_refit.astype(np.float32), "y_refit": y_refit, "prep": prep,
        "index_refit": sequence_index(store, refit), "refit_log": refit["log1p_auction_volume"].to_numpy(float),
        "x_test": x_test.astype(np.float32), "index_test": sequence_index(store, test), "test_base": test_base,
        "test_keys": test[["date", "permno"]].copy(), "y_test_raw": test["auction_volume"].to_numpy(float),
        "rows": {"tune": len(tune), "validation": len(validation), "refit": len(refit), "test": len(test)},
    }
    del tune, validation, refit, test
    gc.collect()
    return data


def tabular_mask(prep, mode):
    features = TARGET_DAY_CALENDAR if mode == "sequence" else cu.RATIO_SET
    return cu.feature_mask(prep, features)


def run_sequence_experiment(candidates, store, out_dir, tag, reference_dir, reference_tag):
    """Train each candidate on every fold and save predictions next to 16a's LightGBM forecasts.

    candidates: list of dicts {name, mode}. The 16a prediction file for each year supplies the
    context columns, the frozen ridge forecast and the LightGBM forecasts, so every model is
    scored on exactly the same test rows.
    """
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, history_rows = [], []
    started = time.time()
    log = lambda message: print(f"[{time.time() - started:6.0f}s] {message}", flush=True)  # noqa: E731

    for test_year in cu.TEST_YEARS:
        data = prepare_fold(test_year, store)
        log(f"{test_year} rows={data['rows']}")
        reference = cu.sql(f"SELECT * FROM read_parquet('{reference_dir}/predictions/{reference_tag}_predictions_{test_year}.parquet')")
        reference["date"] = pd.to_datetime(reference["date"]).astype("datetime64[ns]")
        output = data["test_keys"].assign(date=pd.to_datetime(data["test_keys"]["date"]).astype("datetime64[ns]"))
        output = output.merge(reference, on=["date", "permno"], how="left", validate="one_to_one")
        if output["frozen_prediction"].isna().any():
            raise AssertionError("16a predictions did not align with the test rows.")

        for number, candidate in enumerate(candidates):
            mode = candidate["mode"]
            tune_mask, refit_mask = tabular_mask(data["prep_tune"], mode), tabular_mask(data["prep"], mode)
            log(f"  {candidate['name']} ({mode}): tuning")
            validation = {"x_tab": data["x_validation"][:, tune_mask], "index": data["index_validation"],
                          "base": data["validation_base"], "y_log": data["y_validation_log"],
                          "train_log": data["tune_log"]}
            model, best_epoch, history = train(mode, data["x_tune"][:, tune_mask], data["y_tune"], store,
                                               data["index_tune"], cu.SEED + test_year + number,
                                               TRAIN["max_epochs"], validation, log=log)
            history_rows += [{"test_year": test_year, "candidate": candidate["name"], "phase": "tune", **h}
                             for h in history]
            del model
            gc.collect()
            log(f"  {candidate['name']}: refit for {best_epoch} epochs")
            model, _, history = train(mode, data["x_refit"][:, refit_mask], data["y_refit"], store,
                                      data["index_refit"], cu.SEED + 300 + test_year + number,
                                      best_epoch, log=log)
            history_rows += [{"test_year": test_year, "candidate": candidate["name"], "phase": "refit", **h}
                             for h in history]
            residual = predict(model, data["x_test"][:, refit_mask], store, data["index_test"])
            prediction_log = cu.final_log_prediction(residual, data["test_base"], data["refit_log"])
            prediction_raw = np.maximum(np.expm1(prediction_log), 0)
            output[candidate["name"]] = prediction_raw
            metric_rows.append({"test_year": test_year, "candidate": candidate["name"], "family": "neural network",
                                "selected_config": 0, "iterations": best_epoch,
                                **forecast_metrics(data["y_test_raw"], prediction_raw)})
            log(f"  {candidate['name']}: test log RMSE {metric_rows[-1]['rmse_log']:.4f}")
            del model
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        path = predictions_dir / f"{tag}_predictions_{test_year}.parquet"
        cu.con.register("prediction_output", output)
        cu.con.execute(f"COPY prediction_output TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        cu.con.unregister("prediction_output")
        log(f"wrote {len(output):,} predictions for {test_year}")
        del data, output, reference
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    history = pd.DataFrame(history_rows)
    metrics.to_csv(out_dir / "nn_metrics.csv", index=False)
    history.to_csv(out_dir / "training_history.csv", index=False)
    return metrics, history
