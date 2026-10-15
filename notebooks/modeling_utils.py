"""Small leakage-safe preprocessing and ridge helpers for capstone notebooks."""

import numpy as np
import pandas as pd


def numeric_array(series):
    return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def fit_preprocessor(frame, continuous, boolean, categorical):
    params = {
        "continuous": {}, "boolean": {}, "categorical": {},
        "output_names": [], "output_sources": [],
    }
    columns = []

    for feature in continuous:
        raw = numeric_array(frame[feature])
        finite = raw[np.isfinite(raw)]
        missing = ~np.isfinite(raw)
        if len(finite) == 0:
            use_log = False
            transformed = raw.copy()
            lower = upper = median = 0.0
            scale = 1.0
        else:
            nonnegative = np.nanmin(finite) >= 0
            med_abs = max(abs(np.nanmedian(finite)), 1e-12)
            skew_ratio = np.nanquantile(np.abs(finite), 0.99) / med_abs
            use_log = bool(nonnegative and skew_ratio >= 20)
            transformed = np.log1p(np.clip(raw, 0, None)) if use_log else raw.copy()
            finite_t = transformed[np.isfinite(transformed)]
            lower, upper = np.nanquantile(finite_t, [0.005, 0.995])
            clipped = np.clip(transformed, lower, upper)
            median = float(np.nanmedian(clipped))
            filled = np.where(np.isfinite(clipped), clipped, median)
            scale = float(np.std(filled))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
        clipped = np.clip(transformed, lower, upper)
        filled = np.where(np.isfinite(clipped), clipped, median)
        columns.append(((filled - median) / scale).astype(np.float32))
        params["output_names"].append(feature)
        params["output_sources"].append(feature)
        add_missing = bool(missing.any())
        if add_missing:
            columns.append(missing.astype(np.float32))
            params["output_names"].append(f"{feature}__missing")
            params["output_sources"].append(feature)
        params["continuous"][feature] = {
            "log1p": use_log, "lower": float(lower), "upper": float(upper),
            "median": float(median), "scale": float(scale), "missing_indicator": add_missing,
        }

    for feature in boolean:
        raw = numeric_array(frame[feature])
        missing = ~np.isfinite(raw)
        columns.append(np.where(missing, 0.0, raw).astype(np.float32))
        params["output_names"].append(feature)
        params["output_sources"].append(feature)
        add_missing = bool(missing.any())
        if add_missing:
            columns.append(missing.astype(np.float32))
            params["output_names"].append(f"{feature}__missing")
            params["output_sources"].append(feature)
        params["boolean"][feature] = {"missing_indicator": add_missing}

    for feature in categorical:
        values = frame[feature].astype("string").fillna("MISSING")
        levels = sorted(values.unique().tolist())
        reference = levels[0]
        modeled_levels = levels[1:]
        for level in modeled_levels:
            columns.append((values == level).to_numpy(dtype=np.float32))
            params["output_names"].append(f"{feature}=={level}")
            params["output_sources"].append(feature)
        columns.append((~values.isin(levels)).to_numpy(dtype=np.float32))
        params["output_names"].append(f"{feature}==OTHER")
        params["output_sources"].append(feature)
        params["categorical"][feature] = {
            "levels": levels, "reference": reference, "modeled_levels": modeled_levels,
        }

    return np.column_stack(columns).astype(np.float32), params


def apply_preprocessor(frame, params, continuous, boolean, categorical):
    columns = []
    for feature in continuous:
        p = params["continuous"][feature]
        raw = numeric_array(frame[feature])
        missing = ~np.isfinite(raw)
        transformed = np.log1p(np.clip(raw, 0, None)) if p["log1p"] else raw
        clipped = np.clip(transformed, p["lower"], p["upper"])
        filled = np.where(np.isfinite(clipped), clipped, p["median"])
        columns.append(((filled - p["median"]) / p["scale"]).astype(np.float32))
        if p["missing_indicator"]:
            columns.append(missing.astype(np.float32))

    for feature in boolean:
        p = params["boolean"][feature]
        raw = numeric_array(frame[feature])
        missing = ~np.isfinite(raw)
        columns.append(np.where(missing, 0.0, raw).astype(np.float32))
        if p["missing_indicator"]:
            columns.append(missing.astype(np.float32))

    for feature in categorical:
        p = params["categorical"][feature]
        values = frame[feature].astype("string").fillna("MISSING")
        for level in p["modeled_levels"]:
            columns.append((values == level).to_numpy(dtype=np.float32))
        columns.append((~values.isin(p["levels"])).to_numpy(dtype=np.float32))

    matrix = np.column_stack(columns).astype(np.float32)
    assert matrix.shape[1] == len(params["output_names"])
    return matrix


def fit_ridge(matrix, target, strength, weights=None):
    x = np.column_stack([np.ones(len(matrix), dtype=np.float32), matrix]).astype(np.float64)
    y = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(y)
    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        valid &= np.isfinite(w) & (w > 0)
    x, y, w = x[valid], y[valid], w[valid]
    w = w / np.mean(w)
    root_w = np.sqrt(w)
    xw = x * root_w[:, None]
    yw = y * root_w
    gram = (xw.T @ xw) / len(xw)
    rhs = (xw.T @ yw) / len(xw)
    penalty = np.eye(x.shape[1]) * strength
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, rhs)


def predict_ridge(matrix, coefficients):
    return coefficients[0] + matrix.astype(np.float64) @ coefficients[1:]


def forecast_metrics(y_raw, prediction_raw):
    y = np.asarray(y_raw, dtype=float)
    p = np.asarray(prediction_raw, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], np.maximum(p[valid], 0)
    y_log, p_log = np.log1p(y), np.log1p(p)
    return {
        "observations": len(y),
        "mae_raw": np.mean(np.abs(y - p)),
        "wape": np.sum(np.abs(y - p)) / np.sum(np.abs(y)),
        "mae_log": np.mean(np.abs(y_log - p_log)),
        "rmse_log": np.sqrt(np.mean((y_log - p_log) ** 2)),
        "spearman": pd.Series(y_log).rank(pct=True).corr(pd.Series(p_log).rank(pct=True)),
        "aggregate_bias": np.sum(p) / np.sum(y) - 1,
    }
