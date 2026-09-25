"""SEMAI Forecast: "will this greenhouse need misting in the next 2 hours?"

Single source of truth for the feature set, the label, the saved model and the
2025 replay used by the console (docs/SPEC.md §6 step 2).

    label(t)  = 1 if dry(t+1) or dry(t+2), where dry = air VPD > 1.2 kPa
    features  = FEATURES below, computed at hour t on the continuous hourly series

Public API
    build_features(df)   -> (X, y, meta)   X: 17-column DataFrame, y: labels (NaN
                            where undefined), meta: DataFrame[time, vpd, dry]
    load_model()         -> dict or None (cached; None when models/forecast.joblib
                            is missing)
    available()          -> bool
    available_dates()    -> ["2025-01-01", ..., "2025-12-31"]
    replay(date)         -> 24 dicts {hour, t, rh, vpd, prob, actual_dry, dry_now}
"""
from __future__ import annotations

import os
import re
from datetime import date as _date

import numpy as np
import pandas as pd

from semai.vpd import VPD_DRY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "forecast.joblib")

FEATURES = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "t", "rh", "vpd",
    "t_lag1", "t_lag2", "t_lag3",
    "rh_lag1", "rh_lag2", "rh_lag3",
    "vpd_lag1", "vpd_lag2", "vpd_lag3",
    "dvpd",
]
LAGS = (1, 2, 3)
HORIZON_HOURS = (1, 2)          # label looks at t+1 and t+2
THRESHOLD = 0.5
TEST_YEAR = 2025
LABEL_DEFINITION = (
    "label(t) = 1 if air VPD > 1.2 kPa at t+1 or t+2 (misting needed within the "
    "next 2 h); NaN where t+1 or t+2 falls off the end of the series"
)
SPLIT_DEFINITION = (
    "time split: train = rows with year <= 2024 whose label window (t+1, t+2) stays "
    "inside 2024; test = every row with year == 2025"
)


# ----------------------------------------------------------------------------
# Features and label
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame):
    """Build the 17 SEMAI Forecast features and the 2-hour-ahead label.

    `df` must have columns time (datetime64), t (°C), rh (%), vpd (kPa) on a
    continuous hourly grid, sorted by time. Lags and the label are computed on
    the whole series (features use only past hours; the label uses future
    hours). Nothing is dropped here: X, y and meta share df's index, and rows
    with NaN (first 3 hours for lags, last 2 hours for the label) are left in
    so callers decide what to drop.

    Returns (X, y, meta): X is a DataFrame with exactly FEATURES in order, y a
    float Series (1.0 / 0.0 / NaN), meta a DataFrame with time, vpd, dry.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_features expects a pandas DataFrame")
    for col in ("time", "t", "rh", "vpd"):
        if col not in df.columns:
            raise ValueError(f"build_features: missing column {col!r}")

    d = df[["time", "t", "rh", "vpd"]].copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values("time", kind="stable")

    time = d["time"]
    t = d["t"].astype(float).to_numpy()
    rh = d["rh"].astype(float).to_numpy()
    vpd = d["vpd"].astype(float).to_numpy()
    hour = time.dt.hour.to_numpy().astype(float)
    doy = time.dt.dayofyear.to_numpy().astype(float)

    X = pd.DataFrame(index=d.index)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["t"] = t
    X["rh"] = rh
    X["vpd"] = vpd
    for k in LAGS:
        X[f"t_lag{k}"] = d["t"].astype(float).shift(k).to_numpy()
    for k in LAGS:
        X[f"rh_lag{k}"] = d["rh"].astype(float).shift(k).to_numpy()
    for k in LAGS:
        X[f"vpd_lag{k}"] = d["vpd"].astype(float).shift(k).to_numpy()
    X["dvpd"] = X["vpd"].to_numpy() - X["vpd_lag1"].to_numpy()
    X = X[FEATURES]

    dry = pd.Series(vpd > VPD_DRY, index=d.index)
    dry_f = dry.astype(float)
    fut = [dry_f.shift(-h) for h in HORIZON_HOURS]        # NaN at the tail
    y = pd.concat(fut, axis=1).max(axis=1, skipna=False)   # NaN if any future missing
    y.name = "label"

    meta = pd.DataFrame({"time": time, "vpd": vpd, "dry": dry.to_numpy()}, index=d.index)
    return X, y, meta


# ----------------------------------------------------------------------------
# Model loading (cached)
# ----------------------------------------------------------------------------
_MODEL_CACHE: dict = {"loaded": False, "bundle": None}


def load_model():
    """Return the saved model bundle dict, or None when models/forecast.joblib
    is missing. Cached in a module global after the first call."""
    if _MODEL_CACHE["loaded"]:
        return _MODEL_CACHE["bundle"]
    bundle = None
    if os.path.exists(MODEL_PATH):
        import joblib
        bundle = joblib.load(MODEL_PATH)
    _MODEL_CACHE["loaded"] = True
    _MODEL_CACHE["bundle"] = bundle
    return bundle


def reset_cache():
    """Forget the cached model and replay frame (used after retraining)."""
    _MODEL_CACHE["loaded"] = False
    _MODEL_CACHE["bundle"] = None
    _REPLAY_CACHE["frame"] = None


def available() -> bool:
    return load_model() is not None


# ----------------------------------------------------------------------------
# Replay of 2025
# ----------------------------------------------------------------------------
def available_dates() -> list:
    """All 365 days of the test year as 'YYYY-MM-DD' strings."""
    days = pd.date_range(f"{TEST_YEAR}-01-01", f"{TEST_YEAR}-12-31", freq="D")
    return [d.strftime("%Y-%m-%d") for d in days]


_REPLAY_CACHE: dict = {"frame": None}
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _replay_frame() -> pd.DataFrame:
    """2025 rows with features, label, and (if the model exists) probability.
    Built lazily from the cached CSV and kept in a module global."""
    if _REPLAY_CACHE["frame"] is not None:
        return _REPLAY_CACHE["frame"]
    from semai import penang_data
    df = penang_data.load()
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    X, y, meta = build_features(df)
    is_test = meta["time"].dt.year.to_numpy() == TEST_YEAR
    Xt = X.loc[is_test]
    frame = pd.DataFrame({
        "time": meta.loc[is_test, "time"].to_numpy(),
        "t": df.loc[is_test, "t"].astype(float).to_numpy(),
        "rh": df.loc[is_test, "rh"].astype(float).to_numpy(),
        "vpd": meta.loc[is_test, "vpd"].to_numpy(),
        "dry_now": meta.loc[is_test, "dry"].to_numpy(),
        "label": y.loc[is_test].to_numpy(),
    })
    feat_ok = ~Xt.isna().any(axis=1).to_numpy()
    prob = np.full(len(frame), np.nan)
    bundle = load_model()
    if bundle is not None and feat_ok.any():
        model = bundle["model"]
        feats = bundle.get("features", FEATURES)
        prob[feat_ok] = model.predict_proba(Xt.loc[feat_ok, feats])[:, 1]
    frame["prob"] = prob
    frame["date"] = pd.Series(frame["time"]).dt.strftime("%Y-%m-%d").to_numpy()
    frame["hour"] = pd.Series(frame["time"]).dt.hour.to_numpy()
    _REPLAY_CACHE["frame"] = frame
    return frame


def _parse_date(date: str) -> str:
    if not isinstance(date, str):
        raise ValueError(f"date must be a 'YYYY-MM-DD' string, got {type(date).__name__}")
    m = _DATE_RE.match(date.strip())
    if not m:
        raise ValueError(f"malformed date {date!r}, expected YYYY-MM-DD")
    yy, mm, dd = (int(g) for g in m.groups())
    try:
        d = _date(yy, mm, dd)
    except ValueError as e:
        raise ValueError(f"invalid date {date!r}: {e}") from None
    if d.year != TEST_YEAR:
        raise ValueError(f"replay only covers {TEST_YEAR}, got {date!r}")
    return d.strftime("%Y-%m-%d")


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def replay(date: str) -> list:
    """Hourly replay of one 2025 day.

    Returns 24 dicts {hour, t, rh, vpd, prob, actual_dry, dry_now}:
      prob        model probability that misting is needed within 2 h
                  (None if the model is missing or features are unavailable)
      actual_dry  the label: whether VPD > 1.2 actually followed at t+1 or t+2
                  (None only for the last hours of 2025-12-31)
      dry_now     VPD(t) > 1.2
    Raises ValueError for a malformed or non-2025 date.
    """
    key = _parse_date(date)
    frame = _replay_frame()
    day = frame[frame["date"].to_numpy() == key]
    if len(day) != 24:
        raise ValueError(f"expected 24 hourly rows for {key}, found {len(day)}")
    out = []
    for row in day.itertuples(index=False):
        lab = row.label
        out.append({
            "hour": int(row.hour),
            "t": float(row.t),
            "rh": float(row.rh),
            "vpd": float(row.vpd),
            "prob": _f(row.prob),
            "actual_dry": None if (lab is None or np.isnan(lab)) else bool(lab >= 0.5),
            "dry_now": bool(row.dry_now),
        })
    out.sort(key=lambda r: r["hour"])
    return out


if __name__ == "__main__":  # tiny self-check
    import time as _time
    from semai import penang_data
    X, y, meta = build_features(penang_data.load())
    assert list(X.columns) == FEATURES and len(X) == len(y) == len(meta)
    assert int(X.isna().any(axis=1).sum()) == 3 and int(y.isna().sum()) == 2
    print("features OK:", X.shape, "labels NaN:", int(y.isna().sum()))
    print("model available:", available())
    t0 = _time.perf_counter(); r = replay("2025-03-15"); t1 = _time.perf_counter()
    _ = replay("2025-03-15"); t2 = _time.perf_counter()
    print(r[:3])
    print(f"replay first call {1000*(t1-t0):.1f} ms, second call {1000*(t2-t1):.2f} ms")
