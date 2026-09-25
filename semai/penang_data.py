"""Penang hourly weather (Open-Meteo / ERA5 archive), 2023-01-01 .. 2025-12-31.

Fetch once, cache to data/penang_hourly_2023_2025.csv, copy to results/.
Everything downstream (SEMAI Forecast, "why VPD") reads the CSV.

Columns: time (local Asia/Kuala_Lumpur, ISO), t (°C), rh (%), vpd (kPa), dry (bool)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from semai.vpd import air_vpd, VPD_DRY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
CSV_PATH = os.path.join(DATA_DIR, "penang_hourly_2023_2025.csv")
RAW_JSON_PATH = os.path.join(DATA_DIR, "penang_hourly_raw.json")
META_PATH = os.path.join(DATA_DIR, "penang_hourly_meta.json")

ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=5.4482&longitude=100.29"
    "&start_date=2023-01-01&end_date=2025-12-31"
    "&hourly=temperature_2m,relative_humidity_2m"
    "&timezone=Asia%2FKuala_Lumpur"
)
EXPECTED_ROWS = 26304


def _fetch_raw(retries: int = 3) -> dict:
    """GET the archive JSON with 3 retries and backoff (SPEC §3)."""
    last = None
    for i in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, timeout=120)
            r.raise_for_status()
            j = r.json()
            if "hourly" not in j:
                raise ValueError(f"no 'hourly' key in response: {str(j)[:200]}")
            return j
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i * 2)
    raise RuntimeError(f"Open-Meteo archive unreachable after {retries} tries: {last}")


def raw_to_frame(j: dict) -> pd.DataFrame:
    h = j["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "t": pd.to_numeric(pd.Series(h["temperature_2m"]), errors="coerce"),
        "rh": pd.to_numeric(pd.Series(h["relative_humidity_2m"]), errors="coerce"),
    })
    df["vpd"] = air_vpd(df["t"].values, df["rh"].values)
    df["dry"] = df["vpd"] > VPD_DRY
    return df


def load(force_fetch: bool = False) -> pd.DataFrame:
    """Return the Penang hourly frame, fetching + caching on first use."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH) and not force_fetch:
        df = pd.read_csv(CSV_PATH, parse_dates=["time"])
        return df

    # Offline fallback: the CSV is also committed under results/ (it is the
    # project's dataset file), so a fresh clone can replay 2025 without the
    # network. Copy it back into data/ and use it.
    results_csv = os.path.join(RESULTS_DIR, "penang_hourly_2023_2025.csv")
    if os.path.exists(results_csv) and not force_fetch:
        shutil.copyfile(results_csv, CSV_PATH)
        return pd.read_csv(CSV_PATH, parse_dates=["time"])

    fetched_at = None
    if os.path.exists(RAW_JSON_PATH) and not force_fetch:
        with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        # fetch time of the cached raw file = its mtime (UTC)
        fetched_at = datetime.fromtimestamp(os.path.getmtime(RAW_JSON_PATH), tz=timezone.utc)
    else:
        fetched_at = datetime.now(tz=timezone.utc)
        j = _fetch_raw()
        with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(j, f)

    df = raw_to_frame(j)
    df.to_csv(CSV_PATH, index=False)
    shutil.copyfile(CSV_PATH, os.path.join(RESULTS_DIR, "penang_hourly_2023_2025.csv"))
    meta = {
        "url": ARCHIVE_URL,
        "fetched_at_utc": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(df)),
        "expected_rows": EXPECTED_ROWS,
        "latitude": j.get("latitude"),
        "longitude": j.get("longitude"),
        "timezone": j.get("timezone"),
        "elevation_m": j.get("elevation"),
        "source": "Open-Meteo Historical Weather API (ERA5 reanalysis), CC BY 4.0",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return df


def meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"url": ARCHIVE_URL}


if __name__ == "__main__":
    df = load()
    print(f"rows={len(df)} (expected {EXPECTED_ROWS})")
    print(df.head(3).to_string())
    print(df.tail(3).to_string())
    print("NaN t:", int(df['t'].isna().sum()), "NaN rh:", int(df['rh'].isna().sum()))
    print("share dry (vpd>1.2): %.3f" % df["dry"].mean())
    print(meta())
