"""Pull today's Penang outlook (Open-Meteo *weather forecast*, not the archive)
into data/today.json for the console's "today's Penang outlook" block.

Run this the night before the demo while online (docs/SPEC.md §6 step 2.7).
The console never fetches anything itself; it only reads this file.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import requests  # noqa: E402

from semai.vpd import air_vpd, VPD_DRY  # noqa: E402

URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=5.4482&longitude=100.29"
    "&hourly=temperature_2m,relative_humidity_2m"
    "&past_days=1&forecast_days=2&timezone=Asia%2FKuala_Lumpur"
)
OUT = os.path.join(ROOT, "data", "today.json")


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    last = None
    for i in range(3):
        try:
            r = requests.get(URL, timeout=60)
            r.raise_for_status()
            j = r.json()
            if "hourly" not in j:
                raise ValueError("no 'hourly' in response")
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i * 2)
    else:
        print(f"fetch_today: FAILED after 3 tries: {last}", file=sys.stderr)
        return 1

    h = j["hourly"]
    rows = []
    for t, temp, rh in zip(h["time"], h["temperature_2m"], h["relative_humidity_2m"]):
        vpd = None
        if temp is not None and rh is not None:
            vpd = round(float(air_vpd(temp, rh)), 3)
        rows.append({
            "time": t,
            "t": temp,
            "rh": rh,
            "vpd": vpd,
            "dry": (vpd is not None and vpd > VPD_DRY),
        })
    now_utc = datetime.now(tz=timezone.utc)
    out = {
        "label": "today's Penang outlook (weather forecast)",
        "source": "Open-Meteo Forecast API (weather forecast, not measurements)",
        "url": URL,
        "fetched_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_at_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "timezone": j.get("timezone", "Asia/Kuala_Lumpur"),
        "latitude": j.get("latitude"),
        "longitude": j.get("longitude"),
        "hourly": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    n_dry = sum(1 for r in rows if r["dry"])
    print(f"fetch_today: wrote {OUT} with {len(rows)} hours ({n_dry} dry) at {out['fetched_at_local']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
