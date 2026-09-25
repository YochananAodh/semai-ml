"""RigLogger: append every rig reading to logs/rig_YYYYMMDD.csv (SPEC section 6, Step 6.2).

Columns: ts (laptop local ISO time with milliseconds), vpd, t_air, rh, t_leaf,
water, light, status.  NaN / None are written as empty cells.  The header is
written when a file is created, files roll over by local date, every write is
flushed, and a failing write never crashes the app (it prints to stderr).

That CSV is future training data for free: the forecast model can later be
retrained on the greenhouse's own readings.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(ROOT, "logs")

COLUMNS = ("ts", "vpd", "t_air", "rh", "t_leaf", "water", "light", "status")


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.2f}"
    s = str(v)
    if any(c in s for c in (",", '"', "\n", "\r")):
        s = '"' + s.replace('"', '""') + '"'
    return s


class RigLogger:
    def __init__(self, log_dir: str = DEFAULT_LOG_DIR):
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._fh = None
        self._date = None
        self._path = None
        self.rows_written = 0
        self.write_errors = 0

    @property
    def path(self) -> str | None:
        return self._path

    def _open_for(self, day: str) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"rig_{day}.csv")
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", encoding="utf-8", newline="")
        if new:
            self._fh.write(",".join(COLUMNS) + "\n")
            self._fh.flush()
        self._date = day
        self._path = path

    def log(self, reading: dict) -> None:
        """Append one reading. Thread-safe; never raises."""
        now = datetime.now()
        day = now.strftime("%Y%m%d")
        with self._lock:
            try:
                if self._fh is None or day != self._date:
                    self._open_for(day)
                cells = [now.isoformat(timespec="milliseconds")]
                for k in COLUMNS[1:]:
                    cells.append(_cell(reading.get(k)))
                self._fh.write(",".join(cells) + "\n")
                self._fh.flush()
                self.rows_written += 1
            except Exception as e:  # noqa: BLE001
                self.write_errors += 1
                print(f"[logger] write failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                # drop the handle so the next call reopens it
                try:
                    if self._fh is not None:
                        self._fh.close()
                except Exception:  # noqa: BLE001
                    pass
                self._fh = None

    # RigClient.on_reading callback
    __call__ = log

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:  # noqa: BLE001
                    pass
                self._fh = None


if __name__ == "__main__":  # tiny self-check into a temp folder
    import tempfile
    d = tempfile.mkdtemp()
    lg = RigLogger(d)
    lg.log({"vpd": float("nan"), "t_air": 30.0, "rh": 60.0, "t_leaf": 28.5, "water": 40, "light": None,
            "status": "MOCK_OPTIMAL"})
    lg.log({"vpd": 1.23, "t_air": 30.0, "rh": 60.0, "t_leaf": 28.5, "water": 40, "light": 1200,
            "status": "OPTIMAL"})
    lg.close()
    txt = open(lg.path, encoding="utf-8").read().splitlines()
    assert txt[0] == ",".join(COLUMNS), txt[0]
    assert txt[1].split(",")[1] == "" and txt[1].split(",")[6] == "", txt[1]
    assert txt[2].split(",")[1] == "1.23", txt[2]
    print("logger.py self-check OK:", lg.path)
