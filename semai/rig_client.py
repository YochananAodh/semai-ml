"""RigClient: the console's only link to the rig (SPEC section 6, Step 6.1).

Two daemon threads:
  (a) data poller  - GET <base>/api/data every poll_s with a timeout_s timeout,
                     keeps only the 7 known keys, JSON null -> NaN (never 0),
                     calls on_reading(reading) after every good reading.
  (b) stream reader - ONE long-lived GET <base>/stream (stream=True), splits
                     the MJPEG byte stream on the FFD8...FFD9 JPEG markers and
                     keeps only the latest decoded frame.

Both threads reconnect automatically with exponential backoff 1, 2, 4, 8, 10 s
(reset to 1 s after a success).  There is never more than one /stream request
alive: the stream thread opens a new request only after the previous response
has been closed (SPEC section 2 rule 5 - the ESP32 has only 3 HTTP workers).
"""
from __future__ import annotations

import math
import sys
import threading
import time
from urllib.parse import urlparse

import numpy as np
import requests

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is in requirements, but degrade anyway
    cv2 = None

FLOAT_KEYS = ("vpd", "t_air", "rh", "t_leaf")
INT_KEYS = ("water", "light")
KNOWN_KEYS = FLOAT_KEYS + INT_KEYS + ("status",)

REAL_RIG_HOST = "192.168.4.1"
CONNECTED_WINDOW_S = 6.0
BACKOFF_MAX_S = 10.0
STREAM_BUFFER_CAP = 2 * 1024 * 1024  # ~2 MB


def _log(msg: str) -> None:
    print(f"[rig_client] {msg}", file=sys.stderr, flush=True)


def _to_float(v) -> float:
    """JSON value -> float; null / non-numeric / bool -> NaN (never 0)."""
    if v is None or isinstance(v, bool):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return float("nan")
    return float("nan")


def _to_int(v):
    """JSON value -> int, or None when null / non-numeric."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return int(round(v))
    if isinstance(v, str):
        try:
            return int(round(float(v.strip())))
        except ValueError:
            return None
    return None


def parse_reading(j) -> dict:
    """Keep only the 7 known keys and coerce their types defensively."""
    if not isinstance(j, dict):
        j = {}
    out = {k: _to_float(j.get(k)) for k in FLOAT_KEYS}
    for k in INT_KEYS:
        out[k] = _to_int(j.get(k))
    status = j.get("status")
    out["status"] = status if isinstance(status, str) and status else "UNKNOWN"
    return out


class RigClient:
    def __init__(self, base_url: str, poll_s: float = 2.0, timeout_s: float = 2.0, on_reading=None):
        self._base_url = base_url.rstrip("/")
        self.poll_s = float(poll_s)
        self.timeout_s = float(timeout_s)
        self.on_reading = on_reading

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_data: dict | None = None
        self._last_data_ts: float | None = None
        self._last_frame: np.ndarray | None = None
        self._last_frame_ts: float | None = None
        self._stream_connections_opened = 0
        self._poll_thread: threading.Thread | None = None
        self._stream_thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._poll_thread is not None:
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="rig-data-poller", daemon=True)
        self._stream_thread = threading.Thread(target=self._stream_loop, name="rig-stream-reader", daemon=True)
        self._poll_thread.start()
        self._stream_thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        for th in (self._poll_thread, self._stream_thread):
            if th is not None and th.is_alive():
                th.join(join_timeout)
        self._poll_thread = None
        self._stream_thread = None

    # ------------------------------------------------------------ properties
    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def is_real_rig(self) -> bool:
        try:
            return urlparse(self._base_url).hostname == REAL_RIG_HOST
        except Exception:  # noqa: BLE001
            return False

    @property
    def connected(self) -> bool:
        with self._lock:
            ts = self._last_data_ts
        return ts is not None and (time.time() - ts) <= CONNECTED_WINDOW_S

    @property
    def last_data(self) -> dict | None:
        with self._lock:
            return dict(self._last_data) if self._last_data is not None else None

    @property
    def last_data_ts(self) -> float | None:
        with self._lock:
            return self._last_data_ts

    @property
    def last_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    @property
    def last_frame_ts(self) -> float | None:
        with self._lock:
            return self._last_frame_ts

    @property
    def stream_connections_opened(self) -> int:
        with self._lock:
            return self._stream_connections_opened

    # ------------------------------------------------------------ data poller
    def _poll_loop(self) -> None:
        backoff = 1.0
        failures = 0
        while not self._stop.is_set():
            wait = self.poll_s
            try:
                r = requests.get(self._base_url + "/api/data", timeout=self.timeout_s)
                r.raise_for_status()
                reading = parse_reading(r.json())
                now = time.time()
                with self._lock:
                    self._last_data = reading
                    self._last_data_ts = now
                if failures:
                    _log(f"/api/data back after {failures} failed polls")
                failures = 0
                backoff = 1.0
                if self.on_reading is not None:
                    try:
                        self.on_reading(dict(reading))
                    except Exception as e:  # noqa: BLE001
                        _log(f"on_reading raised {e!r} (ignored)")
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures <= 3 or failures % 20 == 0:
                    _log(f"/api/data failed ({failures}): {type(e).__name__}: {str(e)[:120]}; retry in {backoff:.0f} s")
                wait = backoff
                backoff = min(BACKOFF_MAX_S, backoff * 2)
            self._stop.wait(wait)

    # ------------------------------------------------------------ stream reader
    def _handle_jpeg(self, jpg: bytes) -> bool:
        if cv2 is None:
            return False
        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        now = time.time()
        with self._lock:
            self._last_frame = img
            self._last_frame_ts = now
        return True

    def _stream_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            resp = None
            frames = 0
            try:
                # Exactly one /stream request at a time: the previous response is
                # always closed in the finally block before this line runs again.
                resp = requests.get(self._base_url + "/stream", stream=True, timeout=(2, 10))
                resp.raise_for_status()
                with self._lock:
                    self._stream_connections_opened += 1
                    n = self._stream_connections_opened
                _log(f"/stream opened (#{n})")
                buf = bytearray()
                for chunk in resp.iter_content(chunk_size=4096):
                    if self._stop.is_set():
                        break
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    while True:
                        s = buf.find(b"\xff\xd8")
                        if s < 0:
                            # no start marker: keep only the last byte (could be a lone 0xFF)
                            if len(buf) > 1:
                                del buf[:-1]
                            break
                        e = buf.find(b"\xff\xd9", s + 2)
                        if e < 0:
                            if s > 0:
                                del buf[:s]  # drop multipart headers before the JPEG
                            if len(buf) > STREAM_BUFFER_CAP:
                                _log("stream buffer exceeded cap without an end marker; dropping it")
                                buf.clear()
                            break
                        jpg = bytes(buf[s:e + 2])
                        del buf[:e + 2]
                        if self._handle_jpeg(jpg):
                            frames += 1
                            backoff = 1.0
                _log(f"/stream ended after {frames} frames")
            except Exception as e:  # noqa: BLE001
                _log(f"/stream error after {frames} frames: {type(e).__name__}: {str(e)[:120]}; retry in {backoff:.0f} s")
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:  # noqa: BLE001
                        pass
                    resp = None
            if self._stop.is_set():
                break
            self._stop.wait(backoff)
            backoff = min(BACKOFF_MAX_S, backoff * 2)


if __name__ == "__main__":  # tiny self-check of the parser
    r = parse_reading({"vpd": None, "t_air": "abc", "rh": 60, "t_leaf": 27.5, "water": 40.0,
                       "light": None, "status": 5, "p_baro": 1008.2})
    assert math.isnan(r["vpd"]) and math.isnan(r["t_air"]) and r["rh"] == 60.0 and r["t_leaf"] == 27.5
    assert r["water"] == 40 and r["light"] is None and r["status"] == "UNKNOWN"
    assert "p_baro" not in r and set(r) == set(KNOWN_KEYS)
    print("rig_client.py parse self-check OK")
