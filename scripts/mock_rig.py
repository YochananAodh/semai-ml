"""Mock SEMAI rig (SPEC section 6, Step 5).

A Flask server on 127.0.0.1:8081 that mimics the real ESP32-S3-CAM rig
(SPEC section 4) closely enough to develop and soak-test the console offline:

    GET /api/data        -> JSON {vpd, t_air, rh, t_leaf, water, light, status, p_baro}
    GET /stream          -> long-lived multipart/x-mixed-replace MJPEG, 800x600, ~5 fps
    GET /api/mock/stats  -> {"stream_connections_open", "stream_connections_total",
                            "data_requests", "uptime_s"}   (mock only, not on the rig)

Usage:
    python scripts/mock_rig.py [--port 8081] [--images FOLDER] [--period 300]

Frame source for /stream, in this order:
    1. --images FOLDER : cycle through jpg/png files, resized to 800x600, new picture every 2 s
    2. webcam 0        : cv2.VideoCapture(0) if it opens and returns a frame within 2 s
    3. generated       : a solid colour that slowly changes hue, a large "MOCK" label,
                         a timestamp and a frame counter
All three modes stamp a small "MOCK" tag in a corner.  If the webcam fails
mid-stream the server falls back to generated frames and never crashes.

Every status string is prefixed MOCK_ so the console can never mistake
this server for the real rig (SPEC section 2 rule 6).

NOTE on the real rig: the ESP32-S3 HTTP server has only 3 worker slots, so a
client that opens more than one /stream at a time starves /api/data.  This mock
does NOT enforce that limit; it only counts open /stream connections
(/api/mock/stats) so the console's soak test can prove it opens exactly one.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import threading
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
from flask import Flask, Response, jsonify  # noqa: E402

from semai.vpd import VPD_DRY, VPD_OPT_HIGH, VPD_OPT_LOW, VPD_SAT, leaf_vpd  # noqa: E402

random.seed(0)

FRAME_W, FRAME_H = 800, 600
STREAM_FPS = 5.0
IMAGE_HOLD_S = 2.0  # --images mode: change picture every 2 s
NULL_FIELD_RATE = 1 / 20.0  # ~1 reading in 20 has one null float
ERR_SENSOR_RATE = 1 / 60.0  # ~1 reading in 60 is MOCK_ERR_SENSOR with all four null

app = Flask(__name__)

# ---------------------------------------------------------------- shared state
_lock = threading.Lock()
_t0 = time.time()
_stats = {
    "stream_connections_open": 0,
    "stream_connections_total": 0,
    "data_requests": 0,
}
_period_s = 300.0
_images_dir: str | None = None
_image_files: list[str] = []
_image_cache: dict[str, np.ndarray] = {}
_webcam: cv2.VideoCapture | None = None
_webcam_ok = False
_source_name = "generated"
_frame_counter = 0
_frame_lock = threading.Lock()  # cv2.VideoCapture and the counter are shared by all /stream generators

# hysteresis state for the synthetic status (mirrors semai.vpd.rig_state_sequence)
_prev_state = "OPTIMAL"


# ---------------------------------------------------------------- synthetic day
def _day_curve(now: float) -> dict:
    """Smooth synthetic day curve that cycles once per period (default 300 s)."""
    phase = ((now - _t0) % _period_s) / _period_s  # 0..1
    # "sun" shape: cold/humid at phase 0, hot/dry at phase 0.5, back to cold at 1
    s = 0.5 - 0.5 * math.cos(2 * math.pi * phase)  # 0 -> 1 -> 0
    t_air = 24.0 + 9.0 * s  # 24..33 C
    rh = 90.0 - 35.0 * s  # 90..55 %
    t_leaf = t_air - 1.5 + 0.4 * math.sin(2 * math.pi * phase * 3.0)
    light = int(round(200 + 3300 * s))  # raw ADC 200..3500
    water = int(round(100 * (1.0 - phase)))  # drains 100 -> 0 then resets
    water = max(0, min(100, water))
    vpd = float(leaf_vpd(t_leaf, t_air, rh))
    return {
        "vpd": vpd,
        "t_air": t_air,
        "rh": rh,
        "t_leaf": t_leaf,
        "water": water,
        "light": light,
    }


def _rig_state(vpd: float) -> str:
    """Rig thresholds with hysteresis (same logic as semai.vpd.rig_state_sequence)."""
    global _prev_state
    if vpd != vpd:  # NaN keeps the previous state
        s = _prev_state
    elif vpd > VPD_DRY:
        s = "DRY"
    elif vpd < VPD_SAT:
        s = "SATURATED"
    elif VPD_OPT_LOW <= vpd <= VPD_OPT_HIGH:
        s = "OPTIMAL"
    else:
        s = _prev_state
    _prev_state = s
    return s


_STATE_TO_STATUS = {
    "OPTIMAL": "MOCK_OPTIMAL",
    "DRY": "MOCK_HIGH_VPD_MISTING",
    "SATURATED": "MOCK_LOW_VPD_PURGE",
}


def _r2(x):
    return None if x is None else round(float(x), 2)


@app.route("/api/data")
def api_data():
    with _lock:
        _stats["data_requests"] += 1
        d = _day_curve(time.time())
        state = _rig_state(d["vpd"])
        status = _STATE_TO_STATUS[state]
        if d["water"] < 5:  # tank nearly empty: the real rig reports ERR_LOW_WATER
            status = "MOCK_ERR_LOW_WATER"
        # ~1 in 60: whole sensor error, all four floats null
        if random.random() < ERR_SENSOR_RATE:
            out = {
                "vpd": None, "t_air": None, "rh": None, "t_leaf": None,
                "water": d["water"], "light": d["light"], "status": "MOCK_ERR_SENSOR",
            }
        else:
            out = {
                "vpd": _r2(d["vpd"]),
                "t_air": _r2(d["t_air"]),
                "rh": _r2(d["rh"]),
                "t_leaf": _r2(d["t_leaf"]),
                "water": int(d["water"]),
                "light": int(d["light"]),
                "status": status,
            }
            # ~1 in 20: one of the four floats is null (sensor glitch)
            if random.random() < NULL_FIELD_RATE:
                out[random.choice(["vpd", "t_air", "rh", "t_leaf"])] = None
        # extra key the real firmware may also emit; the console must ignore it
        out["p_baro"] = _r2(1008.0 + 3.0 * math.sin((time.time() - _t0) / 97.0))
    return jsonify(out)


@app.route("/api/mock/stats")
def api_stats():
    with _lock:
        out = dict(_stats)
    out["uptime_s"] = round(time.time() - _t0, 2)
    out["frame_source"] = _source_name
    return jsonify(out)


# ---------------------------------------------------------------- frame sources
def _stamp_mock(frame: np.ndarray) -> np.ndarray:
    """Small MOCK tag in the bottom-right corner (all modes)."""
    cv2.rectangle(frame, (FRAME_W - 118, FRAME_H - 40), (FRAME_W - 8, FRAME_H - 8), (0, 0, 0), -1)
    cv2.putText(frame, "MOCK", (FRAME_W - 110, FRAME_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def _generated_frame(n: int) -> np.ndarray:
    hue = int((time.time() * 6) % 180)  # slowly changing hue (OpenCV hue 0..179)
    hsv = np.full((FRAME_H, FRAME_W, 3), (hue, 140, 200), dtype=np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.putText(frame, "MOCK", (170, 330), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 14, cv2.LINE_AA)
    cv2.putText(frame, "MOCK", (170, 330), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 0), 5, cv2.LINE_AA)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {n}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "generated mock frame - no camera", (20, FRAME_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _load_image(path: str) -> np.ndarray | None:
    img = _image_cache.get(path)
    if img is None:
        raw = cv2.imread(path)
        if raw is None:
            return None
        img = cv2.resize(raw, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        if len(_image_cache) < 64:
            _image_cache[path] = img
    return img.copy()


def _images_frame(n: int) -> np.ndarray | None:
    if not _image_files:
        return None
    idx = int((time.time() - _t0) / IMAGE_HOLD_S) % len(_image_files)
    img = _load_image(_image_files[idx])
    if img is None:
        return None
    name = os.path.basename(_image_files[idx])
    cv2.putText(img, f"{name}  #{n}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _webcam_frame() -> np.ndarray | None:
    global _webcam, _webcam_ok
    if _webcam is None or not _webcam_ok:
        return None
    try:
        ok, frame = _webcam.read()
    except Exception as e:  # noqa: BLE001
        print(f"[mock_rig] webcam read raised {e!r}; falling back to generated frames", flush=True)
        ok, frame = False, None
    if not ok or frame is None:
        _webcam_ok = False
        try:
            _webcam.release()
        except Exception:  # noqa: BLE001
            pass
        _webcam = None
        print("[mock_rig] webcam stopped delivering frames; falling back to generated frames", flush=True)
        return None
    return cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)


def _next_frame() -> np.ndarray:
    with _frame_lock:
        return _next_frame_locked()


def _next_frame_locked() -> np.ndarray:
    global _frame_counter, _source_name
    _frame_counter += 1
    n = _frame_counter
    frame = None
    if _image_files:
        frame = _images_frame(n)
        if frame is not None:
            _source_name = "images"
    if frame is None and _webcam_ok:
        frame = _webcam_frame()
        if frame is not None:
            _source_name = "webcam"
    if frame is None:
        frame = _generated_frame(n)
        _source_name = "generated"
    return _stamp_mock(frame)


def _open_webcam(timeout_s: float = 2.0) -> bool:
    """Try webcam 0; succeed only if it opens and returns a frame within timeout_s."""
    global _webcam, _webcam_ok
    result = {"cap": None, "ok": False, "timed_out": False}

    def worker():
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and not result["timed_out"]:
                    result["cap"] = cap
                    result["ok"] = True
                    return
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            print(f"[mock_rig] webcam open raised {e!r}", flush=True)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        result["timed_out"] = True  # a late success releases its own capture
        print(f"[mock_rig] webcam 0 did not answer within {timeout_s} s; using generated frames", flush=True)
        return False
    if result["ok"]:
        _webcam = result["cap"]
        _webcam_ok = True
        return True
    return False


def _mjpeg_generator():
    """Yield multipart JPEG parts at ~5 fps. Counts itself open while running."""
    with _lock:
        _stats["stream_connections_open"] += 1
        _stats["stream_connections_total"] += 1
        n_open = _stats["stream_connections_open"]
        n_total = _stats["stream_connections_total"]
    print(f"[mock_rig] /stream connect  (open={n_open}, total={n_total}, source={_source_name})", flush=True)
    try:
        interval = 1.0 / STREAM_FPS
        while True:
            t_start = time.time()
            try:
                frame = _next_frame()
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                data = jpg.tobytes()
            except Exception as e:  # noqa: BLE001
                print(f"[mock_rig] frame error {e!r}; sending generated frame", flush=True)
                ok, jpg = cv2.imencode(".jpg", _stamp_mock(_generated_frame(-1)))
                data = jpg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                   + data + b"\r\n")
            dt = time.time() - t_start
            if dt < interval:
                time.sleep(interval - dt)
    except GeneratorExit:
        pass
    finally:
        with _lock:
            _stats["stream_connections_open"] -= 1
            n_open = _stats["stream_connections_open"]
        print(f"[mock_rig] /stream disconnect (open={n_open})", flush=True)


@app.route("/stream")
def stream():
    resp = Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return ("SEMAI mock rig. Endpoints: /api/data, /stream, /api/mock/stats\n",
            200, {"Content-Type": "text/plain"})


# ---------------------------------------------------------------- main
def main(argv=None):
    global _period_s, _images_dir, _image_files
    ap = argparse.ArgumentParser(description="SEMAI mock rig (SPEC section 6 step 5)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--images", default=None, help="folder of jpg/png to cycle through on /stream")
    ap.add_argument("--period", type=float, default=300.0, help="seconds per synthetic day cycle")
    args = ap.parse_args(argv)
    _period_s = max(5.0, float(args.period))

    if args.images:
        _images_dir = args.images
        files = []
        for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend(glob.glob(os.path.join(args.images, pat)))
        _image_files = sorted(set(files))
        if _image_files:
            print(f"[mock_rig] frame source: {len(_image_files)} images from {args.images}", flush=True)
        else:
            print(f"[mock_rig] --images {args.images}: no jpg/png found; trying webcam", flush=True)
    if not _image_files:
        if _open_webcam():
            print("[mock_rig] frame source: webcam 0", flush=True)
        else:
            print("[mock_rig] frame source: generated MOCK frames", flush=True)

    print(f"[mock_rig] listening on http://127.0.0.1:{args.port}  (period={_period_s:.0f} s)", flush=True)
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False, debug=False)
    finally:
        if _webcam is not None:
            try:
                _webcam.release()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
