"""SEMAI Console (SPEC section 6, Step 6).

    python -m semai.app [--rig URL] [--roi 0.7] [--port 8000]

One offline page at http://127.0.0.1:8000 with live rig readings, the camera
frame with the Wilt Watch verdict, and the SEMAI Forecast replay of 2025.
Nothing is loaded from the internet (SPEC section 2 rule 4).

Endpoints:
    GET  /                         the page
    GET  /api/state                live state (never an HTTP error)
    GET  /frame.jpg                latest frame with ROI + verdict overlay
    GET  /api/replay?date=YYYY-MM-DD
    GET  /api/today                data/today.json
    POST /api/capture?label=healthy|wilted

semai.forecast and semai.wilt are imported lazily and the console degrades to
"forecast model unavailable" / "wilt model unavailable" when they are missing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
from flask import Flask, Response, jsonify, render_template, request  # noqa: E402

from semai.logger import RigLogger  # noqa: E402
from semai.rig_client import FLOAT_KEYS, INT_KEYS, RigClient  # noqa: E402

REPLAY_LABEL = "Replay of 2025 (model never saw this year) · outdoor Penang air"
TODAY_PATH = os.path.join(ROOT, "data", "today.json")
RIG_FRAMES_DIR = os.path.join(ROOT, "data", "rig_frames")
CAPTURE_LABELS = ("healthy", "wilted")
WILT_PERIOD_S = 2.0
FRAME_W, FRAME_H = 800, 600

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))
app.config["JSON_SORT_KEYS"] = False

# ------------------------------------------------------------------ runtime state
STATE = {
    "client": None,          # RigClient
    "logger": None,          # RigLogger
    "roi": 0.7,
    "rig_url": "http://192.168.4.1",
}
_wilt_lock = threading.Lock()
WILT = {"available": False, "label": None, "p_wilted": None, "ms": None, "updated_ts": None, "error": None}


def _log(msg: str) -> None:
    print(f"[app] {msg}", flush=True)


# ------------------------------------------------------------------ lazy model imports
def _get_forecast():
    """Return semai.forecast when importable and available(), else None."""
    try:
        import importlib
        mod = importlib.import_module("semai.forecast")
        if bool(mod.available()):
            return mod
    except Exception as e:  # noqa: BLE001
        _get_forecast.last_error = f"{type(e).__name__}: {str(e)[:120]}"
    return None


_get_forecast.last_error = None


def _get_wilt():
    """Return semai.wilt when importable and available(), else None."""
    try:
        import importlib
        mod = importlib.import_module("semai.wilt")
        if bool(mod.available()):
            return mod
    except Exception as e:  # noqa: BLE001
        _get_wilt.last_error = f"{type(e).__name__}: {str(e)[:120]}"
    return None


_get_wilt.last_error = None


# ------------------------------------------------------------------ helpers
def _json_num(v):
    """float NaN/inf -> None so the JSON is valid (never emit NaN)."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _clean_data(d: dict | None):
    if d is None:
        return None
    out = {}
    for k in FLOAT_KEYS + INT_KEYS:
        out[k] = _json_num(d.get(k))
    out["status"] = d.get("status") if isinstance(d.get("status"), str) else "UNKNOWN"
    return out


def _roi_box(h: int, w: int, roi: float):
    """Centred ROI box (x0, y0, x1, y1); prefers semai.wilt.roi_box when present."""
    mod = _get_wilt()
    if mod is not None:
        try:
            x0, y0, x1, y1 = mod.roi_box(h, w, roi)
            return int(x0), int(y0), int(x1), int(y1)
        except Exception:  # noqa: BLE001
            pass
    roi = min(1.0, max(0.05, float(roi)))
    bw, bh = int(w * roi), int(h * roi)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    return x0, y0, x0 + bw, y0 + bh


def _is_mock(client: RigClient | None, data: dict | None) -> bool:
    if client is None:
        return True
    if not client.is_real_rig:
        return True
    st = (data or {}).get("status")
    return isinstance(st, str) and st.startswith("MOCK")


def _placeholder_frame(text: str) -> np.ndarray:
    img = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
    cv2.putText(img, text, (60, FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    return img


def _encode_jpeg(img: np.ndarray) -> bytes:
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return jpg.tobytes()


# ------------------------------------------------------------------ wilt background thread
def _wilt_loop():
    """Re-run the wilt model on the latest frame every ~2 s (not per /frame.jpg)."""
    while True:
        t0 = time.time()
        try:
            mod = _get_wilt()
            if mod is None:
                with _wilt_lock:
                    WILT["available"] = False
                    WILT["error"] = _get_wilt.last_error
            else:
                client = STATE["client"]
                frame = client.last_frame if client is not None else None
                if frame is None:
                    with _wilt_lock:
                        WILT["available"] = True
                        WILT["label"] = None
                        WILT["p_wilted"] = None
                        WILT["ms"] = None
                else:
                    res = mod.predict(frame, roi=float(STATE["roi"]))
                    with _wilt_lock:
                        WILT["available"] = True
                        WILT["label"] = str(res.get("label"))
                        WILT["p_wilted"] = _json_num(res.get("p_wilted"))
                        WILT["ms"] = _json_num(res.get("ms"))
                        WILT["updated_ts"] = time.time()
                        WILT["error"] = None
        except Exception as e:  # noqa: BLE001
            with _wilt_lock:
                WILT["available"] = False
                WILT["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            _log(f"wilt inference failed: {WILT['error']}")
        dt = time.time() - t0
        time.sleep(max(0.2, WILT_PERIOD_S - dt))


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html",
                           rig_url=STATE["rig_url"],
                           roi=STATE["roi"],
                           replay_label=REPLAY_LABEL)


@app.route("/api/state")
def api_state():
    try:
        client: RigClient | None = STATE["client"]
        now = time.time()
        data = client.last_data if client is not None else None
        data_ts = client.last_data_ts if client is not None else None
        frame_ts = client.last_frame_ts if client is not None else None
        with _wilt_lock:
            wilt = dict(WILT)
        out = {
            "connected": bool(client.connected) if client is not None else False,
            "rig_url": STATE["rig_url"],
            "mock": _is_mock(client, data),
            "data": _clean_data(data),
            "data_age_s": (round(now - data_ts, 2) if data_ts else None),
            "frame_age_s": (round(now - frame_ts, 2) if frame_ts else None),
            "wilt": {
                "available": bool(wilt["available"]),
                "label": wilt["label"] if wilt["available"] else None,
                "p_wilted": wilt["p_wilted"] if wilt["available"] else None,
                "ms": wilt["ms"] if wilt["available"] else None,
                "roi": float(STATE["roi"]),
                "updated_ts": wilt["updated_ts"] if wilt["available"] else None,
            },
            "forecast_available": _get_forecast() is not None,
            "stream_connections_opened": int(client.stream_connections_opened) if client is not None else 0,
            "server_time": datetime.now().isoformat(timespec="milliseconds"),
        }
        return jsonify(out)
    except Exception as e:  # noqa: BLE001
        _log(f"/api/state fell back: {type(e).__name__}: {e}")
        return jsonify({
            "connected": False, "rig_url": STATE.get("rig_url"), "mock": True, "data": None,
            "data_age_s": None, "frame_age_s": None,
            "wilt": {"available": False, "label": None, "p_wilted": None, "ms": None,
                     "roi": STATE.get("roi", 0.7), "updated_ts": None},
            "forecast_available": False, "stream_connections_opened": 0,
            "server_time": datetime.now().isoformat(timespec="milliseconds"),
            "error": f"{type(e).__name__}: {str(e)[:120]}",
        })


@app.route("/frame.jpg")
def frame_jpg():
    try:
        client: RigClient | None = STATE["client"]
        frame = client.last_frame if client is not None else None
        frame_ts = client.last_frame_ts if client is not None else None
        if frame is None:
            img = _placeholder_frame("no camera frame yet")
        else:
            img = frame  # last_frame is already a copy
            h, w = img.shape[:2]
            x0, y0, x1, y1 = _roi_box(h, w, STATE["roi"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
            with _wilt_lock:
                wilt = dict(WILT)
            if wilt["available"] and wilt["label"] is not None:
                p = wilt["p_wilted"]
                txt = f"{wilt['label']}  p_wilted={p:.2f}" if p is not None else str(wilt["label"])
                colour = {"WILTED": (0, 0, 255), "HEALTHY": (0, 200, 0)}.get(wilt["label"], (0, 200, 255))
            elif wilt["available"]:
                txt, colour = "wilt: waiting for a frame", (200, 200, 200)
            else:
                txt, colour = "wilt model unavailable", (0, 0, 255)
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, colour, 2, cv2.LINE_AA)
            age = (time.time() - frame_ts) if frame_ts else float("nan")
            age_txt = f"frame age {age:.1f} s"
            cv2.putText(img, age_txt, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, age_txt, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        data = _encode_jpeg(img)
    except Exception as e:  # noqa: BLE001
        _log(f"/frame.jpg fell back: {type(e).__name__}: {e}")
        data = _encode_jpeg(_placeholder_frame("frame error"))
    resp = Response(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


_DATE_RE = re.compile(r"^2025-(\d{2})-(\d{2})$")


@app.route("/api/replay")
def api_replay():
    date = (request.args.get("date") or "").strip()
    if not _DATE_RE.match(date):
        return jsonify({"error": "date must be a 2025 date in YYYY-MM-DD form", "date": date}), 400
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "not a real calendar date", "date": date}), 400
    mod = _get_forecast()
    if mod is None:
        return jsonify({"available": False, "date": date, "label": REPLAY_LABEL,
                        "reason": "forecast model unavailable", "hours": []})
    try:
        hours = mod.replay(date)
    except ValueError as e:
        return jsonify({"error": str(e)[:200], "date": date}), 400
    except Exception as e:  # noqa: BLE001
        _log(f"forecast.replay failed: {type(e).__name__}: {e}")
        return jsonify({"available": False, "date": date, "label": REPLAY_LABEL,
                        "reason": f"replay failed: {type(e).__name__}", "hours": []})
    clean = []
    for h in hours:
        clean.append({
            "hour": int(h.get("hour", 0)),
            "t": _json_num(h.get("t")),
            "rh": _json_num(h.get("rh")),
            "vpd": _json_num(h.get("vpd")),
            "prob": _json_num(h.get("prob")),
            "actual_dry": (None if h.get("actual_dry") is None else bool(h.get("actual_dry"))),
            "dry_now": bool(h.get("dry_now", False)),
        })
    return jsonify({"available": True, "date": date, "label": REPLAY_LABEL, "hours": clean})


@app.route("/api/today")
def api_today():
    try:
        with open(TODAY_PATH, "r", encoding="utf-8") as f:
            j = json.load(f)
        if not isinstance(j, dict) or "hourly" not in j:
            raise ValueError("today.json has no 'hourly' key")
        j["available"] = True
        return jsonify(j)
    except Exception as e:  # noqa: BLE001
        return jsonify({"available": False, "reason": "run scripts/fetch_today.py while online",
                        "detail": f"{type(e).__name__}: {str(e)[:120]}"})


@app.route("/api/capture", methods=["POST"])
def api_capture():
    label = (request.args.get("label") or "").strip().lower()
    if label not in CAPTURE_LABELS:
        return jsonify({"ok": False, "error": "label must be healthy or wilted"}), 400
    client: RigClient | None = STATE["client"]
    frame = client.last_frame if client is not None else None
    if frame is None:
        return jsonify({"ok": False, "error": "no camera frame yet"}), 409
    folder = os.path.join(RIG_FRAMES_DIR, label)
    os.makedirs(folder, exist_ok=True)
    now = datetime.now()
    name = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}.jpg"
    path = os.path.join(folder, name)
    ok = cv2.imwrite(path, frame)  # RAW frame, no overlay
    if not ok:
        return jsonify({"ok": False, "error": "could not write file"}), 500
    count = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    _log(f"captured {rel} ({count} in {label})")
    return jsonify({"ok": True, "path": rel, "label": label, "count": count})


# ------------------------------------------------------------------ startup
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m semai.app", description="SEMAI Console")
    ap.add_argument("--rig", default="http://192.168.4.1", help="rig base URL (default the real rig)")
    ap.add_argument("--roi", type=float, default=0.7, help="centred ROI fraction for Wilt Watch")
    ap.add_argument("--port", type=int, default=8000)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    STATE["rig_url"] = args.rig.rstrip("/")
    STATE["roi"] = float(args.roi)
    for lab in CAPTURE_LABELS:
        os.makedirs(os.path.join(RIG_FRAMES_DIR, lab), exist_ok=True)

    logger = RigLogger()
    client = RigClient(STATE["rig_url"], poll_s=2.0, timeout_s=2.0, on_reading=logger.log)
    STATE["logger"] = logger
    STATE["client"] = client
    client.start()
    threading.Thread(target=_wilt_loop, name="wilt-inference", daemon=True).start()

    mock = not client.is_real_rig
    _log(f"rig URL: {STATE['rig_url']}")
    _log(f"mock: {mock} (MOCK DATA banner shows when the host is not 192.168.4.1 or a status starts with MOCK)")
    _log(f"roi: {STATE['roi']}")
    _log(f"page: http://127.0.0.1:{args.port}")
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False, debug=False)
    finally:
        client.stop()
        logger.close()


if __name__ == "__main__":
    main()
