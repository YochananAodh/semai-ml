"""Rig-photo check for SEMAI Wilt Watch (docs/SPEC.md §6 step 4.7).

Evaluates semai.wilt.predict on real frames captured from the rig camera by the
console's "Capture healthy" / "Capture wilted" buttons:

    data/rig_frames/healthy/*.jpg|jpeg|png
    data/rig_frames/wilted/*.jpg|jpeg|png

When both folders are empty it writes a "pending" results/wilt_rig_eval.json
and exits 0. Otherwise it writes accuracy (UNSURE counted as wrong, and also
excluding UNSURE), the confusion matrix, per-file predictions and the 50 %
baseline, and prints a table.

Run:  python scripts/eval_rig_frames.py [--roi 0.7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RIG_DIR = os.path.join(ROOT, "data", "rig_frames")
OUT = os.path.join(ROOT, "results", "wilt_rig_eval.json")
IMG_EXT = (".jpg", ".jpeg", ".png")
BASELINE_ACC = 0.5
CLASSES = ("healthy", "wilted")


def list_frames() -> dict[str, list[str]]:
    out = {}
    for c in CLASSES:
        d = os.path.join(RIG_DIR, c)
        os.makedirs(d, exist_ok=True)
        out[c] = sorted(os.path.join(d, n) for n in os.listdir(d) if n.lower().endswith(IMG_EXT))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roi", type=float, default=0.7, help="centred ROI fraction (default 0.7)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = list_frames()
    n_h, n_w = len(frames["healthy"]), len(frames["wilted"])
    checked = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if n_h == 0 and n_w == 0:
        res = {
            "status": "pending — no rig photos captured yet",
            "n_healthy": 0,
            "n_wilted": 0,
            "baseline_accuracy": BASELINE_ACC,
            "roi": args.roi,
            "folders": ["data/rig_frames/healthy", "data/rig_frames/wilted"],
            "how_to": "capture ~30 healthy and ~30 wilted frames with the console buttons, "
                      "then re-run python scripts/eval_rig_frames.py (SPEC §10)",
            "checked_at_utc": checked,
        }
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"{res['status']} -> {OUT}")
        return 0

    import cv2

    import semai.wilt as W

    if not W.available():
        res = {
            "status": f"error - wilt model unavailable ({W._unavailable_reason()})",
            "n_healthy": n_h, "n_wilted": n_w, "baseline_accuracy": BASELINE_ACC,
            "roi": args.roi, "checked_at_utc": checked,
        }
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(res["status"])
        return 1

    preds = []
    for truth in CLASSES:
        for path in frames[truth]:
            fr = cv2.imread(path, cv2.IMREAD_COLOR)
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            if fr is None:
                preds.append({"file": rel, "true": truth, "label": "UNREADABLE", "p_wilted": None, "ms": None})
                continue
            r = W.predict(fr, roi=args.roi)
            preds.append({"file": rel, "true": truth, "label": r["label"],
                          "p_wilted": round(r["p_wilted"], 4), "ms": round(r["ms"], 1)})

    n = len(preds)
    correct = sum(1 for p in preds if p["label"] == p["true"].upper())
    n_unsure = sum(1 for p in preds if p["label"] == "UNSURE")
    n_unreadable = sum(1 for p in preds if p["label"] == "UNREADABLE")
    decided = [p for p in preds if p["label"] in ("HEALTHY", "WILTED")]
    acc_excl = (sum(1 for p in decided if p["label"] == p["true"].upper()) / len(decided)) if decided else None
    conf = {t: {"HEALTHY": 0, "WILTED": 0, "UNSURE": 0, "UNREADABLE": 0} for t in CLASSES}
    for p in preds:
        conf[p["true"]][p["label"]] += 1
    ms = [p["ms"] for p in preds if p["ms"] is not None]

    res = {
        "status": "done",
        "checked_at_utc": checked,
        "roi": args.roi,
        "n_healthy": n_h,
        "n_wilted": n_w,
        "n_total": n,
        "accuracy": correct / n,
        "accuracy_note": "UNSURE (and unreadable files) counted as wrong",
        "accuracy_excluding_unsure": acc_excl,
        "n_decided": len(decided),
        "n_unsure": n_unsure,
        "n_unreadable": n_unreadable,
        "confusion": conf,
        "baseline_accuracy": BASELINE_ACC,
        "mean_ms": (sum(ms) / len(ms)) if ms else None,
        "predictions": preds,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    print(f"{'file':60s} {'true':8s} {'pred':10s} {'p_wilted':>8s} {'ms':>7s}")
    for p in preds:
        pw = "-" if p["p_wilted"] is None else f"{p['p_wilted']:.3f}"
        msv = "-" if p["ms"] is None else f"{p['ms']:.0f}"
        print(f"{p['file'][-60:]:60s} {p['true']:8s} {p['label']:10s} {pw:>8s} {msv:>7s}")
    print(f"\nrig frames: {n_h} healthy, {n_w} wilted -> accuracy {res['accuracy'] * 100:.1f} % "
          f"(UNSURE counted wrong; {n_unsure} unsure), "
          f"excluding UNSURE {'-' if acc_excl is None else f'{acc_excl * 100:.1f} %'}, "
          f"baseline {BASELINE_ACC * 100:.0f} %")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
