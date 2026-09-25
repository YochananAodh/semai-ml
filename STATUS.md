# STATUS.md — SEMAI autonomous build, 25 Sep 2026

Built in one unattended Claude Code run from `docs/SPEC.md` (v2). Written 2026-09-25 14:09 local (Asia/Kuala_Lumpur).
Every number below is read from `results/*.json` and `results/soak_test.txt` by the script that wrote this file; nothing is retyped.

## Steps

| Step | Result | Notes |
|---|---|---|
| 1 Environment | ✅ done | `.venv` on Python 3.12.13 (uv-managed, addressed by full path because `py -3.12` does not resolve here); all packages incl. torch 2.14.0+cpu installed first try; `results/ENV.md`. |
| 2 SEMAI Forecast | ✅ done | 26,304 Penang hours fetched first try; trained on 2023–2024, tested on 2025 only (asserted); model + 3 baselines, permutation importance, `F1`–`F4`, `replay(date)`; `fetch_today.py` run. Test set is 8,758 h (spec expected ~8,759: last two hours of 2025 have no label). |
| 3 Why VPD, not humidity | ✅ done | Hysteresis replica, RH-rule misses, dry-by-hour; `W1`–`W3`; `why_vpd.json`. All numbers match §8. |
| 4 SEMAI Wilt Watch | ✅ done | 904 Kaggle photos first try; MobileNetV3-Large features (torch worked, no fallback); 71 near-duplicate pairs grouped; grouped vs naive CV; `R1`–`R2`; `predict()` in ~7 ms; rig-photo check pending. Grouped CV 83.5 % is 1 point under the spec's ~84.5 % (reported as measured, cause in DECISIONS.md). |
| 5 Mock rig | ✅ done | `scripts/mock_rig.py` on 127.0.0.1:8081: §4 keys, MOCK_ statuses (incl. ERR_SENSOR and ERR_LOW_WATER), ~1/20 nulls, MJPEG from `--images` / webcam / generated frames, `/api/mock/stats`. |
| 6 SEMAI Console | ✅ done | `python -m semai.app`: one `/stream` thread with backoff, 2 s polling, CSV logger, offline page (no http(s) in templates/static), MOCK DATA banner, "uncalibrated", replay label, nulls as "—", capture buttons, all five endpoints. |
| 7 Soak test + results pack + notebook | ✅ done | Soak test **PASS** (9/9 checks). Pack: RESULTS, EXPLAINER, JUDGE_QA, SLIDES, DEMO_CHECKLIST in `results/` (5 documents audited number-by-number: 845 numbers checked, all sourced; 0 serious findings remain after fixes). Notebook executed with nbconvert: 0 errors, 9 figures. The first pack run was interrupted by the account spend limit and re-run (DECISIONS.md). |
| 8 Finish | ✅ done | Committed as 3fac819 on branch master (this STATUS.md follows in a second commit); `results/` copied to the vault folder; this file. |

No step needed a §3 fallback. Every deviation from a §8 expected number is reported as measured (nothing tuned) and explained in `DECISIONS.md`.

## Headline numbers

**Penang air, 26,304 hours (2023–2025, Open-Meteo / ERA5).**
- VPD > 1.2 kPa: 13.7 % of hours; VPD < 0.6: 59.9 %; 0.8–1.1: 11.6 %.
- The RH < 60 % rule misses 2,428 of 3,604 dry-stress hours = **67.4 %**. Typical missed hour: 30.1 °C, 66 % RH, 1.41 kPa.
- Dry share peaks at 50.2 % at 14:00.

**SEMAI Forecast, tested on 2025 only: 8,758 hours, 24.3 % positive (trained on 17,539 hours of 2023–2024).**

| | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| Always "no" | 75.7 % | 0.000 | 0.000 | 0.000 | — |
| Persistence | 89.7 % | 0.772 | 0.839 | 0.715 | 0.836 (binary) |
| Hour-of-day climatology | 84.1 % | 0.702 | 0.644 | 0.773 | 0.907 |
| **Model** | **93.8 %** | **0.865** | 0.919 | 0.817 | 0.981 |

Model confusion (2025): tn 6,476, fp 153, fn 390, tp 1,739. Top permutation importance: vpd 0.464, hour_cos 0.102, dvpd 0.043.

**SEMAI Wilt Watch, 904 photos (452 healthy / 452 wilted), baseline 50 %.**
- Near-duplicates: 71 pairs covering 124 images, 0 with mismatched labels.
- Grouped 5-fold CV (headline): **83.5 % ± 1.1** (F1 0.836, recall 0.843, AUC 0.912); naive 5-fold CV: 85.6 % ± 2.0 (92 duplicate groups leak across folds).
- Feature extraction 13.6 s on 4 threads; `predict()` mean 7 ms, max 9 ms per 800×600 frame (target < 300 ms).
- Rig-photo check: **pending — no rig photos captured yet** (0 healthy, 0 wilted frames).

**Soak test (PASS):** 297 `/api/state` polls, 0 errors; console recovered 1.0 s after the mock restart (limit 10 s); max 1 `/stream` connection open; MOCK flag true throughout; nulls kept as null.

## §7 self-checks (run independently by the orchestrator, not by the builders)

- ✅ Forecast test set is 2025 only and no 2025 row is in training (asserted in `train_forecast.py`, and re-derived independently: n_test 8,758; re-scoring `models/forecast.joblib` on 2025 reproduces accuracy/F1/AUC to 4 decimals).
- ✅ Wilt CV folds never split a duplicate group (asserted in `train_wilt.py`; re-derived independently from `data/wilt_features.npz`: 71 pairs, 0 groups split in 5 folds).
- ✅ Every figure that shows a model number shows its baseline (F1–F4 carry persistence; R1–R2 carry the 50 % line; checked by eye on all 9 PNGs).
- ✅ `grep` of `semai/templates` and `semai/static` for `http://` and `https://`: no hits at all (the rig URL is injected by the server).
- ✅ No Kaggle image in `results/` (no .jpg/.jpeg; only the 9 matplotlib PNGs).
- ✅ Every results-pack number checked against `results/*.json` (agent audit + scripted token check).

## Needs a human (not part of this run)

- **One 30-minute rig session before Mon 5 Oct 2026** (§10): production firmware on the boards, laptop on `VPD-Greenhouse-S3`, `python -m semai.app`, plain background and white light, capture ~30 healthy and ~30 wilted frames with the console buttons, then `python scripts/eval_rig_frames.py` and re-run the results pack. Until then the rig-photo line stays "pending". Note: a dark or empty camera frame reads as WILTED; the camera must see the plant.
- The night before the demo, run `scripts/fetch_today.py` while online (see `results/DEMO_CHECKLIST.md`).

## Exact demo commands (Windows, from `C:\Dev\semai-ml`)

```
REM night before, online
.venv\Scripts\python.exe scripts\fetch_today.py

REM at the booth: power the rig, join Wi-Fi VPD-Greenhouse-S3 (password: ask the hardware team), then
.venv\Scripts\python.exe -m semai.app
REM open http://127.0.0.1:8000 in the laptop browser. Never open http://192.168.4.1.

REM fallback without the rig (red MOCK DATA banner)
.venv\Scripts\python.exe scripts\mock_rig.py
.venv\Scripts\python.exe -m semai.app --rig http://127.0.0.1:8081

REM optional 5-minute soak test / reproduce everything
.venv\Scripts\python.exe scripts\soak_test.py
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\semai_results.ipynb
```

Results pack: `results/RESULTS.md`, `results/EXPLAINER.md`, `results/JUDGE_QA.md`, `results/SLIDES.md`, `results/DEMO_CHECKLIST.md`. Decisions and fallbacks: `DECISIONS.md`.
