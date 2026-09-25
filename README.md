# SEMAI — laptop brain for a leaf-VPD misting greenhouse

**Team:** Brand New Developers · **Competition:** PSC GET Talent (judging 17 Oct 2026)

The rig (a Raspberry Pi Pico H running the misting loop, and an ESP32-S3-CAM serving readings and a camera stream over its own Wi-Fi) is finished and frozen. This repo is the laptop side:

| Part | What it does | Data |
|---|---|---|
| **SEMAI Forecast** | Supervised model: *will this greenhouse need misting in the next 2 hours?* | 3 years of real Penang weather (Open-Meteo / ERA5) |
| **SEMAI Wilt Watch** | Supervised image model: is the plant under the camera healthy or wilted? | 904 real labelled photos (Kaggle) |
| **SEMAI Console** | One offline laptop web app: live rig readings, camera with the wilt verdict, forecast replay | the rig's own Wi-Fi |

> **For GETT 2026 judges: what's where**
> - **Source code:** `semai/` (models, console) and `scripts/` (training, evaluation, fake rig)
> - **Dataset documentation:** [DATASETS.md](DATASETS.md)
> - **Trained model files:** `models/forecast.joblib` (SEMAI Forecast) and `models/wilt.joblib` (SEMAI Wilt Watch)
> - **Results and figures:** `results/`, written up in [results/RESULTS.md](results/RESULTS.md)
> - **Notebook:** `notebooks/semai_results.ipynb` re-runs the scripts and shows every figure
>
> **How it was built:** the rig, its sensors and its VPD thresholds are the team's own work. The laptop software was written with an AI coding assistant (Claude Code) from the spec in `docs/SPEC.md`. Every choice is logged in [DECISIONS.md](DECISIONS.md), and every reported number can be re-run from the scripts.

Everything was built in one unattended run on 25 Sep 2026 from [docs/SPEC.md](docs/SPEC.md). Every choice and fallback is in [DECISIONS.md](DECISIONS.md); the outcome of each step is in [STATUS.md](STATUS.md). All numbers live in `results/*.json` and are written up in [results/RESULTS.md](results/RESULTS.md).

## Setup (once, online)

```
py -3.12 -m venv .venv          # any Python 3.12/3.13
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Reproduce the results (online for the two data downloads, then offline)

```
.venv\Scripts\python.exe -m semai.penang_data        # fetch + cache Penang weather (Open-Meteo archive)
.venv\Scripts\python.exe scripts\train_forecast.py   # SEMAI Forecast: train on 2023-2024, test on 2025
.venv\Scripts\python.exe scripts\why_vpd.py          # "Why VPD, not humidity" figures
.venv\Scripts\python.exe scripts\train_wilt.py       # Wilt Watch: grouped 5-fold CV + final model
.venv\Scripts\python.exe scripts\eval_rig_frames.py  # rig-photo check (pending until photos exist)
.venv\Scripts\python.exe scripts\fetch_today.py      # today's Penang outlook -> data/today.json
```

`notebooks/semai_results.ipynb` is a thin wrapper that re-runs the same scripts and shows every figure.

## Demo (offline, at the booth)

1. Power the rig and join Wi-Fi `VPD-Greenhouse-S3` (password: ask the hardware team).
2. `.venv\Scripts\python.exe -m semai.app` and open `http://127.0.0.1:8000`.
3. Never open `http://192.168.4.1` in a browser: the ESP32 has only 3 HTTP workers and the console already holds the one camera stream.

Fallback without the rig (shows a red **MOCK DATA** banner):

```
.venv\Scripts\python.exe scripts\mock_rig.py              # mock rig on 127.0.0.1:8081 (webcam 0 or generated frames)
.venv\Scripts\python.exe -m semai.app --rig http://127.0.0.1:8081
```

Soak test (5 minutes, mock + console, with a mock restart at 2 min): `.venv\Scripts\python.exe scripts\soak_test.py`.

## Layout

```
semai/       vpd.py penang_data.py forecast.py wilt.py rig_client.py logger.py app.py templates/ static/
scripts/     train_forecast.py why_vpd.py train_wilt.py eval_rig_frames.py fetch_today.py mock_rig.py soak_test.py
notebooks/   semai_results.ipynb
results/     figures (PNG, 200 dpi), metrics JSON, the results pack (RESULTS, EXPLAINER, JUDGE_QA, SLIDES, DEMO_CHECKLIST)
models/      the two trained model files (forecast.joblib, wilt.joblib)
data/ logs/  gitignored (data downloads, rig logs)
```

## Hard rules we kept

No synthetic training data; no accuracy reported on training data; a time split for the forecast (train ≤ 2024, test 2025) and grouped cross-validation for the photos (near-duplicates never straddle a fold); every model number shown beside its baseline; the console loads nothing from the internet; one camera connection; honest labels (MOCK DATA banner, "uncalibrated" water level, replay label); Kaggle photos are cite-only and never appear in figures or slides.

## Data sources and licences

- Penang hourly weather 2023–2025: Open-Meteo Historical Weather API (ERA5 reanalysis), CC BY 4.0. Cached copy: `results/penang_hourly_2023_2025.csv`.
- Houseplant photos: Kaggle dataset `russellchan/healthy-and-wilted-houseplant-images` (452 healthy, 452 wilted). Used for training only; cite-only.
- Today's outlook: Open-Meteo Forecast API (a weather forecast, not a measurement), fetched the night before while online.
