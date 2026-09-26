# SEMAI: predicting greenhouse misting two hours ahead

**Team:** Brand New Developers · **Competition:** PSC GET Talent 2026, Machine Learning (software) · judging 17 Oct 2026

Our tabletop greenhouse rig mists when the air's vapour-pressure deficit (VPD) rises above 1.2 kPa, but it can only react once the air is already dry. **SEMAI Forecast** is a supervised model that answers, for every hour: *will this greenhouse need misting in the next 2 hours?* That gives small growers in Penang an early warning before the hot afternoon.

> **For GETT 2026 judges: the ML entry**
> - **Notebook:** [`SEMAI_Forecast.ipynb`](SEMAI_Forecast.ipynb). The whole project, top to bottom: load, clean, label, features, time split, baselines, two models, evaluation, top-5 features with evidence, and a live demo. Runs in under a minute.
> - **Dataset:** `dataset/penang_weather_hourly_2023_2025_raw.csv` (26,304 rows × 26 columns, exactly as downloaded) and `dataset/penang_weather_hourly_2023_2025_clean.csv` (26,301 rows × 31 columns, written by the notebook). See [DATASETS.md](DATASETS.md).
> - **Trained model:** `models/semai_forecast_gb.joblib`
> - **Figures:** `figures/` (all written by the notebook)
> - **Poster and references:** `submission/SEMAI-Poster-A2.pdf` and `submission/SEMAI-References.pdf`
>
> **How it was built:** the rig, its sensors and its VPD threshold are the team's own work. The code was written with an AI coding assistant (Claude) and checked by the team. Every number below comes from the notebook and can be re-run.

## Results on 2025, a year the model never saw

Trained on 2023–2024 (17,541 hours), tested on all of 2025 (8,758 hours). 24 % of 2025 hours need misting within 2 hours.

| Method | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Always "no" | 0.760 | 0.000 | 0.000 | 0.000 | — |
| "Dry right now" rule | 0.899 | 0.839 | 0.717 | 0.773 | — |
| Usual dry hours rule | 0.839 | 0.635 | 0.772 | 0.697 | 0.905 |
| Logistic regression | 0.942 | 0.881 | 0.879 | 0.880 | 0.984 |
| **Gradient boosting** | **0.945** | **0.898** | **0.872** | **0.885** | **0.985** |

- The model catches 1,831 of 2,100 misting-due hours in 2025 (87.2 %), with 209 false alarms. The "dry right now" rule catches 71.7 %.
- "Always no" already scores 76 % accuracy while catching nothing, so we judge on F1, not accuracy.
- Top 5 features (drop in F1 when shuffled): VPD now 0.287, hour of day (cos) 0.034, soil moisture 0.026, 1-hour change in sunlight 0.022, relative humidity 0.020.
- Why VPD at all: a "mist when humidity drops below 60 %" rule misses 2,379 of Penang's 3,555 dry-stress hours (66.9 %).

## Run the notebook

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\jupyter.exe notebook SEMAI_Forecast.ipynb
```

Then **Run → Run All Cells**. No internet is needed: the dataset is in `dataset/`.

The saved outputs were produced with Python 3.12, scikit-learn 1.9.1 and pandas 3.0 on Windows. Other versions or machines can move the gradient-boosting scores by about ±0.003; the conclusions don't change.

## Data source and licence

Penang hourly weather, 1 Jan 2023 – 31 Dec 2025, from the Open-Meteo Historical Weather API (ERA5 reanalysis). Licence CC BY 4.0: weather data by Open-Meteo.com. Real data only, nothing synthetic. Full details in [DATASETS.md](DATASETS.md).

---

## The rest of this repo: the booth demo

These parts run at the booth beside the rig. They are **not** part of the GETT ML entry, which is the notebook above.

| Part | What it does | Data |
|---|---|---|
| **SEMAI Console** | One offline laptop web app: live rig readings, the camera, and a forecast replay | the rig's own Wi-Fi |
| **Console forecast** | An earlier, smaller version of SEMAI Forecast (temperature and humidity only, 17 features) that the console uses | `models/forecast.joblib` |
| **SEMAI Wilt Watch** | Image model: is the plant under the camera healthy or wilted? | 904 labelled photos (Kaggle), not redistributed |

The rig (a Raspberry Pi Pico H running the misting loop, and an ESP32-S3-CAM serving readings and a camera stream over its own Wi-Fi) is finished and frozen. The console's code was built in one run on 25 Sep 2026 from [docs/SPEC.md](docs/SPEC.md). Every choice is in [DECISIONS.md](DECISIONS.md), the outcome of each step is in [STATUS.md](STATUS.md), and its numbers are in `results/` (written up in [results/RESULTS.md](results/RESULTS.md)).

### Reproduce the console's models (online for the two data downloads, then offline)

```
.venv\Scripts\python.exe -m semai.penang_data        # fetch + cache Penang temperature and humidity
.venv\Scripts\python.exe scripts\train_forecast.py   # console forecast: train on 2023-2024, test on 2025
.venv\Scripts\python.exe scripts\why_vpd.py          # "Why VPD, not humidity" figures
.venv\Scripts\python.exe scripts\train_wilt.py       # Wilt Watch: grouped 5-fold CV + final model
.venv\Scripts\python.exe scripts\eval_rig_frames.py  # rig-photo check (pending until photos exist)
.venv\Scripts\python.exe scripts\fetch_today.py      # today's Penang outlook -> data/today.json
```

`notebooks/semai_results.ipynb` re-runs those scripts and shows their figures.

### Demo (offline, at the booth)

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
SEMAI_Forecast.ipynb   the GETT ML entry
dataset/     raw + cleaned Penang weather CSVs, and the exact download URL
figures/     every figure and table the notebook writes
models/      semai_forecast_gb.joblib (the entry), forecast.joblib + wilt.joblib (booth demo)
submission/  A2 poster and references (PDF)
semai/       console code: vpd.py penang_data.py forecast.py wilt.py rig_client.py logger.py app.py templates/ static/
scripts/     console training, evaluation and fake rig
notebooks/   semai_results.ipynb (console results)
results/     console figures, metrics JSON and write-ups
data/ logs/  gitignored (downloads, rig logs)
```

## Hard rules we kept

No synthetic training data; no accuracy reported on training data; a time split (train ≤ 2024, test 2025) with an assert that no 2025 row is used in training; every model number shown beside its baseline; the console loads nothing from the internet; honest labels (MOCK DATA banner, replay label); Kaggle photos are cite-only and never appear in figures, slides or the poster.
