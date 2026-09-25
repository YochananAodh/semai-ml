# SEMAI — Autonomous Build Spec (v2 · one run · today)

**Product:** SEMAI, a leaf-VPD misting system with a laptop "brain" · **Team:** Brand New Developers · **Competition:** PSC GET Talent, judging **17 Oct 2026**
**Builder:** Claude Code (Fable), **one unattended run, finished today (25 Sep 2026)** · **Repo:** `C:\Dev\semai-ml` · **Replaces:** v1 of this spec

---

## 0. How to run it

Open a terminal in `C:\Dev\semai-ml` and start Claude Code with permissions pre-approved for this repo:

```
claude --dangerously-skip-permissions
```

Paste this single prompt:

> Read `docs/SPEC.md` completely, then build all of it, steps 1 to 8 in order, in this one session. Do not ask me anything. When something is unclear, take the default in the spec. When something fails, use the fallback in §3. Log every decision and fallback in `DECISIONS.md` and keep going. Finish by writing `STATUS.md` and printing its contents.

**Autonomy rules for Claude Code:**

1. **Never ask the user a question, and never stop to wait for approval.** There is no one to answer.
2. If a choice isn't covered here, pick the simplest option that satisfies the hard rules, write one line about it in `DECISIONS.md`, and continue.
3. **Never stop early.** If a step can't be completed even with its fallback, record that in `STATUS.md` and move on to the next step. The run ends only after step 8.
4. **Never fake a result.** If a number misses its expected value, report the real number. Missing a target is allowed. Inventing a result is not.

---

## 1. What SEMAI is (one product, three parts)

The rig is **finished and frozen**:
- A Raspberry Pi Pico H runs the leaf-VPD misting control loop.
- An ESP32-S3-CAM serves live readings and a camera stream over its own Wi-Fi.

This repo adds SEMAI's laptop brain:

| Part | What it does | Data |
|---|---|---|
| **SEMAI Forecast** | Supervised model: *"will this greenhouse need misting in the next 2 hours?"* | 3 years of real Penang weather (Open-Meteo / ERA5) |
| **SEMAI Wilt Watch** | Supervised image model: is the plant under the camera healthy or wilted? | 904 real labelled photos (Kaggle) |
| **SEMAI Console** | One offline laptop web app: live rig readings, camera with the wilt verdict, and forecast replay | the rig's own Wi-Fi |

There's also a results pack so the team can present it (step 7).

There are no alternatives to compare. Build exactly this.

---

## 2. Hard rules

1. **Never touch the rig's firmware or hardware.** `C:\GDrive\ObsidianSecondMind\10-Work\Projects\PSC-ML-Hardware\klesf26\` is read-only reference. Don't edit, build or flash anything, and don't run `pio`.
2. **No synthetic training data.** Never report accuracy measured on training data.
3. **Fair tests only.** SEMAI Forecast uses a time split: train on 2023–2024, test on 2025. Wilt Watch uses 5-fold cross-validation with near-duplicate images grouped. **Every number is shown next to its baseline.**
4. **Offline demo.** The console never loads anything from the internet: no CDNs, no runtime downloads.
5. **One camera connection.** The console opens exactly one `/stream` connection to the rig and polls `/api/data` gently. The ESP32 only has 3 HTTP workers.
6. **Honest UI.**
   - Mock data always shows a red **MOCK DATA** banner.
   - Forecast replay is labelled **"Replay of 2025 (model never saw this year) · outdoor Penang air"**.
   - Water level is labelled **"uncalibrated"**.
7. **Dataset photos are cite-only.** Never put Kaggle images in figures, slides or public pages.
8. **Stay in bounds.** Only write inside `C:\Dev\semai-ml\` and the vault results folder in step 8.

---

## 3. Defaults and fallbacks (use these instead of asking)

| Situation | Do this |
|---|---|
| Python version | Use `py -3.12` or `py -3.13` if present (`py -0p`), otherwise the default `python`. |
| `torch`/`torchvision` won't install | Retry with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. If it still fails, build Wilt Watch with the **no-torch fallback**: colour histograms (HSV, 16 bins × 3) + HOG on 128×128 grayscale (scikit-image) → the same logistic regression. Report that model's real accuracy and log the fallback. |
| Any other package fails | Try it once without a version pin. If it still fails, skip what depends on it, then log it. |
| Open-Meteo archive unreachable | Retry 3× with backoff. If it still fails, SEMAI Forecast can't be trained, so record that in STATUS and continue with the other steps. |
| Kaggle download fails | Retry 3×. If it still fails, Wilt Watch is marked unavailable, and the console camera panel shows the live frame with "wilt model unavailable". |
| A number misses its expected range (§8) | Report the real number in RESULTS.md with a one-line likely cause. Don't tune on the test set to hit the target. |
| `jupyter nbconvert --execute` fails | Keep the scripts as the reproducible source, delete the broken notebook, and log it. |
| No webcam for the mock stream | Mock stream serves generated frames: a solid colour plus a timestamp, clearly labelled MOCK. |
| A stage takes > 60 min | Ship the simplest working version of it and move on. |

---

## 4. Rig facts (verified from `klesf26\src\main2.cpp`, production build)

- **Wi-Fi AP:** SSID `VPD-Greenhouse-S3`, password: ask the hardware team.
- **Host:** `http://192.168.4.1`.
- **`GET /api/data` returns JSON with the keys `vpd, t_air, rh, t_leaf, water, light, status`.**
  - `vpd`, `t_air`, `rh` and `t_leaf` are floats or `null`. `null` means the sensor is invalid, so show it as "—" and **never treat it as 0**.
  - `water` is an int from 0 to 100, uncalibrated.
  - `light` is a raw ADC int.
  - `status` is one of `OPTIMAL`, `HIGH_VPD_MISTING`, `LOW_VPD_PURGE`, `ERR_LOW_WATER`, `ERR_SENSOR`, `WAITING_FOR_PICO` or `PICO_OFFLINE`.
  - Ignore unknown extra keys such as `p_baro` or `anom`.
- **`GET /stream`** is MJPEG at 800×600 and long-lived. The Pico updates every 2 s.
- **Rig thresholds:**
  - DRY (mist) when VPD > **1.2** kPa
  - OPTIMAL 0.8–1.1
  - SATURATED (fan purge) when < **0.6**
  - gaps keep the previous state
- **VPD formula:** `SVP(T) = 0.6108·exp(17.27·T/(T+237.3))` kPa. Air VPD = `SVP(T_air)·(1−RH/100)`. Leaf VPD = `SVP(T_leaf) − SVP(T_air)·RH/100`.

---

## 5. Repo layout

```
C:\Dev\semai-ml\
  CLAUDE.md  README.md  DECISIONS.md  STATUS.md  requirements.txt  .gitignore
  docs\SPEC.md
  semai\  __init__.py  vpd.py  penang_data.py  forecast.py  wilt.py
          rig_client.py  logger.py  app.py  templates\index.html  static\app.js  static\style.css
  scripts\ train_forecast.py  why_vpd.py  train_wilt.py  eval_rig_frames.py
           fetch_today.py  mock_rig.py  soak_test.py
  notebooks\ semai_results.ipynb      # thin wrapper that re-runs the scripts and shows every figure
  data\  models\  logs\               # gitignored
  results\                            # figures, metrics JSON and the results pack (committed)
```

`CLAUDE.md` should contain: *"Read docs/SPEC.md first. Follow §2 hard rules and §3 fallbacks. Never ask the user; log decisions in DECISIONS.md."*

---

## 6. Build steps (in order, all in this session)

### Step 1 — Environment
- `git init`.
- Create `.venv` per §3.
- `requirements.txt`: numpy, pandas, scikit-learn, scikit-image, requests, matplotlib, joblib, pillow, opencv-python, flask, torch, torchvision, jupyter, nbconvert.
- Install. Write the versions to `results/ENV.md`.

### Step 2 — SEMAI Forecast
1. **Fetch the data.**
   - URL: `https://archive-api.open-meteo.com/v1/archive?latitude=5.4482&longitude=100.29&start_date=2023-01-01&end_date=2025-12-31&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FKuala_Lumpur`
   - Cache to `data/penang_hourly_2023_2025.csv` and copy to `results/`. It's public, and it's the project's dataset file.
   - Expect 26,304 rows.
2. **Build the label.**
   - `vpd = SVP(T)·(1−RH/100)`
   - `dry = vpd > 1.2`
   - **label = 1 if `dry` at t+1 or t+2**
3. **Build the features at hour t:**
   - `sin/cos(2π·hour/24)` and `sin/cos(2π·dayofyear/365)`
   - `T`, `RH` and `vpd` at t and at lags of 1, 2 and 3 hours
   - `dvpd = vpd(t) − vpd(t−1)`
   - Drop any rows with NaN.
4. **Split, train and score.**
   - Train on years ≤ 2024. Test on 2025 only.
   - Model: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)` with a 0.5 threshold.
   - Baselines, all on 2025:
     - always "no"
     - persistence (predict `dry(t)`)
     - hour-of-day climatology, with its threshold chosen on training data to maximise F1
   - Metrics: accuracy, F1, precision, recall, AUC and the confusion matrix.
   - Also compute permutation importance on the 2025 test set.
5. **Save outputs.**
   - `models/forecast.joblib` and `results/forecast_metrics.json` (every number, plus the URL, fetch time, versions and seed).
   - `forecast.replay(date)` returns the hourly `{hour, t, rh, vpd, prob, actual_dry}` for any 2025 date.
6. **Draw the figures** in `results/`, PNG at 200 dpi, plain and readable with units on the axes:
   - `F1_model_vs_baselines.png`
   - `F2_confusion_2025.png`
   - `F3_feature_importance.png`
   - `F4_replay_3days.png`: three 2025 days (the driest day, a mixed day and the wettest day, picked automatically by daily max VPD), each showing actual VPD, the 1.2 line and the model's probability.
7. **Today's view.** `scripts/fetch_today.py` pulls `https://api.open-meteo.com/v1/forecast?latitude=5.4482&longitude=100.29&hourly=temperature_2m,relative_humidity_2m&past_days=1&forecast_days=2&timezone=Asia%2FKuala_Lumpur` into `data/today.json`. Run it once now. The console shows it as *"today's Penang outlook (weather forecast)"* with its fetch time.

### Step 3 — "Why VPD, not humidity" (same data, no ML) → `scripts/why_vpd.py`
1. Using a replica of the rig state machine with hysteresis, compute the share of hours in each state.
2. Compare the rule "mist when RH < 60%" with "mist when VPD > 1.2":
   - How many dry-stress hours does the RH rule miss?
   - What does a typical missed hour look like (median T, RH and VPD)?
3. Compute the share of dry hours by hour of day.
4. Draw the figures:
   - `W1_rig_states_penang.png`
   - `W2_rh_rule_misses.png`
   - `W3_dry_by_hour.png`
5. Save `results/why_vpd.json`.

### Step 4 — SEMAI Wilt Watch → `scripts/train_wilt.py`
1. **Download the photos.**
   - URL: `https://www.kaggle.com/api/v1/datasets/download/russellchan/healthy-and-wilted-houseplant-images` (no login needed).
   - Unzip to `data/wilt/`. You should get 904 images: 452 healthy and 452 wilted.
2. **Extract features.** Use frozen ImageNet **MobileNetV3-Large** features (torchvision `IMAGENET1K_V2`, classifier set to Identity, 960-d). If torch isn't available, use the no-torch fallback in §3.
3. **Group near-duplicates.**
   - L2-normalise the features.
   - Pairs with cosine similarity > 0.97 count as the same image group. Merge them with union-find.
   - Log the number of pairs, the number of images involved, and any pairs whose two labels differ.
4. **Train and test.**
   - Model: `StandardScaler` → `LogisticRegression(C=0.1, max_iter=3000)`.
   - Evaluate with 5-fold `StratifiedGroupKFold(shuffle=True, random_state=0)` as the headline, and also report plain 5-fold CV to show why grouping matters. The baseline is 50%.
5. **Train the final model** on all 904 images and save it as `models/wilt.joblib` (the scaler and LR head; the MobileNet backbone comes from torchvision weights, which must be downloaded now and cached).
6. **Inference in `semai/wilt.py`:**
   - `predict(frame_bgr, roi=0.7)` returns `{label: HEALTHY|WILTED|UNSURE, p_wilted}`.
   - The result is UNSURE when `0.35 < p < 0.65`.
   - It must take under 300 ms per frame on CPU. Time it and record the result.
7. **Rig-photo check.** `scripts/eval_rig_frames.py` evaluates `data/rig_frames/{healthy,wilted}/` when the folders hold photos. When they're empty, it writes `{"status": "pending — no rig photos captured yet"}` to `results/wilt_rig_eval.json` and exits cleanly.
8. **Save outputs.**
   - Figures: `R1_wilt_cv_grouped_vs_naive.png` and `R2_wilt_confusion.png`. **No dataset images.**
   - Metrics: `results/wilt_metrics.json`.

### Step 5 — Mock rig → `scripts/mock_rig.py`
- It's a Flask app on `127.0.0.1:8081` that mimics §4 exactly.
- `/api/data` returns a smooth synthetic day curve cycling every 5 min. `status` values are prefixed `MOCK_`, and roughly 1 reading in 20 has a `null` field.
- `/stream` serves MJPEG from `--images <folder>` if given, otherwise from webcam 0, otherwise from generated MOCK frames.

### Step 6 — SEMAI Console → `semai/app.py` (`python -m semai.app [--rig URL] [--roi 0.7]`)
1. **`rig_client.py`**
   - Poll `/api/data` every 2 s with a 2 s timeout. Treat `null` as NaN, and ignore unknown keys.
   - Read `/stream` on **one** background thread with `requests(stream=True)`, parsing JPEGs by their `FFD8…FFD9` markers. Keep only the latest frame.
   - Reconnect automatically with backoff from 1 s up to 10 s.
   - Expose `connected`, `last_data_ts` and `last_frame_ts`.
2. **`logger.py`** appends every reading, stamped with laptop time, to `logs/rig_YYYYMMDD.csv`. That's future training data for free.
3. **Page** at `127.0.0.1:8000`: a single offline page readable on a 1366×768 laptop.
   - Header: **SEMAI**, a connection dot, and a MOCK banner whenever the rig URL isn't `192.168.4.1` or any status starts with `MOCK`.
   - **Live panel:**
     - leaf VPD (large), air T, RH, leaf T, light, and water (labelled "uncalibrated")
     - status as a coloured chip, plus "updated N s ago"
     - null values show as "—"
   - **Camera panel:**
     - the latest frame with an ROI box, the verdict and `p_wilted`, re-run every ~2 s
     - buttons **Capture healthy** and **Capture wilted**, which save the raw frame to `data/rig_frames/<label>/`
   - **Forecast panel:**
     - date picker (2025 only) plus play/pause, animating that day hour by hour: actual VPD, the 1.2 line, the model's probability and whether misting actually followed
     - a small "today's Penang outlook" block from `data/today.json`, showing its fetch time
     - charts drawn with inline SVG or canvas in vanilla JS, with no libraries
4. **Endpoints:**
   - `GET /api/state`
   - `GET /frame.jpg` (with overlay)
   - `GET /api/replay?date=`
   - `GET /api/today`
   - `POST /api/capture?label=`
5. The page polls `/api/state` and refreshes `/frame.jpg` once a second.

### Step 7 — Soak test, then the results pack
1. **Soak test.** `scripts/soak_test.py` starts the mock rig and the console, runs for **5 minutes**, and kills and restarts the mock at the 2-minute mark. It checks:
   - the console recovers within 10 s
   - `/api/state` never errors
   - nulls show as "—"
   - only one `/stream` connection is open
   - the MOCK banner flag is true

   Write the results to `results/soak_test.txt`.
2. **Results pack.** Write these in plain English for Form 4 students. Every number must be **read from `results/*.json`**, never retyped:
   - **`results/RESULTS.md`:** the Penang "why VPD" findings; SEMAI Forecast against every baseline; Wilt Watch, grouped vs naive, plus the rig-photo line (pending); what we did **not** do, and why; data sources and licences.
   - **`results/EXPLAINER.md`** (~2 pages): how each part works, what a baseline is, why the splits are fair, and what F1 and recall mean, worked through with our own numbers. The reader has 90 minutes to learn it.
   - **`results/JUDGE_QA.md`:** 15 likely questions with short, honest answers. It must include:
     - Why VPD and not humidity? (the RH-rule miss figure)
     - Why not just use a weather forecast? (the model runs offline from local recent readings, and the next step is retraining on the greenhouse's own logs, which the console already collects)
     - What data did you train on?
     - How do you know it isn't overfitting?
     - Does the ML run on the Pico? (no: control and safety run on the Pico, and learning runs on the laptop hub)
     - What if a sensor fails?
     - Why could camera accuracy be lower on your rig?
     - Did you use AI? (yes, as a coding assistant; the data, experiments and results were run and checked by the team)
   - **`results/SLIDES.md`:** slide-by-slide content for the pitch pair, with one figure and one spoken sentence per slide, in this order: problem → Penang insight → what we built → SEMAI Forecast → Wilt Watch → results → limits → future.
   - **`results/DEMO_CHECKLIST.md`:**
     - Night before: run `fetch_today.py` while online, and charge the laptop.
     - At the booth: power the rig, join `VPD-Greenhouse-S3`, run `python -m semai.app`, and never open `192.168.4.1` in a browser.
     - Fallback: run the mock with a webcam, with the MOCK banner showing.
3. **Notebook.** `notebooks/semai_results.ipynb` re-runs the scripts and displays every figure. Execute it with nbconvert.

### Step 8 — Finish
1. Commit everything that isn't gitignored.
2. Copy `results/` to `C:\GDrive\ObsidianSecondMind\10-Work\Projects\PSC-ML-Hardware\SEMAI-ML\results\` (create it if missing; overwriting is fine).
3. Write `STATUS.md`:
   - each step with ✅ done, ⚠️ done with fallback, or ❌ not done, with the reason
   - the headline numbers, read from the JSON files
   - anything needing a human (just the rig-photo session in §10)
   - the exact commands to run the demo
4. Print `STATUS.md`.

---

## 7. Self-checks (run them, record the results in STATUS.md, never stop on a miss)

- The forecast test set is 2025 only, and no 2025 row is in training. Assert both in code.
- The wilt CV folds never split a duplicate group. Assert it.
- Every figure and table that shows a model number also shows its baseline.
- `grep` the page and templates for `http://` and `https://`. The only allowed hits are the rig URL and the mock URL.
- No Kaggle image is saved in `results/`.

---

## 8. Expected numbers (from the 25 Sep test runs, used to sanity-check the build)

**Penang data, 26,304 hours:**
- VPD > 1.2: 13.7% of hours; VPD < 0.6: 59.9%; 0.8–1.1: 11.6%.
- The RH < 60% rule misses 2,428 of 3,604 dry-stress hours, which is **67%**. The typical missed hour is 30.1 °C, 66% RH, 1.41 kPa.
- The dry share peaks at about 50% at 14:00.

**SEMAI Forecast, tested on 2025:** 8,759 hours, of which 24.3% are positive.

| | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| Persistence | ~89.7% | ~0.77 | ~0.84 | ~0.72 | — |
| Model | **~93.8%** | **~0.865** | ~0.92 | ~0.82 | ~0.981 |

A difference of ±1 point is normal because ERA5 data gets revised.

**Wilt Watch:**
- 71 near-duplicate pairs covering 124 images, none with mismatched labels.
- Naive CV ~85.6%, **grouped CV ~84.5%**, baseline 50%.
- Feature extraction takes about 30 s on 2 CPU cores.

---

## 9. Out of scope

- Anything that runs on the Pico or ESP32.
- Plant-disease datasets.
- The team's old synthetic camera model and its ML firmware build.
- Cloud services, phone apps, accounts.
- Fine-tuning experiments.
- Multi-enclosure trials.

---

## 10. After today (humans only, not part of this run)

- **One 30-minute rig session, before Mon 5 Oct:**
  - Put the production firmware on the boards and connect the laptop to `VPD-Greenhouse-S3`.
  - Run `python -m semai.app`, with a plain background and white light.
  - Capture about 30 frames of a healthy plant and about 30 of a cutting that has wilted for 3 hours.
  - Run `python scripts/eval_rig_frames.py` and re-run step 7's results pack.
- **Freeze:** Mon 5 Oct. **Judging:** 17 Oct.
