# SEMAI results

*Team Brand New Developers. Every number on this page was read from a file in `results/` (`forecast_metrics.json`, `why_vpd.json`, `wilt_metrics.json`, `wilt_rig_eval.json`, `soak_test.txt`). Nothing was typed from memory. Re-run the scripts and the numbers come back.*

**How to read this page.** VPD (vapour-pressure deficit, in kPa) is how "thirsty" the air is. It depends on temperature and humidity together. The rig calls the air DRY when VPD is above 1.2 kPa and mists the leaves. Everything below compares our models with a simple baseline. A baseline is the dumbest guess that costs nothing. A model only counts if it beats its baseline.

---

## 1. Why VPD, not humidity (Penang, 26,304 hours)

We took three years of real Penang outdoor air: one reading every hour from 1 Jan 2023 to 31 Dec 2025. That is 26,304 hours. We worked out the VPD of every hour and asked three questions.

### 1a. How often is the air dry, optimal or saturated?

First, without any memory (each hour judged on its own):

| Band (air VPD) | Hours | Share of hours |
|---|---|---|
| DRY, above 1.2 kPa | 3,604 | 13.7 % |
| gap, 1.1 to 1.2 kPa | 726 | 2.8 % |
| OPTIMAL, 0.8 to 1.1 kPa | 3,064 | 11.6 % |
| gap, 0.6 to 0.8 kPa | 3,142 | 11.9 % |
| SATURATED, below 0.6 kPa | 15,768 | 59.9 % |

Second, the way the real rig does it. The rig has hysteresis, which means the gaps keep the previous state, so it does not flicker on and off. We ran a copy of the rig's rules over all three years:

| Rig state | All 3 years | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| DRY (misting) | 4,069 h, 15.5 % | 7.9 % | 15.3 % | 23.2 % |
| OPTIMAL | 4,809 h, 18.3 % | 19.5 % | 17.1 % | 18.2 % |
| SATURATED (fan purge) | 17,426 h, 66.2 % | 72.6 % | 67.6 % | 58.6 % |

Two things to say to a judge. Penang air is saturated most of the time, so the fan matters as much as the mist. And the dry share grew from 7.9 % in 2023 to 23.2 % in 2025, so the weather is not standing still. See `results/W1_rig_states_penang.png`: grouped bars, one group for all three years and one per year, with the percentage printed on every bar.

### 1b. What does a plain humidity rule miss?

Many cheap systems mist when relative humidity (RH) drops below 60 %. We compared that rule with the rig's VPD rule on the same 26,304 hours.

- There were 3,604 dry-stress hours (VPD above 1.2 kPa).
- The RH < 60 % rule fired on 1,176 of them and missed 2,428.
- That is 67.4 % of the dry-stress hours missed, with 0 false alarms.

A typical missed hour is warm and only moderately humid: median 30.1 °C, 66.0 % RH, VPD 1.41 kPa. The middle half of missed hours sits between 29.5 and 30.7 °C, 64.0 and 69.0 % RH, and 1.30 and 1.58 kPa. The plant is thirsty, but a humidity-only rule sees "66 %" and does nothing.

Moving the humidity threshold does not fix it. Either you keep missing dry hours, or you start misting when the plant is fine:

| Rule | Hours the rule fires | Dry-stress hours missed | Missed share | False alarms |
|---|---|---|---|---|
| RH < 55 % | 618 | 2,986 | 82.9 % | 0 |
| RH < 60 % | 1,176 | 2,428 | 67.4 % | 0 |
| RH < 65 % | 1,979 | 1,626 | 45.1 % | 1 |
| RH < 70 % | 3,300 | 391 | 10.8 % | 87 |
| RH < 75 % | 5,146 | 0 | 0.0 % | 1,542 |

See `results/W2_rh_rule_misses.png`. It is a scatter of temperature against humidity for a random sample of 4,000 of the 26,304 hours. The black curve is VPD = 1.2 kPa, the red dashed line is RH = 60 %. The orange dots in the shaded wedge (340 in the sample) are dry-stress hours the humidity rule misses; the blue dots below the line (193 in the sample) are the ones it catches.

### 1c. When in the day is it dry?

We split the 26,304 hours into 24 slots by hour of day (1,096 hours in each slot) and counted how often each slot was dry.

- The peak is 14:00, when 50.2 % of hours are dry.
- 13:00 (45.8 %) and 15:00 (46.8 %) are almost as bad.
- From 04:00 to 06:00 the dry share is 0.0 %.
- The mean VPD by hour peaks at 1.30 kPa at 14:00 and bottoms out at 0.20 kPa at 07:00.

See `results/W3_dry_by_hour.png`: the top panel is the dry share per hour with the 14:00 bar highlighted in orange; the bottom panel is the mean VPD per hour with the 1.2 kPa line, which the average only crosses in the early afternoon.

---

## 2. SEMAI Forecast against every baseline

**The question the model answers:** "Will the air be DRY (VPD above 1.2 kPa) in the next 1 or 2 hours?" It uses 17 inputs measured now: the hour of day and the day of year (each as a sine and cosine pair), temperature, humidity and VPD now and 1, 2 and 3 hours ago, and the change in VPD over the last hour. The model is a HistGradientBoostingClassifier from scikit-learn (a boosted decision-tree model). It stopped early after 110 of its 300 allowed rounds.

**The fair test.** Trained on 2023 and 2024: 17,539 hours, of which 12.4 % were "yes". Tested on 2025 only: 8,758 hours, of which 24.3 % were "yes". The model never saw a single 2025 hour. There is also a leakage guard: the last 2 training rows of 2024 were dropped, because their "next 2 hours" reached into 2025. Training stops at 21:00 on 31 Dec 2024.

**The baselines, all scored on the same 8,758 hours of 2025:**

- Always "no": never mist.
- Persistence: say "yes" if it is dry right now.
- Hour-of-day climatology: say "yes" at the hours that were usually dry in 2023–2024. Its threshold (0.250) was chosen on training data only; it fires from 10:00 to 16:00.

| Method | Accuracy | F1 | Precision | Recall | AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| Always "no" | 75.7 % | 0.000 | 0.000 | 0.000 | not defined | 6,629 | 0 | 2,129 | 0 |
| Persistence | 89.7 % | 0.772 | 0.839 | 0.715 | 0.836 | 6,336 | 293 | 606 | 1,523 |
| Hour-of-day climatology | 84.1 % | 0.702 | 0.644 | 0.773 | 0.907 | 5,719 | 910 | 484 | 1,645 |
| **SEMAI Forecast (model)** | **93.8 %** | **0.865** | **0.919** | **0.817** | **0.981** | 6,476 | 153 | 390 | 1,739 |

What the columns mean, in one line each. TP is a "yes" that was right; FP is a false alarm; FN is a missed dry spell; TN is a "no" that was right. Accuracy is the share of all hours the method got right. Precision is "when it said yes, how often was it right?". Recall is "of all the real dry spells, how many did it catch?". F1 is one number that balances precision and recall (1.000 is perfect). AUC is how well the method ranks hours from least to most likely dry (0.500 is a coin toss, 1.000 is perfect).

The story in plain words. Always "no" already gets 75.7 % because most hours are not dry, which is why accuracy alone is a poor judge. Persistence is a strong baseline at 89.7 %. The model still beats it on every column: it cuts false alarms from 293 to 153 and missed dry spells from 606 to 390 out of 2,129 real ones.

**What the model pays attention to.** Permutation importance shuffles one input at a time and measures how much F1 drops. The top five on the 2025 test:

| Input | F1 drop (mean) | Spread (std) |
|---|---|---|
| VPD now | 0.464 | 0.010 |
| hour of day (cosine) | 0.102 | 0.003 |
| VPD change over the last hour | 0.043 | 0.004 |
| humidity now | 0.032 | 0.004 |
| hour of day (sine) | 0.021 | 0.002 |

VPD right now matters most, then the time of day and the VPD trend. Every lagged input (1, 2 or 3 hours ago) scores 0.001 or lower, so the lags add almost nothing.

**Three replay days.** The console can replay any 2025 day hour by hour. `results/F4_replay_3days.png` shows three picked automatically by daily maximum VPD: 7 Feb 2025, the driest day (max 3.72 kPa); 24 Nov 2025, the wettest day (max 0.44 kPa); and 22 Feb 2025, a mixed day (max 1.65 kPa, closest to the 2025 median of 1.65 kPa). On the dry day the model's probability climbs above 0.5 as the VPD curve heads for the 1.2 kPa line and drops again in the evening. On the wet day the probability stays flat near 0 all day. On the mixed day it rises for the short dry spell around midday and falls back.

**Figures.** `results/F1_model_vs_baselines.png`: grouped bars of accuracy, F1, precision and recall for all four methods. `results/F2_confusion_2025.png`: the persistence and model confusion matrices side by side, with counts and row percentages. `results/F3_feature_importance.png`: the 17 inputs ranked by F1 drop with error bars. `results/F4_replay_3days.png`: the three replay days.

---

## 3. SEMAI Wilt Watch: grouped versus naive cross-validation

**What it is.** A photo goes through a frozen MobileNetV3-Large network (trained on ImageNet, never changed by us) that turns it into 960 numbers. A small logistic regression (a simple yes/no classifier) reads those numbers and says healthy or wilted. Turning all the training photos into numbers took 13.6 seconds on 4 CPU threads.

**The data.** The Kaggle dataset "Healthy and Wilted Houseplant Images" by russellchan: 904 real photos, 452 healthy and 452 wilted. We only cite it; no photo is copied into our results, slides or pages.

**Near-duplicates.** Internet datasets contain copies of the same photo. We compared every pair of feature vectors and called two photos the same if their cosine similarity (a 0-to-1 score of how alike two photos' number-lists are) was above 0.97. We found 71 such pairs covering 124 images. Chaining the pairs into groups (union-find) gave 59 groups of 2 or more photos (largest group 3), and 0 pairs had mismatched labels. Cross-validation (CV) means splitting the photos into 5 folds and testing on each fold in turn with a model trained on the other 4. If a copy of a photo is in the training folds and another copy is in the test fold, the test is cheating. Grouped CV keeps every duplicate group on one side. Naive CV does not.

| | Grouped 5-fold CV (headline) | Naive 5-fold CV | Baseline |
|---|---|---|---|
| Fold accuracies | 83.4 %, 82.3 %, 82.3 %, 84.5 %, 85.0 % | 85.6 %, 86.2 %, 81.8 %, 87.3 %, 87.2 % | 50.0 % |
| Mean accuracy (± std) | **83.5 % ± 1.1** | 85.6 % ± 2.0 | 50.0 % |
| Precision | 0.830 | 0.863 | |
| Recall | 0.843 | 0.847 | |
| F1 | 0.836 | 0.855 | |
| AUC | 0.912 | 0.935 | |
| Confusion (TN, FP, FN, TP) | 374, 78, 71, 381 | 391, 61, 69, 383 | |
| Duplicate groups split across train and test | 0 | 92 | |

The baseline is 50.0 %: always say one class, because the set is exactly balanced (452 and 452). The headline is the grouped result, 83.5 % against 50.0 %. The naive test lets 92 duplicate groups leak across the split and looks better at 85.6 %. That 2.1-point gap is the cheating bonus, and it is why we report the grouped number. Wilted is the positive class: of 452 wilted photos, 381 were caught and 71 were missed; of 452 healthy photos, 78 were wrongly called wilted.

**Speed at the camera.** The console looks at the central region of the frame (70 % of the width and height) and re-runs the check about every 2 seconds. On an 800x600 frame the whole check took 7.1 ms on average, 7.0 ms median and 9.2 ms at worst over 20 frames (18 ms for the first warm-up call), against a target of 300 ms. The verdict is UNSURE when the wilted probability is between 0.35 and 0.65.

**Figures.** `results/R1_wilt_cv_grouped_vs_naive.png`: green bars (grouped) beside orange bars (naive) for each fold and the mean, with the 50 % baseline as a dashed line. `results/R2_wilt_confusion.png`: the pooled grouped-CV confusion matrix with counts and row percentages.

**Rig-photo check: pending.** `results/wilt_rig_eval.json` says "pending — no rig photos captured yet", with 0 healthy and 0 wilted rig frames and the same 50.0 % baseline. The human step (SPEC section 10) is one 30-minute session at the rig before the freeze on Mon 5 Oct 2026: capture about 30 healthy and about 30 wilted frames with the console's Capture buttons, run `scripts/eval_rig_frames.py`, then re-run the results pack. Expect the rig number to be lower than 83.5 %. The training photos are of many species from many angles; the rig has one plant, one fixed camera angle, greenhouse lighting and mist droplets on the lens. We also saw that a dark or empty frame (the laptop webcam with its lid closed) was called WILTED with high confidence. The model has only ever seen plants, so the camera must actually see the plant.

---

## 4. SEMAI Console and the soak test

The console is one offline web page on the laptop. It polls the rig's `/api/data` every 2 seconds, holds exactly one `/stream` camera connection (the ESP32 has only 3 HTTP workers, so nobody should open `192.168.4.1` in a browser at the same time), shows a red MOCK DATA banner whenever it is not talking to the real rig, labels the water level "uncalibrated", labels the forecast replay "Replay of 2025 (model never saw this year) · outdoor Penang air", shows a missing sensor value as "—" and never as 0, and logs every reading to `logs/rig_YYYYMMDD.csv` as future training data. A mock rig (`scripts/mock_rig.py`) fakes the real one for rehearsals.

**Soak test** (`results/soak_test.txt`, run 2026-09-25 12:34:09). The test ran the mock rig and the console for 300 seconds and killed and restarted the mock at 120 seconds. Its numbers:

- `/api/state` polls: 297, errors: 0; polls with connected=true: 297.
- null float fields seen in `/api/state`: 26; float fields equal to exactly 0 (suspect null->0): 0.
- max stream_connections_open: 1.
- recovered after restart: 1.0 s (limit 10 s); `/stream` back after restart: 6.0 s.

Its checks, quoted:

```
PASS  mock_started
PASS  console_started
PASS  page_has_em_dash_for_null
PASS  page_has_MOCK_DATA_string
PASS  console_recovers_within_10s
PASS  api_state_never_errors
PASS  nulls_kept_as_null_not_zero
PASS  only_one_stream_connection
PASS  mock_banner_flag_always_true

OVERALL: PASS
```

---

## 5. What we did not do, and why

- **Nothing on the Pico or the ESP32 was changed.** The rig is finished and frozen. Control and safety stay on the Pico; learning and the screens run on the laptop.
- **No plant-disease datasets.** Wilt Watch answers one question, wilted or not.
- **The team's old synthetic camera model and its firmware build were not used.** No synthetic training data anywhere.
- **No cloud, phone app or accounts.** The demo must work offline on the rig's own Wi-Fi.
- **No fine-tuning of the image backbone.** With 904 photos, a frozen network plus a small classifier is the honest choice.
- **No multi-enclosure trials.** One rig, one plant.
- **No tuning on the test set.** The spec's expected numbers were only used as a sanity check afterwards.
- **No accuracy on training data is reported anywhere** (SPEC section 2).

**AI use.** Claude Code (Anthropic) was used as a coding assistant to write the scripts and documents in one unattended run on 25 Sep 2026. The data, experiments and results were run on the team's laptop and are checked by the team. Every number is in `results/*.json` and can be re-derived with the scripts.

---

## 6. Data sources and licences

| Data | Source | Size | Licence | Used for |
|---|---|---|---|---|
| Penang hourly temperature and humidity, 1 Jan 2023 to 31 Dec 2025 | Open-Meteo Historical Weather API (ERA5 reanalysis), fetched 2026-09-25 03:49 UTC | 26,304 hours | CC BY 4.0 | Why-VPD findings and SEMAI Forecast |
| "Healthy and Wilted Houseplant Images" by russellchan | Kaggle, `https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images` | 904 photos (452 healthy, 452 wilted) | see the Kaggle dataset page | Wilt Watch training only; cite-only, no image copied |
| MobileNetV3-Large weights (IMAGENET1K_V2) | torchvision | 960 features per photo | BSD 3-Clause (torchvision) | Frozen image backbone |

"ERA5 reanalysis" means a weather archive built by combining real measurements with a physics model of the atmosphere. The Penang data is outdoor air, not greenhouse air. The greenhouse's own logs, which the console records, are the next training data. Software: Python 3.12.13, scikit-learn 1.9.1, torch 2.14.0 (CPU only); the full list is in `results/ENV.md`.

---

## 7. Two honest deviations from the spec's expected numbers

Both are recorded in `DECISIONS.md`.

1. **The 2025 test set has 8,758 hours, not about 8,759.** The last two hours of 2025 have no "next 2 hours" to label, so they are dropped by the label definition itself. Nothing else was removed.
2. **Wilt Watch grouped CV came out at 83.5 %, about 1 point under the spec's rough expectation.** The most likely cause is a different fold assignment with this scikit-learn version (1.9.1); the fold-to-fold spread is 1.1 points, so the gap is within noise. Nothing was tuned to close it. The naive result (85.6 %) matches the spec, which is exactly why the naive number is not the headline.
