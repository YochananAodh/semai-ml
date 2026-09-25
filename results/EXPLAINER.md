# SEMAI Explainer — how it works, in plain words

*For the presenting team. Every number below was read from a file in `results/` (named in brackets). If you can say each "one breath" summary at the end without notes, you are ready.*

## 1. The big picture

SEMAI is one product with three parts. The rig is finished and frozen: a Raspberry Pi Pico H runs the misting control loop, and an ESP32-S3-CAM serves live readings and a camera stream over its own Wi-Fi. Nothing in this repo runs on those boards. **Control and safety stay on the Pico. Learning and the screens run on the laptop.**

| Part | Question it answers |
|---|---|
| SEMAI Forecast | Will the air be dry enough to need misting in the next 1 or 2 hours? |
| SEMAI Wilt Watch | Is the plant under the camera healthy or wilted? |
| SEMAI Console | One offline web page showing the live rig, the camera verdict and the forecast replay |

## 2. What VPD is, and why we use it instead of humidity

**VPD** (vapour-pressure deficit, in kPa) is how "thirsty" the air is. It depends on temperature **and** humidity together: warm air can hold more water, so 66 % humidity in 30 °C air is much thirstier than the same 66 % humidity in cooler air. The rig mists when leaf VPD goes above 1.2 kPa, is happy between 0.8 and 1.1 kPa, and runs a fan purge below 0.6 kPa. The gaps between those bands are a **hysteresis rule**: the rig only changes state when VPD crosses an outer threshold, so it does not flip-flop (`why_vpd.json`, `thresholds`).

On 26,304 hours of real outdoor Penang air (2023 to 2025, `why_vpd.json`) the air was dry (VPD above 1.2 kPa) for 13.7 % of hours, saturated (below 0.6 kPa) for 59.9 %, and in the optimal band for only 11.6 %. So a Penang greenhouse is mostly too wet, but there are still 3,604 dry-stress hours to catch, and with the hysteresis rule the dry share rose from 7.9 % of hours in 2023 to 23.2 % in 2025 (`W1_rig_states_penang.png`).

**Why not a humidity rule?** "Mist when humidity is below 60 %" catches only 1,176 of the 3,604 dry-stress hours and **misses 2,428, which is 67.4 %**. The typical missed hour is 30.1 °C, 66.0 % humidity and 1.414 kPa: warm, moderately humid, still thirsty (`W2_rh_rule_misses.png`). Loosening the rule does not fix it: "below 70 %" still misses 391 hours and adds 87 false alarms; "below 75 %" misses none but raises 1,542 false alarms (`rh_threshold_table`). VPD gets both right because it uses temperature and humidity together. Dry hours cluster in the afternoon, peaking at 50.2 % of hours at 14:00 (`W3_dry_by_hour.png`), which is why hour of day is a useful forecast input.

## 3. SEMAI Forecast — how it works

Every hour the model looks at 17 inputs measured now: hour of day and day of year (as sine and cosine so 23:00 sits next to 00:00), temperature, humidity and VPD now and 1, 2 and 3 hours ago, and the VPD change over the last hour. It answers one yes/no question: "will VPD go above 1.2 kPa in the next 1 or 2 hours?"

The model is a gradient-boosted tree classifier (scikit-learn `HistGradientBoostingClassifier`): many small decision trees built one after another, each fixing the mistakes of the last. It was allowed 300 rounds but stopped early after 110 (`forecast_metrics.json`, `model_iterations_used`), and says "yes" at probability 0.5 or more.

The data is hourly outdoor temperature and humidity for Penang from the Open-Meteo Historical Weather API, 2023 to 2025, CC BY 4.0 (`forecast_metrics.json`, `data`). It is **ERA5 reanalysis (a weather archive built from real measurements plus a physics model)**: a weather-model reconstruction of past hours, not a thermometer in Penang, and outdoor air, not greenhouse air. The greenhouse's own logs, which the console records, are the next training data.

**Results on 2025 (8,758 hours the model never saw; 24.3 % of them needed misting)** (`forecast_metrics.json`, `metrics`; chart `F1_model_vs_baselines.png`):

| Method | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| Always "no" | 75.7 % | 0.000 | 0.000 | 0.000 | not defined |
| Persistence (say "yes" if dry now) | 89.7 % | 0.772 | 0.839 | 0.715 | 0.836 |
| Hour-of-day climatology (say "yes" at the hours that were usually dry in 2023–2024) | 84.1 % | 0.702 | 0.644 | 0.773 | 0.907 |
| **SEMAI Forecast (model)** | **93.8 %** | **0.865** | **0.919** | **0.817** | **0.981** |

`F4_replay_3days.png` shows three real 2025 days: the driest (max 3.72 kPa, 2025-02-07), the wettest (max 0.44 kPa, 2025-11-24) and a mixed day (max 1.65 kPa, 2025-02-22). On the dry and mixed days the probability line rises above 0.5 before VPD crosses 1.2, which is exactly the early warning we want.

## 4. What a baseline is, and why we always show one

A **baseline** is the dumbest sensible method; it tells you what score you get for free. A model is only worth having if it clearly beats the baseline on the same test data. Always "no" scores 75.7 % accuracy only because 75.7 % of 2025 hours did not need misting; it never mists, so its recall is 0.000. Accuracy alone can fool you. Persistence ("if it is dry now, it stays dry") is the real competitor at 89.7 %; the model's 93.8 % is 4.1 points better, and that gap (0.093 in F1) is what the machine learning earned. For Wilt Watch the photo set is balanced (452 healthy, 452 wilted), so always guessing one class scores 50 %; our grouped test scores 83.5 %. Judges will ask "compared to what?". Always answer with the baseline number.

## 5. Why the tests are fair

**Forecast: a time split.** Trained on 2023 and 2024 (17,539 hours), tested on 2025 only (8,758 hours). Shuffling the years together would let the model memorise the weather either side of each test hour. There is a trap at the year boundary: each label looks 2 hours ahead, so the training hours at 22:00 and 23:00 on 31 Dec 2024 would have labels made from 2025 test data. Those 2 training rows were dropped (`n_train_dropped_for_leakage`) and the code asserts that no training label touches 2025. The test set is 8,758 rather than the spec's rough 8,759 because the last 2 hours of 2025 have no "next 2 hours" and get no label (`DECISIONS.md`).

**Wilt Watch: grouped cross-validation.** With only 904 photos we use **5-fold cross-validation**: split into 5 parts, train on 4, test on the 5th, repeat so every photo is tested once and never in its own training set. The trap is **near-duplicate photos**: 71 pairs of feature vectors were more than 0.97 similar (cosine), covering 124 images in 59 groups, the largest of 3, with no mismatched labels (`wilt_metrics.json`, `duplicates`). We keep each group inside one fold (`StratifiedGroupKFold`); the naive split let 92 groups straddle train and test.

| Cross-validation | Mean accuracy | Spread (std) | F1 | Duplicate groups split |
|---|---|---|---|---|
| Naive 5-fold (leaky) | 85.6 % | 2.0 points | 0.855 | 92 |
| **Grouped 5-fold (headline)** | **83.5 %** | 1.1 points | **0.836** | **0** |
| Baseline (always one class) | 50 % | — | — | — |

The naive number is 2.1 points higher **only because it cheats**: the model has seen a copy of some test photos. That gap is why we report the grouped number (`R1_wilt_cv_grouped_vs_naive.png`; grouped folds 83.4 %, 82.3 %, 82.3 %, 84.5 %, 85.0 %). Honest note: 83.5 % is 1 point under the spec's rough expectation of about 84.5 %, most likely a different fold assignment with this scikit-learn version; nothing was tuned to close the gap (`DECISIONS.md`).

## 6. What precision, recall and F1 mean, with our own numbers

A **confusion matrix** counts four things. TP: we said "mist" and it was needed. FP: we said "mist" but it was not. FN: we said "no" but misting was needed. TN: we said "no" and it was not needed.

**Forecast model on 2025** (`metrics.model.confusion`; `F2_confusion_2025.png`): TP 1,739, FP 153, FN 390, TN 6,476.

- **Precision** ("when we say mist, how often are we right?") = TP / (TP + FP) = 1,739 / 1,892 = **0.919**.
- **Recall** ("of the hours that needed misting, how many did we catch?") = TP / (TP + FN) = 1,739 / 2,129 = **0.817**.
- **F1** (harmonic mean, which punishes you if either is low) = 2 × 0.919 × 0.817 / (0.919 + 0.817) = 1.502 / 1.736 = **0.865**.
- **Accuracy** = (TP + TN) / all = 8,215 / 8,758 = **93.8 %**.

All match the JSON to 3 decimals. In plant terms: we give early warning for about 82 % of the hours that will need misting, and fewer than 1 in 10 warnings are false alarms. Persistence on the same hours (`metrics.persistence.confusion`) has TP 1,523, FP 293, FN 606, TN 6,336: it misses 606 dry hours where the model misses 390.

**Wilt Watch, grouped 5-fold, pooled over 904 photos** (`cv_grouped.confusion`; `R2_wilt_confusion.png`; "positive" = wilted): TP 381, FP 78, FN 71, TN 374.
Precision = 381 / 459 = **0.830**. Recall = 381 / 452 = **0.843**. F1 = 2 × 0.830 × 0.843 / (0.830 + 0.843) = 1.399 / 1.673 = **0.836**. Accuracy = 755 / 904 = **83.5 %**. Again all match the JSON. The camera spots about 84 % of wilted plants; it wrongly calls 78 healthy plants wilted and misses 71 wilted ones.

**AUC** (0.5 = guessing, 1.0 = perfect) measures how well the model *ranks* dry above wet at every threshold, not just 0.5: forecast model 0.981 against persistence 0.836; Wilt Watch grouped 0.912.

## 7. Permutation importance — which inputs matter

**Permutation importance** asks: if we scramble one input column on the 2025 test set (so it becomes noise) and score again, how much does F1 drop? A big drop means the model relies on that input (`permutation_importance`; `F3_feature_importance.png`):

| Input | F1 drop when shuffled |
|---|---|
| VPD right now (`vpd`) | 0.464 |
| Hour of day (`hour_cos`) | 0.102 |
| VPD change over the last hour (`dvpd`) | 0.043 |

So the model mostly needs the current VPD, then the time of day, then whether VPD is rising or falling. The lagged values add almost nothing. That matches common sense and the afternoon peak in `W3`: the model is not doing anything magical.

## 8. SEMAI Wilt Watch — how it works

1. **Look at the middle of the frame.** The frame is cropped to a central region of interest covering 0.7 of the width and height (`model.roi_default`), so the plant, not the background, is judged.
2. **Turn the photo into numbers.** A frozen ImageNet **MobileNetV3-Large** network converts the crop into 960 numbers describing shapes, textures and colours. "Frozen" means we never changed its weights; we only borrow its eyes. Extracting features for all 904 photos took 13.6 s on the laptop CPU.
3. **A small classifier decides.** The 960 numbers are scaled (`StandardScaler`) and fed to a logistic regression (C = 0.1), which outputs a probability that the plant is wilted.
4. **Three verdicts.** WILTED above 0.65, HEALTHY below 0.35, **UNSURE** in between (`model.unsure_band`). Saying "unsure" is more honest than guessing.
5. **Fast.** One prediction takes about 7 ms (mean 7.1 ms, median 7.0 ms, worst 9.2 ms of 20 frames), far under the 300 ms target (`predict_timing_ms`).

**Training data.** The Kaggle dataset "Healthy and Wilted Houseplant Images" by russellchan: 904 photos, 452 healthy and 452 wilted, cite-only; no photo is copied into results, slides or pages. Its licence was not verified; say "see the Kaggle dataset page".

**Why rig accuracy could be lower.** Different plant species, one fixed camera angle, greenhouse lighting, mist on the lens. In testing a dark or empty frame was classified WILTED with high confidence: the model has only ever seen plants, so the camera must actually see the plant. **Rig-photo check: pending.** `wilt_rig_eval.json` reports 0 healthy and 0 wilted rig frames. The human step is one 30-minute rig session before Mon 5 Oct 2026: capture about 30 healthy and 30 wilted frames with the console buttons, run `scripts/eval_rig_frames.py`, re-run the pack. Until then the honest answer is "we have not yet measured accuracy on our own rig".

## 9. SEMAI Console — how it works

The console is one offline Flask page on the laptop that never loads anything from the internet. It:

- polls the rig's `/api/data` every 2 s, keeps exactly **one** `/stream` camera connection (the ESP32 has only 3 HTTP workers, so never open the rig's address in a browser at the same time), and logs every reading to a daily CSV as free future training data;
- shows leaf VPD large, the other readings with water "uncalibrated" and a missing value as "—" never 0, the camera frame with its Wilt Watch verdict and **Capture healthy / wilted** buttons, plus a replay of any 2025 day labelled "Replay of 2025 (model never saw this year) · outdoor Penang air" and "today's Penang outlook (weather forecast)" fetched the night before;
- shows a red **MOCK DATA** banner whenever it is not talking to the real rig.

A soak test (`results/soak_test.txt`) ran the console against the mock rig for 300 s and killed the mock at 120 s: the console recovered 1.0 s after the restart (limit 10 s), 297 `/api/state` polls had 0 errors, at most 1 `/stream` connection was open, 26 null readings stayed null rather than becoming 0, and every check passed (OVERALL: PASS).

## 10. What we did not do, on purpose

Nothing on the Pico or ESP32 was changed. No plant-disease datasets, no old synthetic camera model, no cloud, phone app or accounts, no fine-tuning of the image backbone, no multi-enclosure trials, no synthetic training data, and no tuning on the test set. Accuracy on training data is never reported. **AI use:** Claude Code (Anthropic) was used as a coding assistant in one unattended run on 25 Sep 2026; the data, experiments and results ran on the team's laptop and are checked by the team. Every number is in `results/*.json` (Python 3.12.13, CPU only, `ENV.md`).

## 11. How to say it in one breath

**VPD, not humidity.** "VPD combines temperature and humidity into one number for how thirsty the air is. On three years of Penang air a plain 'humidity below 60 %' rule missed 67.4 % of the dry-stress hours; VPD catches them all."

**SEMAI Forecast.** "A gradient-boosted tree model trained on Penang 2023 and 2024 predicts whether misting will be needed in the next 2 hours. On 2025, which it never saw, it scores 93.8 % accuracy and F1 0.865, against 89.7 % and 0.772 for the 'stays as it is now' baseline."

**SEMAI Wilt Watch.** "A frozen MobileNet turns the camera frame into 960 numbers and a logistic regression says healthy, wilted or unsure. On 904 real photos with near-duplicates kept in one fold it scores 83.5 % against a 50 % baseline, in about 7 ms. We have not yet tested it on our own rig; that is the next step."

**SEMAI Console.** "One offline laptop page: live rig readings, the camera with its wilt verdict, and a replay of the 2025 forecast, all logged for future training. It keeps one camera connection, never hides a bad sensor as zero, and shows a red MOCK DATA banner whenever it is not on the real rig."

**Why trust it.** "Every model number sits next to a baseline, every test is on data the model never saw, and every number is in a JSON file you can open."
