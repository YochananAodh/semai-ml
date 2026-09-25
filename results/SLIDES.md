# SEMAI — slide-by-slide script for the pitch pair

**Who this is for:** the two presenters. Each slide below gives you a title, ONE figure, the bullet points that go on the slide, and ONE sentence to say out loud. Practise the sentence until it sounds like you, not like a script.

**Where the numbers come from:** every number on these slides was read from a file in `results/` (`forecast_metrics.json`, `why_vpd.json`, `wilt_metrics.json`, `wilt_rig_eval.json`). Nothing was typed from memory. If a judge doubts a number, the file is the proof.

**Two words you will use a lot:**

- **VPD** (vapour-pressure deficit, in kPa) is how "thirsty" the air is. It depends on temperature AND humidity together. High VPD pulls water out of leaves.
- **Baseline** is the simplest possible guess. We show every model number next to its baseline so the judges can see what the model adds.

Slide order is fixed: problem → Penang insight → what we built → SEMAI Forecast → Wilt Watch → results → limits → future.

---

## Slide 1 — The problem: Penang air swings between soaked and thirsty

**Figure:** `results/W1_rig_states_penang.png` (bars: share of hours in each rig state, for all three years and for 2023, 2024 and 2025 separately)

**On the slide:**

- We replayed our rig's rules on 26,304 hours of real Penang air (2023 to 2025).
- The air was SATURATED (fan needed) 66.2 % of the time and DRY (mist needed) 15.5 % of the time.
- The dry share is climbing: 7.9 % in 2023, 15.3 % in 2024, 23.2 % in 2025.
- A plant cannot wait for a human to notice. The greenhouse has to react on its own.

**Say:** "Penang air is too wet two-thirds of the time and too dry more and more often, so our greenhouse has to manage itself."

**If a judge asks** what the rig states mean or where the rules come from: see RESULTS.md section 1a and EXPLAINER.md (the VPD section): DRY is VPD above 1.2 kPa, OPTIMAL 0.8–1.1 kPa, SATURATED below 0.6 kPa, from the rig firmware.

---

## Slide 2 — The Penang insight: humidity alone misses most dry hours

**Figure:** `results/W2_rh_rule_misses.png` (scatter of temperature against humidity; orange dots are dry-stress hours that a humidity-only rule misses; the black curve is VPD = 1.2 kPa, the red dashed line is RH = 60 %)

**On the slide:**

- A simple rule many people use: "mist when humidity drops below 60 %".
- On the same 26,304 hours, that rule misses 2,428 of the 3,604 dry-stress hours. That is 67.4 %.
- A typical missed hour looks harmless: 30.1 °C and 66.0 % humidity, yet the VPD is 1.41 kPa, above the 1.2 kPa mist line.
- This is why SEMAI controls on VPD, not on humidity.

**Say:** "Two out of three dry-stress hours in Penang look fine on a humidity meter, so we control on VPD instead."

**If a judge asks** "Why VPD and not humidity?": see JUDGE_QA.md (first question). The 67.4 % figure is in `why_vpd.json` under `rh_rule_vs_vpd_rule`.

---

## Slide 3 — What we built: a finished rig plus a laptop brain

**Figure:** no figure: a photo of the rig (Pico H, ESP32-S3-CAM, mister, plant) next to a screenshot of the SEMAI Console page. Never a dataset photo.

**On the slide:**

- The rig is finished and frozen. A Raspberry Pi Pico H runs the leaf-VPD misting loop. An ESP32-S3-CAM serves readings and a camera stream over its own Wi-Fi.
- SEMAI Forecast: a model that answers "will the air be dry within the next two hours?"
- SEMAI Wilt Watch: a camera model that says HEALTHY, WILTED or UNSURE.
- SEMAI Console: one offline web page on the laptop that shows live readings, the camera verdict and the forecast replay, and logs every reading for future training.

**Say:** "Control and safety stay on the Pico; the laptop adds the learning, the camera verdict and the screens, all offline."

**If a judge asks** "Does the ML run on the Pico?" or "Why offline?": see JUDGE_QA.md (the Pico question and the weather-forecast question).

---

## Slide 4 — SEMAI Forecast: two hours ahead, from what the sensors see now

**Figure:** `results/F4_replay_3days.png` (three 2025 days: purple line is the actual air VPD, blue line is the model's probability, pink shading marks hours where misting really was needed within two hours)

**On the slide:**

- Inputs: temperature, humidity and VPD now and over the last 3 hours, plus time of day and time of year. 17 numbers in total.
- Trained on 2023 and 2024 (17,539 hours). Tested on 2025 only (8,758 hours). The model never saw 2025.
- Driest 2025 day (7 Feb, max 3.72 kPa): the probability jumps to nearly 1 about an hour before the VPD crosses the mist line.
- Wettest day (24 Nov, max 0.44 kPa): the probability stays flat at 0. No false alarms.

**Say:** "On the driest day of 2025, the model raised the alarm an hour before the air actually crossed the mist line."

**If a judge asks** "What data did you train on?" or "How do you know it isn't overfitting?": see JUDGE_QA.md. Overfitting means a model has memorised its training data and fails on new data; the 2025 test is our proof it did not.

---

## Slide 5 — SEMAI Wilt Watch: a fair test with duplicates kept together

**Figure:** `results/R1_wilt_cv_grouped_vs_naive.png` (green bars: grouped cross-validation, our headline; orange bars: naive cross-validation; dashed line: the 50 % baseline)

**On the slide:**

- 904 real houseplant photos from Kaggle (452 healthy, 452 wilted), used for training only and never shown on our slides.
- We found 71 near-duplicate photo pairs covering 124 photos, and kept each group on one side of every test fold.
- Grouped test (fair): 83.5 % accuracy. Naive test (duplicates leak): 85.6 %. Baseline: 50 %.
- The naive number looks better because the model is quietly re-seeing photos it trained on.

**Say:** "We report the lower, fairer number, 83.5 %, because the easier test lets the model peek at copies of its own training photos."

**If a judge asks** why grouped is lower than naive, or what cross-validation is: see JUDGE_QA.md (the overfitting question). Cross-validation means testing five times, each time on a different fifth of the photos the model did not train on.

---

## Slide 6 — Results: SEMAI Forecast beats every baseline on 2025

**Figure:** `results/F1_model_vs_baselines.png` (grouped bars: accuracy, F1, precision and recall for the model and three baselines, all on the same 8,758 hours of 2025)

**On the slide (table):**

| Method (tested on 2025, 8,758 hours) | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Always "no" | 75.7 % | 0.000 | 0.000 | 0.000 |
| Persistence ("is it dry right now?") | 89.7 % | 0.772 | 0.839 | 0.715 |
| Hour-of-day climatology (usual dry hours of 2023–2024) | 84.1 % | 0.702 | 0.644 | 0.773 |
| **SEMAI Forecast (model)** | **93.8 %** | **0.865** | **0.919** | **0.817** |

- Recall means "of the hours that really needed misting, how many did we catch?" The model catches 0.817; persistence catches 0.715.
- Precision means "when we said mist, how often were we right?" The model: 0.919; persistence: 0.839.
- The model's AUC (how well it ranks dry hours above wet ones, 1.0 is perfect) is 0.981.

**Say:** "Against the best simple guess, the model catches ten percentage points more of the dry hours and raises fewer false alarms."

**If a judge asks** what F1 or recall mean, or why accuracy alone is misleading (always "no" already scores 75.7 %): see JUDGE_QA.md and EXPLAINER.md. The confusion matrix is in `results/F2_confusion_2025.png`.

---

## Slide 7 — Results: Wilt Watch, and what it gets wrong

**Figure:** `results/R2_wilt_confusion.png` (a two-by-two grid: rows are the true label, columns are the model's answer, from the grouped test)

**On the slide:**

- Grouped test on all 904 photos: 83.5 % accuracy against a 50 % baseline. F1 0.836, precision 0.830, recall 0.843.
- 381 of the 452 wilted plants were caught; 71 were called healthy.
- 374 of the 452 healthy plants were correct; 78 were called wilted.
- Speed on the laptop CPU: about 7 ms per frame (worst case 9 ms), far under the 300 ms target.

**Say:** "Wilt Watch spots five out of six wilted plants in a fair test, and it runs in a few milliseconds on a plain laptop."

**If a judge asks** why the number is not higher, or about the plants it gets wrong: see JUDGE_QA.md (the camera-accuracy question). The UNSURE band (wilted probability between 0.35 and 0.65) is there so a borderline plant is flagged for a human, not guessed.

---

## Slide 8 — Limits: what we have not shown yet

**Figure:** no figure: a screenshot of the console with the red "MOCK DATA" banner showing, and the line "pending — no rig photos captured yet" from `results/wilt_rig_eval.json`.

**On the slide:**

- Wilt Watch has been tested on Kaggle photos only. The check on our own rig's camera is still pending: 0 healthy and 0 wilted rig frames so far.
- Rig accuracy could be lower: a different plant species, one fixed camera angle, greenhouse light and mist droplets on the lens.
- The forecast was trained on outdoor Penang air, not greenhouse air. The console is already logging the greenhouse's own readings to fix that.
- Nothing on the Pico or ESP32 was changed. No cloud, no phone app, no synthetic data, and no tuning on the test set.

**Say:** "Our camera number comes from internet photos, so the next test is on our own plant under our own camera, and we will report whatever it says."

**If a judge asks** "Why could camera accuracy be lower on your rig?" or "What if a sensor fails?": see JUDGE_QA.md. Be honest: a dark or empty frame was once called WILTED with high confidence, because the model has only ever seen plants.

---

## Slide 9 — Future: teach SEMAI with its own greenhouse

**Figure:** no figure: a photo of the plant under the rig camera, with the console's "Capture healthy" and "Capture wilted" buttons visible.

**On the slide:**

- One rig session before the freeze: capture about 30 healthy and about 30 wilted frames with the console buttons, run `scripts/eval_rig_frames.py`, and re-run the results pack.
- Retrain SEMAI Forecast on the greenhouse's own logs, which the console records every day.
- Add more of our own plant photos so Wilt Watch learns our species and our lighting.
- Keep every result re-derivable: the scripts and `results/*.json` are the record.

**Say:** "Every reading the console logs today is tomorrow's training data, so SEMAI gets better simply by running in our greenhouse."

**If a judge asks** "Why not just use a weather forecast?" or "Did you use AI?": see JUDGE_QA.md. Short honest answer on AI: Claude Code was used as a coding assistant for the scripts and documents (25 Sep 2026); the data, experiments and results were run on the team's laptop and are checked by the team.

---

## Presenter's number sheet (read these from the JSON, do not memorise them wrongly)

| Number on a slide | Value | File and path |
|---|---|---|
| Penang hours analysed | 26,304 | `why_vpd.json` → `data.n_hours` |
| SATURATED share (with hysteresis), all years | 66.2 % | `why_vpd.json` → `rig_states_hysteresis.overall.SATURATED.share_pct` |
| DRY share, all years / 2023 / 2024 / 2025 | 15.5 % / 7.9 % / 15.3 % / 23.2 % | `why_vpd.json` → `rig_states_hysteresis.*.DRY.share_pct` |
| Dry-stress hours missed by the RH < 60 % rule | 2,428 of 3,604 (67.4 %) | `why_vpd.json` → `rh_rule_vs_vpd_rule` |
| Typical missed hour | 30.1 °C, 66.0 % RH, 1.41 kPa | `why_vpd.json` → `rh_rule_vs_vpd_rule.typical_missed_hour.*.median` |
| Forecast inputs | 17 | `forecast_metrics.json` → `features` (length) |
| Training hours / test hours | 17,539 / 8,758 | `forecast_metrics.json` → `n_train`, `n_test` |
| Driest / wettest 2025 day | 2025-02-07 (3.72 kPa) / 2025-11-24 (0.44 kPa) | `forecast_metrics.json` → `replay_days` |
| Model accuracy / F1 / precision / recall / AUC | 93.8 % / 0.865 / 0.919 / 0.817 / 0.981 | `forecast_metrics.json` → `metrics.model` |
| Persistence accuracy / F1 / precision / recall | 89.7 % / 0.772 / 0.839 / 0.715 | `forecast_metrics.json` → `metrics.persistence` |
| Climatology (usual dry hours) accuracy / F1 / precision / recall | 84.1 % / 0.702 / 0.644 / 0.773 | `forecast_metrics.json` → `metrics.climatology` |
| Always "no" accuracy | 75.7 % | `forecast_metrics.json` → `metrics.always_no.accuracy` |
| Kaggle photos | 904 (452 healthy, 452 wilted) | `wilt_metrics.json` → `dataset` |
| Near-duplicate pairs / photos involved | 71 / 124 | `wilt_metrics.json` → `duplicates` |
| Grouped CV accuracy / F1 / precision / recall | 83.5 % / 0.836 / 0.830 / 0.843 | `wilt_metrics.json` → `cv_grouped` |
| Naive CV accuracy | 85.6 % | `wilt_metrics.json` → `cv_naive.mean_accuracy` |
| Wilt Watch baseline | 50 % | `wilt_metrics.json` → `baseline.accuracy` |
| Grouped confusion (tn / fp / fn / tp) | 374 / 78 / 71 / 381 | `wilt_metrics.json` → `cv_grouped.confusion` |
| Predict time per frame (mean / max / target) | 7 ms / 9 ms / 300 ms | `wilt_metrics.json` → `predict_timing_ms` |
| UNSURE band | 0.35 to 0.65 | `wilt_metrics.json` → `model.unsure_band` |
| Rig-photo check | pending, 0 healthy, 0 wilted | `wilt_rig_eval.json` |

Not measured yet (so do not say a number for it): Wilt Watch accuracy on the rig's own camera, and how the forecast behaves on greenhouse air rather than outdoor air.
