# SEMAI: 15 questions the judges may ask, with honest answers

For the pitch pair. Read each answer out loud once, then say it in your own words.
Every number below was read from a file in `results/` (`forecast_metrics.json`, `why_vpd.json`, `wilt_metrics.json`, `wilt_rig_eval.json`, `ENV.md`). If a judge asks for a number that is not here, say "we did not measure that" rather than guessing.

Three words you will need:

| Word | One-line meaning |
|---|---|
| VPD | Vapour-pressure deficit, in kPa. How "thirsty" the air is. It depends on temperature and humidity together. |
| Baseline | A very simple rule we compare the model against. If the model cannot beat it, the model is not worth having. |
| Test set | Data the model never saw while learning. Only test-set numbers are reported here. |

---

## 1. Why VPD and not humidity?

Humidity alone does not tell you whether a leaf is losing water. The same 66 % humidity is fine at night but stressful at 30 °C in the afternoon, because warm air can hold much more water. We checked this on 26,304 hours of real Penang weather (2023 to 2025). There were 3,604 dry-stress hours (VPD above 1.2 kPa), and the rule "mist when RH < 60 %" catches only 1,176 of them and misses 2,428, which is 67.4 % of all dry-stress hours. A typical missed hour is 30.1 °C, 66 % RH and 1.41 kPa: it does not look dry on a humidity meter, but the plant is under stress.

See `results/W2_rh_rule_misses.png`: the orange dots in the shaded "miss zone" are dry-stress hours that sit above the red RH = 60 % line but below the black VPD = 1.2 kPa curve. Loosening the humidity rule does not fix it either: at "RH < 70 %" the rule still misses 391 hours (10.8 %) and now adds 87 false alarms, and at "RH < 75 %" it misses nothing but raises 1,542 false alarms (misting when the plant does not need it).

| Humidity rule | Dry-stress hours missed | Share missed | False alarms |
|---|---|---|---|
| RH < 55 % | 2,986 | 82.9 % | 0 |
| RH < 60 % | 2,428 | 67.4 % | 0 |
| RH < 65 % | 1,626 | 45.1 % | 1 |
| RH < 70 % | 391 | 10.8 % | 87 |
| RH < 75 % | 0 | 0.0 % | 1,542 |

---

## 2. Why not just use a weather forecast?

A weather forecast needs the internet, and it describes the sky over Penang, not the air inside one greenhouse. SEMAI Forecast needs no internet: it is a file on the laptop that takes the last 3 hours of temperature and humidity readings plus the time of day. In the demo it replays 2025 hours the model never saw; wiring it to the rig's live log is the next step, and the console already writes that log to `logs/rig_YYYYMMDD.csv`. When we scrambled each input in turn to see which one matters (`results/F3_feature_importance.png`), VPD right now mattered most: the F1 score fell by 0.464 out of 0.865 when we scrambled the current VPD, but by only 0.102 for the hour of day and 0.043 for the last-hour trend. We do still show "today's Penang outlook" on the console, but it is fetched the night before and clearly labelled as a weather forecast. The honest limit is that the model was trained on outdoor air; the next step after the live wiring is to retrain it on the greenhouse's own logs.

---

## 3. What data did you train on?

Two public datasets, and nothing synthetic. For SEMAI Forecast: 26,304 hourly readings of Penang temperature and humidity from 1 Jan 2023 to 31 Dec 2025, from the Open-Meteo Historical Weather API (ERA5 reanalysis, licence CC BY 4.0), fetched on 25 Sep 2026. We trained on 2023 and 2024 (17,539 hours) and tested on 2025 (8,758 hours), which the model never saw. For SEMAI Wilt Watch: the Kaggle dataset "Healthy and Wilted Houseplant Images" by russellchan, 904 real photos (452 healthy, 452 wilted); for its licence see the Kaggle dataset page, and no photo is copied into our results or slides. We found 71 near-duplicate photo pairs covering 124 images, and kept the copies together so they could not leak between training and test.

---

## 4. How do you know it isn't overfitting?

Overfitting means the model memorised the training data and would fail on new data. We guarded against it in four ways, and every number we quote is from data the model never saw.

| Guard | What we did | Number from the results files |
|---|---|---|
| Time split | Forecast trained on 2023-2024, tested on 2025 only; 2 training rows at the end of 2024 were dropped because their labels reached into 2025 | 17,539 train hours, 8,758 test hours |
| Early stopping | The gradient-boosting model stopped adding trees when validation stopped improving | 110 iterations used out of a maximum of 300 |
| Baselines | The model must beat simple rules on the same 2025 hours | Model accuracy 93.8 % vs persistence 89.7 %, climatology 84.1 %, always "no" 75.7 % |
| Grouped cross-validation | Wilt Watch near-duplicates never sit on both sides of a fold | 0 duplicate groups split in the grouped test vs 92 in the naive test |

The grouped Wilt Watch accuracy (83.5 %) is lower than the naive one (85.6 %), and we report the lower, fairer number as the headline. The climatology baseline's threshold was chosen on training data, not on the test set. Nothing was tuned to hit a target: our grouped accuracy came out 1 point under the rough figure in our plan, and we left it.

---

## 5. Does the ML run on the Pico?

No. The Raspberry Pi Pico H runs the misting control loop and the safety checks, exactly as it did before this project, and we changed nothing on the Pico or the ESP32-S3-CAM. The learning and the screens run on the laptop, which reads the rig's live readings and camera stream over the rig's own Wi-Fi network. This split is deliberate: if the laptop crashes, the plant is still misted, because the control loop does not depend on the laptop. A Pico could not run the image model anyway; even on the laptop's CPU the wilt check takes about 7 ms per frame.

---

## 6. What if a sensor fails?

The rig reports the failed reading as `null` and sets its status to `ERR_SENSOR`. The console shows a null as "—" and never treats it as 0, because a fake 0 kPa would look like saturated air on the screen and would be written into the log as if it were a real reading. The laptop only reads from the rig; it cannot switch the mister or the fan. The Pico keeps its own safety logic and does not need the laptop to react. We rehearsed this with a mock rig that deliberately puts a null into about 1 reading in 20, so the "—" path is exercised every time we practise. In the recorded soak test (`results/soak_test.txt`) the console saw 26 null sensor values and turned 0 of them into a zero. The forecast is only shown where all its inputs exist; a missing hour gets no probability, never a made-up one.

---

## 7. Why could camera accuracy be lower on your rig?

Our 83.5 % accuracy comes from 904 internet photos of many houseplant species, taken from many angles in many kinds of light. The rig has one fixed camera angle, greenhouse lighting, mist droplets that can land on the lens, and a plant species that may not be in the training set. We also found that a dark or empty frame (for example the laptop webcam with the lid closed during testing) was called WILTED with high confidence: the model has only ever seen plants, so the camera must actually see the plant. That is why the verdict says UNSURE when the wilted probability is between 0.35 and 0.65, and why we have a rig-photo check. That check is still pending: `results/wilt_rig_eval.json` currently holds 0 healthy and 0 wilted rig frames, and we will capture about 30 of each before the freeze on Mon 5 Oct 2026.

---

## 8. Did you use AI?

Yes, openly. Claude Code (Anthropic) was used as a coding assistant to write the scripts and documents in one unattended run on 25 Sep 2026. The data downloads, experiments and results were run on the team's own laptop, and the team checks them: every number is stored in `results/*.json` and can be re-derived by running the scripts again. The AI did not choose the problem, the rig thresholds or the fairness rules; those were set by the team before the run. We think a coding assistant is a tool like a calculator, and the honest thing is to say so.

---

## 9. What does 93.8 % actually mean for a farmer?

It means that, over the 8,758 test hours of 2025, the model's answer to "will misting be needed within the next 2 hours?" was right 93.8 % of the time. The more useful view is the confusion matrix (`results/F2_confusion_2025.png`). Of the 2,129 hours where misting really was needed, the model warned in 1,739 and missed 390; it also raised 153 false alarms. The simple "is it dry right now?" rule (persistence) on the same hours caught 1,523, missed 606 and raised 293 false alarms. So compared with the simple rule, the model gives the farmer 216 more early warnings and 140 fewer false alarms across the year, but it still misses some, and it was measured on outdoor air, not inside a greenhouse.

---

## 10. What is F1, and why not just report accuracy?

Accuracy can flatter a lazy model. Misting is needed in only 24.3 % of the 2025 hours, so a model that always says "no" scores 75.7 % accuracy while being useless (its F1 is 0.000). F1 combines two things: precision (when the model says "mist", how often is it right?) and recall (of the hours that really needed misting, how many did it catch?). Our model's precision is 0.919 and its recall is 0.817, giving an F1 of 0.865; the persistence baseline scores 0.772 and the always-"no" rule 0.000 (`results/F1_model_vs_baselines.png`). We also report AUC, which asks: if you pick one hour that needed misting and one that did not, how often does the model give the first one the higher score? 0.981 means almost always; the simple persistence rule scores 0.836.

---

## 11. Why forecast 2 hours ahead, and not more?

Two hours is long enough to act and short enough to be right. Dry stress in Penang follows the sun: the share of dry hours climbs from 8.2 % at 10:00 to a peak of 50.2 % at 14:00 and falls back to 3.2 % by 20:00 (`results/W3_dry_by_hour.png`). A 2-hour warning lets the system start misting, top up the tank or open shading before the plant is stressed, which the rig's react-when-dry rule cannot do. Because the model only looks at the last 3 hours, and the older lags add almost nothing (the best lag feature moves F1 by 0.001), it would have little to go on for a much longer horizon. Longer horizons are a fair next experiment, but we did not test them, so we cannot quote a number.

---

## 12. How fast is it? Can a laptop keep up?

Yes, easily, and nothing needs a GPU. The wilt check on one 800x600 test frame (the camera's frame size) takes 7 ms on average and 9 ms at worst, against our target of 300 ms, so re-running it every 2 seconds is not a strain. Extracting features for all 904 training photos took 13.6 seconds using 4 CPU threads. Training the forecast model on 17,539 hours took 0.2 seconds. The whole stack is Python 3.12.13 with scikit-learn 1.9.1 and a CPU-only torch 2.14.0 build, listed in `results/ENV.md`.

---

## 13. What happens with no internet at the venue?

Nothing breaks, because the demo is designed to be offline. The laptop joins the rig's own Wi-Fi network (`VPD-Greenhouse-S3`), the console page loads no fonts, scripts or images from the internet, and both models are files on the laptop's disk. The only internet-dependent item is "today's Penang outlook", which we fetch the night before with `scripts/fetch_today.py` and display with its fetch time. If the rig itself fails at the booth, we run the mock rig with a webcam, and the console shows a red "MOCK DATA" banner so nobody is misled.

---

## 14. How would you know if the model is wrong?

We already know where it is wrong on 2025: 390 missed hours and 153 false alarms out of 8,758 (`results/F2_confusion_2025.png`). On the rig, the console logs every reading with the laptop time, so after a day we can line up "the model said mist within 2 hours" against what the VPD actually did and count the same four boxes again. For the camera, the UNSURE band (probability 0.35 to 0.65) is the model saying "do not trust me here", and the capture buttons let us collect labelled rig frames to score it properly. We have also been honest that the grouped photo accuracy was 1 point below our rough expectation, and that the rig-photo check is still pending with 0 frames. A model that is never checked against new data is one you should not trust.

---

## 15. What would you do with more time?

Four things, in order. First, the 30-minute rig session: capture about 30 healthy and about 30 wilted frames and run `scripts/eval_rig_frames.py`, so the camera number is measured on our plant, not on internet photos. Second, retrain SEMAI Forecast on the greenhouse's own logs once a few weeks have been collected, because outdoor air is only a stand-in. Third, watch for drift: in the Penang data the rig's DRY state covered 7.9 % of hours in 2023 but 23.2 % in 2025, so a model trained on old years can go stale. Fourth, the things we deliberately left out: no changes to the Pico or ESP32, no plant-disease datasets, no fine-tuning of the image backbone, no cloud or phone app, and no multi-enclosure trials. The console itself has already passed a 300-second soak test (`results/soak_test.txt`): the mock rig was killed at 120 s, the console recovered 1.0 s after the restart (limit 10 s), all 297 status polls succeeded with 0 errors, and only 1 camera stream connection was ever open.

---

## Numbers at a glance (for the back of your hand)

| Item | Model | Baseline | Source file |
|---|---|---|---|
| Forecast accuracy, 2025 test (8,758 h) | 93.8 % | 89.7 % (persistence) | `forecast_metrics.json` |
| Forecast F1 | 0.865 | 0.772 (persistence) | `forecast_metrics.json` |
| Forecast recall / precision | 0.817 / 0.919 | 0.715 / 0.839 (persistence) | `forecast_metrics.json` |
| Wilt Watch accuracy, grouped 5-fold CV (904 photos) | 83.5 % | 50.0 % (always one class) | `wilt_metrics.json` |
| Wilt Watch accuracy, naive CV (leaky, not the headline) | 85.6 % | 50.0 % | `wilt_metrics.json` |
| Wilt Watch on rig photos | pending, 0 frames | 50.0 % | `wilt_rig_eval.json` |
| RH < 60 % rule misses | 2,428 of 3,604 dry-stress hours (67.4 %) | - | `why_vpd.json` |
| Wilt check speed | 7 ms mean, 9 ms max | target 300 ms | `wilt_metrics.json` |
