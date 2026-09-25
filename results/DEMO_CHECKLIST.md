# SEMAI demo checklist

This is the list to tick off before and during the booth demo. Print it, or keep it open on a phone.

Every number below was read from the files in `results/` (`forecast_metrics.json`, `wilt_metrics.json`, `wilt_rig_eval.json`, `why_vpd.json`) or from `results\soak_test.txt`. The commands are for Windows, typed in a terminal opened in `C:\Dev\semai-ml`. They all use the project's own Python, `.venv\Scripts\python.exe`.

Two words you will see a lot:

- **Rig** = the greenhouse box. A Raspberry Pi Pico H runs the misting loop. An ESP32-S3-CAM (a small Wi-Fi board with a camera) sends the readings and the video.
- **Console** = the SEMAI web page on the laptop. It shows the live readings, the camera with the wilt verdict, and the forecast replay. It works offline.

---

## 1. Night before (online)

Do these at home, with internet, the evening before the demo.

- [ ] **Fetch today's Penang outlook.** This is the only step that needs the internet. The console never goes online by itself; it only reads this file.

  ```
  cd C:\Dev\semai-ml
  .venv\Scripts\python.exe scripts\fetch_today.py
  ```

  The script prints one line ending with the local time it fetched the data. That time should be tonight.

- [ ] **Check `data\today.json` has a fresh fetch time.** Open the file in Notepad. The line `"fetched_at_local"` near the top must show tonight's date and time. If it shows an older date, run the command above again.

- [ ] **Charge the laptop to 100 %.** Bring the charger too. The camera stream and the wilt model keep the CPU busy.

- [ ] **Check the two model files exist.** In File Explorer, or in a terminal:

  ```
  dir models\forecast.joblib
  dir models\wilt.joblib
  ```

  Both files must be listed. Without `forecast.joblib` the replay panel says "forecast model unavailable". Without `wilt.joblib` the camera panel says "wilt model unavailable".

- [ ] **Check the MobileNet weights are cached.** Wilt Watch uses a ready-trained image network called MobileNetV3-Large. Its weights (the numbers inside the network) live in a cache file outside the repo. The path is recorded in `results\wilt_metrics.json` under `features.weights_cache_file`. In this build it is:

  `C:\Users\seana\.cache\torch\hub\checkpoints\mobilenet_v3_large-5c1a4163.pth`

  Check that file exists on the demo laptop. If it is missing, the model would try to download it at the booth, where there is no internet, and the camera panel would show "wilt model unavailable". To force the cache to fill while you are still online, run the console once tonight (`.venv\Scripts\python.exe -m semai.app --rig http://127.0.0.1:8081` with the mock rig running, see section 3) and confirm the camera panel shows a verdict.

- [ ] **Run the soak test once.** A soak test runs the whole thing for a while and checks nothing falls over. This one runs the mock rig and the console for 5 minutes and restarts the mock at the 2-minute mark.

  ```
  .venv\Scripts\python.exe scripts\soak_test.py
  ```

  Then glance at `results\soak_test.txt`. Every check should say it passed (console recovers after the restart, `/api/state` never errors, nulls show as "—", only one `/stream` connection, MOCK banner flag true). The soak test was run once on 25 Sep 2026 (5 minutes, mock restarted at 2 minutes) and every check passed: the console recovered 1.0 s after the restart, `/api/state` never errored over 297 polls, nulls stayed as dashes, only one `/stream` connection was open, and the MOCK banner flag was always true. Run it again tonight and check `results\soak_test.txt` still says OVERALL: PASS.

- [ ] **Close every other program that uses the webcam.** Teams, Zoom, the Camera app, browser tabs with video calls. Only one program can hold the webcam. This matters for the fallback (section 3).

- [ ] **Pack:** laptop, charger, the rig, the rig's power supply, a healthy plant, a plain sheet of card or cloth for the background, and a white light (a desk lamp or a phone torch).

---

## 2. At the booth

### 2.1 Start up (in this order)

- [ ] **Power the rig.** Wait until the `VPD-Greenhouse-S3` network appears in the laptop's Wi-Fi list; that means the boards have booted. The ESP32 makes its own Wi-Fi network; it does not use the venue's Wi-Fi.

- [ ] **Join the rig's Wi-Fi on the laptop.** Network name (SSID): `VPD-Greenhouse-S3`. Password: `<rig Wi-Fi password>`. Windows will say "No internet". That is correct and expected.

- [ ] **Start the console.**

  ```
  cd C:\Dev\semai-ml
  .venv\Scripts\python.exe -m semai.app
  ```

  Leave this terminal window open. Do not close it during the demo.

- [ ] **Open the page** in a browser: `http://127.0.0.1:8000`

- [ ] **Check the connection dot is green.** It is next to the SEMAI title at the top left. Green means the console is receiving readings from the rig. Red means it is not (see the table in section 4).

- [ ] **Check there is NO red MOCK DATA banner.** If the banner shows, the console is not talking to the real rig. Stop, and go through section 4.

- [ ] **NEVER open `http://192.168.4.1` in a browser.** Not on the laptop, not on a phone. The ESP32 can only serve 3 web requests at once (it has 3 HTTP workers). The console already holds the one camera stream. A second viewer can stall the stream for everyone. If a judge asks to see the raw rig page, say "the console is the rig's only viewer, on purpose".

### 2.2 What each panel should show

| Panel | What you should see | What to say |
|---|---|---|
| **Live** (left) | Leaf VPD in large numbers, then air T (°C), RH (%), leaf T (°C), light (raw ADC) and water (%). A coloured status chip (OPTIMAL, HIGH_VPD_MISTING, LOW_VPD_PURGE, or an ERR status). "updated N s ago" with N staying small. | "VPD is how thirsty the air is. It comes from temperature and humidity together. The rig mists when leaf VPD goes above 1.2 kPa, purges with the fan below 0.6 kPa, and 0.8 to 1.1 kPa is the comfortable band." |
| Water level | A number with the word "uncalibrated" beside it. | "The tank sensor is not calibrated yet, so we label it honestly." |
| **Camera** (middle) | The live picture from the rig's camera with a box drawn in the middle (the region the model looks at, 70 % of the width and height). Under it a verdict: HEALTHY (green), WILTED (red) or UNSURE (amber), plus the wilted probability. It refreshes about every 2 s. | "This is Wilt Watch. It is trained on 904 real houseplant photos. On a fair test it scored 83.5 % against a 50 % coin-flip baseline. UNSURE means the probability is between 0.35 and 0.65, so the model is not confident either way." |
| Capture buttons | Two buttons, Capture healthy and Capture wilted. | Do not press these during the demo unless you mean to save a photo. They save raw frames to `data\rig_frames\` for the rig-photo check. |
| **Forecast replay** (right) | A date picker (2025 only), a play button, and a chart with the actual VPD line, the 1.2 kPa line and the model's probability. The label reads "Replay of 2025 (model never saw this year) · outdoor Penang air". | "The model is asked one question: will VPD go above 1.2 kPa within the next 2 hours? It learned from 2023 and 2024 and is tested on 2025, which it never saw." |
| Today's outlook | A small strip under the replay: "today's Penang outlook (weather forecast)" with the fetch time from last night. | "This is a weather forecast we fetched last night while online, so the demo itself stays offline." |

### 2.3 If a value shows "—"

A dash means the rig sent `null` for that sensor. `null` means the sensor reading was invalid at that moment. The console shows a dash on purpose and never shows 0, because 0 would look like a real reading.

Say: "That dash means the sensor did not give a valid reading just now. The console shows a dash rather than a fake zero. Control and safety stay on the Pico, so a bad reading never reaches the misting logic through the laptop."

If the dash does not clear on its own, check the rig's sensor cable after the demo. Do not restart anything mid-demo just for one dash.

### 2.4 How to use the replay

The replay shows one 2025 day, hour by hour. Pick the date, press play, and let it run. Three good days to pick (all from `results\forecast_metrics.json`, `replay_days`). They are the same three days drawn in `results\F4_replay_3days.png`, so you can practise with the figure first.

| Day to pick | Why | What you will see |
|---|---|---|
| **2025-02-07** (the driest day of 2025) | Highest daily peak VPD: 3.72 kPa. | The air was already dry (above 1.2 kPa) for 11 of the 24 hours. The model's probability climbs above 0.5 in the morning, before the VPD line crosses 1.2 kPa, stays near 1.0 through the afternoon, then drops in the evening. It was above 0.5 for 10 hours. |
| **2025-02-22** (a mixed day) | Its peak VPD, 1.65 kPa, is the closest to the middle of all 2025 days. | Wet morning, dry early afternoon, wet again by evening. The VPD line crosses 1.2 kPa for only 3 hours. The probability rises in the late morning and falls again by mid-afternoon; it was above 0.5 for 5 hours. This is the interesting one, because the model has to catch a short dry spell. |
| **2025-11-24** (the wettest day) | Lowest daily peak VPD: 0.44 kPa. | The VPD line never gets near 1.2 kPa. The probability stays flat near 0 all day. No misting needed, and the model agrees. |

Spoken line for the replay: "Over all 8,758 test hours of 2025, the model was right 93.8 % of the time. The simple rule 'if it is dry now, say yes' was right 89.7 % of the time. On the score that punishes missed dry hours, called F1, the model got 0.865 and the simple rule got 0.772."

F1 in one line: a score from 0 to 1 that rewards catching the dry hours without raising false alarms.

### 2.5 Camera hygiene

- [ ] Keep the camera pointed at the plant. Leaves should fill most of the box.
- [ ] Plain background behind the plant (the card or cloth you packed).
- [ ] White light on the plant. Avoid coloured venue lighting and strong backlight.
- [ ] Wipe mist droplets off the lens before the demo. Check again after each misting cycle.
- [ ] **A dark or empty frame reads as WILTED.** The model has only ever seen plants. During testing, a black frame (the laptop webcam with the lid closed) was called WILTED with high confidence. If the verdict looks wrong, first check what the camera can actually see.

### 2.6 Honest things to say if asked

- The rig-photo check is still pending. `results\wilt_rig_eval.json` says "pending — no rig photos captured yet": 0 healthy and 0 wilted rig frames so far. The team still needs one 30-minute session at the rig (about 30 healthy and about 30 wilted frames via the Capture buttons, then `.venv\Scripts\python.exe scripts\eval_rig_frames.py`). See `docs\SPEC.md` section 10 for the deadline.
- Wilt Watch scored 83.5 % on the fair (grouped) test and 85.6 % on the naive test that lets near-copies of the same photo leak into both halves (a coin-flip baseline scores 50 %). The fair number is the one we quote.
- Camera accuracy on the rig could be lower than 83.5 %: different plant species, one fixed camera angle, greenhouse lighting and mist on the lens.
- The forecast was trained on outdoor Penang air, not greenhouse air. The console logs every reading to `logs\rig_YYYYMMDD.csv`, and those logs are the next training data.
- Each wilt verdict takes about 7 ms on the laptop CPU (target was under 300 ms).

---

## 3. Fallback: no rig, or the rig will not connect

Use the mock rig. It is a small program that pretends to be the rig on the laptop itself. It gives smooth fake readings and a picture from the laptop webcam (or a folder of photos, or a generated colour frame). All its status names start with `MOCK_`, and roughly 1 reading in 20 has a `null` so you can show the dash.

- [ ] **Terminal 1: start the mock rig.** Use the webcam:

  ```
  cd C:\Dev\semai-ml
  .venv\Scripts\python.exe scripts\mock_rig.py
  ```

  Or use a folder of your own photos (jpg or png, changes picture every 2 s):

  ```
  .venv\Scripts\python.exe scripts\mock_rig.py --images C:\path\to\your\photos
  ```

  If the webcam is busy or missing, the mock prints a message and uses generated frames stamped MOCK. That is fine.

- [ ] **Terminal 2: start the console pointed at the mock.**

  ```
  cd C:\Dev\semai-ml
  .venv\Scripts\python.exe -m semai.app --rig http://127.0.0.1:8081
  ```

- [ ] **Open `http://127.0.0.1:8000`** and confirm the red **MOCK DATA** banner is showing across the top. If it is not showing, you are not on the mock; stop and check which command you ran.

- [ ] **Say out loud, at the start, that this is mock data.** For example: "Our rig is not connected right now, so the readings you see are from a simulator. The red banner says MOCK DATA so nobody is misled. The camera and the forecast replay are real." Then continue with sections 2.2 to 2.5 as normal. The replay panel works exactly the same on the mock.

- [ ] If you use the laptop webcam, point it at the plant with the lid open. A closed lid gives a black frame, and a black frame reads as WILTED.

---

## 4. If something breaks

| Symptom | Likely cause | Do this |
|---|---|---|
| Connection dot red, "updated — s ago", readings all "—" | Laptop is not on the rig's Wi-Fi, or the rig is off | Check Windows Wi-Fi shows `VPD-Greenhouse-S3` connected. Check the rig has power. The console reconnects by itself: in the soak test it recovered 1.0 s after a restart and the stream 6.0 s after; give it 10 s. |
| Red MOCK DATA banner when you expected the real rig | Console was started with `--rig http://127.0.0.1:8081`, or the rig is sending `MOCK_` statuses | Stop the console (Ctrl+C in its terminal) and start it again with plain `.venv\Scripts\python.exe -m semai.app`. |
| Camera panel black or dark, verdict WILTED | Lid closed, lens covered, or no light on the plant | Open the lid, uncover the lens, switch on the white light, point at the plant. |
| Camera panel says "wilt: waiting for a camera frame" | Stream has not started yet, or someone opened `192.168.4.1` in a browser | Wait 10 s. Close any browser tab showing `192.168.4.1`. |
| Camera panel says "wilt model unavailable" | `models\wilt.joblib` is missing, or the MobileNet weights are not cached | Check `models\wilt.joblib` exists. Check the weights file from section 1 exists. Restart the console. |
| Replay says "forecast model unavailable" | `models\forecast.joblib` is missing | Check `models\forecast.joblib` exists. Restart the console. |
| Page stuck, numbers frozen, "updated N s ago" growing | Console process hung or the terminal was closed | In the console terminal press Ctrl+C, then run `.venv\Scripts\python.exe -m semai.app` again (add `--rig http://127.0.0.1:8081` if you are on the mock). Refresh the browser. |
| Stream stops after a visitor opens the rig page on a phone | The ESP32's 3 HTTP workers are all busy | Ask them to close it. Restart the console so it reconnects to `/stream`. |
| Today's outlook strip is empty or shows an old date | `data\today.json` missing or stale | Nothing to fix at the booth (needs internet). Say it is last night's forecast, or that it was not fetched. |
| Mock rig will not start (an "address already in use" / socket error) | An older mock is still running | Close the old terminal, or run `.venv\Scripts\python.exe scripts\mock_rig.py --port 8082` and `.venv\Scripts\python.exe -m semai.app --rig http://127.0.0.1:8082`. |

Golden rule: restarting `python -m semai.app` is always safe. It never touches the rig. Control and safety stay on the Pico whatever the laptop does.

---

## 5. Where the numbers in this checklist come from

| Number | File and key |
|---|---|
| 8,758 test hours; 93.8 % vs 89.7 % accuracy; F1 0.865 vs 0.772 | `results\forecast_metrics.json` → `n_test`, `metrics.model`, `metrics.persistence` |
| 2025-02-07 (3.72 kPa), 2025-02-22 (1.65 kPa), 2025-11-24 (0.44 kPa); hours dry and hours above 0.5 | `results\forecast_metrics.json` → `replay_days[]` (`daily_max_vpd`, `hours[].dry_now`, `hours[].prob`) |
| 904 photos; 83.5 % grouped; 85.6 % naive; 50 % baseline; about 7 ms per verdict, 300 ms target; 70 % box; UNSURE band 0.35 to 0.65 | `results\wilt_metrics.json` → `dataset.n_images`, `cv_grouped.mean_accuracy`, `cv_naive.mean_accuracy`, `baseline.accuracy`, `predict_timing_ms`, `model.roi_default`, `model.unsure_band` |
| Weights cache path | `results\wilt_metrics.json` → `features.weights_cache_file` |
| Rig-photo check pending, 0 healthy, 0 wilted | `results\wilt_rig_eval.json` → `status`, `n_healthy`, `n_wilted` |
| 1.2, 0.8 to 1.1 and 0.6 kPa thresholds | `results\why_vpd.json` → `thresholds` |
| Soak test: 5 minutes, restart at 2 minutes, recovered 1.0 s, stream back 6.0 s, 297 polls with 0 errors, 10 s limit, OVERALL: PASS | `results\soak_test.txt` → header line, "console recovered", "/stream back after restart", "/api/state polls", "recovered after restart ... (limit 10 s)", last line |
