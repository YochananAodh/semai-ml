# DECISIONS.md — SEMAI build log (25 Sep 2026)

Every choice not spelled out in `docs/SPEC.md`, and every fallback taken, one line each, in order.

## Step 1 — Environment
- Python: `py -3.12` does not resolve on this machine because the 3.12 install is registered under the tag `Astral/CPython3.12.13` (uv-managed). Used its full path (`%APPDATA%\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe`) to create `.venv`. Same interpreter the spec asks for, just addressed differently.
- Kaggle URL: a HEAD request returns 404 but GET follows a 302 to Google Cloud Storage and works, so the download uses GET with redirects.
- `pip install -r requirements.txt` succeeded on the first attempt (torch 2.14.0+cpu and torchvision 0.29.0+cpu came straight from PyPI), so no §3 install fallback was needed.
- Open-Meteo archive and the Kaggle zip both downloaded on the first attempt (no retries needed). Raw archive JSON kept in `data/penang_hourly_raw.json` (gitignored) beside the CSV so the CSV can be rebuilt offline.
- `scripts/fetch_today.py` was written and run during Step 1 (11:52 local) so that the console builder could rely on `data/today.json` existing. It computes VPD per forecast hour with the same formula as the rig.

## Steps 2–6 — built in parallel
- Steps 2, 3, 4 and 5+6 were built by four parallel builder agents, each followed by an adversarial verifier and a fix pass. Their per-part decisions are listed below under their own headings.

### Step 2 — SEMAI Forecast (builder: done)
- Label leakage guard: the two training rows at 2024-12-31 22:00 and 23:00 have label windows (t+1, t+2) that reach into 2025, so they were dropped from training (n_train_dropped_for_leakage = 2); asserted in code that every remaining train label window ends before 2025 (last train hour 2024-12-31 21:00, window ends 23:00).
- Test set = every 2025 hour with a defined label: 8,758 rows. The last 2 hours of 2025-12-31 (22:00, 23:00) have no t+2/t+1 in the data, get NaN labels, and are dropped per the task text; this is why the count is 8,758 rather than the SPEC's ~8,759.
- NaN dropping: the first 3 hours of 2023 (lags undefined) and the last 2 hours of 2025 (label undefined) are the only 5 rows dropped; the CSV has no other NaN.
- build_features() drops nothing itself: it returns X, y and meta on the same index with NaN left in place, so training and replay share one function and callers decide what to drop (replay keeps all 8,760 hours of 2025 since features are defined for all of them).
- Climatology threshold: grid over the 24 distinct training hourly positive rates; predict 1 when rate(hour) >= thr; the smallest rate with the highest training F1 wins (chosen 0.2503, training F1 0.494; positive hours 10-16). The hourly rate itself is the score for AUC (labelled auc_kind = 'hourly rate').
- Persistence AUC is the AUC of the binary prediction and is stored with auc_kind = 'binary'; always_no has auc = null. Model AUC uses predict_proba (auc_kind = 'probability').
- Permutation importance uses the exact call in the task (n_repeats=10, random_state=0, scoring='f1', n_jobs=4) on the 8,758 test rows; stored sorted descending by mean.
- replay(): probabilities for all 8,760 hours of 2025 are computed once at warm-up and cached in a module global together with the feature frame, so each replay() call is a filtered lookup (~0.6 ms). actual_dry is None for the last two hours of 2025-12-31 only; prob is None only if a feature is NaN (never happens in practice).
- replay() date validation: rejects non-strings, non-YYYY-MM-DD, impossible calendar dates and any year other than 2025 with ValueError.
- F4 uses a dual axis (VPD left, probability right) because the task text explicitly mandates it, even though general chart guidance discourages dual axes; one shared legend sits under the three panels to avoid overlapping the curves.
- Figures show the model beside a baseline everywhere: F1 grouped bars for 4 methods, F2 persistence left / model right, F3 and F4 carry 'model F1 0.865 vs persistence baseline F1 0.772' in their titles.
- Every number in the figures is in forecast_metrics.json: the four metric blocks with confusion matrices, permutation importance mean/std, the three replay days with daily max VPD and all 72 hourly rows (t, rh, vpd, prob, actual_dry, dry_now), and the 2025 median daily max VPD (1.653 kPa).
- The metrics JSON also records the 12 in-code self-checks (name, passed, detail), the train/test time ranges, model iterations used (110 of max 300, early stopping), and all library versions.
- Colours: fixed categorical slots from the dataviz reference palette (yellow always_no, orange persistence, aqua climatology, blue model, violet VPD, red shading); text stays in ink colours.
- No §3 fallback was needed for this step.

### Step 3 — Why VPD, not humidity (builder: done)
- Per-year hysteresis shares are sliced from ONE continuous 3-year rig_state_sequence run (the rig runs continuously), not re-run per year; only the first hour of a year could differ.
- Threshold-band boundaries follow the inequalities in semai.vpd exactly: dry = vpd>1.2, gap_1.1_to_1.2 = (1.1,1.2], optimal = [0.8,1.1], gap_0.6_to_0.8 = [0.6,0.8), saturated = vpd<0.6; an assertion checks the five bands partition every hour.
- missed_share_of_dry_pct uses dry-stress hours as the denominator (share of dry-stress hours missed), as the task describes.
- typical_missed_hour reports p25/median/p75 for t, rh and vpd only for the headline RH<60 rule; the 55-75% threshold table carries counts/shares only.
- W1 drawn as grouped bars (overall + 2023/2024/2025 x DRY/OPTIMAL/SATURATED), percentages annotated on every bar.
- W2 scatter sample: numpy default_rng(0).choice(n, 4000, replace=False); 'not dry' points drawn in recessive gray, caught in blue, missed in orange; the miss zone between the RH=60 line and the VPD=1.2 curve is lightly shaded and labelled in the legend; miss count and percentage are in the title.
- W3 uses a second PANEL (shared x) for mean VPD rather than a second y-axis, to satisfy the dataviz skill's one-axis rule while meeting the task's 'second axis or second panel' option; peak bar highlighted and annotated.
- Colours taken from the dataviz skill's pre-validated categorical palette (blue #2a78d6, orange #eb6834, aqua #1baf7a, red #e34948 for threshold lines); the node validator was not re-run since the reference palette is documented as validated.
- Shares are stored as percentages rounded to 3 decimals; hour-of-day mean VPD rounded to 4 decimals; raw hour counts are stored alongside every share so nothing is lost to rounding.
- computed_at_utc is the only field expected to change between runs; the idempotency check diffs the JSON with that one line excluded.
- Script asserts: rows == penang_data.EXPECTED_ROWS (26304), no NaN in VPD, strictly continuous hourly timestamps; it sorts by time before running the state machine so order is chronological regardless of CSV order.
- No §3 fallback was needed for this step.

### Step 4 — SEMAI Wilt Watch (builder: done)
- Preprocessing identity: semai/wilt.py is the single source of the feature extractor (weights.transforms(): resize 232, centre-crop 224, ImageNet mean/std, BILINEAR) and scripts/train_wilt.py imports the underscore helpers (_load_extractor / extractor['fn']) rather than duplicating the code, so training and inference cannot diverge. The joblib's feature_extractor string selects torch vs the fallback extractor at inference time.
- Public surface of semai/wilt.py kept to exactly available(), roi_box(), predict(); the extra diagnostic helper is named _unavailable_reason() (underscore) so it does not widen the required API, and the console can still read it if wanted.
- predict() ms is measured inside predict (crop + preprocessing + backbone + LR head) and returned as the 'ms' key; the JSON timing block additionally times 20 synthetic frames from outside the call after one warm-up call, and reports mean/median/max/min plus the warm-up ms.
- Timing frames are uniform-noise 800x600 BGR arrays from numpy default_rng(0); the label they produce is meaningless and is not reported as a result.
- Duplicate group ids are the connected components of the >0.97 cosine graph (union-find, path compression); singletons are their own group, so n_groups counts all 839 components and n_groups_2plus the 59 multi-image ones.
- Per-fold accuracy, mean and std (population std, numpy default) are reported alongside pooled out-of-fold precision/recall/F1/AUC/confusion; pooled accuracy equals the fold mean here because folds are near-equal in size.
- Baseline is the always-one-class 50 % (452/452 balanced set); it is drawn as a dashed line in R1, stated in the R2 title, and stored as baseline_accuracy in both JSON files.
- Kaggle licence written as 'see the Kaggle dataset page' because I do not know with certainty what that page states.
- training accuracy is deliberately not computed or written (SPEC §2 rule 2); the 6-image predict smoke test is labelled in the JSON as a plumbing check on training data, not an accuracy claim.
- Feature cache validity: data/wilt_features.npz is reused only if its extractor name, ordered file list and labels match the current dataset; otherwise features are re-extracted.
- Weight download retries: mobilenet_v3_large(weights=...) is wrapped in 3 tries with 3/6 s sleeps; any failure raises and train_wilt.py then switches to the §3 HSV+HOG fallback and records it in 'fallbacks' (path exercised only via a unit smoke test in the scratchpad, since torch worked).
- eval_rig_frames.py 'done' path: UNSURE and unreadable files count as wrong for 'accuracy'; 'accuracy_excluding_unsure' drops them; confusion is a per-true-class dict with HEALTHY/WILTED/UNSURE/UNREADABLE columns; it returns exit 1 only if photos exist but the model is unavailable.
- The rig-eval 'done' path was tested by monkeypatching RIG_DIR/OUT to the scratchpad with 4 copied dataset photos, so no dataset photo was ever placed under data/rig_frames/ or results/; the scratch copies were deleted afterwards.
- R2 figure was widened to 7.2x6.0 in with a three-line title after the first render's title collided with the colourbar ticks.
- No §3 fallback was needed for this step.

### Orchestrator fixes after the Step 2–4 verifiers (all verifier findings were "minor"; no fix agent was needed)
- `semai/penang_data.py`: `load()` now falls back to the committed `results/penang_hourly_2023_2025.csv` before ever touching the network, so a fresh clone can replay 2025 offline (verifier note on the console's `/api/replay`).
- `scripts/eval_rig_frames.py`: the pending status string now uses the spec's exact wording with an em dash ("pending — no rig photos captured yet").
- `scripts/train_wilt.py`: the 6-image plumbing check on training photos keeps only the per-frame timing in `wilt_metrics.json` (labels go to stdout), so nobody can read it as a training-set accuracy; the empty `decisions` list now points at this file; `R2` is saved with `bbox_inches="tight"` so the row labels are not clipped.
- `semai/wilt.py`: `predict()` raises a clear `ValueError` for a non-uint8 frame instead of failing inside PIL.
- `scripts/why_vpd.py`: the console summary no longer assumes at least one missed hour exists (harmless on real data, where 2,428 are missed).
- The wilt grouped-CV accuracy is 83.5 %, 1.0 point under the spec's ~84.5 %. Reported as measured; likely cause is a different `StratifiedGroupKFold` fold assignment (sklearn 1.9.1, group-id numbering); the fold-to-fold std is 1.1 points. Nothing was tuned.
- The forecast test set has 8,758 rows, not the spec's ~8,759: the last two hours of 2025-12-31 have no t+1/t+2 label and are dropped, per the spec's own label definition.

### Steps 5–6 — Mock rig and SEMAI Console (builder: done)
- Mock status hysteresis is implemented inline in mock_rig.py (_rig_state) with the same thresholds/logic as semai.vpd.rig_state_sequence, because the mock needs one stateful step per request rather than a whole sequence.
- Mock null injection: a MOCK_ERR_SENSOR reading (1/60) never also gets the 1/20 single-null treatment; water and light stay ints even during MOCK_ERR_SENSOR.
- Mock frame counter and hue are time/counter based; --images mode picks the picture by wall-clock (every 2 s) so it stays 2 s per picture regardless of fps.
- RigClient: water/light that are null or non-numeric become None (spec only defines NaN mapping for the four floats); status that is not a non-empty string becomes 'UNKNOWN'; bool JSON values are treated as invalid (NaN/None).
- RigClient data poller: after a failed poll the next attempt waits the backoff (1,2,4,8,10 s) instead of poll_s; after a success it returns to the 2 s cadence.
- RigClient stream buffer: when no FFD8 is found, only the last byte is kept (a lone 0xFF); if a start marker is found but no end marker and the buffer exceeds ~2 MB, the buffer is dropped.
- Logger writes floats with 2 dp, ints as-is, NaN/None/inf as empty; header is also written if an existing file is empty.
- /api/state 'mock' is true when the rig host is not 192.168.4.1 OR the latest status starts with 'MOCK'; before the first reading data is null and mock reflects only the host rule.
- /api/replay validates the date itself (regex ^2025-MM-DD$ plus strptime) before calling semai.forecast.replay, so a non-2025 or malformed date is 400 even when the forecast model is missing; a ValueError raised by replay() is also mapped to 400.
- /frame.jpg overlay uses semai.wilt.roi_box when available, else a centred box of roi*w by roi*h; verdict colours: WILTED red, HEALTHY green, UNSURE amber; 'wilt: waiting for a frame' is shown when the model is available but no frame has arrived.
- The wilt background thread re-tries the lazy import every ~2 s so the panel switches from 'wilt model unavailable' to live verdicts as soon as semai/wilt.py lands, without restarting the console.
- Capture file name is YYYYMMDD_HHMMSS_mmm.jpg from datetime.now(); the returned path uses forward slashes relative to the repo root.
- Page layout: CSS grid 52 px header + one row of three panels (280 px live | 1fr camera | 1fr forecast), html/body overflow hidden; canvases are 560x300 (replay) and 560x90 (today) scaled by CSS.
- Replay chart reveals the VPD and probability lines progressively up to the current hour and loops from hour 23 back to 0; a date change or Play with no data reloads /api/replay; the replay loads the default 2025-03-15 automatically once /api/state reports forecast_available.
- Today's outlook strip shows the first 24 hourly entries at or after now (30 min grace); if fewer than 24 remain in today.json it falls back to the last 24 entries and labels them '(latest available)'; the strip is re-fetched every 5 minutes.
- Status chip colour is decided after stripping a MOCK_ prefix; the chip text keeps the full MOCK_ prefix; unknown statuses are grey.
- The mock harness ran on port 8091 to avoid colliding with any other agent; the end-to-end harness used the specified 8081/8000 after confirming both ports were free.
- The browser look was served by a self-cleaning launcher; I killed its two server processes by PID (never by image name) once the checks were done.
- No §3 fallback was needed for these steps.

### Orchestrator fixes after the Steps 5–6 verifier (three "minor" findings; no fix agent needed)
- `scripts/mock_rig.py`: emits `MOCK_ERR_LOW_WATER` when the synthetic tank drops below 5 %, so every §4 status family (incl. the red ERR chip) is exercised by the mock; one lock now guards the shared frame source (`cv2.VideoCapture` is not thread-safe if a second viewer opens `/stream`).
- `semai/static/app.js`: a typed date outside 2025 now shows the server's error text instead of "forecast model unavailable".
- Note for the demo team (from the builder's tests): with the laptop lid closed the webcam gives a black frame and Wilt Watch calls it WILTED with p ≈ 0.95. The model has only seen plants; point the camera at the plant.
- `scripts/soak_test.py` was written by the orchestrator against the endpoint contracts; it also does a static check that the page maps null to "—" because a headless test cannot render the DOM.

## Step 7 — Soak test, results pack, notebook
- Soak test ran once for the full 5 minutes against the mock rig (webcam frames) with the restart at 2 min: OVERALL PASS (see `results/soak_test.txt`). The console's `connected` flag is judged on a 6 s window, so the test detects recovery from a *fresh* reading (data age < 4 s) rather than from the flag alone.
- The results pack was written by one agent per document, each forbidden to type a number that it had not just read from `results/*.json` or `results/soak_test.txt`, then audited number-by-number by a second agent against the same files, then fixed and re-audited where needed.
- `notebooks/semai_results.ipynb` re-runs the four scripts via subprocess and displays the figures from `results/`; it was executed with `jupyter nbconvert --execute --inplace`.

## Step 8 — Finish
- Git had no user identity on this machine; set a repo-local one (a team identity) so the commit carries the team name. Nothing was changed globally.
- The first results-pack run was cut off by the account's monthly spend limit after the five writers had finished but before their audits ran (the `RESULTS.md` writer was cut off after writing the file, before its report). On the user's "try again" the audit-and-fix pass was re-launched for all five documents, and a scripted number check (every numeric token in each document looked up against `results/*.json`, `soak_test.txt` and `ENV.md`) was run as an independent fallback: only shown arithmetic, differences of sourced numbers, section numbers and an alternate port were unmatched.
- Results-pack audit outcome (five auditors, 260 / 214 / 165 / 145 / 61 numbers checked): RESULTS.md and SLIDES.md had only minor findings; EXPLAINER.md, JUDGE_QA.md and DEMO_CHECKLIST.md had one or two major ones (a stale "soak test not present" sentence written before the soak test ran; an overstated claim that the forecast is fed live rig readings; a wrong description of the connection dot as having a grey state; a sentence implying the laptop could switch the fan) which were fixed by fix agents and, for the last one, by the orchestrator. Remaining minors (one-decimal timings, plain-English glosses for cosine similarity / union-find / ERA5 / climatology, the invented "22 °C" comparison, the backbone licence, a dangling cross-reference, a quoted error message the mock never prints) were fixed by the orchestrator. EXPLAINER.md stays longer than the spec's "~2 pages" (about 2,600 words) because trimming further would drop content the spec explicitly asks for.
