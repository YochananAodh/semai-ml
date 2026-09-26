# Dataset documentation

SEMAI uses public datasets for training and testing, plus data the rig collects itself. No synthetic data is used for training or for any reported result.

## 1. Penang hourly weather, 2023–2025 (the GETT ML entry, `SEMAI_Forecast.ipynb`)

| | |
|---|---|
| **Source** | Open-Meteo Historical Weather API (`archive-api.open-meteo.com`), built on the ERA5 reanalysis (Copernicus / ECMWF) |
| **Licence** | CC BY 4.0. Attribution: "Weather data by Open-Meteo.com" |
| **Location** | Penang, 5.4482° N, 100.29° E |
| **Period** | 1 Jan 2023 – 31 Dec 2025, hourly, local time (Asia/Kuala_Lumpur) |
| **Raw file** | `dataset/penang_weather_hourly_2023_2025_raw.csv`: **26,304 rows × 26 columns** (time + 25 weather variables), exactly as downloaded. The first 3 lines are Open-Meteo's location metadata; the header row carries the units. |
| **Cleaned file** | `dataset/penang_weather_hourly_2023_2025_clean.csv`: **26,301 rows × 31 columns** (time, 29 features, the label), written by the notebook |
| **Exact request** | `dataset/SOURCE_URL.txt` |

**Raw variables (25)**

| Group | Variables |
|---|---|
| Air | temperature, relative humidity, dew point, apparent temperature, vapour-pressure deficit (VPD) |
| Sun and cloud | shortwave, direct and diffuse radiation; sunshine duration; total, low, mid and high cloud cover |
| Wind | speed, gusts, direction |
| Water | precipitation, rain, FAO-56 reference evapotranspiration (ET₀) |
| Soil | temperature and moisture, 0–7 cm |
| Other | surface pressure, sea-level pressure, is-day flag, weather code |

**Cleaning (notebook section 2)**

- Column names: units stripped from the headers; `time` parsed as a date-time.
- Checks: 0 missing values, 0 duplicate hours, 0 skipped hours, 0 values out of physical range.
- Dropped 4 redundant columns: `rain` (identical to `precipitation`), `pressure_msl` (`surface_pressure` shifted by the ground height), `is_day` (covered by the hour of day) and `weather_code` (a category code derived from the other variables).
- Wind direction (0–360°) encoded as sine and cosine, so 359° and 1° are close. Sunshine duration converted from seconds to minutes.
- The first hour (no previous hour for the 1-hour changes) and the last two hours (no future for the label) are removed: 26,304 → 26,301 rows.

**Label**

`mist_next_2h` = 1 if the dataset's own `vapour_pressure_deficit` is above **1.2 kPa**, the rig's misting threshold, at hour t+1 or t+2. In other words, "misting will be needed within 2 hours". 16.1 % of hours are 1.

**Features (29)**

- 20 from the dataset: temperature, relative humidity, dew point, apparent temperature, VPD now; shortwave, direct and diffuse radiation, sunshine; total, low, mid and high cloud cover; wind speed, gusts; precipitation, ET₀; soil temperature, soil moisture; surface pressure.
- 9 engineered: hour of day and day of year (sine and cosine each), wind direction (sine and cosine), and the 1-hour change in VPD, temperature and radiation.

**Split, by time**

- Training: 2023–2024, 17,541 hours (ends 31 Dec 2024 21:00, so no training label reaches into 2025).
- Testing: all of 2025, 8,758 hours. The notebook asserts that no 2025 row is used in training.

**Limits**

- ERA5 is a reanalysis: a gridded estimate, not a weather-station reading.
- It describes **outdoor** Penang air, not the air inside a greenhouse. The next step is retraining on the rig's own logs (section 4).

## 2. Penang temperature and humidity, 2023–2025 (booth console only)

The console's smaller forecast (`models/forecast.joblib`, 17 features) was trained on the same location and period, but only temperature and relative humidity, with VPD computed as `SVP(t) × (1 − rh/100)`, `SVP(T) = 0.6108·exp(17.27·T/(T+237.3))`. File: `results/penang_hourly_2023_2025.csv`; rebuild with `python -m semai.penang_data`. It is not part of the GETT ML entry.

## 3. Healthy and Wilted Houseplant Images (SEMAI Wilt Watch, booth console only)

| | |
|---|---|
| **Source** | Kaggle, "Healthy and Wilted Houseplant Images" by russellchan: https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images |
| **Licence** | See the dataset's Kaggle page. The images are **not redistributed** in this repo. |
| **Size** | 904 photos: 452 healthy, 452 wilted |
| **Get it** | `python scripts/train_wilt.py` downloads it into `data/wilt/`, which is git-ignored |

Each photo becomes a 960-number feature vector from a frozen MobileNetV3-Large network (ImageNet weights); a logistic-regression head is scored with 5-fold cross-validation that keeps near-duplicate photos on one side of every fold: accuracy 83.5 % ± 1.1 against a 50 % baseline. The photos are used for training only and never appear in our figures, slides or poster. Wilt Watch is not part of the GETT ML entry.

## 4. Data the rig collects (future training data)

- **Readings log.** The SEMAI Console appends every rig reading, time-stamped, to `logs/rig_YYYYMMDD.csv` (git-ignored). The columns are `vpd, t_air, rh, t_leaf, water, light, status`.
- **Camera frames.** The console's **Capture healthy** and **Capture wilted** buttons save camera frames to `data/rig_frames/<label>/`.

## Trained model files

| File | Model | Used by |
|---|---|---|
| `models/semai_forecast_gb.joblib` | scikit-learn `HistGradientBoostingClassifier` (learning rate 0.05, up to 300 iterations; scikit-learn's automatic early stopping ended it at 114), trained on 2023–2024 with the 29 features above, decision threshold 0.5. Saved as a dict: `model`, `features`, `threshold`, `mist_kpa`. | **GETT ML entry** (the notebook) |
| `models/forecast.joblib` | Earlier `HistGradientBoostingClassifier` on temperature and humidity only (17 features) | booth console |
| `models/wilt.joblib` | `StandardScaler` → `LogisticRegression(C=0.1)` on MobileNetV3-Large features | booth console |
