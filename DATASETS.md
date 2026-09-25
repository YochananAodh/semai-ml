# Dataset documentation

SEMAI uses two public datasets for training and testing, plus data the rig collects itself. No synthetic data is used for training or for any reported result.

## 1. Penang hourly weather, 2023–2025 (SEMAI Forecast)

| | |
|---|---|
| **Source** | Open-Meteo Historical Weather API (`archive-api.open-meteo.com`), built on the ERA5 reanalysis (Copernicus / ECMWF) |
| **Licence** | CC BY 4.0. Attribution: "Weather data by Open-Meteo.com" |
| **Location** | Penang, 5.4482° N, 100.29° E |
| **Period** | 1 Jan 2023 – 31 Dec 2025, hourly, local time (Asia/Kuala_Lumpur) |
| **Rows** | 26,304 hours |
| **File in this repo** | `results/penang_hourly_2023_2025.csv` |
| **Rebuild it** | `python -m semai.penang_data` (fetch and cache) |

Exact request:

```
https://archive-api.open-meteo.com/v1/archive?latitude=5.4482&longitude=100.29&start_date=2023-01-01&end_date=2025-12-31&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FKuala_Lumpur
```

**Columns**

| Column | Meaning | Unit |
|---|---|---|
| `time` | Hour, local time | — |
| `t` | Air temperature at 2 m | °C |
| `rh` | Relative humidity at 2 m | % |
| `vpd` | Air vapour-pressure deficit, computed as `SVP(t) × (1 − rh/100)` with `SVP(T) = 0.6108·exp(17.27·T/(T+237.3))` | kPa |
| `dry` | `vpd > 1.2`, the rig's misting threshold | true/false |

**How it is used**

- **Label:** 1 if the air is dry (`vpd > 1.2`) at t+1 or t+2 hours. In other words, "misting will be needed within 2 hours".
- **Features (17):**
  - hour of day and day of year, as sine and cosine pairs
  - `t`, `rh` and `vpd` now and 1, 2 and 3 hours ago
  - the change in `vpd` over the last hour
- **Split, by time:**
  - Training uses 2023–2024 (17,539 hours). Two late-2024 rows whose labels reach into 2025 are dropped.
  - Testing uses 2025 only (8,758 hours).
  - No 2025 row is used in training, and the code asserts this.

**Limits**

- ERA5 is a reanalysis: a gridded estimate, not a weather-station reading.
- It describes **outdoor** Penang air, not the air inside a greenhouse. The next step is retraining on the rig's own logs (section 3).

## 2. Healthy and Wilted Houseplant Images (SEMAI Wilt Watch)

| | |
|---|---|
| **Source** | Kaggle, "Healthy and Wilted Houseplant Images" by russellchan: https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images |
| **Licence** | See the dataset's Kaggle page. The images are **not redistributed** in this repo. |
| **Size** | 904 photos: 452 healthy, 452 wilted |
| **Get it** | `python scripts/train_wilt.py` downloads it into `data/wilt/`, which is git-ignored |

**How it is used**

1. **Features.** Each photo becomes a 960-number feature vector from a frozen MobileNetV3-Large network, pretrained on ImageNet (torchvision `IMAGENET1K_V2`).
2. **Near-duplicates.** Photos whose feature vectors have cosine similarity above 0.97 are grouped together. That gives 71 pairs covering 124 images, none with mismatched labels.
3. **Evaluation.** 5-fold cross-validation, with each duplicate group kept on one side of every fold:
   - accuracy **83.5 % ± 1.1**, against a 50 % baseline
   - the naive, ungrouped CV scores 85.6 %
4. **Presentation.** The photos are used for training only. They never appear in our figures or slides.

## 3. Data the rig collects (future training data)

- **Readings log.** The SEMAI Console appends every rig reading, time-stamped, to `logs/rig_YYYYMMDD.csv` (git-ignored). The columns are `vpd, t_air, rh, t_leaf, water, light, status`.
- **Camera frames.** The console's **Capture healthy** and **Capture wilted** buttons save camera frames to `data/rig_frames/<label>/`. `scripts/eval_rig_frames.py` scores Wilt Watch on them. That result is pending until the team's photo session.

## Trained model files

| File | Model | Size |
|---|---|---|
| `models/forecast.joblib` | scikit-learn `HistGradientBoostingClassifier` (learning rate 0.05, early stopping at 110 iterations), trained on 2023–2024, decision threshold 0.5 | 0.4 MB |
| `models/wilt.joblib` | `StandardScaler` → `LogisticRegression(C=0.1)` head on the 960-number MobileNetV3-Large features. The backbone weights download from torchvision on first use. | 0.03 MB |

- **Loading:** `semai/forecast.py` loads the forecast model and `semai/wilt.py` loads the camera model.
- **Results:** every reported number is in `results/*.json` and written up in [results/RESULTS.md](results/RESULTS.md).
