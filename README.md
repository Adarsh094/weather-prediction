# 🌦️ Smart Weather Prediction System

**Predict the weather for any future date — today, tomorrow, next week, or six months from now.**

A complete Machine Learning + Time Series Forecasting project that combines a
live weather API with five trained forecasting models, wrapped in an
interactive Streamlit application.

---

## 📖 Project Overview

Most beginner weather projects ask the user to *type in* today's humidity and
pressure, then predict rain from those numbers. That is a classification
exercise, not a forecast — you cannot use it to answer *"will it rain on
15 September?"*, because nobody knows September's humidity yet.

This project solves the real problem. You pick a **city** and a **date**, and
the system works out how to answer:

```
                     user picks a city and a date
                                 |
                 how many days ahead is that date?
                    /                          \
     0 to 7 days ahead                    more than 7 days
             |                                    |
    LIVE WEATHER API                 TRAINED TIME-SERIES MODELS
 (OpenWeatherMap / Open-Meteo)   (Prophet / ARIMA / LSTM / RF / XGBoost)
             \                                    /
              ----------> weather parameters <----
                                 |
                   Rain classifier → rain probability
                                 |
              Rain / No Rain  +  honest confidence score
```

Both routes end at the same place, so the app always shows the same set of
numbers — only the badge at the top tells you which engine produced them.

---

## ✨ Features

- **Any future date** — from today out to a year ahead
- **Automatic routing** — live API for the next 7 days, trained models beyond that
- **Six cities** — Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad
- **Five forecasting models compared** — Random Forest, XGBoost, Prophet, ARIMA, LSTM
- **A climatology baseline** that every model must beat to be deployed
- **Rolling-origin backtesting** — the professional way to measure forecast error
- **Honest confidence scores** derived from measured error, not invented
- **Four interactive Plotly charts** + feature importance
- **10 years of real weather data** (ERA5 reanalysis), not a synthetic CSV
- **Graceful degradation** — if the API is unreachable, the models take over
- **Works with no API key** out of the box

---

## 📊 Dataset Description

Real historical daily weather from the **Open-Meteo ERA5 archive** — the same
reanalysis dataset professional meteorologists use for historical studies.

| Property | Value |
|----------|-------|
| Source | Open-Meteo ERA5 archive (free, no API key) |
| Period | 2016-07-22 → 2026-07-22 (**10 years**) |
| Cities | 6 |
| Rows | **21,918** daily observations (3,653 per city) |
| Target | `rain` — 1 if precipitation ≥ 1.0 mm, else 0 |

### Columns

| Column | Meaning | Unit |
|--------|---------|------|
| `date` | Calendar day | date |
| `city` | City name | text |
| `temperature` | Daily mean air temperature | °C |
| `temp_max` / `temp_min` | Daily maximum / minimum | °C |
| `humidity` | Mean relative humidity | % |
| `pressure` | Mean sea-level pressure | hPa |
| `wind_speed` | Mean wind speed | km/h |
| `wind_gust` | Maximum wind speed | km/h |
| `cloud_cover` | Mean cloud cover | % |
| `dew_point` | Mean dew point | °C |
| `precipitation` | Total rainfall | mm |
| `sunshine_seconds` | Sunshine duration | s |
| **`rain`** | **Target** — 1 = Rain, 0 = No Rain | 0/1 |

> **Why generate nothing?** Every number in this project is a real measurement.
> The dataset downloads itself on first run, so the repository stays small and
> anyone can reproduce it exactly.

---

## 🛠️ Technologies Used

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 – 3.14 |
| Data | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost |
| Time Series | Prophet, statsmodels (SARIMAX) |
| Deep Learning | **PyTorch** (LSTM) |
| Visualisation | Plotly, matplotlib, seaborn |
| Web App | Streamlit |
| Model Persistence | Joblib |
| Live Weather | OpenWeatherMap API, Open-Meteo API |

> **Why PyTorch instead of TensorFlow/Keras?** TensorFlow publishes no build for
> Python 3.13+. PyTorch does, and it is equally standard in industry and
> research. The architecture (LSTM layer → dense head) is exactly what a Keras
> version would use.

---

## 🚀 Installation Steps

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

PyTorch is large; for the small CPU-only build use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Download the data and train the models

```bash
python train.py
```

This downloads 10 years of weather for all six cities, cleans it, trains and
compares every model, and saves the winners. **It takes about 4 minutes.**

Useful options:

```bash
python train.py --cities Delhi Mumbai   # train only these cities
python train.py --quick                 # skip the slow models (LSTM, ARIMA)
```

### 3. Run the web app

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

### 4. (Optional) Use OpenWeatherMap instead of Open-Meteo

The app works with **no API key** — it falls back to Open-Meteo automatically.
To use OpenWeatherMap, get a free key from
[openweathermap.org/api](https://openweathermap.org/api) and set it:

```bash
# Windows
setx OWM_API_KEY "your_key_here"

# macOS / Linux
export OWM_API_KEY="your_key_here"
```

Restart the app and the sidebar will show **OpenWeatherMap** as the provider.

---

## 📁 Project Structure

```
Smart_Weather_Prediction/
│
├── data/
│   ├── raw/
│   │   ├── weather_history.csv        # 21,918 rows, all cities
│   │   └── city_*.csv                 # per-city download cache (resumable)
│   └── processed/
│       └── weather_clean.csv          # cleaned, gap-filled, with the target
│
├── models/
│   ├── delhi/                         # one folder per city
│   │   ├── forecaster_temperature.pkl
│   │   ├── forecaster_humidity.pkl
│   │   ├── forecaster_pressure.pkl
│   │   ├── forecaster_wind_speed.pkl
│   │   ├── forecaster_cloud_cover.pkl
│   │   ├── rain_classifier.pkl
│   │   ├── climatology.pkl
│   │   └── metadata.json              # metrics, reliability, importances
│   ├── mumbai/ … hyderabad/
│   ├── model_comparison.csv           # all five models, all cities
│   ├── rain_model_comparison.csv
│   └── training_summary.json
│
├── notebook/
│   └── Smart_Weather_Prediction.ipynb # full analysis, step by step
│
├── utils/                             # clean architecture: one job per module
│   ├── config.py                      # paths, cities, constants
│   ├── data_loader.py                 # downloading historical weather
│   ├── preprocessing.py               # cleaning, gap filling, the target
│   ├── features.py                    # feature engineering
│   ├── models.py                      # the five-model zoo + baseline
│   ├── evaluation.py                  # MAE / RMSE / R² + classification
│   ├── forecasting.py                 # loading models, producing forecasts
│   ├── predictor.py                   # the API-vs-model routing logic
│   └── visualization.py               # every Plotly chart
│
├── app.py                             # the Streamlit application
├── train.py                           # the full training pipeline
├── forecast.py                        # command-line forecasting tool
├── weather_api.py                     # OpenWeatherMap + Open-Meteo clients
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

| What you want | Command |
|---------------|---------|
| Train everything | `python train.py` |
| Launch the web app | `streamlit run app.py` |
| Forecast from the terminal | `python forecast.py Delhi 2026-09-15` |
| Next 14 days as a table | `python forecast.py Mumbai --days 14` |
| Check the live API | `python weather_api.py Chennai` |
| Re-download the data | `python -m utils.data_loader` |
| Open the analysis | `jupyter notebook notebook/Smart_Weather_Prediction.ipynb` |

---

## 📈 Model Performance

All numbers below are **measured**, produced by `python train.py` on
2026-07-28. Reproduce them by running it yourself.

### Forecasting daily mean temperature

Each model was trained on the first 9 years and asked to forecast the **final,
completely unseen year**. Averaged over all six cities:

| Rank | Model | MAE (°C) | RMSE (°C) | R² |
|------|-------|---------|-----------|-----|
| 🥇 | **Prophet** | **1.219** | **1.563** | **0.786** |
| 🥈 | *Climatology (baseline)* | *1.242* | *1.609* | *0.767* |
| 🥉 | ARIMA (SARIMAX + Fourier) | 1.253 | 1.612 | 0.766 |
| 4 | XGBoost | 1.464 | 1.885 | 0.668 |
| 5 | Random Forest | 1.541 | 1.934 | 0.635 |
| 6 | LSTM (PyTorch) | 1.648 | 2.118 | 0.601 |

### 🔍 The most important finding in this project

**Prophet barely beats the climatology baseline** — its average RMSE is just
**2.9%** lower (1.563 vs 1.609 °C). Per city the margin ranges from **−2.5%**
(Kolkata, where the baseline *wins*) to **+15.5%** (Mumbai), averaging **+5.3%**.

The baseline is not a model at all — it just says *"the temperature on
15 September will be the average of the last ten 15th of Septembers"*.

This is the honest truth about long-range weather forecasting, and the project
reports it rather than hiding it:

- Beyond ~10 days the atmosphere is **chaotic**. No model can track individual
  weather systems that far out.
- What a long-range model *can* predict is the **seasonal pattern plus a slow
  trend** — which is most of the signal, and is exactly what the baseline
  captures too.
- The LSTM does **worse** than the baseline. A deep network is not automatically
  better; recursive multi-step rollout accumulates error at every step.

**This is why the baseline is a deployable model here.** For Kolkata, plain
climatology beat all five trained models, so climatology is what actually ships
for Kolkata. Deploying a sophisticated model that loses to a ten-year average
would be bad engineering, however impressive it looks in a report.

### Per-city results

| City | Deployed model | Skill vs baseline | Rain model | Accuracy | F1 | ROC-AUC |
|------|---------------|-------------------|-----------|----------|-----|---------|
| Delhi | Prophet | +2.8% | Random Forest | 0.882 | 0.749 | 0.932 |
| Mumbai | Prophet | +15.5% | Random Forest | 0.899 | 0.856 | 0.975 |
| Bengaluru | ARIMA | +0.7% | XGBoost | 0.866 | 0.819 | 0.957 |
| Chennai | Prophet | +3.7% | Random Forest | 0.789 | 0.734 | 0.891 |
| **Kolkata** | **Climatology** | **−2.5%** | Random Forest | 0.893 | 0.882 | 0.976 |
| Hyderabad | Prophet | +11.9% | Random Forest | 0.907 | 0.832 | 0.953 |

### Rain / No Rain classifier

Averaged over all six cities, scored on the unseen final year:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|-------|----------|-----------|--------|-----|---------|-------|
| **Random Forest** | **0.873** | 0.860 | **0.771** | **0.813** | **0.947** | **0.086** |
| XGBoost | 0.865 | **0.874** | 0.727 | 0.793 | 0.945 | 0.096 |

The **Brier score** (0.086) measures whether the probabilities mean anything —
when the model says "70% chance", does it rain about 70% of the time? This
matters because the app shows a percentage, not just a yes/no.

### How the confidence score is calculated

The confidence figure is **not** invented. `train.py` runs a **rolling-origin
backtest** — training at six different points in history and forecasting
forward from each — then measures the error at each forecast distance:

| Horizon | Samples | MAE (°C) | RMSE (°C) | Reliability |
|---------|---------|----------|-----------|-------------|
| 1–7 days | 42 | 1.90 | 2.33 | 0.865 |
| 8–30 days | 138 | 1.76 | 2.24 | 0.874 |
| 31–90 days | 360 | 2.17 | 2.72 | 0.845 |
| 91–180 days | 540 | 2.28 | 2.84 | 0.838 |

*(Delhi shown; every city has its own profile in `models/<city>/metadata.json`.)*

```
reliability = 1 − MAE / (2 × natural variability),  clipped to [0.35, 0.95]
confidence  = classifier_confidence × reliability × 100
```

A plain train/test split would have given only **7** samples for the "1–7 days"
bucket — one single week, which is meaningless. Rolling-origin backtesting
gives 42, so the number can be trusted.

---

## 📸 Charts and Screenshots

### Generated automatically by the notebook

Running `notebook/Smart_Weather_Prediction.ipynb` produces these into `images/`:

| Chart | File | What it shows |
|-------|------|---------------|
| Seasonal cycle | `seasonal_cycle.png` | Every year overlaid — signal vs noise |
| Forecast vs actual | `forecast_vs_actual.png` | Models against the unseen year |
| Model comparison | `model_comparison.png` | RMSE and R² for all six models |
| Error vs horizon | `error_vs_horizon.png` | How error grows with distance |
| Cyclical encoding | `cyclical_encoding.png` | Why sin/cos beats a raw day number |
| Fourier harmonics | `fourier_harmonics.png` | The seasonal basis functions |
| Correlation heatmap | `correlation_heatmap.png` | How parameters relate |
| Rain box plots | `rain_boxplots.png` | Rainy vs dry conditions |
| Monthly rainfall | `monthly_rainfall.png` | The monsoon, per city |
| Class balance | `class_balance.png` | Rain-day rate per city |
| Calibration | `calibration.png` | Does "70%" really mean 70%? |
| Feature importance | `feature_importance.png` | What drives the rain decision |
| API/model hand-off | `handoff.png` | Where the routing switches |

### App screenshots

Take these from the running app and drop them into `images/`:

| View | Suggested filename |
|------|--------------------|
| Main prediction screen | `app_prediction.png` |
| Live API route (next 7 days) | `app_api_route.png` |
| Model route (months ahead) | `app_model_route.png` |
| Charts tab | `app_charts.png` |
| Model comparison tab | `app_models.png` |

---

## ⚠️ Known Limitations

Stated plainly, because a project that hides these is not trustworthy:

1. **Visibility is displayed but never used as a model input.** The ERA5
   archive returns `null` for visibility on every historical day, so there is
   nothing to train on. The live API does provide it, so the app shows it for
   near-term dates. Inventing a decade of fake visibility to satisfy a
   requirement would have been worse than the honest gap.
2. **Long-range forecasts are seasonal expectations, not real forecasts.**
   Beyond ~2 weeks the app tells you what is *normal* for that date, adjusted
   for trend. The UI says so explicitly whenever the date is more than 30 days
   out.
3. **Six cities only.** Adding more is a three-line edit in `utils/config.py`
   followed by `python train.py`.
4. **Daily resolution.** No hourly forecasts, and no distinction between a
   drizzle and a downpour — only "rain day" vs "not a rain day" at the 1 mm
   threshold.
5. **OpenWeatherMap's free tier gives 5 days**, not 7. With no key, Open-Meteo
   covers 16 days. The routing boundary (`API_FORECAST_DAYS = 7`) is
   configurable in `utils/config.py`.

---

## 🔮 Future Improvements

- **Hourly forecasts** instead of daily aggregates
- **Rainfall amount** (regression) as well as rain/no-rain
- **More cities**, auto-geocoded from any place name
- **Ensemble the models** rather than picking a single winner — averaging
  Prophet and ARIMA would likely beat both
- **SHAP values** for per-prediction explanations
- **Hyperparameter tuning** with Optuna
- **Extreme-weather alerts** (heatwave, heavy rainfall warnings)
- **Deploy to Streamlit Community Cloud** for a public URL
- **Retrain on a schedule** so the models never go stale

---

## 👤 Author

Built as a Machine Learning + Time Series Forecasting project.

**Concepts demonstrated:** time-series forecasting, rolling-origin backtesting,
model comparison and selection, baseline-relative evaluation, probability
calibration, API integration with graceful degradation, clean modular
architecture, and honest reporting of model limitations.

---

## 📄 License

Free to use for educational purposes.

Weather data: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0),
ERA5 reanalysis by [Copernicus / ECMWF](https://climate.copernicus.eu/).
