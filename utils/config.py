"""
utils/config.py
===============

Central configuration for the **Smart Weather Prediction System**.

Every path, city, column name and tunable constant lives here, so no other
module has to hard-code them. If you want to add a city or change the forecast
horizon, this is the only file you need to edit.
"""

import os

# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebook")

RAW_HISTORY_FILE = os.path.join(RAW_DATA_DIR, "weather_history.csv")
PROCESSED_HISTORY_FILE = os.path.join(PROCESSED_DATA_DIR, "weather_clean.csv")
MODEL_COMPARISON_FILE = os.path.join(MODELS_DIR, "model_comparison.csv")
RAIN_COMPARISON_FILE = os.path.join(MODELS_DIR, "rain_model_comparison.csv")
TRAINING_SUMMARY_FILE = os.path.join(MODELS_DIR, "training_summary.json")


def ensure_directories():
    """Create every project folder if it does not already exist."""
    for folder in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
                   IMAGES_DIR, NOTEBOOK_DIR):
        os.makedirs(folder, exist_ok=True)


def city_model_dir(city):
    """Return (and create) the folder that holds one city's saved models."""
    folder = os.path.join(MODELS_DIR, city.replace(" ", "_").lower())
    os.makedirs(folder, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------
# To add a city: copy a line and fill in its latitude, longitude and timezone.
# Nothing else in the project needs to change - re-run `python train.py`.

CITIES = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "timezone": "Asia/Kolkata",
                  "country": "India"},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "timezone": "Asia/Kolkata",
                  "country": "India"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "timezone": "Asia/Kolkata",
                  "country": "India"},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707, "timezone": "Asia/Kolkata",
                  "country": "India"},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639, "timezone": "Asia/Kolkata",
                  "country": "India"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "timezone": "Asia/Kolkata",
                  "country": "India"},
}

DEFAULT_CITY = "Delhi"
CITY_NAMES = list(CITIES.keys())


# ---------------------------------------------------------------------------
# Historical data settings
# ---------------------------------------------------------------------------

HISTORY_YEARS = 10          # how many years of history to download
ARCHIVE_LAG_DAYS = 6        # the ERA5 archive is ~5 days behind real time

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

DATE_COLUMN = "date"
CITY_COLUMN = "city"

# The weather parameters this project forecasts and displays.
TEMPERATURE = "temperature"
HUMIDITY = "humidity"
PRESSURE = "pressure"
WIND_SPEED = "wind_speed"
CLOUD_COVER = "cloud_cover"
DEW_POINT = "dew_point"
PRECIPITATION = "precipitation"
VISIBILITY = "visibility"
RAIN = "rain"

# Parameters we build a time-series forecaster for.
FORECAST_TARGETS = [TEMPERATURE, HUMIDITY, PRESSURE, WIND_SPEED, CLOUD_COVER]

# The headline target used for the five-model comparison (MAE / RMSE / R2).
PRIMARY_TARGET = TEMPERATURE

# Inputs to the Rain / No Rain classifier.
#
# NOTE: `visibility` is deliberately NOT in this list. The ERA5 historical
# archive does not contain visibility (it returns null for every past day), so
# there is nothing to train on. We still *display* visibility whenever the live
# API provides it - see README, "Known limitations".
RAIN_FEATURES = [
    TEMPERATURE,
    HUMIDITY,
    PRESSURE,
    WIND_SPEED,
    CLOUD_COVER,
    DEW_POINT,
]

# A day counts as "Rain" when at least this much precipitation falls.
# 1.0 mm is a common meteorological threshold for a "rain day".
RAIN_THRESHOLD_MM = 1.0


# ---------------------------------------------------------------------------
# Forecast routing (Requirement 14)
# ---------------------------------------------------------------------------
# Dates within this many days of today are served by the live weather API.
# Anything further ahead is served by the trained time-series models.

API_FORECAST_DAYS = 7

# The longest future date the app will let the user pick.
MAX_FORECAST_DAYS = 365


# ---------------------------------------------------------------------------
# Model training settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_DAYS = 365             # final year held out for the backtest
FOURIER_ORDER = 4           # harmonics used to encode the yearly cycle

# LSTM (PyTorch)
LSTM_LOOKBACK = 30          # days of history fed into the network
LSTM_HIDDEN_SIZE = 48
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 64
LSTM_LEARNING_RATE = 0.005

# ARIMA / SARIMAX
ARIMA_ORDER = (2, 0, 2)     # (p, d, q); yearly season handled by Fourier exog


# ---------------------------------------------------------------------------
# Weather API settings
# ---------------------------------------------------------------------------
# OpenWeatherMap needs a free API key. Set it as an environment variable:
#
#     Windows :  setx OWM_API_KEY "your_key_here"
#     Linux   :  export OWM_API_KEY="your_key_here"
#
# If no key is found, the project automatically falls back to Open-Meteo,
# which is free and needs no key at all - so the app always works.

OWM_API_KEY = os.environ.get("OWM_API_KEY", "").strip()
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

API_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------

UNITS = {
    TEMPERATURE: "°C",
    HUMIDITY: "%",
    PRESSURE: "hPa",
    WIND_SPEED: "km/h",
    CLOUD_COVER: "%",
    DEW_POINT: "°C",
    VISIBILITY: "km",
    PRECIPITATION: "mm",
}

PRETTY_NAMES = {
    TEMPERATURE: "Temperature",
    HUMIDITY: "Humidity",
    PRESSURE: "Pressure",
    WIND_SPEED: "Wind Speed",
    CLOUD_COVER: "Cloud Cover",
    DEW_POINT: "Dew Point",
    VISIBILITY: "Visibility",
    PRECIPITATION: "Precipitation",
}

# Colour palette used by every chart, so the whole app looks like one system.
COLORS = {
    "primary": "#2563eb",
    "secondary": "#0891b2",
    "warm": "#ea580c",
    "cool": "#0ea5e9",
    "rain": "#1e40af",
    "no_rain": "#f59e0b",
    "grid": "#94a3b8",
    "forecast": "#7c3aed",
}
