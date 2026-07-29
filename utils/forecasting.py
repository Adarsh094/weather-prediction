"""
utils/forecasting.py
====================

**Forecasting module.**

Loads the models saved by `train.py` and turns them into actual future weather
values for any date the user asks for.

One `CityForecaster` object holds everything needed for one city:

* a forecaster per weather parameter (temperature, humidity, pressure, ...)
* the Rain / No Rain classifier
* the city's climatology (the 10-year "normal" for every day of the year)
* the backtest results, which tell us how much to trust a given horizon
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from utils import config, features

FORECASTER_FILE = "forecaster_{target}.pkl"
RAIN_CLASSIFIER_FILE = "rain_classifier.pkl"
CLIMATOLOGY_FILE = "climatology.pkl"
METADATA_FILE = "metadata.json"


# ===========================================================================
# Saving (called by train.py)
# ===========================================================================

# Joblib compression level for saved models. A Random Forest pickle is mostly
# repeated tree structure, so zlib shrinks it by about 75% with no change to
# the predictions at all - it just takes a moment longer to load. That matters
# when the models have to be deployed to a cloud host.
COMPRESSION = ("zlib", 6)


def save_city_artifacts(city, forecasters, rain_classifier, climatology_data,
                        metadata):
    """Write one city's trained models and reference data to `models/<city>/`."""
    folder = config.city_model_dir(city)

    for target, model in forecasters.items():
        joblib.dump(model,
                    os.path.join(folder, FORECASTER_FILE.format(target=target)),
                    compress=COMPRESSION)

    joblib.dump(rain_classifier, os.path.join(folder, RAIN_CLASSIFIER_FILE),
                compress=COMPRESSION)
    joblib.dump(climatology_data, os.path.join(folder, CLIMATOLOGY_FILE),
                compress=COMPRESSION)

    with open(os.path.join(folder, METADATA_FILE), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return folder


def build_climatology_pack(city_frame):
    """Pre-compute every "what is normal for this date" lookup the app needs."""
    pack = {
        "targets": {
            target: features.climatology(city_frame, target)
            for target in config.FORECAST_TARGETS
        },
        "rain_frequency": features.rain_frequency_by_day(city_frame),
        "monthly_rainfall": features.monthly_rainfall(city_frame),
        "history_start": city_frame[config.DATE_COLUMN].min(),
        "history_end": city_frame[config.DATE_COLUMN].max(),
    }
    return pack


# ===========================================================================
# Loading and using (called by the app)
# ===========================================================================

class CityForecaster:
    """Everything needed to forecast one city, loaded from disk."""

    def __init__(self, city):
        self.city = city
        self.folder = config.city_model_dir(city)

        if not os.path.exists(os.path.join(self.folder, METADATA_FILE)):
            raise FileNotFoundError(
                f"No trained models found for {city}. Run 'python train.py' first."
            )

        with open(os.path.join(self.folder, METADATA_FILE),
                  encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.forecasters = {}
        for target in config.FORECAST_TARGETS:
            path = os.path.join(self.folder,
                                FORECASTER_FILE.format(target=target))
            if os.path.exists(path):
                self.forecasters[target] = joblib.load(path)

        self.rain_classifier = joblib.load(
            os.path.join(self.folder, RAIN_CLASSIFIER_FILE)
        )
        self.climatology = joblib.load(
            os.path.join(self.folder, CLIMATOLOGY_FILE)
        )

    # -- basic information -------------------------------------------------
    @property
    def best_model_name(self):
        return self.metadata.get("best_forecast_model", "unknown")

    @property
    def rain_model_name(self):
        return self.metadata.get("best_rain_model", "unknown")

    @property
    def history_end(self):
        return pd.Timestamp(self.metadata.get("history_end"))

    # -- forecasting weather parameters ------------------------------------
    def forecast_parameters(self, dates):
        """Predict every weather parameter for the given dates.

        Returns a DataFrame with one row per date and one column per parameter.
        """
        index = features._as_datetime_index(dates)
        result = pd.DataFrame({config.DATE_COLUMN: index})

        for target, model in self.forecasters.items():
            values = model.predict(index)
            result[target] = self._apply_physical_limits(target, values)

        # Dew point is derived, not forecast separately - it is a exact
        # function of temperature and humidity.
        if config.TEMPERATURE in result and config.HUMIDITY in result:
            from utils.preprocessing import dew_point_from
            result[config.DEW_POINT] = dew_point_from(
                result[config.TEMPERATURE].to_numpy(),
                result[config.HUMIDITY].to_numpy(),
            )

        return result

    @staticmethod
    def _apply_physical_limits(target, values):
        """Stop a model from predicting 110% humidity or negative wind."""
        limits = {
            config.HUMIDITY: (0.0, 100.0),
            config.CLOUD_COVER: (0.0, 100.0),
            config.WIND_SPEED: (0.0, 200.0),
            config.PRESSURE: (900.0, 1080.0),
            config.TEMPERATURE: (-50.0, 55.0),
        }
        if target in limits:
            low, high = limits[target]
            return np.clip(values, low, high)
        return values

    # -- rain -------------------------------------------------------------
    def predict_rain(self, weather_frame):
        """Turn forecast weather parameters into a rain probability.

        `weather_frame` must contain the columns in `config.RAIN_FEATURES`
        plus a date column (used to work out the season).
        """
        frame = weather_frame.copy()

        # Fill any parameter the caller did not supply with the climatological
        # normal for that date, so the classifier always gets a full row.
        for target in config.RAIN_FEATURES:
            if target not in frame.columns or frame[target].isna().any():
                normal = self.climatology_value(
                    target, frame[config.DATE_COLUMN]
                )
                if target not in frame.columns:
                    frame[target] = normal
                else:
                    frame[target] = frame[target].fillna(pd.Series(normal))

        season = features.seasonal_features(frame[config.DATE_COLUMN])
        model_input = pd.concat(
            [
                frame[config.RAIN_FEATURES].reset_index(drop=True),
                season[["sin_1", "cos_1", "month"]].reset_index(drop=True),
            ],
            axis=1,
        )[features.rain_feature_names()]

        probabilities = self.rain_classifier.predict_proba(model_input)[:, 1]
        return probabilities

    # -- climatology -------------------------------------------------------
    def climatology_value(self, target, dates):
        """The 10-year average value of `target` for those days of the year."""
        index = features._as_datetime_index(dates)
        series = self.climatology["targets"].get(target)
        if series is None:
            return np.full(len(index), np.nan)
        return series.reindex(index.dayofyear).to_numpy(dtype=float)

    def historical_rain_chance(self, dates):
        """How often it has actually rained on this date over the last 10 years."""
        index = features._as_datetime_index(dates)
        return (
            self.climatology["rain_frequency"]
            .reindex(index.dayofyear)
            .to_numpy(dtype=float)
        )

    def monthly_rainfall(self):
        """Average rainfall per calendar month, in mm."""
        return self.climatology["monthly_rainfall"]

    # -- confidence --------------------------------------------------------
    def horizon_reliability(self, days_ahead):
        """How much to trust a forecast this far ahead, as a factor 0-1.

        This is **not** invented. It comes straight from the backtest in
        `train.py`: the best model forecast a full unseen year, and we measured
        its R2 separately for 1-7, 8-30, 31-90 and 91-365 days ahead. Those
        measured R2 values (clipped to a sensible range) become the factor.
        """
        profile = self.metadata.get("horizon_reliability", {})
        for bucket, factor in sorted(
            ((int(k.split("-")[0]), v) for k, v in profile.items()),
            reverse=True,
        ):
            if days_ahead >= bucket:
                return float(factor)
        return float(profile.get("1-7", 0.9)) if profile else 0.9


def load_all_cities(cities=None):
    """Load every city's forecaster. Missing cities are skipped, not fatal."""
    loaded, missing = {}, []
    for city in (cities or config.CITY_NAMES):
        try:
            loaded[city] = CityForecaster(city)
        except (FileNotFoundError, OSError):
            missing.append(city)
    return loaded, missing


def available_cities():
    """List the cities that actually have trained models on disk."""
    ready = []
    for city in config.CITY_NAMES:
        folder = config.city_model_dir(city)
        if os.path.exists(os.path.join(folder, METADATA_FILE)):
            ready.append(city)
    return ready
