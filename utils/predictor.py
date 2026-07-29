r"""
utils/predictor.py
==================

**Prediction module** - the single entry point the Streamlit app calls.

This is where Requirement 14 lives: the app never decides *how* to answer a
question, it just asks `predict_weather(city, date)` and this module works out
the right route.

```
                        user picks a city and a date
                                    |
                    how many days ahead is that date?
                       /                          \
        0 to 7 days ahead                    more than 7 days
                |                                    |
     LIVE WEATHER API                    TRAINED TIME-SERIES MODELS
  (OpenWeatherMap / Open-Meteo)     (Prophet / LSTM / ARIMA / RF / XGB)
                \                                    /
                 ----------> weather parameters <----
                                    |
                      Rain classifier -> rain probability
                                    |
                   Rain / No Rain  +  confidence score
```

Both routes end at exactly the same place, so the app displays the same set of
numbers no matter which path produced them.
"""

import datetime as dt

import numpy as np
import pandas as pd

from utils import config
from utils.forecasting import CityForecaster
from weather_api import WeatherAPIError, get_provider

# How much to trust the live API itself. Real forecast skill decays with
# distance, so day 0 is nearly certain and day 7 noticeably less so.
API_RELIABILITY_BY_DAY = {
    0: 0.97, 1: 0.95, 2: 0.93, 3: 0.90, 4: 0.87, 5: 0.84, 6: 0.81, 7: 0.78,
}

SOURCE_API = "Live Weather API"
SOURCE_MODEL = "Time-Series Model"


def days_until(target_date, today=None):
    """How many days ahead the requested date is (0 = today, negative = past)."""
    today = today or dt.date.today()
    if isinstance(target_date, (dt.datetime, pd.Timestamp)):
        target_date = target_date.date()
    return (target_date - today).days


def choose_route(target_date, today=None,
                 api_days=config.API_FORECAST_DAYS):
    """Decide which engine answers this date. Returns 'api' or 'model'."""
    ahead = days_until(target_date, today)
    if 0 <= ahead <= api_days:
        return "api"
    return "model"


# ===========================================================================
# The main entry point
# ===========================================================================

def predict_weather(city, target_date, forecaster=None, provider=None,
                    today=None):
    """Predict the weather for one city on one date.

    Returns a dictionary with every field the app displays:

        route, source, days_ahead, date, city
        temperature, humidity, pressure, wind_speed, cloud_cover,
        visibility, dew_point
        rain_probability (0-100), rain_label, confidence (0-100)
        normal_temperature, historical_rain_chance
        model_used, warnings
    """
    forecaster = forecaster or CityForecaster(city)
    ahead = days_until(target_date, today)
    target_timestamp = pd.Timestamp(target_date)

    warnings_list = []

    if ahead < 0:
        warnings_list.append(
            "That date is in the past. Showing the model's reconstruction, "
            "not a real observation."
        )

    route = choose_route(target_date, today)

    if route == "api":
        result = _predict_via_api(
            city, target_timestamp, ahead, forecaster, provider, warnings_list
        )
    else:
        result = _predict_via_model(
            city, target_timestamp, ahead, forecaster, warnings_list
        )

    # ---- context that both routes share --------------------------------
    result["city"] = city
    result["date"] = target_timestamp
    result["days_ahead"] = ahead
    result["normal_temperature"] = float(
        forecaster.climatology_value(config.TEMPERATURE, [target_timestamp])[0]
    )
    result["historical_rain_chance"] = float(
        forecaster.historical_rain_chance([target_timestamp])[0]
    )
    result["warnings"] = warnings_list
    return result


# ===========================================================================
# Route 1 - live weather API (0 to 7 days ahead)
# ===========================================================================

def _predict_via_api(city, target_timestamp, ahead, forecaster, provider,
                     warnings_list):
    """Fetch the real forecast, then run it through the rain classifier."""
    provider = provider or get_provider()

    try:
        forecast = provider.get_daily_forecast(
            city, days=max(ahead + 1, config.API_FORECAST_DAYS)
        )
    except WeatherAPIError as error:
        warnings_list.append(
            f"The live weather API could not be reached ({error}). "
            "Falling back to the trained time-series model."
        )
        return _predict_via_model(
            city, target_timestamp, ahead, forecaster, warnings_list
        )

    forecast[config.DATE_COLUMN] = pd.to_datetime(
        forecast[config.DATE_COLUMN]
    ).dt.normalize()
    match = forecast[forecast[config.DATE_COLUMN] == target_timestamp.normalize()]

    if match.empty:
        warnings_list.append(
            "The API forecast did not cover that date, so the trained "
            "time-series model was used instead."
        )
        return _predict_via_model(
            city, target_timestamp, ahead, forecaster, warnings_list
        )

    row = match.iloc[0]
    values = {
        parameter: (float(row[parameter])
                    if parameter in row and pd.notna(row[parameter]) else None)
        for parameter in (config.TEMPERATURE, config.HUMIDITY, config.PRESSURE,
                          config.WIND_SPEED, config.CLOUD_COVER,
                          config.VISIBILITY, config.DEW_POINT)
    }

    # Run the API's numbers through OUR trained classifier - this is the
    # "pass these values to the trained ML model" step of the brief.
    classifier_input = pd.DataFrame([{
        config.DATE_COLUMN: target_timestamp,
        **{k: v for k, v in values.items() if k in config.RAIN_FEATURES},
    }])
    model_probability = float(forecaster.predict_rain(classifier_input)[0])

    # The API also publishes its own rain probability. Averaging the two is
    # more reliable than trusting either alone (a simple ensemble).
    api_probability = (float(row["rain_chance"]) / 100.0
                       if "rain_chance" in row and pd.notna(row["rain_chance"])
                       else None)

    if api_probability is not None:
        rain_probability = 0.5 * model_probability + 0.5 * api_probability
        blend_note = "average of the API's probability and our classifier"
    else:
        rain_probability = model_probability
        blend_note = "our trained classifier"

    reliability = API_RELIABILITY_BY_DAY.get(max(0, min(ahead, 7)), 0.78)
    classifier_confidence = max(rain_probability, 1.0 - rain_probability)

    return {
        "route": "api",
        "source": f"{SOURCE_API} ({row.get('source', provider.name)})",
        **values,
        "rain_probability": round(rain_probability * 100, 1),
        "rain_label": "Rain" if rain_probability >= 0.5 else "No Rain",
        "confidence": round(classifier_confidence * reliability * 100, 1),
        "model_used": f"{forecaster.rain_model_name} classifier "
                      f"({blend_note})",
        "description": row.get("description", ""),
        "precipitation": (float(row[config.PRECIPITATION])
                          if config.PRECIPITATION in row else None),
    }


# ===========================================================================
# Route 2 - trained time-series models (beyond the API range)
# ===========================================================================

def _predict_via_model(city, target_timestamp, ahead, forecaster,
                       warnings_list):
    """Forecast the parameters with the trained models, then classify rain."""
    parameters = forecaster.forecast_parameters([target_timestamp])
    row = parameters.iloc[0]

    values = {
        parameter: (float(row[parameter]) if parameter in row else None)
        for parameter in (config.TEMPERATURE, config.HUMIDITY, config.PRESSURE,
                          config.WIND_SPEED, config.CLOUD_COVER,
                          config.DEW_POINT)
    }
    # Visibility has no historical record to train on (see README), so the
    # long-range route cannot forecast it.
    values[config.VISIBILITY] = None

    rain_probability = float(forecaster.predict_rain(parameters)[0])

    reliability = forecaster.horizon_reliability(max(ahead, 1))
    classifier_confidence = max(rain_probability, 1.0 - rain_probability)

    if ahead > 30:
        warnings_list.append(
            f"This date is {ahead} days away. Beyond about 2 weeks no model "
            "can predict individual weather systems - what you are seeing is "
            "the expected seasonal pattern for that date, not a real forecast."
        )

    return {
        "route": "model",
        "source": f"{SOURCE_MODEL} ({forecaster.best_model_name})",
        **values,
        "rain_probability": round(rain_probability * 100, 1),
        "rain_label": "Rain" if rain_probability >= 0.5 else "No Rain",
        "confidence": round(classifier_confidence * reliability * 100, 1),
        "model_used": (f"{forecaster.best_model_name} forecaster + "
                       f"{forecaster.rain_model_name} classifier"),
        "description": "",
        "precipitation": None,
    }


# ===========================================================================
# Multi-day helpers (used by the charts)
# ===========================================================================

def forecast_series(city, start_date, days, forecaster=None, provider=None,
                    today=None):
    """Build a day-by-day forecast table, routing each day appropriately.

    Used for the "next N days" charts: the first few days come from the live
    API and the rest from the trained models, joined into one continuous table.
    """
    forecaster = forecaster or CityForecaster(city)
    provider = provider or get_provider()
    today = today or dt.date.today()

    dates = pd.date_range(pd.Timestamp(start_date), periods=days, freq="D")

    api_rows, model_dates = {}, []
    api_limit = min(config.API_FORECAST_DAYS, provider.max_forecast_days)

    # --- 1. Ask the API once for everything it can cover ----------------
    api_dates = [d for d in dates if 0 <= days_until(d, today) <= api_limit]
    if api_dates:
        try:
            api_forecast = provider.get_daily_forecast(city, days=api_limit + 1)
            api_forecast[config.DATE_COLUMN] = pd.to_datetime(
                api_forecast[config.DATE_COLUMN]
            ).dt.normalize()
            api_rows = {
                row[config.DATE_COLUMN]: row
                for _, row in api_forecast.iterrows()
            }
        except WeatherAPIError:
            api_rows = {}

    # --- 2. Everything the API did not cover goes to the models ---------
    model_dates = [d for d in dates if d.normalize() not in api_rows]
    model_frame = (forecaster.forecast_parameters(model_dates)
                   if model_dates else pd.DataFrame())
    model_lookup = {
        pd.Timestamp(row[config.DATE_COLUMN]).normalize(): row
        for _, row in model_frame.iterrows()
    } if not model_frame.empty else {}

    # --- 3. Stitch the two sources into one table -----------------------
    records = []
    for date in dates:
        key = date.normalize()
        if key in api_rows:
            row, source = api_rows[key], SOURCE_API
        elif key in model_lookup:
            row, source = model_lookup[key], SOURCE_MODEL
        else:
            continue

        record = {config.DATE_COLUMN: date, "source": source}
        for parameter in (config.TEMPERATURE, config.HUMIDITY, config.PRESSURE,
                          config.WIND_SPEED, config.CLOUD_COVER,
                          config.DEW_POINT, config.VISIBILITY):
            record[parameter] = (float(row[parameter])
                                 if parameter in row and pd.notna(row[parameter])
                                 else np.nan)
        record["api_rain_chance"] = (
            float(row["rain_chance"])
            if "rain_chance" in row and pd.notna(row["rain_chance"]) else np.nan
        )
        records.append(record)

    table = pd.DataFrame(records)
    if table.empty:
        return table

    # --- 4. One rain probability for every day --------------------------
    model_probability = forecaster.predict_rain(table) * 100
    table["model_rain_chance"] = model_probability
    table["rain_probability"] = np.where(
        table["api_rain_chance"].notna(),
        0.5 * table["api_rain_chance"] + 0.5 * model_probability,
        model_probability,
    )
    table["rain_label"] = np.where(
        table["rain_probability"] >= 50, "Rain", "No Rain"
    )
    table["normal_temperature"] = forecaster.climatology_value(
        config.TEMPERATURE, table[config.DATE_COLUMN]
    )
    table["historical_rain_chance"] = forecaster.historical_rain_chance(
        table[config.DATE_COLUMN]
    )
    return table
