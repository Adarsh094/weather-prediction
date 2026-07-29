"""
utils/features.py
=================

**Feature Engineering module.**

This project needs features that work for **any future date**, including dates
months away. That rules out the usual "yesterday's temperature" lag features,
because when you forecast 90 days ahead there is no yesterday to look at.

So we build two different feature sets:

``seasonal features`` (used by Random Forest and XGBoost)
    Calendar-only: where the date sits in the yearly cycle, plus a slow trend.
    These can be computed for **any** date, past or future, which is exactly
    what "predict weather for any future date" requires.

``lag features`` (used for short-range checks and the LSTM)
    Recent history: yesterday, the last week, the last month.

### Why sine and cosine instead of just "day 1 to 365"?

If we fed the model the raw day number, 31 December (day 365) and 1 January
(day 1) would look 364 days apart - even though they are next-door neighbours
with almost identical weather. Projecting the day onto a circle with sine and
cosine fixes this: the two days end up right next to each other.

Adding several *harmonics* (k = 1, 2, 3, 4) lets the model describe a yearly
shape more complex than a single smooth wave - which matters a lot for a
monsoon climate, where the rainy season arrives quite suddenly.
"""

import numpy as np
import pandas as pd

from utils import config

# The reference date that the linear "trend" feature is measured from.
# Fixed so that training and prediction always agree.
TREND_ORIGIN = pd.Timestamp("2000-01-01")


def _as_datetime_index(dates):
    """Accept a Series, DatetimeIndex, list or single date and normalise it."""
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    if isinstance(dates, pd.Series):
        return pd.DatetimeIndex(pd.to_datetime(dates))
    if isinstance(dates, (list, tuple, np.ndarray)):
        return pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DatetimeIndex([pd.to_datetime(dates)])


def fourier_terms(dates, order=config.FOURIER_ORDER):
    """Return the sine/cosine harmonics that describe the yearly cycle."""
    index = _as_datetime_index(dates)
    day_of_year = index.dayofyear.to_numpy(dtype=float)
    # Fraction of the way through the year, 0.0 -> 1.0
    angle = 2.0 * np.pi * day_of_year / 365.25

    terms = {}
    for k in range(1, order + 1):
        terms[f"sin_{k}"] = np.sin(k * angle)
        terms[f"cos_{k}"] = np.cos(k * angle)
    return pd.DataFrame(terms, index=range(len(index)))


def seasonal_features(dates, order=config.FOURIER_ORDER):
    """Build the calendar-only feature table that works for any date.

    Columns
    -------
    sin_k / cos_k : position in the yearly cycle (the seasonal shape)
    trend_years   : years since 2000 - lets the model learn a slow warming
                    or drying trend instead of assuming the climate is static
    month         : the calendar month, a coarse extra hint
    day_of_year   : the raw day number, useful for tree splits
    """
    index = _as_datetime_index(dates)

    frame = fourier_terms(index, order=order)
    frame["trend_years"] = (
        (index - TREND_ORIGIN).days.to_numpy(dtype=float) / 365.25
    )
    frame["month"] = index.month.to_numpy(dtype=float)
    frame["day_of_year"] = index.dayofyear.to_numpy(dtype=float)
    return frame


def seasonal_feature_names(order=config.FOURIER_ORDER):
    """The exact column order produced by `seasonal_features`."""
    names = []
    for k in range(1, order + 1):
        names.extend([f"sin_{k}", f"cos_{k}"])
    names.extend(["trend_years", "month", "day_of_year"])
    return names


def add_lag_features(frame, column, lags=(1, 2, 3, 7, 14, 30),
                     rolling_windows=(7, 30)):
    """Add "what happened recently" columns for short-range models.

    Every lag is shifted by at least one day, so a row never contains
    information from its own day or later - that would be **data leakage**.
    """
    frame = frame.sort_values(config.DATE_COLUMN).copy()

    for lag in lags:
        frame[f"{column}_lag_{lag}"] = frame[column].shift(lag)

    for window in rolling_windows:
        # shift(1) first so the window ends *yesterday*, never today.
        frame[f"{column}_roll_mean_{window}"] = (
            frame[column].shift(1).rolling(window, min_periods=1).mean()
        )
        frame[f"{column}_roll_std_{window}"] = (
            frame[column].shift(1).rolling(window, min_periods=2).std()
        )

    return frame


def lag_feature_names(column, lags=(1, 2, 3, 7, 14, 30),
                      rolling_windows=(7, 30)):
    """The column names produced by `add_lag_features`."""
    names = [f"{column}_lag_{lag}" for lag in lags]
    for window in rolling_windows:
        names.extend([
            f"{column}_roll_mean_{window}",
            f"{column}_roll_std_{window}",
        ])
    return names


def build_seasonal_training_set(city_frame, target):
    """Return (X, y) for a seasonal regressor on one city and one parameter."""
    city_frame = city_frame.sort_values(config.DATE_COLUMN)
    features = seasonal_features(city_frame[config.DATE_COLUMN])
    target_values = city_frame[target].to_numpy(dtype=float)
    return features, target_values


def build_rain_training_set(city_frame):
    """Return (X, y) for the Rain / No Rain classifier.

    The classifier answers a different question from the forecasters:
    *given these weather conditions, is it raining?* So its inputs are the
    weather parameters themselves, plus the season (some months are simply
    much wetter than others).
    """
    city_frame = city_frame.sort_values(config.DATE_COLUMN)

    features = city_frame[config.RAIN_FEATURES].reset_index(drop=True)
    season = seasonal_features(city_frame[config.DATE_COLUMN])
    # Only the first harmonic and the month - enough to express "monsoon
    # season" without letting the classifier memorise individual dates.
    season = season[["sin_1", "cos_1", "month"]].reset_index(drop=True)

    features = pd.concat([features, season], axis=1)
    target = city_frame[config.RAIN].to_numpy(dtype=int)
    return features, target


def rain_feature_names():
    """The exact column order the rain classifier expects."""
    return list(config.RAIN_FEATURES) + ["sin_1", "cos_1", "month"]


def climatology(city_frame, target, window_days=7):
    """Average value for each day of the year, smoothed over a window.

    This is the "normal" weather for that date, computed from all 10 years.
    It is the baseline every forecast model has to beat, and the app shows it
    so the user can see how unusual a forecast is.
    """
    frame = city_frame.copy()
    frame["day_of_year"] = frame[config.DATE_COLUMN].dt.dayofyear

    daily_mean = frame.groupby("day_of_year")[target].mean()
    daily_mean = daily_mean.reindex(range(1, 367)).interpolate(
        limit_direction="both"
    )

    # Wrap the series around so 1 January is smoothed with late December.
    tripled = pd.concat([daily_mean, daily_mean, daily_mean])
    smoothed = tripled.rolling(window_days, center=True, min_periods=1).mean()
    smoothed = smoothed.iloc[len(daily_mean):2 * len(daily_mean)]
    smoothed.index = daily_mean.index
    return smoothed


def monthly_rainfall(city_frame):
    """Average total rainfall per calendar month - used by the charts."""
    frame = city_frame.copy()
    frame["month"] = frame[config.DATE_COLUMN].dt.month
    frame["year"] = frame[config.DATE_COLUMN].dt.year

    per_month_per_year = frame.groupby(["year", "month"])[
        config.PRECIPITATION
    ].sum()
    return per_month_per_year.groupby("month").mean()


def rain_frequency_by_day(city_frame, window_days=15):
    """Historical chance of rain for each day of the year, as a percentage."""
    frame = city_frame.copy()
    frame["day_of_year"] = frame[config.DATE_COLUMN].dt.dayofyear

    frequency = frame.groupby("day_of_year")[config.RAIN].mean() * 100
    frequency = frequency.reindex(range(1, 367)).interpolate(
        limit_direction="both"
    )

    tripled = pd.concat([frequency, frequency, frequency])
    smoothed = tripled.rolling(window_days, center=True, min_periods=1).mean()
    smoothed = smoothed.iloc[len(frequency):2 * len(frequency)]
    smoothed.index = frequency.index
    return smoothed
