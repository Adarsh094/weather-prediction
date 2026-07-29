"""
utils/preprocessing.py
======================

**Data Preprocessing module.**

Turns the raw download into a clean, model-ready table:

1. Sort by city and date, and make sure no calendar day is missing.
2. Fill gaps (time-series data must never have holes - a model that sees a
   jump from 3 March to 9 March learns the wrong thing).
3. Remove duplicate dates.
4. Cap physically impossible sensor values.
5. Build the binary `rain` target from the precipitation column.
"""

import numpy as np
import pandas as pd

from utils import config

# Hard physical limits. Anything outside these is a data error, not weather.
VALID_RANGES = {
    config.TEMPERATURE: (-60.0, 60.0),
    config.HUMIDITY: (0.0, 100.0),
    config.PRESSURE: (870.0, 1085.0),      # world record low/high, sea level
    config.WIND_SPEED: (0.0, 250.0),
    config.CLOUD_COVER: (0.0, 100.0),
    config.DEW_POINT: (-60.0, 40.0),
    config.PRECIPITATION: (0.0, 2000.0),
}


def reindex_to_daily(city_frame):
    """Insert a row for every missing calendar day, so the series is regular."""
    city_frame = city_frame.sort_values(config.DATE_COLUMN)
    full_index = pd.date_range(
        city_frame[config.DATE_COLUMN].min(),
        city_frame[config.DATE_COLUMN].max(),
        freq="D",
    )
    city_frame = (
        city_frame.set_index(config.DATE_COLUMN)
        .reindex(full_index)
        .rename_axis(config.DATE_COLUMN)
        .reset_index()
    )
    return city_frame


def fill_missing_values(city_frame):
    """Fill gaps in a way that respects the time ordering.

    * Numeric columns are interpolated in time (a missing Tuesday becomes the
      average of Monday and Wednesday), which is far more sensible than the
      column median for a time series.
    * Any remaining gap at the very start or end is filled forwards/backwards.
    * `city` is a text column, so it is simply forward/back filled.
    """
    city_frame[config.CITY_COLUMN] = (
        city_frame[config.CITY_COLUMN].ffill().bfill()
    )

    numeric_columns = city_frame.select_dtypes(include="number").columns
    city_frame[numeric_columns] = (
        city_frame[numeric_columns]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    return city_frame


def clip_to_physical_ranges(data):
    """Cap values that are physically impossible (bad sensor readings)."""
    report = {}
    for column, (low, high) in VALID_RANGES.items():
        if column not in data.columns:
            continue
        bad = int(((data[column] < low) | (data[column] > high)).sum())
        if bad:
            report[column] = bad
        data[column] = data[column].clip(low, high)
    return data, report


def add_rain_target(data):
    """Create the binary target: 1 = Rain, 0 = No Rain.

    A day counts as a rain day when at least `RAIN_THRESHOLD_MM` of
    precipitation falls. Using a threshold instead of `> 0` avoids labelling a
    trace of 0.1 mm - which nobody would call rain - as a rainy day.
    """
    data[config.RAIN] = (
        data[config.PRECIPITATION] >= config.RAIN_THRESHOLD_MM
    ).astype(int)
    return data


def clean_history(raw_data, verbose=True):
    """Run the whole preprocessing pipeline and return the clean DataFrame."""
    if verbose:
        print("Preprocessing")
        print(f"   rows in       : {len(raw_data):,}")

    data = raw_data.copy()
    data[config.DATE_COLUMN] = pd.to_datetime(data[config.DATE_COLUMN])

    # --- 1. Duplicates -----------------------------------------------------
    before = len(data)
    data = data.drop_duplicates(
        subset=[config.CITY_COLUMN, config.DATE_COLUMN], keep="last"
    )
    duplicates_removed = before - len(data)

    # --- 2. Regular daily index + gap filling, per city --------------------
    cleaned_cities = []
    total_gaps = 0
    for city, city_frame in data.groupby(config.CITY_COLUMN, sort=False):
        expanded = reindex_to_daily(city_frame)
        total_gaps += int(expanded[config.TEMPERATURE].isna().sum())
        cleaned_cities.append(fill_missing_values(expanded))

    data = pd.concat(cleaned_cities, ignore_index=True)

    # --- 3. Physical range check ------------------------------------------
    data, clip_report = clip_to_physical_ranges(data)

    # --- 4. Target variable ------------------------------------------------
    data = add_rain_target(data)

    # --- 5. Tidy up --------------------------------------------------------
    data = data.sort_values(
        [config.CITY_COLUMN, config.DATE_COLUMN]
    ).reset_index(drop=True)

    if verbose:
        print(f"   duplicates    : {duplicates_removed} removed")
        print(f"   missing days  : {total_gaps} filled by time interpolation")
        print(f"   clipped values: {clip_report if clip_report else 'none'}")
        print(f"   rows out      : {len(data):,}")
        rain_rate = data[config.RAIN].mean() * 100
        print(f"   rain days     : {data[config.RAIN].sum():,} "
              f"({rain_rate:.1f}% of all days)")

    return data


def save_clean_history(data):
    """Write the cleaned table to `data/processed/weather_clean.csv`."""
    config.ensure_directories()
    data.to_csv(config.PROCESSED_HISTORY_FILE, index=False)
    print(f"   saved         : {config.PROCESSED_HISTORY_FILE}")
    return config.PROCESSED_HISTORY_FILE


def summarise(data):
    """Return a small per-city summary table - handy for the notebook/README."""
    rows = []
    for city, frame in data.groupby(config.CITY_COLUMN, sort=False):
        rows.append({
            "City": city,
            "Days": len(frame),
            "From": frame[config.DATE_COLUMN].min().date(),
            "To": frame[config.DATE_COLUMN].max().date(),
            "Avg Temp (°C)": round(frame[config.TEMPERATURE].mean(), 1),
            "Avg Humidity (%)": round(frame[config.HUMIDITY].mean(), 1),
            "Rain Days (%)": round(frame[config.RAIN].mean() * 100, 1),
            "Annual Rain (mm)": round(
                frame[config.PRECIPITATION].sum() / (len(frame) / 365.25)
            ),
        })
    return pd.DataFrame(rows).set_index("City")


def train_test_split_by_time(frame, test_days=config.TEST_DAYS):
    """Split a single city's series chronologically.

    A random split is **wrong** for time series: it would let the model peek at
    the future to predict the past. We always keep the last `test_days` as the
    unseen test period.
    """
    frame = frame.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    if len(frame) <= test_days + 365:
        raise ValueError(
            f"Not enough history ({len(frame)} days) for a {test_days}-day test set."
        )
    split_at = len(frame) - test_days
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def dew_point_from(temperature, humidity):
    """Magnus-Tetens dew point, used when an API does not provide it."""
    humidity = np.clip(humidity, 1.0, 100.0)
    a, b = 17.625, 243.04
    alpha = np.log(humidity / 100.0) + (a * temperature) / (b + temperature)
    return (b * alpha) / (a - alpha)
