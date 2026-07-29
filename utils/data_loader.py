"""
utils/data_loader.py
====================

**Data Collection module.**

Downloads real historical daily weather for every configured city from the
free **Open-Meteo ERA5 archive** and stores it in `data/raw/weather_history.csv`.

Why Open-Meteo and not Kaggle or OpenWeatherMap?

* It is genuinely free and needs **no API key**, so the project runs for anyone
  who clones it.
* It exposes **ERA5 reanalysis** data going back to 1940, which is the same
  dataset professional meteorologists use for historical studies.
* OpenWeatherMap's free tier does *not* include historical data - only the
  current weather and a short forecast. We use OpenWeatherMap for the live
  forecast instead (see `weather_api.py`).

Run it directly to (re)download everything:

    python -m utils.data_loader
"""

import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from utils import config

# The daily variables we ask the archive for, and the friendly names we map
# them to. Keeping this mapping in one place means the rest of the project
# never has to know Open-Meteo's naming scheme.
ARCHIVE_VARIABLES = {
    "temperature_2m_mean": config.TEMPERATURE,
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "relative_humidity_2m_mean": config.HUMIDITY,
    "pressure_msl_mean": config.PRESSURE,
    "wind_speed_10m_mean": config.WIND_SPEED,
    "wind_speed_10m_max": "wind_gust",
    "cloud_cover_mean": config.CLOUD_COVER,
    "dew_point_2m_mean": config.DEW_POINT,
    "precipitation_sum": config.PRECIPITATION,
    "sunshine_duration": "sunshine_seconds",
}


def _call_api(url, params, retries=5):
    """Send a GET request and return the decoded JSON, retrying on failure.

    Free APIs rate-limit heavy users. When the server answers **429 Too Many
    Requests** we back off for much longer than for an ordinary network blip,
    because retrying quickly would just get us blocked again.
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}"

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(
                full_url, timeout=config.API_TIMEOUT_SECONDS
            ) as response:
                return json.loads(response.read())

        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retries:
                # Rate limited: wait 20s, 40s, 60s, 80s ...
                wait_seconds = 20 * attempt
                print(f"\n      rate limited (429); waiting {wait_seconds}s "
                      f"before retry {attempt + 1}/{retries}", flush=True)
                time.sleep(wait_seconds)
                continue
            if attempt == retries:
                raise RuntimeError(
                    f"Weather API request failed after {retries} attempts.\n"
                    f"URL: {full_url}\nError: {error}"
                ) from error
            time.sleep(3 * attempt)

        except Exception as error:            # noqa: BLE001 - retry on anything
            if attempt == retries:
                raise RuntimeError(
                    f"Weather API request failed after {retries} attempts.\n"
                    f"URL: {full_url}\nError: {error}"
                ) from error
            wait_seconds = 3 * attempt
            print(f"\n      request failed ({error}); retrying in "
                  f"{wait_seconds}s", flush=True)
            time.sleep(wait_seconds)

    return None  # pragma: no cover - unreachable, keeps linters happy


def _city_cache_file(city):
    """Where one city's download is cached, so a failed run can resume."""
    safe_name = city.replace(" ", "_").lower()
    return os.path.join(config.RAW_DATA_DIR, f"city_{safe_name}.csv")


def date_range_for_history(years=config.HISTORY_YEARS):
    """Return the (start_date, end_date) strings for the historical download.

    The ERA5 archive lags real time by about five days, so we stop a few days
    before today instead of asking for data that does not exist yet.
    """
    end_date = dt.date.today() - dt.timedelta(days=config.ARCHIVE_LAG_DAYS)
    start_date = end_date - dt.timedelta(days=int(round(365.25 * years)))
    return start_date.isoformat(), end_date.isoformat()


def download_city_history(city, years=config.HISTORY_YEARS, use_cache=True):
    """Download one city's daily weather history as a tidy DataFrame.

    Each city is cached to its own CSV. If the download is interrupted - a
    dropped connection, a rate limit - re-running the script picks up where it
    left off instead of starting from scratch.
    """
    if city not in config.CITIES:
        raise KeyError(f"Unknown city '{city}'. Known cities: {config.CITY_NAMES}")

    config.ensure_directories()
    cache_file = _city_cache_file(city)

    if use_cache and os.path.exists(cache_file):
        cached = pd.read_csv(cache_file, parse_dates=[config.DATE_COLUMN])
        print(f"   {city:<12} loaded from cache ({len(cached)} days)")
        return cached

    place = config.CITIES[city]
    start_date, end_date = date_range_for_history(years)

    params = {
        "latitude": place["lat"],
        "longitude": place["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(ARCHIVE_VARIABLES.keys()),
        "timezone": place["timezone"],
    }

    print(f"   {city:<12} {start_date} -> {end_date} ...", end="", flush=True)
    payload = _call_api(config.OPEN_METEO_ARCHIVE_URL, params)

    daily = payload["daily"]
    frame = pd.DataFrame(daily)
    frame = frame.rename(columns={"time": config.DATE_COLUMN, **ARCHIVE_VARIABLES})
    frame[config.DATE_COLUMN] = pd.to_datetime(frame[config.DATE_COLUMN])
    frame.insert(0, config.CITY_COLUMN, city)

    frame.to_csv(cache_file, index=False)          # cache for a later resume
    print(f" {len(frame)} days")
    return frame


def download_all_cities(years=config.HISTORY_YEARS, save=True, use_cache=True):
    """Download the history for every configured city and stack it together."""
    config.ensure_directories()

    print(f"Downloading {years} years of daily weather for "
          f"{len(config.CITIES)} cities from the Open-Meteo ERA5 archive")

    frames = []
    for position, city in enumerate(config.CITIES):
        frames.append(download_city_history(city, years, use_cache=use_cache))
        if position < len(config.CITIES) - 1:
            time.sleep(5)        # stay well inside the free API's rate limit

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(
        [config.CITY_COLUMN, config.DATE_COLUMN]
    ).reset_index(drop=True)

    if save:
        data.to_csv(config.RAW_HISTORY_FILE, index=False)
        size_mb = os.path.getsize(config.RAW_HISTORY_FILE) / (1024 * 1024)
        print(f"\nSaved {len(data):,} rows to {config.RAW_HISTORY_FILE} "
              f"({size_mb:.1f} MB)")

    return data


def load_raw_history(download_if_missing=True):
    """Read `data/raw/weather_history.csv`, downloading it first if needed."""
    if not os.path.exists(config.RAW_HISTORY_FILE):
        if not download_if_missing:
            raise FileNotFoundError(
                f"{config.RAW_HISTORY_FILE} not found. "
                "Run 'python -m utils.data_loader' to download it."
            )
        return download_all_cities()

    data = pd.read_csv(config.RAW_HISTORY_FILE, parse_dates=[config.DATE_COLUMN])
    return data


def load_clean_history():
    """Read the cleaned dataset produced by `utils.preprocessing`."""
    if not os.path.exists(config.PROCESSED_HISTORY_FILE):
        raise FileNotFoundError(
            f"{config.PROCESSED_HISTORY_FILE} not found. Run 'python train.py' "
            "(or the preprocessing step) first."
        )
    return pd.read_csv(
        config.PROCESSED_HISTORY_FILE, parse_dates=[config.DATE_COLUMN]
    )


if __name__ == "__main__":
    download_all_cities()
