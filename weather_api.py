"""
weather_api.py
==============

**Weather API module** - the live data layer (Requirement 6).

Two providers are implemented behind one common interface:

``OpenWeatherMapProvider``
    The provider named in the project brief. Needs a free API key in the
    ``OWM_API_KEY`` environment variable. Gives current weather and a 5-day
    forecast (3-hourly), which this module aggregates into daily values.

``OpenMeteoProvider``
    A free, **key-free** fallback so the app works for anyone who clones the
    repository without signing up for anything. Gives current weather and a
    16-day daily forecast.

`get_provider()` picks OpenWeatherMap when a key is present and Open-Meteo
otherwise, so the rest of the project never needs to care which one is active.

Every provider returns the **same normalised units**:

    temperature   deg C     pressure     hPa (sea level)
    humidity      %         wind_speed   km/h
    cloud_cover   %         visibility   km
    precipitation mm        rain_chance  % (0-100, may be None)

Run it directly for a quick manual check:

    python weather_api.py
    python weather_api.py Mumbai
"""

import datetime as dt
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from utils import config
from utils.preprocessing import dew_point_from

# How many times to retry a failed request, and how long to wait between tries.
MAX_RETRIES = 4
RATE_LIMIT_WAIT = 15          # seconds to wait after an HTTP 429
RESPONSE_CACHE_SECONDS = 600  # reuse an API answer for 10 minutes

# Which OpenWeatherMap condition groups mean precipitation is falling.
OWM_RAIN_CODES = ("Rain", "Drizzle", "Thunderstorm", "Snow")

# Open-Meteo WMO weather codes that mean precipitation is falling.
WMO_RAIN_CODES = set(range(51, 68)) | set(range(71, 78)) | set(range(80, 100))

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherAPIError(RuntimeError):
    """Raised when a weather provider cannot be reached or returns nonsense."""


def _round_numeric(frame, decimals=2):
    """Round only the number columns - rounding a date column is meaningless."""
    numeric_columns = frame.select_dtypes(include="number").columns
    frame[numeric_columns] = frame[numeric_columns].round(decimals)
    return frame


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
# The app asks for the same forecast several times in one page load (once for
# the current conditions, once for the prediction, once for the charts).
# Without a cache that is three network round-trips for identical data, which
# is slow and a fast way to get rate-limited. We keep each answer for ten
# minutes. A lock makes it safe when Streamlit re-runs the script in threads.

_RESPONSE_CACHE = {}
_CACHE_LOCK = threading.Lock()


def clear_cache():
    """Forget every cached API response (used by tests and the refresh button)."""
    with _CACHE_LOCK:
        _RESPONSE_CACHE.clear()


def _cache_get(key):
    with _CACHE_LOCK:
        entry = _RESPONSE_CACHE.get(key)
    if entry is None:
        return None
    stored_at, payload = entry
    if time.time() - stored_at > RESPONSE_CACHE_SECONDS:
        return None
    return payload


def _cache_put(key, payload):
    with _CACHE_LOCK:
        _RESPONSE_CACHE[key] = (time.time(), payload)


def _get_json(url, params, timeout=config.API_TIMEOUT_SECONDS, use_cache=True):
    """Fetch a URL and decode the JSON body.

    Network calls fail for boring reasons: a dropped connection, a momentary
    rate limit, a slow response. None of those mean "this city has no weather",
    so we retry with a growing pause instead of giving up on the first error.
    Only when every attempt fails do we raise, and the caller then falls back
    to the trained models.
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}"

    if use_cache:
        cached = _cache_get(full_url)
        if cached is not None:
            return cached

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(full_url, timeout=timeout) as response:
                payload = json.loads(response.read())
            if use_cache:
                _cache_put(full_url, payload)
            return payload

        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", "ignore")[:200]
            except Exception:                               # noqa: BLE001
                pass
            last_error = f"HTTP {error.code}. {detail}".strip()

            # 4xx errors other than "too many requests" are our fault - a bad
            # city, a bad key - so retrying would just waste time.
            if error.code != 429 and 400 <= error.code < 500:
                raise WeatherAPIError(
                    f"Weather API returned {last_error}"
                ) from error

            wait = RATE_LIMIT_WAIT * attempt if error.code == 429 else attempt

        except Exception as error:                          # noqa: BLE001
            # Connection reset, timeout, DNS hiccup - all worth retrying.
            last_error = str(error)
            wait = 1.5 * attempt

        if attempt < MAX_RETRIES:
            time.sleep(wait)

    raise WeatherAPIError(
        f"Could not reach the weather API after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ===========================================================================
# Base provider
# ===========================================================================

class BaseWeatherProvider:
    """Interface every live-weather provider implements."""

    name = "base"
    max_forecast_days = 7
    needs_api_key = False

    def get_current_weather(self, city):
        """Return a dict of the weather right now."""
        raise NotImplementedError

    def get_daily_forecast(self, city, days=7):
        """Return a DataFrame with one row per forecast day."""
        raise NotImplementedError

    @staticmethod
    def _place(city):
        if city not in config.CITIES:
            raise KeyError(
                f"Unknown city '{city}'. Known cities: {config.CITY_NAMES}"
            )
        return config.CITIES[city]


# ===========================================================================
# Provider 1 - OpenWeatherMap (needs a free API key)
# ===========================================================================

class OpenWeatherMapProvider(BaseWeatherProvider):
    """Live weather from OpenWeatherMap.

    Free-tier endpoints used:

    * ``/weather``  - current conditions
    * ``/forecast`` - 5 days ahead in 3-hour steps, which we average into days
    """

    name = "OpenWeatherMap"
    max_forecast_days = 5
    needs_api_key = True

    def __init__(self, api_key=None):
        self.api_key = (api_key or config.OWM_API_KEY).strip()
        if not self.api_key:
            raise WeatherAPIError(
                "No OpenWeatherMap API key found. Set the OWM_API_KEY "
                "environment variable, or use the Open-Meteo provider."
            )

    def get_current_weather(self, city):
        place = self._place(city)
        payload = _get_json(f"{config.OWM_BASE_URL}/weather", {
            "lat": place["lat"],
            "lon": place["lon"],
            "appid": self.api_key,
            "units": "metric",          # gives Celsius and metres per second
        })

        main = payload.get("main", {})
        wind = payload.get("wind", {})
        weather = (payload.get("weather") or [{}])[0]

        temperature = float(main.get("temp", float("nan")))
        humidity = float(main.get("humidity", float("nan")))

        return {
            "city": city,
            "time": dt.datetime.fromtimestamp(payload.get("dt", 0)),
            config.TEMPERATURE: temperature,
            config.HUMIDITY: humidity,
            # Sea-level pressure keeps cities comparable regardless of altitude.
            config.PRESSURE: float(main.get("sea_level") or main.get("pressure")),
            config.WIND_SPEED: float(wind.get("speed", 0.0)) * 3.6,  # m/s->km/h
            config.CLOUD_COVER: float(payload.get("clouds", {}).get("all", 0.0)),
            config.VISIBILITY: float(payload.get("visibility", 10000)) / 1000.0,
            config.DEW_POINT: float(dew_point_from(temperature, humidity)),
            "description": weather.get("description", "").title(),
            "is_raining": weather.get("main") in OWM_RAIN_CODES,
            "source": self.name,
        }

    def get_daily_forecast(self, city, days=5):
        place = self._place(city)
        payload = _get_json(f"{config.OWM_BASE_URL}/forecast", {
            "lat": place["lat"],
            "lon": place["lon"],
            "appid": self.api_key,
            "units": "metric",
        })

        rows = []
        for slot in payload.get("list", []):
            main = slot.get("main", {})
            weather = (slot.get("weather") or [{}])[0]
            rows.append({
                config.DATE_COLUMN: pd.to_datetime(slot["dt_txt"]).normalize(),
                config.TEMPERATURE: main.get("temp"),
                config.HUMIDITY: main.get("humidity"),
                config.PRESSURE: main.get("sea_level") or main.get("pressure"),
                config.WIND_SPEED: slot.get("wind", {}).get("speed", 0.0) * 3.6,
                config.CLOUD_COVER: slot.get("clouds", {}).get("all"),
                config.VISIBILITY: slot.get("visibility", 10000) / 1000.0,
                config.PRECIPITATION: slot.get("rain", {}).get("3h", 0.0),
                "rain_chance": slot.get("pop", 0.0) * 100.0,
                "is_raining": weather.get("main") in OWM_RAIN_CODES,
            })

        if not rows:
            raise WeatherAPIError("OpenWeatherMap returned an empty forecast.")

        slots = pd.DataFrame(rows)

        # Collapse the eight 3-hour slots of each day into one daily row.
        daily = slots.groupby(config.DATE_COLUMN).agg({
            config.TEMPERATURE: "mean",
            config.HUMIDITY: "mean",
            config.PRESSURE: "mean",
            config.WIND_SPEED: "mean",
            config.CLOUD_COVER: "mean",
            config.VISIBILITY: "mean",
            config.PRECIPITATION: "sum",       # total rain over the day
            "rain_chance": "max",              # wettest moment of the day
        }).reset_index()

        daily[config.DEW_POINT] = dew_point_from(
            daily[config.TEMPERATURE], daily[config.HUMIDITY]
        )
        daily["description"] = ""
        daily["source"] = self.name
        return _round_numeric(daily.head(days))


# ===========================================================================
# Provider 2 - Open-Meteo (no API key required)
# ===========================================================================

class OpenMeteoProvider(BaseWeatherProvider):
    """Live weather from Open-Meteo - free, no registration, 16-day forecast."""

    name = "Open-Meteo"
    max_forecast_days = 16
    needs_api_key = False

    CURRENT_FIELDS = [
        "temperature_2m", "relative_humidity_2m", "pressure_msl",
        "wind_speed_10m", "cloud_cover", "visibility", "dew_point_2m",
        "precipitation", "weather_code",
    ]

    DAILY_FIELDS = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "relative_humidity_2m_mean", "pressure_msl_mean",
        "wind_speed_10m_mean", "cloud_cover_mean", "dew_point_2m_mean",
        "precipitation_sum", "precipitation_probability_mean", "weather_code",
    ]

    def get_current_weather(self, city):
        place = self._place(city)
        payload = _get_json(config.OPEN_METEO_FORECAST_URL, {
            "latitude": place["lat"],
            "longitude": place["lon"],
            "current": ",".join(self.CURRENT_FIELDS),
            "timezone": place["timezone"],
        })

        current = payload.get("current", {})
        code = int(current.get("weather_code", 0))
        visibility_m = current.get("visibility")

        return {
            "city": city,
            "time": pd.to_datetime(current.get("time")),
            config.TEMPERATURE: current.get("temperature_2m"),
            config.HUMIDITY: current.get("relative_humidity_2m"),
            config.PRESSURE: current.get("pressure_msl"),
            config.WIND_SPEED: current.get("wind_speed_10m"),
            config.CLOUD_COVER: current.get("cloud_cover"),
            config.VISIBILITY: (visibility_m / 1000.0
                                if visibility_m is not None else None),
            config.DEW_POINT: current.get("dew_point_2m"),
            "description": WMO_DESCRIPTIONS.get(code, "Unknown"),
            "is_raining": code in WMO_RAIN_CODES,
            "source": self.name,
        }

    def get_daily_forecast(self, city, days=7):
        place = self._place(city)
        days = int(max(1, min(days, self.max_forecast_days)))

        # Always request the full horizon, then slice. Every caller therefore
        # builds the *same* URL and shares one cached response, instead of
        # making a fresh network call for 7 days, then 8, then 16.
        payload = _get_json(config.OPEN_METEO_FORECAST_URL, {
            "latitude": place["lat"],
            "longitude": place["lon"],
            "daily": ",".join(self.DAILY_FIELDS),
            "hourly": "visibility",
            "forecast_days": self.max_forecast_days,
            "timezone": place["timezone"],
        })

        daily = payload["daily"]
        frame = pd.DataFrame({
            config.DATE_COLUMN: pd.to_datetime(daily["time"]),
            config.TEMPERATURE: daily["temperature_2m_mean"],
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            config.HUMIDITY: daily["relative_humidity_2m_mean"],
            config.PRESSURE: daily["pressure_msl_mean"],
            config.WIND_SPEED: daily["wind_speed_10m_mean"],
            config.CLOUD_COVER: daily["cloud_cover_mean"],
            config.DEW_POINT: daily["dew_point_2m_mean"],
            config.PRECIPITATION: daily["precipitation_sum"],
            "rain_chance": daily["precipitation_probability_mean"],
            "weather_code": daily["weather_code"],
        })

        # Open-Meteo only reports visibility hourly, so average it per day.
        hourly = payload.get("hourly", {})
        if hourly.get("visibility"):
            visibility = pd.DataFrame({
                config.DATE_COLUMN: pd.to_datetime(hourly["time"]),
                config.VISIBILITY: hourly["visibility"],
            })
            visibility[config.DATE_COLUMN] = visibility[
                config.DATE_COLUMN
            ].dt.normalize()
            daily_visibility = (
                visibility.groupby(config.DATE_COLUMN)[config.VISIBILITY]
                .mean() / 1000.0
            )
            frame[config.VISIBILITY] = frame[config.DATE_COLUMN].map(
                daily_visibility
            )
        else:
            frame[config.VISIBILITY] = None

        frame["description"] = frame["weather_code"].map(
            lambda c: WMO_DESCRIPTIONS.get(int(c), "Unknown")
        )
        frame["source"] = self.name
        return _round_numeric(frame.head(days).copy())


# ===========================================================================
# Provider selection
# ===========================================================================

def get_provider(prefer_openweathermap=True, verbose=False):
    """Return the best available provider.

    OpenWeatherMap is used when an API key is configured (it is the provider
    the project brief asks for). Otherwise we fall back to Open-Meteo so the
    application always works.
    """
    if prefer_openweathermap and config.OWM_API_KEY:
        try:
            provider = OpenWeatherMapProvider()
            if verbose:
                print(f"Using {provider.name} (API key found)")
            return provider
        except WeatherAPIError as error:
            if verbose:
                print(f"OpenWeatherMap unavailable ({error}); "
                      f"falling back to Open-Meteo")

    provider = OpenMeteoProvider()
    if verbose:
        print(f"Using {provider.name} (no API key required)")
    return provider


def provider_status():
    """Describe which provider is active - shown in the app sidebar."""
    if config.OWM_API_KEY:
        return {
            "provider": "OpenWeatherMap",
            "key_configured": True,
            "forecast_days": OpenWeatherMapProvider.max_forecast_days,
            "note": "Using your OpenWeatherMap API key.",
        }
    return {
        "provider": "Open-Meteo",
        "key_configured": False,
        "forecast_days": OpenMeteoProvider.max_forecast_days,
        "note": ("No OWM_API_KEY set, so the free Open-Meteo API is being "
                 "used. Set OWM_API_KEY to switch to OpenWeatherMap."),
    }


if __name__ == "__main__":
    chosen_city = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_CITY

    status = provider_status()
    print(f"Provider : {status['provider']}")
    print(f"Note     : {status['note']}\n")

    active_provider = get_provider()

    print(f"CURRENT WEATHER - {chosen_city}")
    print("-" * 52)
    for key, value in active_provider.get_current_weather(chosen_city).items():
        unit = config.UNITS.get(key, "")
        print(f"   {key:<14}: {value} {unit}".rstrip())

    print(f"\n{config.API_FORECAST_DAYS}-DAY FORECAST - {chosen_city}")
    print("-" * 52)
    forecast = active_provider.get_daily_forecast(
        chosen_city, days=config.API_FORECAST_DAYS
    )
    columns = [config.DATE_COLUMN, config.TEMPERATURE, config.HUMIDITY,
               config.PRESSURE, config.WIND_SPEED, config.CLOUD_COVER,
               config.VISIBILITY, "rain_chance"]
    print(forecast[[c for c in columns if c in forecast.columns]].to_string(
        index=False
    ))
