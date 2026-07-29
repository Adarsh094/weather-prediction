"""
forecast.py
===========

**Command-line forecasting tool.**

The same prediction engine the Streamlit app uses, driven from the terminal.
Useful for testing, for a viva demonstration, and for anyone who wants the
numbers without opening a browser.

Examples
--------
    python forecast.py                              # today, default city
    python forecast.py Mumbai                       # today, Mumbai
    python forecast.py Delhi 2026-12-25             # a specific date
    python forecast.py Chennai --days 14            # a 14-day table
    python forecast.py Kolkata --days 90 --csv out.csv

The city and date can be given either positionally (as above) or with flags
(`--city Delhi --date 2026-12-25`), whichever you find easier to remember.
"""

import argparse
import datetime as dt

import pandas as pd

from utils import config
from utils.forecasting import CityForecaster, available_cities
from utils.predictor import forecast_series, predict_weather
from weather_api import get_provider, provider_status


def print_single_prediction(result):
    """Pretty-print one date's full prediction."""
    date_text = pd.Timestamp(result["date"]).strftime("%A, %d %B %Y")

    print("\n" + "=" * 62)
    print(f"  {result['city'].upper()}  -  {date_text}")
    print("=" * 62)

    ahead = result["days_ahead"]
    when = ("today" if ahead == 0 else
            f"{ahead} day{'s' if abs(ahead) != 1 else ''} "
            f"{'ahead' if ahead > 0 else 'ago'}")
    print(f"  Horizon      : {when}")
    print(f"  Answered by  : {result['source']}")
    print(f"  Model used   : {result['model_used']}")

    print("\n  PREDICTED CONDITIONS")
    print("  " + "-" * 58)
    for parameter in (config.TEMPERATURE, config.HUMIDITY, config.PRESSURE,
                      config.WIND_SPEED, config.CLOUD_COVER,
                      config.VISIBILITY, config.DEW_POINT):
        value = result.get(parameter)
        label = config.PRETTY_NAMES.get(parameter, parameter)
        unit = config.UNITS.get(parameter, "")
        if value is None:
            print(f"  {label:<14}: not available for this route")
        else:
            print(f"  {label:<14}: {value:.1f} {unit}")

    print("\n  RAIN PREDICTION")
    print("  " + "-" * 58)
    print(f"  Verdict      : {result['rain_label'].upper()}")
    print(f"  Probability  : {result['rain_probability']:.1f} %")
    print(f"  Confidence   : {result['confidence']:.1f} %")

    print("\n  FOR CONTEXT (from 10 years of history)")
    print("  " + "-" * 58)
    print(f"  Normal temperature for this date : "
          f"{result['normal_temperature']:.1f} °C")
    print(f"  It has rained on this date in    : "
          f"{result['historical_rain_chance']:.0f}% of past years")

    if result["warnings"]:
        print("\n  NOTES")
        print("  " + "-" * 58)
        for warning in result["warnings"]:
            print(f"  ! {warning}")
    print()


def print_forecast_table(table):
    """Print a multi-day forecast as a compact table."""
    display = pd.DataFrame({
        "Date": table[config.DATE_COLUMN].dt.strftime("%Y-%m-%d"),
        "Day": table[config.DATE_COLUMN].dt.strftime("%a"),
        "Temp": table[config.TEMPERATURE].round(1),
        "Humid": table[config.HUMIDITY].round(0),
        "Press": table[config.PRESSURE].round(0),
        "Wind": table[config.WIND_SPEED].round(1),
        "Cloud": table[config.CLOUD_COVER].round(0),
        "Rain%": table["rain_probability"].round(0),
        "Verdict": table["rain_label"],
        "Source": table["source"].str.replace("Live Weather API", "API")
                                 .str.replace("Time-Series Model", "Model"),
    })
    print(display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Forecast the weather for any city and date",
        epilog="Example:  python forecast.py Delhi 2026-09-15",
    )
    # Positional forms, so both of these work:
    #     python forecast.py Delhi 2026-09-15
    #     python forecast.py --city Delhi --date 2026-09-15
    parser.add_argument("city_positional", nargs="?", default=None,
                        metavar="CITY", help="city name (optional)")
    parser.add_argument("date_positional", nargs="?", default=None,
                        metavar="DATE", help="date as YYYY-MM-DD (optional)")
    parser.add_argument("--city", default=None,
                        help=f"city name (default: {config.DEFAULT_CITY})")
    parser.add_argument("--date", default=None,
                        help="target date as YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=None,
                        help="print a table of this many days instead")
    parser.add_argument("--csv", default=None,
                        help="also save the table to this CSV file")
    arguments = parser.parse_args()

    # A flag always wins over the positional form.
    arguments.city = (arguments.city or arguments.city_positional
                      or config.DEFAULT_CITY)
    arguments.date = arguments.date or arguments.date_positional

    # Be forgiving about capitalisation: "delhi" should find "Delhi".
    lookup = {name.lower(): name for name in config.CITY_NAMES}
    arguments.city = lookup.get(arguments.city.lower(), arguments.city)

    if arguments.date:
        try:
            dt.date.fromisoformat(arguments.date)
        except ValueError:
            print(f"'{arguments.date}' is not a valid date. "
                  f"Use the YYYY-MM-DD format, for example 2026-09-15.")
            return

    ready = available_cities()
    if not ready:
        print("No trained models found. Run 'python train.py' first.")
        return
    if arguments.city not in ready:
        print(f"'{arguments.city}' has no trained model. Available: "
              f"{', '.join(ready)}")
        return

    status = provider_status()
    print(f"Live weather provider : {status['provider']}")
    print(f"Routing rule          : dates within "
          f"{config.API_FORECAST_DAYS} days use the live API, "
          f"later dates use the trained models")

    forecaster = CityForecaster(arguments.city)
    provider = get_provider()

    print(f"Deployed forecaster   : {forecaster.best_model_name}")
    print(f"Deployed rain model   : {forecaster.rain_model_name}")

    # ---- multi-day table ------------------------------------------------
    if arguments.days:
        start = (dt.date.fromisoformat(arguments.date) if arguments.date
                 else dt.date.today())
        table = forecast_series(arguments.city, start, arguments.days,
                                forecaster=forecaster, provider=provider)
        print(f"\n{arguments.days}-DAY FORECAST - {arguments.city}\n")
        print_forecast_table(table)

        if arguments.csv:
            table.to_csv(arguments.csv, index=False)
            print(f"\nSaved to {arguments.csv}")
        return

    # ---- single date ----------------------------------------------------
    target = (dt.date.fromisoformat(arguments.date) if arguments.date
              else dt.date.today())
    result = predict_weather(arguments.city, target, forecaster=forecaster,
                             provider=provider)
    print_single_prediction(result)


if __name__ == "__main__":
    main()
