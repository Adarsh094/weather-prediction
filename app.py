"""
app.py
======

**Smart Weather Prediction System - Streamlit web application.**

The user picks a **city** and a **date**, presses **Predict**, and the app
answers with the full expected weather plus a Rain / No Rain verdict.

The important idea (Requirement 14): the app never decides *how* to answer.
It calls `utils.predictor.predict_weather()`, which automatically routes the
question to the live weather API for near dates and to the trained time-series
models for far ones.

Run it with:

    streamlit run app.py
"""

import datetime as dt

import pandas as pd
import streamlit as st

from utils import config
from utils import visualization as viz
from utils.forecasting import CityForecaster, available_cities
from utils.predictor import forecast_series, predict_weather
from weather_api import WeatherAPIError, get_provider, provider_status

# ---------------------------------------------------------------------------
# 1. Page setup - must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Weather Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def current_theme():
    """Detect light or dark so the charts can be styled for the real surface."""
    try:
        theme_type = st.context.theme.type
        if theme_type in ("light", "dark"):
            return theme_type
    except Exception:                                       # noqa: BLE001
        pass
    return st.get_option("theme.base") or "light"


THEME = current_theme()

st.markdown(
    """
    <style>
        .hero {
            background: linear-gradient(120deg, #184f95 0%, #2a78d6 55%,
                                        #3987e5 100%);
            padding: 26px 30px; border-radius: 16px; margin-bottom: 20px;
            color: #ffffff;
        }
        .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
        .hero p  { margin: 8px 0 0 0; opacity: 0.92; font-size: 1rem; }

        .verdict {
            padding: 26px; border-radius: 16px; text-align: center;
            color: #ffffff;
        }
        .verdict-rain    { background: linear-gradient(120deg,#0d366b,#2a78d6); }
        .verdict-no-rain { background: linear-gradient(120deg,#b34a12,#eb6834); }
        .verdict h2 { margin: 0; font-size: 2.2rem; }
        .verdict p  { margin: 6px 0 0 0; opacity: 0.95; }

        .route-badge {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-size: 0.8rem; font-weight: 600; margin-bottom: 6px;
        }
        .route-api   { background: rgba(42,120,214,0.16); color: #2a78d6; }
        .route-model { background: rgba(235,104,52,0.16); color: #eb6834; }

        div.stButton > button {
            width: 100%; font-weight: 600; padding: 0.55rem 0;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌤️ Smart Weather Prediction System</h1>
        <p>Pick a city and <b>any</b> date - today, next week, or months ahead.
        The app uses a live weather API for near dates and trained time-series
        models for the rest.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. Cached loaders - so the app is fast after the first run
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_forecaster(city):
    return CityForecaster(city)


@st.cache_resource(show_spinner=False)
def load_provider():
    return get_provider()


@st.cache_data(show_spinner=False)
def load_history(city):
    """The cleaned historical weather for one city (used by the charts)."""
    from utils.data_loader import load_clean_history
    history = load_clean_history()
    return history[history[config.CITY_COLUMN] == city].copy()


@st.cache_data(ttl=900, show_spinner=False)
def cached_current_weather(city):
    """Live conditions, refreshed at most every 15 minutes."""
    return load_provider().get_current_weather(city)


@st.cache_data(ttl=900, show_spinner=False)
def cached_forecast_series(city, start_date, days):
    return forecast_series(city, start_date, days,
                           forecaster=load_forecaster(city),
                           provider=load_provider())


@st.cache_data(show_spinner=False)
def cached_model_comparison():
    import os
    if os.path.exists(config.MODEL_COMPARISON_FILE):
        return pd.read_csv(config.MODEL_COMPARISON_FILE)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Are there trained models?
# ---------------------------------------------------------------------------

trained_cities = available_cities()

if not trained_cities:
    st.error(
        "**No trained models found.**\n\n"
        "Train the system first - it downloads 10 years of real weather data "
        "and trains every model:\n\n"
        "```\npython train.py\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# 4. Sidebar - city, date, predict
# ---------------------------------------------------------------------------

today = dt.date.today()

with st.sidebar:
    st.header("🔎 Make a Prediction")

    city = st.selectbox("**Select City**", trained_cities, index=0)

    target_date = st.date_input(
        "**Select Date**",
        value=today,
        min_value=today - dt.timedelta(days=365),
        max_value=today + dt.timedelta(days=config.MAX_FORECAST_DAYS),
        help="Any date up to a year ahead. Past dates show what the model "
             "would have said.",
    )

    predict_clicked = st.button("🌦️ Predict Weather", type="primary")

    # Show which engine this date will use, before the user even presses go.
    days_ahead = (target_date - today).days
    if 0 <= days_ahead <= config.API_FORECAST_DAYS:
        st.success(f"**{days_ahead} day(s) ahead** → live weather API")
    else:
        st.info(f"**{days_ahead} day(s) ahead** → trained time-series model")

    st.divider()

    st.subheader("⚡ Quick jumps")
    jump_columns = st.columns(2)
    presets = [("Today", 0), ("Tomorrow", 1),
               ("In 7 days", 7), ("In 30 days", 30)]
    for position, (label, offset) in enumerate(presets):
        if jump_columns[position % 2].button(label, key=f"jump_{offset}"):
            st.session_state["jump_offset"] = offset
            st.rerun()

    st.divider()

    status = provider_status()
    st.subheader("🔌 Data sources")
    st.caption(f"**Live API:** {status['provider']}")
    st.caption(status["note"])

    active_forecaster = load_forecaster(city)
    st.caption(f"**Forecast model:** {active_forecaster.best_model_name}")
    st.caption(f"**Rain model:** {active_forecaster.rain_model_name}")
    st.caption(f"**History:** {active_forecaster.metadata['history_start']} "
               f"to {active_forecaster.metadata['history_end']} "
               f"({active_forecaster.metadata['n_days']:,} days)")

# A quick-jump button overrides the date picker.
if "jump_offset" in st.session_state:
    target_date = today + dt.timedelta(days=st.session_state.pop("jump_offset"))
    predict_clicked = True

forecaster = load_forecaster(city)
history = load_history(city)


# ---------------------------------------------------------------------------
# 5. Current weather (always shown)
# ---------------------------------------------------------------------------

st.subheader(f"📍 Current Weather - {city}")

try:
    now = cached_current_weather(city)
    observed_at = pd.Timestamp(now["time"]).strftime("%d %b %Y, %H:%M")
    st.caption(f"{now.get('description', '')} · observed {observed_at} · "
               f"source: {now['source']}")

    metric_columns = st.columns(6)
    live_metrics = [
        (config.TEMPERATURE, "🌡️ Temperature"),
        (config.HUMIDITY, "💧 Humidity"),
        (config.PRESSURE, "🧭 Pressure"),
        (config.WIND_SPEED, "💨 Wind"),
        (config.VISIBILITY, "👁️ Visibility"),
        (config.CLOUD_COVER, "☁️ Cloud"),
    ]
    for column, (key, label) in zip(metric_columns, live_metrics):
        value = now.get(key)
        unit = config.UNITS.get(key, "")
        column.metric(label,
                      f"{value:.1f} {unit}" if value is not None else "n/a")

except WeatherAPIError as error:
    st.warning(f"Could not load live conditions ({error}). "
               "Forecasts below will use the trained models.")
    now = None


# ---------------------------------------------------------------------------
# 6. The prediction
# ---------------------------------------------------------------------------

st.divider()

if predict_clicked:
    st.session_state["last_prediction_date"] = target_date

active_date = st.session_state.get("last_prediction_date")

if active_date is None:
    st.info("👈 Choose a city and a date in the sidebar, then press "
            "**Predict Weather**.")
else:
    with st.spinner("Predicting..."):
        result = predict_weather(city, active_date, forecaster=forecaster,
                                 provider=load_provider())

    pretty_date = pd.Timestamp(result["date"]).strftime("%A, %d %B %Y")
    st.subheader(f"🎯 Prediction - {city}, {pretty_date}")

    badge_class = "route-api" if result["route"] == "api" else "route-model"
    badge_text = ("LIVE WEATHER API" if result["route"] == "api"
                  else "TIME-SERIES MODEL")
    st.markdown(
        f'<span class="route-badge {badge_class}">{badge_text}</span>'
        f'<br><span style="opacity:0.75;font-size:0.9rem">'
        f'{result["model_used"]}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    verdict_column, detail_column = st.columns([1.1, 1])

    with verdict_column:
        is_rain = result["rain_label"] == "Rain"
        st.markdown(
            f"""
            <div class="verdict {'verdict-rain' if is_rain else 'verdict-no-rain'}">
                <h2>{'🌧️ Rain' if is_rain else '☀️ No Rain'}</h2>
                <p>Chance of rain: <b>{result['rain_probability']:.1f}%</b></p>
                <p>Confidence: <b>{result['confidence']:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with detail_column:
        st.metric("Rain Probability", f"{result['rain_probability']:.1f}%")
        st.progress(min(int(result["rain_probability"]), 100))
        st.metric("Confidence Score", f"{result['confidence']:.1f}%",
                  help="The model's own certainty, scaled down by how "
                       "reliable this forecast distance proved to be in the "
                       "rolling-origin backtest.")

    # ---- the predicted parameters ---------------------------------------
    st.write("")
    parameter_columns = st.columns(6)
    predicted_parameters = [
        (config.TEMPERATURE, "🌡️ Temperature"),
        (config.HUMIDITY, "💧 Humidity"),
        (config.PRESSURE, "🧭 Pressure"),
        (config.WIND_SPEED, "💨 Wind Speed"),
        (config.CLOUD_COVER, "☁️ Cloud Cover"),
        (config.VISIBILITY, "👁️ Visibility"),
    ]
    for column, (key, label) in zip(parameter_columns, predicted_parameters):
        value = result.get(key)
        unit = config.UNITS.get(key, "")
        if value is None:
            column.metric(label, "n/a",
                          help="Not available on this route - see the README "
                               "note about historical visibility.")
        else:
            delta = None
            if key == config.TEMPERATURE:
                difference = value - result["normal_temperature"]
                delta = f"{difference:+.1f} °C vs normal"
            column.metric(label, f"{value:.1f} {unit}", delta=delta,
                          delta_color="off")

    # ---- context + warnings ---------------------------------------------
    context_columns = st.columns(2)
    context_columns[0].info(
        f"**Normal for this date:** "
        f"{result['normal_temperature']:.1f} °C  \n"
        f"(10-year average for {city})"
    )
    context_columns[1].info(
        f"**History says:** it has rained on this date in "
        f"**{result['historical_rain_chance']:.0f}%** of the last 10 years."
    )

    for warning in result["warnings"]:
        st.warning(warning)


# ---------------------------------------------------------------------------
# 7. Forecast table + charts
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📅 Forecast Weather")

horizon = st.slider("How many days ahead to show", 7, 90, 14, step=7)

with st.spinner("Building the forecast..."):
    series = cached_forecast_series(city, today, horizon)

if series.empty:
    st.warning("Could not build a forecast series for this city.")
else:
    display = pd.DataFrame({
        "Date": series[config.DATE_COLUMN].dt.strftime("%a, %d %b"),
        "Temp (°C)": series[config.TEMPERATURE].round(1),
        "Humidity (%)": series[config.HUMIDITY].round(0),
        "Pressure (hPa)": series[config.PRESSURE].round(0),
        "Wind (km/h)": series[config.WIND_SPEED].round(1),
        "Cloud (%)": series[config.CLOUD_COVER].round(0),
        "Rain (%)": series["rain_probability"].round(0),
        "Verdict": series["rain_label"],
        "Source": series["source"],
    })
    st.dataframe(display, width="stretch", hide_index=True, height=280)

    api_days = int((series["source"] == "Live Weather API").sum())
    st.caption(f"{api_days} day(s) came from the live API; "
               f"{len(series) - api_days} from the trained models.")

    # ---- charts ---------------------------------------------------------
    st.divider()
    st.subheader("📊 Charts")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Historical Temperature", "Forecast Temperature",
        "Rain Probability", "Monthly Rainfall",
    ])

    with tab1:
        st.plotly_chart(
            viz.historical_temperature_trend(history, city, years=3,
                                             mode=THEME),
            width="stretch",
        )
        st.caption(
            "The pale line is every single day; the bold line is a 30-day "
            "average. The seasonal cycle is highly predictable - the daily "
            "wobble around it is not. That gap is the whole forecasting "
            "problem in one picture."
        )

    with tab2:
        st.plotly_chart(
            viz.forecast_temperature_trend(series, city, mode=THEME),
            width="stretch",
        )
        st.caption(
            "Blue days come from the live weather API, orange days from the "
            "trained time-series model. The dotted grey line is the 10-year "
            "normal for each date."
        )

    with tab3:
        st.plotly_chart(
            viz.rain_probability_chart(series, city, mode=THEME),
            width="stretch",
        )
        st.caption(
            "Anything above the dashed 50% line is reported as 'Rain'. "
            "Bar height and shade show the same number, so the chart still "
            "reads in greyscale."
        )

    with tab4:
        st.plotly_chart(
            viz.monthly_rainfall_chart(forecaster.monthly_rainfall(), city,
                                       mode=THEME),
            width="stretch",
        )
        st.caption(
            "Average rainfall per month over the full 10-year history. For "
            "Indian cities this is the monsoon drawn straight from the data."
        )


# ---------------------------------------------------------------------------
# 8. Feature importance and model performance
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🧠 Inside the Models")

importance_column, comparison_column = st.columns(2)

with importance_column:
    importance = forecaster.metadata.get("rain_feature_importance", {})
    if importance:
        st.plotly_chart(
            viz.feature_importance_chart(
                importance, mode=THEME,
                title=f"Rain classifier - feature importance ({city})",
            ),
            width="stretch",
        )

with comparison_column:
    comparison = cached_model_comparison()
    if not comparison.empty:
        city_comparison = comparison[comparison["City"] == city]
        if not city_comparison.empty:
            table = city_comparison.set_index("Model")[["MAE", "RMSE", "R2"]]
            st.plotly_chart(
                viz.model_comparison_chart(table, metric="RMSE", mode=THEME),
                width="stretch",
            )

with st.expander("📋 Full model comparison table (MAE / RMSE / R²)"):
    comparison = cached_model_comparison()
    if comparison.empty:
        st.write("Run `python train.py` to generate the comparison table.")
    else:
        st.dataframe(
            comparison[comparison["City"] == city]
            .set_index("Model")[["MAE", "RMSE", "R2", "Bias", "Fit (s)"]]
            .round(4),
            width="stretch",
        )
        metadata = forecaster.metadata
        st.markdown(
            f"**Deployed model:** {metadata['best_forecast_model']}  \n"
            f"**Best of the five trained models:** "
            f"{metadata.get('best_trained_model', 'n/a')}  \n"
            f"**Skill vs the climatology baseline:** "
            f"{metadata.get('skill_vs_climatology_pct', 'n/a')}%"
        )
        if metadata.get("baseline_won"):
            st.warning(
                "For this city the plain 10-year climatology baseline beat "
                "every trained model, so that is what is deployed. Shipping a "
                "more complex model that scores worse would be bad practice."
            )

with st.expander("🎯 How reliable is a forecast this far ahead?"):
    detail = forecaster.metadata.get("horizon_detail", [])
    if detail:
        st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
        st.caption(
            "Measured with a rolling-origin backtest: the model was retrained "
            "at several past dates and asked to forecast forward, then scored "
            "against what actually happened. The reliability column is what "
            "scales the confidence score you see above."
        )

st.divider()
st.caption(
    "Smart Weather Prediction System · historical data: Open-Meteo ERA5 "
    "reanalysis · live forecast: OpenWeatherMap / Open-Meteo · "
    "built with Python, scikit-learn, Prophet, PyTorch and Streamlit."
)

# Vercel entrypoint fallback compatibility export
try:
    from api.index import app
except Exception:
    pass

