"""
utils/visualization.py
======================

**Visualization module** - every interactive chart in the app.

Design rules followed throughout (so the whole app reads as one system):

* **One axis per chart.** Never two y-scales - if two measures have different
  units they get two charts, not one confusing overlay.
* **Colour carries identity, text carries values.** Axis labels, tick labels
  and annotations use ink colours, never the series colour.
* **Thin marks, recessive grid.** 2px lines, hairline gridlines, no chart junk.
* **A legend whenever there are two or more series**, so nothing depends on
  colour alone.
* **Light and dark are both designed**, not an automatic flip.

The palette is the validated default from the project's design tokens:
blue `#2a78d6` / orange `#eb6834` in light mode, stepped to `#3987e5` /
`#d95926` for the dark surface. Both pass colour-blindness separation and
contrast checks against their surface.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import config

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "series_1": "#2a78d6",      # blue   - primary series
        "series_2": "#eb6834",      # orange - secondary series
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                       "#2a78d6", "#256abf", "#184f95", "#0d366b"],
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "rgba(42,120,214,0.10)",
        "surface": "rgba(0,0,0,0)",     # inherit the app background
    },
    "dark": {
        "series_1": "#3987e5",
        "series_2": "#d95926",
        "sequential": ["#184f95", "#256abf", "#2a78d6", "#3987e5",
                       "#5598e7", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "rgba(57,135,229,0.16)",
        "surface": "rgba(0,0,0,0)",
    },
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def get_theme(mode="light"):
    """Return the token set for the requested mode."""
    return THEMES.get(mode, THEMES["light"])


def _base_layout(figure, theme, title=None, y_title=None, x_title=None,
                 height=380, show_legend=True):
    """Apply the shared look to any figure - called by every chart below."""
    figure.update_layout(
        title=(dict(text=title, font=dict(size=15, color=theme["text_primary"]),
                    x=0.0, xanchor="left") if title else None),
        height=height,
        margin=dict(l=10, r=16, t=44 if title else 16, b=10),
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(family=FONT_FAMILY, size=12, color=theme["text_secondary"]),
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=theme["text_secondary"], size=11)),
    )
    figure.update_xaxes(
        title_text=x_title,
        showgrid=False,
        linecolor=theme["axis"],
        tickfont=dict(color=theme["muted"], size=11),
        title_font=dict(color=theme["text_secondary"], size=12),
    )
    figure.update_yaxes(
        title_text=y_title,
        gridcolor=theme["grid"],
        griddash="solid",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=theme["muted"], size=11),
        title_font=dict(color=theme["text_secondary"], size=12),
    )
    return figure


def _sequential_color(value, low, high, theme):
    """Map a number onto the sequential blue ramp (light -> dark = low -> high)."""
    ramp = theme["sequential"]
    if high <= low:
        return ramp[len(ramp) // 2]
    position = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return ramp[int(round(position * (len(ramp) - 1)))]


# ===========================================================================
# Chart 1 - Historical Temperature Trend
# ===========================================================================

def historical_temperature_trend(history, city, years=3, mode="light"):
    """Daily temperature over the last few years, with a 30-day smooth line.

    The raw daily values show how noisy weather really is; the smoothed line
    shows the seasonal cycle underneath that noise. Seeing both together is
    what makes the forecasting problem obvious: the season is predictable, the
    individual day is not.
    """
    theme = get_theme(mode)

    frame = history.sort_values(config.DATE_COLUMN)
    cutoff = frame[config.DATE_COLUMN].max() - pd.DateOffset(years=years)
    frame = frame[frame[config.DATE_COLUMN] >= cutoff]

    smoothed = frame[config.TEMPERATURE].rolling(30, center=True,
                                                 min_periods=1).mean()

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frame[config.DATE_COLUMN], y=frame[config.TEMPERATURE],
        name="Daily temperature", mode="lines",
        line=dict(color=theme["series_1"], width=1),
        opacity=0.30, hovertemplate="%{y:.1f} °C<extra>Daily</extra>",
    ))
    figure.add_trace(go.Scatter(
        x=frame[config.DATE_COLUMN], y=smoothed,
        name="30-day average", mode="lines",
        line=dict(color=theme["series_2"], width=2),
        hovertemplate="%{y:.1f} °C<extra>30-day avg</extra>",
    ))

    return _base_layout(
        figure, theme,
        title=f"Historical temperature - {city} (last {years} years)",
        y_title="Temperature (°C)", height=360,
    )


# ===========================================================================
# Chart 2 - Forecast Temperature Trend
# ===========================================================================

def forecast_temperature_trend(forecast_table, city, mode="light"):
    """Upcoming temperature, showing which engine produced each day.

    The two colours are not decoration: they mark the hand-off point where the
    live weather API stops and the trained time-series model takes over. The
    dashed grey line is the 10-year seasonal normal, so the user can see at a
    glance whether the forecast is unusual for the time of year.
    """
    theme = get_theme(mode)
    frame = forecast_table.sort_values(config.DATE_COLUMN)

    figure = go.Figure()

    # The seasonal normal sits underneath, in ink not series colour.
    if "normal_temperature" in frame:
        figure.add_trace(go.Scatter(
            x=frame[config.DATE_COLUMN], y=frame["normal_temperature"],
            name="10-year normal", mode="lines",
            line=dict(color=theme["muted"], width=1.5, dash="dot"),
            hovertemplate="%{y:.1f} °C<extra>Normal</extra>",
        ))

    from utils.predictor import SOURCE_API

    for source, colour, label in (
        (SOURCE_API, theme["series_1"], "Live API forecast"),
        (None, theme["series_2"], "Model forecast"),
    ):
        part = (frame[frame["source"] == SOURCE_API] if source
                else frame[frame["source"] != SOURCE_API])
        if part.empty:
            continue
        figure.add_trace(go.Scatter(
            x=part[config.DATE_COLUMN], y=part[config.TEMPERATURE],
            name=label, mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(size=8, color=colour,
                        line=dict(width=2, color=theme["surface"])),
            hovertemplate="%{y:.1f} °C<extra>" + label + "</extra>",
        ))

    return _base_layout(
        figure, theme,
        title=f"Forecast temperature - {city}",
        y_title="Temperature (°C)", height=360,
    )


# ===========================================================================
# Chart 3 - Rain Probability
# ===========================================================================

def rain_probability_chart(forecast_table, city, mode="light"):
    """Chance of rain for each upcoming day.

    Bar height and bar colour both encode the same number. That redundancy is
    deliberate: it keeps the chart readable in greyscale, in print, and for
    colour-blind readers.
    """
    theme = get_theme(mode)
    frame = forecast_table.sort_values(config.DATE_COLUMN)
    probabilities = frame["rain_probability"].to_numpy(dtype=float)

    colours = [_sequential_color(p, 0, 100, theme) for p in probabilities]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=frame[config.DATE_COLUMN], y=probabilities,
        marker=dict(color=colours, cornerradius=4,
                    line=dict(width=2, color=theme["surface"])),
        name="Chance of rain",
        hovertemplate="%{y:.0f}%<extra>Chance of rain</extra>",
        showlegend=False,
    ))

    # The decision threshold the app uses to say "Rain" or "No Rain".
    figure.add_hline(
        y=50, line=dict(color=theme["muted"], width=1.5, dash="dash"),
        annotation_text="50% - the Rain / No Rain line",
        annotation_position="top left",
        annotation_font=dict(color=theme["text_secondary"], size=11),
    )

    figure = _base_layout(
        figure, theme, title=f"Chance of rain - {city}",
        y_title="Probability (%)", height=340, show_legend=False,
    )
    figure.update_yaxes(range=[0, 105])
    return figure


# ===========================================================================
# Chart 4 - Monthly Rainfall Trend
# ===========================================================================

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_rainfall_chart(monthly_rainfall, city, mode="light"):
    """Average rainfall per calendar month across the whole 10-year history.

    For Indian cities this chart is the monsoon, drawn from data: a few months
    carry almost the entire year's rain.
    """
    theme = get_theme(mode)

    values = monthly_rainfall.reindex(range(1, 13)).fillna(0.0)
    amounts = values.to_numpy(dtype=float)
    colours = [_sequential_color(v, 0, amounts.max() or 1, theme)
               for v in amounts]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=MONTH_LABELS, y=amounts,
        marker=dict(color=colours, cornerradius=4,
                    line=dict(width=2, color=theme["surface"])),
        hovertemplate="%{y:.0f} mm<extra>%{x}</extra>",
        showlegend=False,
    ))

    # Label only the wettest month - never a number on every bar.
    wettest = int(np.argmax(amounts))
    figure.add_annotation(
        x=MONTH_LABELS[wettest], y=amounts[wettest],
        text=f"{amounts[wettest]:.0f} mm", showarrow=False, yshift=12,
        font=dict(color=theme["text_primary"], size=12),
    )

    return _base_layout(
        figure, theme, title=f"Average monthly rainfall - {city}",
        y_title="Rainfall (mm)", height=340, show_legend=False,
    )


# ===========================================================================
# Chart 5 - Feature importance
# ===========================================================================

def feature_importance_chart(importance, mode="light", title=None):
    """Which weather readings the rain classifier relies on most."""
    theme = get_theme(mode)

    series = pd.Series(importance).sort_values()
    labels = [config.PRETTY_NAMES.get(name, name.replace("_", " ").title())
              for name in series.index]
    values = series.to_numpy(dtype=float)
    colours = [_sequential_color(v, 0, values.max() or 1, theme)
               for v in values]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colours, cornerradius=4,
                    line=dict(width=2, color=theme["surface"])),
        hovertemplate="%{x:.3f}<extra>%{y}</extra>",
        showlegend=False,
    ))

    figure = _base_layout(
        figure, theme, title=title or "What the rain model looks at most",
        x_title="Importance", height=320, show_legend=False,
    )
    figure.update_xaxes(showgrid=True, gridcolor=theme["grid"])
    figure.update_yaxes(showgrid=False)
    return figure


# ===========================================================================
# Chart 6 - Model comparison
# ===========================================================================

def model_comparison_chart(comparison, metric="RMSE", mode="light"):
    """Backtest error for every forecasting model, best first."""
    theme = get_theme(mode)

    frame = comparison.copy()
    ascending = metric != "R2"
    frame = frame.sort_values(metric, ascending=not ascending)

    values = frame[metric].to_numpy(dtype=float)
    is_baseline = frame.index.str.contains("baseline", case=False)
    colours = [theme["muted"] if baseline else theme["series_1"]
               for baseline in is_baseline]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=values, y=list(frame.index), orientation="h",
        marker=dict(color=colours, cornerradius=4,
                    line=dict(width=2, color=theme["surface"])),
        hovertemplate="%{x:.3f}<extra>%{y}</extra>",
        showlegend=False,
    ))

    figure = _base_layout(
        figure, theme,
        title=f"Forecast model comparison - {metric} (lower is better)"
        if ascending else f"Forecast model comparison - {metric}",
        x_title=metric, height=320, show_legend=False,
    )
    figure.update_xaxes(showgrid=True, gridcolor=theme["grid"])
    figure.update_yaxes(showgrid=False)
    return figure


# ===========================================================================
# Chart 7 - Backtest: forecast against what actually happened
# ===========================================================================

def backtest_chart(dates, actual, predicted, model_name, mode="light"):
    """The hidden test year: what the model said versus what really happened."""
    theme = get_theme(mode)

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=dates, y=actual, name="Actual", mode="lines",
        line=dict(color=theme["series_1"], width=1.5), opacity=0.65,
        hovertemplate="%{y:.1f} °C<extra>Actual</extra>",
    ))
    figure.add_trace(go.Scatter(
        x=dates, y=predicted, name=f"{model_name} forecast", mode="lines",
        line=dict(color=theme["series_2"], width=2),
        hovertemplate="%{y:.1f} °C<extra>Forecast</extra>",
    ))

    return _base_layout(
        figure, theme,
        title=f"Backtest - {model_name} vs reality on the unseen year",
        y_title="Temperature (°C)", height=340,
    )
