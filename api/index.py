"""
api/index.py
============
Vercel Serverless Function entry point for Smart Weather Prediction System.

Exposes a Flask WSGI `app` object that Vercel uses to deploy the Python backend.
Serves both JSON API endpoints and an interactive single-page Web Application dashboard.
"""

import datetime as dt
from pathlib import Path
import sys
import os

# Ensure the root directory is on sys.path so utils and weather_api are importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from utils import config
from utils.forecasting import CityForecaster, available_cities
from utils.predictor import forecast_series, predict_weather
from weather_api import WeatherAPIError, provider_status

app = Flask(__name__)


def clean_for_json(obj):
    """Convert numpy / pandas types into standard JSON serializable Python types."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 2)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "app": "Smart Weather Prediction System"})


@app.route("/api/cities", methods=["GET"])
def get_cities():
    cities = available_cities()
    details = []
    for city in cities:
        meta = config.CITIES.get(city, {})
        details.append({
            "name": city,
            "country": meta.get("country", "India"),
            "lat": meta.get("lat"),
            "lon": meta.get("lon")
        })
    return jsonify({"cities": details})


@app.route("/api/predict", methods=["GET"])
def api_predict():
    city = request.args.get("city", "Delhi")
    date_str = request.args.get("date")

    if city not in config.CITIES:
        return jsonify({"error": f"City '{city}' is not supported."}), 400

    if date_str:
        try:
            target_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        target_date = dt.date.today() + dt.timedelta(days=1)

    try:
        forecaster = CityForecaster(city)
        result = predict_weather(city, target_date, forecaster=forecaster)
        cleaned = clean_for_json(result)
        return jsonify(cleaned)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast_series", methods=["GET"])
def api_forecast_series():
    city = request.args.get("city", "Delhi")
    date_str = request.args.get("date")
    days = int(request.args.get("days", 14))

    if city not in config.CITIES:
        return jsonify({"error": f"City '{city}' is not supported."}), 400

    if date_str:
        try:
            start_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        start_date = dt.date.today()

    target_dates = [start_date + dt.timedelta(days=i) for i in range(days)]
    try:
        forecaster = CityForecaster(city)
        df = forecast_series(city, target_dates, forecaster=forecaster)
        records = df.to_dict(orient="records")
        cleaned = clean_for_json(records)
        return jsonify({"city": city, "forecast": cleaned})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# HTML Template for the main dashboard UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Weather Prediction System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --primary: #38bdf8;
            --primary-gradient: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --rain-color: #38bdf8;
            --norain-color: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.12) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .control-panel {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            align-items: flex-end;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        select, input[type="date"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            outline: none;
            transition: all 0.2s;
        }

        select:focus, input[type="date"]:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }

        .btn-submit {
            background: var(--primary-gradient);
            color: #fff;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.2s, opacity 0.2s;
            height: 48px;
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .preset-buttons {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.25rem 0.6rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            cursor: pointer;
        }

        .preset-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            color: var(--text-main);
        }

        .results-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 900px) {
            .results-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            padding: 1.5rem;
        }

        .verdict-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 2rem;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-api {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        .badge-model {
            background: rgba(129, 140, 248, 0.15);
            color: #818cf8;
            border: 1px solid rgba(129, 140, 248, 0.3);
        }

        .verdict-box {
            text-align: center;
            padding: 1.5rem 1rem;
            border-radius: 0.75rem;
            background: rgba(15, 23, 42, 0.4);
            margin-bottom: 1.25rem;
        }

        .verdict-title {
            font-size: 2.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .verdict-sub {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 0.75rem;
            text-align: center;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .chart-container {
            height: 380px;
            width: 100%;
        }

        .spinner {
            display: none;
            text-align: center;
            padding: 3rem;
        }

        .spinner div {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-left-color: var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .warning-box {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fcd34d;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌦️ Smart Weather Prediction System</h1>
            <p>Live API + ML Time-Series Forecasting for Any Future Date</p>
        </header>

        <div class="control-panel">
            <div class="form-group">
                <label for="citySelect">Select City</label>
                <select id="citySelect">
                    <option value="Delhi" selected>Delhi</option>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Bengaluru">Bengaluru</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Hyderabad">Hyderabad</option>
                </select>
            </div>

            <div class="form-group">
                <label for="dateInput">Target Date</label>
                <input type="date" id="dateInput">
                <div class="preset-buttons">
                    <button class="preset-btn" onclick="setPreset(1)">Tomorrow</button>
                    <button class="preset-btn" onclick="setPreset(7)">7 Days</button>
                    <button class="preset-btn" onclick="setPreset(30)">30 Days</button>
                    <button class="preset-btn" onclick="setPreset(180)">6 Months</button>
                </div>
            </div>

            <button class="btn-submit" onclick="runPrediction()">Predict Weather</button>
        </div>

        <div id="spinner" class="spinner">
            <div></div>
            <p>Gathering forecast & running machine learning models...</p>
        </div>

        <div id="results" class="results-grid" style="display: none;">
            <!-- Left Card: Verdict -->
            <div class="card">
                <div class="verdict-header">
                    <span id="routeBadge" class="badge badge-api">Live Weather API</span>
                    <span id="daysAhead" style="font-size: 0.85rem; color: var(--text-muted);">+1 days</span>
                </div>

                <div class="verdict-box">
                    <div id="verdictLabel" class="verdict-title" style="color: var(--norain-color);">No Rain</div>
                    <div class="verdict-sub">
                        Rain Probability: <strong id="rainProb">15%</strong> &nbsp;|&nbsp; 
                        Confidence: <strong id="confidenceVal">88%</strong>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Temperature</div>
                        <div id="tempVal" class="metric-value">28.5 °C</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Humidity</div>
                        <div id="humidityVal" class="metric-value">62 %</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Pressure</div>
                        <div id="pressureVal" class="metric-value">1012 hPa</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Wind Speed</div>
                        <div id="windVal" class="metric-value">12 km/h</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Cloud Cover</div>
                        <div id="cloudVal" class="metric-value">20 %</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Dew Point</div>
                        <div id="dewVal" class="metric-value">18 °C</div>
                    </div>
                </div>

                <div id="warningContainer"></div>
            </div>

            <!-- Right Card: Interactive Chart -->
            <div class="card">
                <h3 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: var(--text-main);">
                    14-Day Temperature & Rain Forecast Trend
                </h3>
                <div id="plotlyChart" class="chart-container"></div>
            </div>
        </div>
    </div>

    <script>
        // Set default date to tomorrow
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        document.getElementById('dateInput').value = tomorrow.toISOString().split('T')[0];

        function setPreset(daysAhead) {
            const d = new Date();
            d.setDate(d.getDate() + daysAhead);
            document.getElementById('dateInput').value = d.toISOString().split('T')[0];
            runPrediction();
        }

        async function runPrediction() {
            const city = document.getElementById('citySelect').value;
            const date = document.getElementById('dateInput').value;
            
            if (!date) return;

            document.getElementById('results').style.display = 'none';
            document.getElementById('spinner').style.display = 'block';

            try {
                // Fetch single prediction
                const res = await fetch(`/api/predict?city=${encodeURIComponent(city)}&date=${date}`);
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Fetch series for chart
                const seriesRes = await fetch(`/api/forecast_series?city=${encodeURIComponent(city)}&date=${date}&days=14`);
                const seriesData = await seriesRes.json();

                updateUI(data, seriesData.forecast);
            } catch (err) {
                console.error(err);
                alert('Failed to connect to weather service.');
            } finally {
                document.getElementById('spinner').style.display = 'none';
            }
        }

        function updateUI(data, forecastSeries) {
            document.getElementById('results').style.display = 'grid';

            // Route Badge
            const badge = document.getElementById('routeBadge');
            badge.innerText = data.source || data.route.toUpperCase();
            badge.className = data.route === 'api' ? 'badge badge-api' : 'badge badge-model';

            // Days Ahead
            const ahead = data.days_ahead;
            document.getElementById('daysAhead').innerText = ahead >= 0 ? `+${ahead} days ahead` : `${ahead} days ago`;

            // Verdict
            const isRain = data.rain_label === 'Rain';
            const vLabel = document.getElementById('verdictLabel');
            vLabel.innerText = isRain ? '☔ Rain Expected' : '☀️ No Rain';
            vLabel.style.color = isRain ? 'var(--rain-color)' : 'var(--norain-color)';

            document.getElementById('rainProb').innerText = `${data.rain_probability}%`;
            document.getElementById('confidenceVal').innerText = `${data.confidence}%`;

            // Metrics
            document.getElementById('tempVal').innerText = data.temperature != null ? `${data.temperature} °C` : 'N/A';
            document.getElementById('humidityVal').innerText = data.humidity != null ? `${data.humidity} %` : 'N/A';
            document.getElementById('pressureVal').innerText = data.pressure != null ? `${data.pressure} hPa` : 'N/A';
            document.getElementById('windVal').innerText = data.wind_speed != null ? `${data.wind_speed} km/h` : 'N/A';
            document.getElementById('cloudVal').innerText = data.cloud_cover != null ? `${data.cloud_cover} %` : 'N/A';
            document.getElementById('dewVal').innerText = data.dew_point != null ? `${data.dew_point} °C` : 'N/A';

            // Warnings
            const warnBox = document.getElementById('warningContainer');
            if (data.warnings && data.warnings.length > 0) {
                warnBox.innerHTML = `<div class="warning-box">⚠️ ${data.warnings.join('<br>')}</div>`;
            } else {
                warnBox.innerHTML = '';
            }

            // Plotly Chart
            if (forecastSeries && forecastSeries.length > 0) {
                const dates = forecastSeries.map(r => r.date);
                const temps = forecastSeries.map(r => r.temperature);
                const probs = forecastSeries.map(r => r.rain_probability);

                const traceTemp = {
                    x: dates,
                    y: temps,
                    name: 'Temperature (°C)',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#38bdf8', width: 3 },
                    marker: { size: 6 }
                };

                const traceRain = {
                    x: dates,
                    y: probs,
                    name: 'Rain Prob (%)',
                    type: 'bar',
                    yaxis: 'y2',
                    marker: { color: 'rgba(129, 140, 248, 0.5)' }
                };

                const layout = {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    margin: { l: 40, r: 40, t: 20, b: 40 },
                    font: { color: '#94a3b8' },
                    xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                    yaxis: { title: 'Temperature (°C)', gridcolor: 'rgba(255,255,255,0.05)' },
                    yaxis2: {
                        title: 'Rain Probability (%)',
                        overlaying: 'y',
                        side: 'right',
                        range: [0, 100],
                        showgrid: false
                    },
                    legend: { orientation: 'h', y: 1.15 }
                };

                Plotly.newPlot('plotlyChart', [traceTemp, traceRain], layout, { responsive: true, displayModeBar: false });
            }
        }

        // Run on load
        window.addEventListener('DOMContentLoaded', () => {
            runPrediction();
        });
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
