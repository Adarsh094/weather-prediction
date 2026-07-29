"""
train.py
========

**The complete training pipeline for the Smart Weather Prediction System.**

For every configured city this script:

1. Loads 10 years of real daily weather (downloading it first if needed).
2. Cleans it - gap filling, duplicate removal, physical range checks.
3. Splits it **chronologically**: the final year is hidden as a test set.
4. Trains and compares five forecasting models plus a climatology baseline:
   Random Forest, XGBoost, Prophet, ARIMA and LSTM.
5. Scores every model with **MAE, RMSE and R2** and picks the winner.
6. Measures how the winner's accuracy decays with forecast distance, which
   becomes the app's honest confidence score.
7. Retrains the winning model family on the full history, once per weather
   parameter.
8. Trains and compares Rain / No Rain classifiers (Random Forest vs XGBoost).
9. Saves everything with Joblib into `models/<city>/`.

Usage
-----
    python train.py                        # every city
    python train.py --cities Delhi Mumbai  # just these
    python train.py --quick                # skip the slow models (LSTM, ARIMA)
    python train.py --no-download          # fail instead of downloading data
"""

import argparse
import datetime as dt
import json
import time
import warnings

import numpy as np
import pandas as pd

from utils import config, evaluation, features, forecasting, models
from utils.data_loader import load_raw_history
from utils.preprocessing import (
    clean_history,
    save_clean_history,
    summarise,
    train_test_split_by_time,
)

warnings.filterwarnings("ignore")

# The forecast distances we measure separately, so the app can tell the user
# how much to trust a 5-day answer versus a 5-month one.
HORIZON_BUCKETS = ((1, 7), (8, 30), (31, 90), (91, 180))

# Rolling-origin backtest settings (see `rolling_origin_backtest`).
BACKTEST_ORIGINS = 6        # how many different "pretend todays" to test from
BACKTEST_HORIZON = 180      # how many days to forecast from each origin
BACKTEST_STEP = 60          # gap between origins, in days


def banner(text, char="="):
    print("\n" + char * 74)
    print(text)
    print(char * 74)


# ===========================================================================
# Step 1 - the five-model comparison, for one city
# ===========================================================================

def compare_forecast_models(city, train_frame, test_frame, target,
                            quick=False):
    """Train every model on the training years, score it on the hidden year."""
    test_dates = test_frame[config.DATE_COLUMN]
    actual = test_frame[target].to_numpy(dtype=float)

    candidates = models.build_all_forecasters()
    if quick:
        slow = {"LSTM", "ARIMA"}
        candidates = [m for m in candidates if m.name not in slow]

    results, predictions, fitted = [], {}, {}

    for model in candidates:
        started = time.time()
        try:
            model.fit(train_frame, target)
            predicted = model.predict(test_dates)
        except Exception as error:                      # noqa: BLE001
            print(f"      {model.name:<24} FAILED: {error}")
            continue

        metrics = evaluation.regression_metrics(actual, predicted)
        elapsed = time.time() - started

        results.append({
            "Model": model.name,
            "MAE": round(metrics["MAE"], 3),
            "RMSE": round(metrics["RMSE"], 3),
            "R2": round(metrics["R2"], 4),
            "Bias": round(metrics["Bias"], 3),
            "Fit (s)": round(elapsed, 1),
        })
        predictions[model.name] = predicted
        fitted[model.name] = model

        print(f"      {model.name:<24} MAE={metrics['MAE']:6.3f}  "
              f"RMSE={metrics['RMSE']:6.3f}  R2={metrics['R2']:7.4f}  "
              f"({elapsed:.1f}s)")

    table = evaluation.build_comparison_table(results, sort_by="RMSE")
    return table, predictions, fitted


def rolling_origin_backtest(model_name, city_frame, target,
                            n_origins=BACKTEST_ORIGINS,
                            horizon=BACKTEST_HORIZON,
                            step=BACKTEST_STEP):
    """Measure forecast error properly, as a function of how far ahead it is.

    ### Why a single train/test split is not enough here

    A plain chronological split gives us exactly **one** 7-day-ahead forecast -
    the first week of the test set. Judging "how good is this model one week
    out?" from a single week is close to meaningless.

    **Rolling-origin backtesting** (also called walk-forward validation) fixes
    this. We pretend "today" is some date in the past, train only on data
    before it, forecast forward, and compare with what really happened. Then we
    slide that pretend-today forward and do it again:

        origin 1:  train |------------|  forecast ->>>>>>
        origin 2:  train |----------------|  forecast ->>>>>>
        origin 3:  train |--------------------|  forecast ->>>>>>

    Pooling the errors from every origin gives many independent samples at each
    forecast distance, so the resulting numbers actually mean something.

    Returns a tidy frame of (days_ahead, actual, predicted) across all origins.
    """
    frame = city_frame.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    last_date = frame[config.DATE_COLUMN].max()

    records = []
    for origin_number in range(n_origins):
        # Work backwards from the most recent usable origin.
        origin_date = last_date - pd.Timedelta(
            days=horizon + origin_number * step
        )
        train_part = frame[frame[config.DATE_COLUMN] <= origin_date]
        future_part = frame[
            (frame[config.DATE_COLUMN] > origin_date)
            & (frame[config.DATE_COLUMN] <= origin_date
               + pd.Timedelta(days=horizon))
        ]

        # Need a few years of history behind each origin to train on.
        if len(train_part) < 730 or future_part.empty:
            continue

        try:
            model = models.build_forecaster_by_name(model_name)
            model.fit(train_part, target)
            predicted = model.predict(future_part[config.DATE_COLUMN])
        except Exception as error:                          # noqa: BLE001
            print(f"      backtest origin {origin_date.date()} failed: {error}")
            continue

        days_ahead = (future_part[config.DATE_COLUMN] - origin_date).dt.days
        records.append(pd.DataFrame({
            "origin": origin_date,
            "days_ahead": days_ahead.to_numpy(),
            "actual": future_part[target].to_numpy(dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }))

    if not records:
        return pd.DataFrame(columns=["origin", "days_ahead", "actual",
                                     "predicted"])
    return pd.concat(records, ignore_index=True)


def measure_horizon_reliability(backtest, natural_spread):
    """Turn the rolling-origin backtest into a trust factor per horizon.

    The factor compares the model's typical error with how much the weather
    naturally varies:

        reliability = 1 - MAE / (2 x standard deviation of the target)

    If the model's average miss is small next to the natural swing of the
    weather, the factor is close to 1. If the model is no better than guessing,
    it falls towards 0. We clip it to a sane 0.35-0.95 so the app never claims
    either certainty or total ignorance.

    We deliberately use **MAE**, not R2, for this. Within a single week the
    temperature barely moves, so "what share of the variance did you explain?"
    is an ill-posed question at short horizons and R2 comes out wildly
    negative. Average error in degrees stays meaningful at every distance.
    """
    profile, detail = {}, []

    if backtest.empty:
        return profile, pd.DataFrame()

    for low, high in HORIZON_BUCKETS:
        window = backtest[
            (backtest["days_ahead"] >= low) & (backtest["days_ahead"] <= high)
        ]
        if len(window) < 10:
            continue

        metrics = evaluation.regression_metrics(window["actual"],
                                                window["predicted"])
        factor = float(np.clip(1.0 - metrics["MAE"] / (2.0 * natural_spread),
                               0.35, 0.95))
        profile[f"{low}-{high}"] = round(factor, 3)
        detail.append({
            "Horizon": f"{low}-{high} days",
            "Samples": int(len(window)),
            "MAE": round(metrics["MAE"], 3),
            "RMSE": round(metrics["RMSE"], 3),
            "Reliability": round(factor, 3),
        })

    return profile, pd.DataFrame(detail)


# ===========================================================================
# Step 2 - the Rain / No Rain classifier
# ===========================================================================

def train_rain_classifier(city, train_frame, test_frame):
    """Compare Random Forest and XGBoost, then keep the better one."""
    x_train, y_train = features.build_rain_training_set(train_frame)
    x_test, y_test = features.build_rain_training_set(test_frame)

    results, fitted = [], {}

    for name, classifier in models.build_rain_classifiers().items():
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        probabilities = classifier.predict_proba(x_test)[:, 1]

        scores = evaluation.classification_metrics(y_test, predicted,
                                                   probabilities)
        results.append({"Model": name,
                        **{k: round(v, 4) for k, v in scores.items()}})
        fitted[name] = classifier

        print(f"      {name:<24} Acc={scores['Accuracy']:.4f}  "
              f"F1={scores['F1']:.4f}  "
              f"AUC={scores.get('ROC-AUC', float('nan')):.4f}")

    table = pd.DataFrame(results).set_index("Model")
    # ROC-AUC judges the quality of the probability itself, which is what the
    # app actually shows the user - so we rank on it rather than accuracy.
    best_name = table["ROC-AUC"].idxmax() if "ROC-AUC" in table else \
        table["F1"].idxmax()

    return table, best_name, fitted


# ===========================================================================
# Step 3 - train one city end to end
# ===========================================================================

def train_city(city, city_frame, quick=False):
    """Run the whole pipeline for a single city and save its models."""
    banner(f"CITY: {city}", "-")

    train_frame, test_frame = train_test_split_by_time(city_frame)
    forecast_start = train_frame[config.DATE_COLUMN].max()

    print(f"   history      : {city_frame[config.DATE_COLUMN].min().date()} "
          f"-> {city_frame[config.DATE_COLUMN].max().date()} "
          f"({len(city_frame):,} days)")
    print(f"   training on  : {len(train_frame):,} days")
    print(f"   testing on   : {len(test_frame):,} days (the final year, unseen)")

    # ---- 1. Compare the five forecasting models ------------------------
    print(f"\n   [1] Forecasting '{config.PRIMARY_TARGET}' - model comparison")
    comparison, predictions, fitted = compare_forecast_models(
        city, train_frame, test_frame, config.PRIMARY_TARGET, quick=quick
    )

    if comparison.empty:
        raise RuntimeError(f"No forecasting model trained successfully for {city}")

    # The best of the five trained models...
    best_trained = evaluation.pick_best_model(comparison, metric="RMSE")
    # ...and the best of everything, the plain climatology baseline included.
    best_name = evaluation.pick_best_model(comparison, metric="RMSE",
                                           exclude=())

    baseline_rmse = comparison.loc[models.ClimatologyBaseline.name, "RMSE"] \
        if models.ClimatologyBaseline.name in comparison.index else np.nan
    skill = evaluation.skill_score(comparison.loc[best_trained, "RMSE"],
                                   baseline_rmse)

    print(f"\n      Best trained model: {best_trained}  "
          f"(RMSE={comparison.loc[best_trained, 'RMSE']:.3f}, "
          f"R2={comparison.loc[best_trained, 'R2']:.4f})")
    print(f"      Skill vs climatology baseline: {skill:+.1f}%")

    if best_name != best_trained:
        # Deploying a model that loses to a 10-year average would be bad
        # engineering, however sophisticated the model is. The baseline is a
        # legitimate forecaster - it implements the same interface - so when it
        # wins, it ships.
        print(f"      NOTE: the climatology baseline beat every trained model "
              f"here, so it is what gets deployed for {city}.")

    print(f"      DEPLOYING: {best_name}")

    # ---- 2. How far ahead can we trust it? -----------------------------
    print(f"\n   [2] Rolling-origin backtest of {best_name} "
          f"({BACKTEST_ORIGINS} origins x {BACKTEST_HORIZON} days)")
    backtest = rolling_origin_backtest(
        best_name, city_frame, config.PRIMARY_TARGET
    )
    natural_spread = float(city_frame[config.PRIMARY_TARGET].std()) or 1.0
    reliability, horizon_table = measure_horizon_reliability(
        backtest, natural_spread
    )
    if horizon_table.empty:
        print("      not enough history for a rolling backtest; "
              "using default reliabilities")
        reliability = {"1-7": 0.90, "8-30": 0.85, "31-90": 0.75, "91-180": 0.65}
    else:
        print(horizon_table.to_string(index=False))

    # ---- 3. Retrain the winner on ALL the data, per parameter ----------
    print(f"\n   [3] Training final '{best_name}' forecasters on full history")
    final_forecasters = {}
    for target in config.FORECAST_TARGETS:
        started = time.time()
        model = models.build_forecaster_by_name(best_name)
        model.fit(city_frame, target)
        final_forecasters[target] = model
        print(f"      {config.PRETTY_NAMES.get(target, target):<14} "
              f"trained ({time.time() - started:.1f}s)")

    # ---- 4. Rain / No Rain classifier ----------------------------------
    print("\n   [4] Rain / No Rain classifier")
    rain_table, best_rain_name, rain_fitted = train_rain_classifier(
        city, train_frame, test_frame
    )
    print(f"      BEST: {best_rain_name}")

    # Retrain the winning classifier on every day we have.
    x_all, y_all = features.build_rain_training_set(city_frame)
    final_classifier = models.build_rain_classifiers()[best_rain_name]
    final_classifier.fit(x_all, y_all)

    # Check the probabilities actually mean something (calibration).
    x_test, y_test = features.build_rain_training_set(test_frame)
    test_probabilities = rain_fitted[best_rain_name].predict_proba(x_test)[:, 1]
    calibration = evaluation.reliability_table(y_test, test_probabilities)

    # ---- 5. Save everything --------------------------------------------
    climatology_pack = forecasting.build_climatology_pack(city_frame)

    importance = final_classifier.feature_importances_
    metadata = {
        "city": city,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "history_start": str(city_frame[config.DATE_COLUMN].min().date()),
        "history_end": str(city_frame[config.DATE_COLUMN].max().date()),
        "n_days": int(len(city_frame)),
        "best_forecast_model": best_name,
        "best_trained_model": best_trained,
        "baseline_won": bool(best_name != best_trained),
        "best_rain_model": best_rain_name,
        "primary_target": config.PRIMARY_TARGET,
        "forecast_metrics": comparison.to_dict(orient="index"),
        "rain_metrics": rain_table.to_dict(orient="index"),
        "horizon_reliability": reliability,
        "horizon_detail": horizon_table.to_dict(orient="records"),
        "backtest_origins": int(backtest["origin"].nunique())
        if not backtest.empty else 0,
        "skill_vs_climatology_pct": (round(skill, 2)
                                     if not np.isnan(skill) else None),
        "rain_feature_importance": dict(
            zip(features.rain_feature_names(), [float(v) for v in importance])
        ),
        "rain_rate_pct": round(float(city_frame[config.RAIN].mean() * 100), 2),
        "calibration": calibration.reset_index().astype(str).to_dict("records"),
    }

    folder = forecasting.save_city_artifacts(
        city, final_forecasters, final_classifier, climatology_pack, metadata
    )
    print(f"\n   saved -> {folder}")

    return {
        "city": city,
        "comparison": comparison,
        "rain_table": rain_table,
        "best_forecast_model": best_name,
        "best_trained_model": best_trained,
        "best_rain_model": best_rain_name,
        "skill": skill,
        "horizon": horizon_table,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train the Smart Weather Prediction System"
    )
    parser.add_argument("--cities", nargs="*", default=None,
                        help="cities to train (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="skip the slow models (LSTM and ARIMA)")
    parser.add_argument("--no-download", action="store_true",
                        help="fail if the dataset is missing instead of "
                             "downloading it")
    arguments = parser.parse_args()

    config.ensure_directories()
    started = time.time()

    banner("SMART WEATHER PREDICTION SYSTEM - TRAINING")

    missing = models.missing_libraries()
    if missing:
        print(f"NOTE: these optional libraries are not installed, so their "
              f"models will be skipped: {', '.join(missing)}")

    # ---- Load and clean -------------------------------------------------
    banner("STEP 1 : DATA COLLECTION")
    raw = load_raw_history(download_if_missing=not arguments.no_download)
    print(f"   rows: {len(raw):,}   cities: {raw[config.CITY_COLUMN].nunique()}")

    banner("STEP 2 : DATA PREPROCESSING")
    clean = clean_history(raw)
    save_clean_history(clean)

    print("\nPer-city summary of the cleaned data:")
    print(summarise(clean).to_string())

    # ---- Train city by city --------------------------------------------
    banner("STEP 3 : MODEL TRAINING AND COMPARISON")
    cities = arguments.cities or list(config.CITIES)
    outcomes = []

    for city in cities:
        city_frame = clean[clean[config.CITY_COLUMN] == city].copy()
        if city_frame.empty:
            print(f"\n   skipping {city}: no data")
            continue
        outcomes.append(train_city(city, city_frame, quick=arguments.quick))

    if not outcomes:
        print("\nNothing was trained.")
        return

    # ---- Aggregate the results -----------------------------------------
    banner("STEP 4 : OVERALL RESULTS")

    all_comparisons = []
    for outcome in outcomes:
        frame = outcome["comparison"].reset_index()
        frame.insert(0, "City", outcome["city"])
        all_comparisons.append(frame)

    combined = pd.concat(all_comparisons, ignore_index=True)
    combined.to_csv(config.MODEL_COMPARISON_FILE, index=False)

    print(f"\nForecasting model comparison "
          f"(target = {config.PRIMARY_TARGET}, averaged over all cities):")
    average = (combined.groupby("Model")[["MAE", "RMSE", "R2"]]
               .mean().round(4).sort_values("RMSE"))
    print(average.to_string())

    print("\nWinner per city (deployed model, and the best of the five "
          "trained models):")
    for outcome in outcomes:
        flag = ("  <- baseline beat every trained model"
                if outcome["best_forecast_model"]
                != outcome["best_trained_model"] else "")
        print(f"   {outcome['city']:<12} deployed: "
              f"{outcome['best_forecast_model']:<24} "
              f"best trained: {outcome['best_trained_model']:<16} "
              f"skill: {outcome['skill']:+5.1f}%{flag}")

    rain_frames = []
    for outcome in outcomes:
        frame = outcome["rain_table"].reset_index()
        frame.insert(0, "City", outcome["city"])
        rain_frames.append(frame)
    rain_combined = pd.concat(rain_frames, ignore_index=True)
    rain_combined.to_csv(config.RAIN_COMPARISON_FILE, index=False)

    print("\nRain classifier comparison (averaged over all cities):")
    rain_average = (rain_combined.groupby("Model")
                    .mean(numeric_only=True).round(4))
    print(rain_average.to_string())

    summary = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "cities": [o["city"] for o in outcomes],
        "primary_target": config.PRIMARY_TARGET,
        "average_forecast_metrics": average.to_dict(orient="index"),
        "average_rain_metrics": rain_average.to_dict(orient="index"),
        "winner_per_city": {o["city"]: o["best_forecast_model"]
                            for o in outcomes},
        "rain_winner_per_city": {o["city"]: o["best_rain_model"]
                                 for o in outcomes},
        "training_seconds": round(time.time() - started, 1),
    }
    with open(config.TRAINING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    banner("TRAINING COMPLETE")
    print(f"   total time : {summary['training_seconds']} s")
    print(f"   comparison : {config.MODEL_COMPARISON_FILE}")
    print(f"   summary    : {config.TRAINING_SUMMARY_FILE}")
    print("\n   Next step  : streamlit run app.py")


if __name__ == "__main__":
    main()
