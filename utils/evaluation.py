"""
utils/evaluation.py
===================

**Model Evaluation module.**

Two different jobs need two different sets of metrics:

Forecasting a number (temperature, humidity, ...) -> **regression metrics**

| Metric | Meaning | Good value |
|--------|---------|-----------|
| **MAE** | Mean Absolute Error: the average miss, in °C | as low as possible |
| **RMSE** | Root Mean Squared Error: like MAE but punishes big misses harder | as low as possible |
| **R²** | Share of the variation the model explains | 1.0 is perfect, 0.0 is no better than always guessing the average |

Predicting Rain / No Rain -> **classification metrics**
(accuracy, precision, recall, F1, ROC-AUC).

### Reading R² for a weather forecast

R² is measured against "always predict the overall average". For a city with
strong seasons, simply knowing the time of year already explains most of the
variation, so R² above 0.9 is normal and does **not** mean the model is
magic. That is exactly why every comparison here also includes the
**climatology baseline** - it shows how much the model adds on top of just
knowing the date.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ===========================================================================
# Regression (forecasting) metrics
# ===========================================================================

def regression_metrics(actual, predicted):
    """Return MAE, RMSE and R² for one forecast."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Ignore any position where either value is missing.
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Bias": np.nan}

    errors = predicted - actual
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "Bias": float(np.mean(errors)),      # + = model runs warm, - = cold
    }


def build_comparison_table(results, sort_by="RMSE"):
    """Turn a list of per-model metric dictionaries into a ranked table."""
    table = pd.DataFrame(results)
    if table.empty:
        return table

    table = table.set_index("Model")
    ascending = sort_by != "R2"          # for R² bigger is better
    return table.sort_values(sort_by, ascending=ascending)


def pick_best_model(table, metric="RMSE", exclude=("Climatology (baseline)",)):
    """Return the name of the best real model (baselines do not count).

    The baseline is excluded on purpose: it is the bar to clear, not a
    candidate for deployment.
    """
    candidates = table.drop(index=[c for c in exclude if c in table.index],
                            errors="ignore")
    if candidates.empty:
        return table.index[0]

    if metric == "R2":
        return candidates[metric].idxmax()
    return candidates[metric].idxmin()


def skill_score(model_rmse, baseline_rmse):
    """How much better than climatology, as a percentage.

    +20% means the model's error is 20% smaller than the baseline's.
    A negative number means the model is *worse* than just using the average.
    """
    if not baseline_rmse or np.isnan(baseline_rmse):
        return np.nan
    return float((1.0 - model_rmse / baseline_rmse) * 100.0)


def horizon_error_profile(actual, predicted, dates, forecast_start,
                          buckets=((1, 7), (8, 30), (31, 90), (91, 365))):
    """Measure error separately for each forecast distance.

    This answers the question that actually matters for this project:
    *how much worse does the forecast get the further out you look?*
    The app uses this profile to compute an honest confidence score.
    """
    dates = pd.DatetimeIndex(dates)
    days_ahead = (dates - pd.Timestamp(forecast_start)).days

    rows = []
    for low, high in buckets:
        mask = (days_ahead >= low) & (days_ahead <= high)
        if mask.sum() == 0:
            continue
        metrics = regression_metrics(np.asarray(actual)[mask],
                                     np.asarray(predicted)[mask])
        rows.append({
            "Horizon": f"{low}-{high} days",
            "Days": int(mask.sum()),
            **{k: round(v, 3) for k, v in metrics.items()},
        })

    return pd.DataFrame(rows).set_index("Horizon") if rows else pd.DataFrame()


# ===========================================================================
# Classification (Rain / No Rain) metrics
# ===========================================================================

def classification_metrics(actual, predicted, probabilities=None):
    """Accuracy, precision, recall, F1 and (if given) ROC-AUC + Brier score."""
    scores = {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
    }

    if probabilities is not None and len(set(actual)) > 1:
        scores["ROC-AUC"] = float(roc_auc_score(actual, probabilities))
        # Brier score = mean squared error of the probability itself.
        # Lower is better; it tells us whether "70%" really means 70%.
        scores["Brier"] = float(brier_score_loss(actual, probabilities))

    return scores


def rain_report(actual, predicted, labels=("No Rain", "Rain")):
    """Full scikit-learn classification report as text."""
    return classification_report(actual, predicted, target_names=list(labels),
                                 zero_division=0)


def rain_confusion(actual, predicted):
    """Confusion matrix as a labelled DataFrame, ready to print or plot."""
    matrix = confusion_matrix(actual, predicted)
    return pd.DataFrame(
        matrix,
        index=["Actual: No Rain", "Actual: Rain"],
        columns=["Predicted: No Rain", "Predicted: Rain"],
    )


def reliability_table(actual, probabilities, n_bins=10):
    """Check whether the predicted probabilities are trustworthy.

    Group every prediction by its stated probability, then compare with what
    actually happened. On days the model said "70% chance", did it rain about
    70% of the time? A well-calibrated model matches closely - which matters
    here because the app shows the user a rain **percentage**, not just a
    yes/no answer.
    """
    frame = pd.DataFrame({
        "actual": np.asarray(actual, dtype=float),
        "probability": np.asarray(probabilities, dtype=float),
    })
    frame["bucket"] = pd.cut(
        frame["probability"], bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )

    grouped = frame.groupby("bucket", observed=True).agg(
        Days=("actual", "size"),
        Predicted=("probability", "mean"),
        Actual=("actual", "mean"),
    )
    grouped["Predicted %"] = (grouped["Predicted"] * 100).round(1)
    grouped["Actual %"] = (grouped["Actual"] * 100).round(1)
    grouped["Gap"] = (grouped["Actual %"] - grouped["Predicted %"]).round(1)

    return grouped[["Days", "Predicted %", "Actual %", "Gap"]]
