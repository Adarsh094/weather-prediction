"""
utils/models.py
===============

**The model zoo.**

Five very different forecasting approaches, all wrapped behind the *same*
small interface so `train.py`, `forecast.py` and the notebook can treat them
interchangeably:

    model.fit(city_frame, target)      # learn from history
    model.predict(dates)               # -> numpy array of predicted values

| Class | Family | How it sees the problem |
|-------|--------|-------------------------|
| `RandomForestForecaster` | Machine Learning | "What is normal for this date?" |
| `XGBoostForecaster` | Gradient Boosting | Same, but corrects its own errors |
| `ProphetForecaster` | Statistical decomposition | trend + yearly season + noise |
| `ArimaForecaster` | Classical time series | today depends on recent days |
| `LSTMForecaster` | Deep Learning (PyTorch) | learns patterns in 30-day windows |

### Why not TensorFlow/Keras for the LSTM?

TensorFlow does not publish builds for Python 3.13+. **PyTorch** does, and it
is equally standard in industry and research, so the LSTM here is written in
PyTorch. The network architecture (LSTM layer -> dense output) is exactly what
a Keras version would use.

### A note on honesty

Long-range weather forecasting is genuinely hard. Beyond about ten days the
atmosphere is chaotic and no model can track individual weather systems. What
these models *can* do is predict the **seasonal normal plus a trend**, which is
what everybody means by "what will the weather be like in September?".
`utils/evaluation.py` compares every model against a plain climatology
baseline so you can see exactly how much value each one adds.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from utils import config, features

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Optional dependencies. Each model reports whether its library is installed,
# so the project still runs (with fewer models) if one of them is missing.
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:                                    # pragma: no cover
    HAS_XGBOOST = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:                                    # pragma: no cover
    HAS_PROPHET = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:                                    # pragma: no cover
    HAS_STATSMODELS = False

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:                                    # pragma: no cover
    HAS_TORCH = False

from sklearn.ensemble import RandomForestRegressor     # noqa: E402


# ===========================================================================
# Base class
# ===========================================================================

class BaseForecaster:
    """The common interface every forecaster in this project implements."""

    name = "Base"
    available = True

    def __init__(self):
        self.target = None
        self.last_train_date = None
        self.fitted = False

    def fit(self, city_frame, target):
        """Learn from one city's history. `city_frame` has date + target."""
        raise NotImplementedError

    def predict(self, dates):
        """Return predicted values for the given dates as a numpy array."""
        raise NotImplementedError

    def _remember_training_span(self, city_frame, target):
        self.target = target
        self.last_train_date = pd.Timestamp(
            city_frame[config.DATE_COLUMN].max()
        )
        self.fitted = True

    @staticmethod
    def _to_index(dates):
        return features._as_datetime_index(dates)

    def feature_importance(self):
        """Return a Series of feature importances, or None if not applicable."""
        return None


# ===========================================================================
# 1 + 2. Seasonal regressors: Random Forest and XGBoost
# ===========================================================================

class SeasonalRegressorForecaster(BaseForecaster):
    """Predicts a weather parameter from calendar features alone.

    The model learns the shape of the year - "in Delhi, day 190 is hot and
    humid; day 15 is cold and dry" - plus a slow multi-year trend. Because the
    inputs are pure calendar maths, it can predict **any** date, whether that
    is tomorrow or three years from now, with no recursion and no drift.
    """

    def __init__(self, estimator, name):
        super().__init__()
        self.estimator = estimator
        self.name = name
        self.feature_names = features.seasonal_feature_names()

    def fit(self, city_frame, target):
        x_train, y_train = features.build_seasonal_training_set(
            city_frame, target
        )
        self.estimator.fit(x_train[self.feature_names], y_train)
        self._remember_training_span(city_frame, target)
        return self

    def predict(self, dates):
        index = self._to_index(dates)
        x_future = features.seasonal_features(index)[self.feature_names]
        return np.asarray(self.estimator.predict(x_future), dtype=float)

    def feature_importance(self):
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return None
        return pd.Series(importances, index=self.feature_names)


def make_random_forest(target=None):
    """Random Forest tuned for smooth seasonal curves."""
    return SeasonalRegressorForecaster(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=8,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        name="Random Forest",
    )


def make_xgboost(target=None):
    """XGBoost - gradient boosting, usually a shade better than the forest."""
    if not HAS_XGBOOST:
        return None
    return SeasonalRegressorForecaster(
        xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        name="XGBoost",
    )


# ===========================================================================
# 3. Prophet
# ===========================================================================

class ProphetForecaster(BaseForecaster):
    """Facebook/Meta's Prophet.

    Prophet splits a series into pieces it can explain:

        y(t) = trend(t) + yearly season(t) + weekly season(t) + noise

    It was designed for exactly this kind of business-friendly forecasting:
    strong yearly seasonality, a slow trend, and the ability to name any future
    date. It also gives an uncertainty band for free.
    """

    name = "Prophet"

    def __init__(self):
        super().__init__()
        self.model = None
        self.available = HAS_PROPHET

    def fit(self, city_frame, target):
        if not HAS_PROPHET:
            raise ImportError("prophet is not installed")

        training = pd.DataFrame({
            "ds": pd.to_datetime(city_frame[config.DATE_COLUMN]),
            "y": city_frame[target].to_numpy(dtype=float),
        })

        self.model = Prophet(
            yearly_seasonality=10,      # 10 Fourier terms for the yearly cycle
            weekly_seasonality=False,   # weather does not care what day it is
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        self.model.fit(training)
        self._remember_training_span(city_frame, target)
        return self

    def predict(self, dates):
        index = self._to_index(dates)
        future = pd.DataFrame({"ds": index})
        forecast = self.model.predict(future)
        return forecast["yhat"].to_numpy(dtype=float)

    def predict_interval(self, dates):
        """Return (lower, upper) 80% uncertainty bounds - used by the charts."""
        index = self._to_index(dates)
        forecast = self.model.predict(pd.DataFrame({"ds": index}))
        return (
            forecast["yhat_lower"].to_numpy(dtype=float),
            forecast["yhat_upper"].to_numpy(dtype=float),
        )


# ===========================================================================
# 4. ARIMA (SARIMAX with Fourier seasonality)
# ===========================================================================

class ArimaForecaster(BaseForecaster):
    """Classical Box-Jenkins time series model.

    Plain ARIMA has no idea that weather repeats every year. The textbook fix
    is a *seasonal* ARIMA with period m=365, but fitting that on daily data is
    computationally impossible (the model would need 365 seasonal lags).

    The standard professional workaround, used here, is **SARIMAX with Fourier
    terms**: the yearly cycle is handed to the model as external regressors
    (sine/cosine harmonics) while ARIMA handles the short-term autocorrelation
    on top. This is the same trick Hyndman recommends for long seasonal periods.
    """

    name = "ARIMA"

    def __init__(self, order=config.ARIMA_ORDER):
        super().__init__()
        self.order = order
        self.result = None
        self.available = HAS_STATSMODELS

    def fit(self, city_frame, target):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is not installed")

        frame = city_frame.sort_values(config.DATE_COLUMN)
        series = frame[target].to_numpy(dtype=float)
        dates = pd.DatetimeIndex(frame[config.DATE_COLUMN])
        exog = features.fourier_terms(dates).to_numpy()

        model = SARIMAX(
            series,
            exog=exog,
            order=self.order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = model.fit(disp=False, maxiter=200)

        # Kept so the model can be rebuilt cheaply when it is loaded again -
        # see __getstate__ below.
        self._train_values = series
        self._train_dates = dates
        self._remember_training_span(frame, target)
        return self

    # -- keep the saved file small ----------------------------------------
    def __getstate__(self):
        """Save the fitted parameters, not the whole Kalman filter output.

        A SARIMAXResults object carries the filtered and smoothed state for
        every one of the ~3,300 training days. For this model that is a
        `12 x 12 x 3300` covariance array - about 2 MB per parameter, or
        nearly 10 MB for one city, and none of it is needed to forecast.

        Storing just the fitted parameters plus the training series (~30 KB)
        and re-running the Kalman *filter* on load reproduces the identical
        model. Filtering is not fitting: the parameters are already known, so
        there is no optimisation to redo and it takes well under a second.
        """
        state = self.__dict__.copy()
        result = state.pop("result", None)
        state["_fitted_params"] = (np.asarray(result.params, dtype=float)
                                   if result is not None else None)
        return state

    def __setstate__(self, state):
        """Rebuild the fitted model from its parameters."""
        params = state.pop("_fitted_params", None)
        self.__dict__.update(state)
        self.result = None

        if params is None or not HAS_STATSMODELS:
            return

        exog = features.fourier_terms(self._train_dates).to_numpy()
        model = SARIMAX(
            self._train_values,
            exog=exog,
            order=self.order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        # `filter` applies known parameters; `fit` would re-optimise them.
        self.result = model.filter(params)

    def predict(self, dates):
        index = self._to_index(dates)

        # SARIMAX forecasts forward step by step, so we forecast far enough to
        # cover the furthest requested date and then pick out the days we want.
        horizon = int((index.max() - self.last_train_date).days)
        if horizon < 1:
            horizon = 1

        future_dates = pd.date_range(
            self.last_train_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        exog_future = features.fourier_terms(future_dates).to_numpy()
        forecast = np.asarray(
            self.result.forecast(steps=horizon, exog=exog_future), dtype=float
        )

        lookup = pd.Series(forecast, index=future_dates)
        # Any requested date inside the training period falls back to the
        # closest forecast value we have (the app only asks for future dates).
        return lookup.reindex(index).ffill().bfill().to_numpy(dtype=float)


# ===========================================================================
# 5. LSTM (PyTorch)
# ===========================================================================

if HAS_TORCH:

    class _LSTMNetwork(nn.Module):
        """LSTM layer followed by a small dense head."""

        def __init__(self, n_features, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """Deep-learning forecaster built on PyTorch.

    **Input** at every timestep: the (scaled) weather value for that day plus
    the sine/cosine of where that day sits in the year. Giving the network the
    calendar position is what stops a long forecast from drifting away into a
    flat line - it always knows which season it is walking into.

    **Training**: slide a 30-day window over the history; the network learns to
    predict day 31 from days 1-30.

    **Forecasting**: roll forward one day at a time, feeding each prediction
    back in as the next input (recursive multi-step forecasting).
    """

    name = "LSTM"

    def __init__(self, lookback=config.LSTM_LOOKBACK,
                 hidden_size=config.LSTM_HIDDEN_SIZE,
                 epochs=config.LSTM_EPOCHS,
                 batch_size=config.LSTM_BATCH_SIZE,
                 learning_rate=config.LSTM_LEARNING_RATE):
        super().__init__()
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.network = None
        self.mean = 0.0
        self.std = 1.0
        self.recent_values = None      # last `lookback` scaled values
        self.recent_dates = None
        self.available = HAS_TORCH

    # -- helpers ----------------------------------------------------------
    def _scale(self, values):
        return (values - self.mean) / self.std

    def _unscale(self, values):
        return values * self.std + self.mean

    @staticmethod
    def _season_columns(dates):
        index = features._as_datetime_index(dates)
        angle = 2.0 * np.pi * index.dayofyear.to_numpy(dtype=float) / 365.25
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def _build_windows(self, scaled, season):
        """Turn a flat series into (samples, lookback, n_features) windows."""
        stacked = np.column_stack([scaled, season])   # value + sin + cos
        x_list, y_list = [], []
        for start in range(len(scaled) - self.lookback):
            x_list.append(stacked[start:start + self.lookback])
            y_list.append(scaled[start + self.lookback])
        return np.asarray(x_list, dtype=np.float32), np.asarray(
            y_list, dtype=np.float32
        )

    # -- interface --------------------------------------------------------
    def fit(self, city_frame, target, verbose=False):
        if not HAS_TORCH:
            raise ImportError("torch is not installed")

        torch.manual_seed(config.RANDOM_STATE)
        np.random.seed(config.RANDOM_STATE)

        frame = city_frame.sort_values(config.DATE_COLUMN)
        values = frame[target].to_numpy(dtype=float)
        dates = pd.DatetimeIndex(frame[config.DATE_COLUMN])

        # Standardise: neural networks train far better on centred data.
        self.mean = float(values.mean())
        self.std = float(values.std()) or 1.0
        scaled = self._scale(values)
        season = self._season_columns(dates)

        x_train, y_train = self._build_windows(scaled, season)
        x_tensor = torch.from_numpy(x_train)
        y_tensor = torch.from_numpy(y_train)

        self.network = _LSTMNetwork(n_features=3, hidden_size=self.hidden_size)
        optimiser = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        loss_function = nn.MSELoss()

        n_samples = len(x_tensor)
        self.network.train()
        for epoch in range(self.epochs):
            order = torch.randperm(n_samples)
            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                batch = order[start:start + self.batch_size]
                optimiser.zero_grad()
                prediction = self.network(x_tensor[batch])
                loss = loss_function(prediction, y_tensor[batch])
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"        epoch {epoch + 1:>3}/{self.epochs}  "
                      f"loss={epoch_loss / n_samples:.4f}")

        self.network.eval()

        # Remember the tail of the series - that is where forecasting starts.
        self.recent_values = scaled[-self.lookback:].copy()
        self.recent_dates = dates[-self.lookback:]
        self._remember_training_span(frame, target)
        return self

    def predict(self, dates):
        index = self._to_index(dates)
        horizon = int((index.max() - self.last_train_date).days)
        if horizon < 1:
            horizon = 1

        future_dates = pd.date_range(
            self.last_train_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        buffer_values = list(self.recent_values)
        buffer_dates = list(self.recent_dates)
        predictions = []

        with torch.no_grad():
            for future_date in future_dates:
                window_dates = pd.DatetimeIndex(buffer_dates[-self.lookback:])
                window = np.column_stack([
                    np.asarray(buffer_values[-self.lookback:], dtype=np.float32),
                    self._season_columns(window_dates).astype(np.float32),
                ])
                x_input = torch.from_numpy(window[None, :, :].astype(np.float32))
                next_scaled = float(self.network(x_input).item())

                predictions.append(next_scaled)
                buffer_values.append(next_scaled)     # feed the answer back in
                buffer_dates.append(future_date)

        lookup = pd.Series(
            self._unscale(np.asarray(predictions, dtype=float)),
            index=future_dates,
        )
        return lookup.reindex(index).ffill().bfill().to_numpy(dtype=float)

    # -- make the model safe to save with joblib --------------------------
    def __getstate__(self):
        """Store the network's weights, not the live PyTorch object."""
        state = self.__dict__.copy()
        network = state.pop("network", None)
        state["_network_state"] = (
            {k: v.cpu().numpy() for k, v in network.state_dict().items()}
            if network is not None else None
        )
        return state

    def __setstate__(self, state):
        """Rebuild the network from the saved weights."""
        network_state = state.pop("_network_state", None)
        self.__dict__.update(state)
        self.network = None
        if network_state is not None and HAS_TORCH:
            self.network = _LSTMNetwork(
                n_features=3, hidden_size=self.hidden_size
            )
            self.network.load_state_dict(
                {k: torch.tensor(v) for k, v in network_state.items()}
            )
            self.network.eval()


# ===========================================================================
# Climatology baseline (not a "model", but the bar every model must clear)
# ===========================================================================

class ClimatologyBaseline(BaseForecaster):
    """Predicts the 10-year average weather for that day of the year.

    This is what a farmer with a good almanac would tell you. If a fancy model
    cannot beat it, the fancy model is not earning its keep - so we always
    report it alongside the real models.
    """

    name = "Climatology (baseline)"

    def __init__(self):
        super().__init__()
        self.day_of_year_mean = None

    def fit(self, city_frame, target):
        self.day_of_year_mean = features.climatology(city_frame, target)
        self._remember_training_span(city_frame, target)
        return self

    def predict(self, dates):
        index = self._to_index(dates)
        day_numbers = index.dayofyear
        return self.day_of_year_mean.reindex(day_numbers).to_numpy(dtype=float)


# ===========================================================================
# Registry
# ===========================================================================

def build_all_forecasters():
    """Create one fresh instance of every available forecasting model."""
    candidates = [
        make_random_forest(),
        make_xgboost(),
        ProphetForecaster() if HAS_PROPHET else None,
        ArimaForecaster() if HAS_STATSMODELS else None,
        LSTMForecaster() if HAS_TORCH else None,
        ClimatologyBaseline(),
    ]
    return [model for model in candidates if model is not None]


def build_forecaster_by_name(name):
    """Create a single forecaster by its display name."""
    for model in build_all_forecasters():
        if model.name == name:
            return model
    raise KeyError(f"Unknown forecaster '{name}'")


def missing_libraries():
    """Report which optional libraries are absent, for a clear warning."""
    missing = []
    if not HAS_XGBOOST:
        missing.append("xgboost")
    if not HAS_PROPHET:
        missing.append("prophet")
    if not HAS_STATSMODELS:
        missing.append("statsmodels")
    if not HAS_TORCH:
        missing.append("torch (LSTM)")
    return missing


# ===========================================================================
# Rain classifiers
# ===========================================================================

def build_rain_classifiers():
    """Random Forest and XGBoost classifiers for the Rain / No Rain decision.

    ### Why the forest is deliberately small

    The obvious instinct is a big forest - 400 deep trees. Measured against the
    held-out year across all six cities, that turned out to be **worse**:

        400 trees, depth 14, leaf 4   ->  ROC-AUC 0.9470,  19.0 MB
        120 trees, depth  8, leaf 16  ->  ROC-AUC 0.9504,   2.6 MB

    The deep forest was memorising individual days rather than learning the
    relationship between humidity, cloud cover and rain. Constraining it both
    improved every metric and cut the saved model by 87%, which matters when
    the models have to be uploaded to a cloud host.

    A smaller model that scores better is not a compromise - it is the
    correct model.
    """
    from sklearn.ensemble import RandomForestClassifier

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=16,
            class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    }

    if HAS_XGBOOST:
        classifiers["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    return classifiers
