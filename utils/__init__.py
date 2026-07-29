"""
Smart Weather Prediction System - reusable modules.

Each module has one clear job (clean architecture):

    config          settings, paths, city list
    data_loader     downloading and reading historical weather
    preprocessing   cleaning, gap filling, building the rain target
    features        feature engineering for the models
    models          the model zoo (Random Forest, XGBoost, Prophet, LSTM, ARIMA)
    evaluation      MAE / RMSE / R2 and the comparison tables
    forecasting     turning trained models into future weather values
    predictor       the single entry point the app calls
    visualization   all Plotly charts
"""

__version__ = "1.0.0"
