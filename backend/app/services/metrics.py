import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = np.where(y_true != 0, y_true, np.nan)
    return np.nanmean(np.abs((y_true - y_pred) / non_zero)) * 100


def nash_sutcliffe_efficiency(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": float(mape),
        "NSE": float(nse),
        "ExplainedVariance": float(explained_variance),
    }
