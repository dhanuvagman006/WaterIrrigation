import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_monsoon"] = df["month"].isin(config.SEASON_MAPPING["monsoon"]).astype(int)
    return df


def build_sequences(features: np.ndarray, targets: np.ndarray, lookback: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    seq_targets = []
    for idx in range(lookback, len(features) - horizon + 1):
        sequences.append(features[idx - lookback : idx])
        seq_targets.append(targets[idx + horizon - 1])
    return np.array(sequences), np.array(seq_targets)


def time_series_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.15, test_ratio: float = 0.15):
    total = len(X)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_end = total - val_size - test_size
    val_end = total - test_size
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test
