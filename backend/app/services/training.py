import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from app import config
from app.services import features, metrics as metrics_module
from app.services.models import (
    CNNLSTMModel,
    FeedForwardModel,
    GRUModel,
    LSTMModel,
    TCNModel,
    TransformerModel,
)


def _prepare_data(df: pd.DataFrame):
    df = features.add_time_features(df)
    X = df[config.FEATURE_COLUMNS].to_numpy()
    y = df[config.TARGET_COLUMN].to_numpy().reshape(-1, 1)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    X_seq, y_seq = features.build_sequences(X_scaled, y_scaled, config.LOOKBACK_HOURS, config.FORECAST_HORIZON)
    return X_seq, y_seq, feature_scaler, target_scaler


def _make_loaders(X_train, y_train, X_val, y_val, batch_size: int, feedforward: bool = False):
    if feedforward:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def _train_model(model: nn.Module, train_loader, val_loader, config_params: Dict, device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config_params["learning_rate"])
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 6
    wait = 0

    for _ in range(config_params["epochs"]):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return history


def _evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, target_scaler: StandardScaler, device: torch.device, feedforward: bool = False):
    model.eval()
    if feedforward:
        X_test = X_test.reshape(X_test.shape[0], -1)
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    y_true = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(preds)
    return y_true.flatten(), y_pred.flatten()


def _build_model(model_name: str, num_features: int, config_params: Dict):
    if model_name == "lstm":
        return LSTMModel(num_features, config_params["hidden_size"], config_params["num_layers"], config_params["dropout"])
    if model_name == "gru":
        return GRUModel(num_features, config_params["hidden_size"], config_params["num_layers"], config_params["dropout"])
    if model_name == "cnn_lstm":
        return CNNLSTMModel(
            num_features,
            config_params["conv_channels"],
            config_params["kernel_size"],
            config_params["hidden_size"],
            config_params["num_layers"],
            config_params["dropout"],
        )
    if model_name == "transformer":
        return TransformerModel(
            num_features,
            config_params["d_model"],
            config_params["nhead"],
            config_params["num_layers"],
            config_params["dropout"],
        )
    if model_name == "tcn":
        return TCNModel(num_features, config_params["num_channels"], config_params["kernel_size"], config_params["dropout"])
    if model_name == "ann":
        input_size = num_features * config.LOOKBACK_HOURS
        return FeedForwardModel(input_size, config_params["hidden_sizes"], config_params["dropout"])
    raise ValueError(f"Unknown model {model_name}")


def train_all_models(data_path: Path = config.DATA_PATH, output_dir: Path = config.MODEL_DIR, artifacts_dir: Path = config.ARTIFACTS_DIR):
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    X_seq, y_seq, feature_scaler, target_scaler = _prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = features.time_series_split(X_seq, y_seq)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, artifacts_dir / "feature_scaler.pkl")
    joblib.dump(target_scaler, artifacts_dir / "target_scaler.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    histories = {}
    predictions = {}

    for model_name, config_params in tqdm(config.MODEL_CONFIGS.items(), desc="Training models"):
        feedforward = model_name == "ann"
        train_loader, val_loader = _make_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            config_params["batch_size"],
            feedforward=feedforward,
        )

        model = _build_model(model_name, X_seq.shape[2], config_params).to(device)
        history = _train_model(model, train_loader, val_loader, config_params, device)
        y_true, y_pred = _evaluate_model(model, X_test, y_test, target_scaler, device, feedforward=feedforward)
        model_metrics = metrics_module.compute_metrics(y_true, y_pred)
        results[model_name] = {
            "metrics": model_metrics,
            "hyperparameters": config_params,
        }
        histories[model_name] = history
        predictions[model_name] = {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }

        model_path = output_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)

    model_registry = {
        "models": {name: f"{name}.pt" for name in config.MODEL_CONFIGS.keys()},
        "best_model": _select_best_model(results),
    }

    (artifacts_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (artifacts_dir / "model_registry.json").write_text(json.dumps(model_registry, indent=2))
    (artifacts_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    (artifacts_dir / "training_history.json").write_text(json.dumps(histories, indent=2))
    return results


def _select_best_model(results: Dict) -> str:
    best_model = None
    best_score = float("inf")
    for model_name, payload in results.items():
        score = payload["metrics"]["RMSE"]
        if score < best_score:
            best_score = score
            best_model = model_name
    return best_model
