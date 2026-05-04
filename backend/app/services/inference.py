import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from app import config
from app.services import features
from app.services.models import (
    CNNLSTMModel,
    FeedForwardModel,
    GRUModel,
    LSTMModel,
    TCNModel,
    TransformerModel,
)


class InferenceEngine:
    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR, model_dir: Path = config.MODEL_DIR):
        self.artifacts_dir = artifacts_dir
        self.model_dir = model_dir
        self.feature_scaler = joblib.load(artifacts_dir / "feature_scaler.pkl")
        self.target_scaler = joblib.load(artifacts_dir / "target_scaler.pkl")
        self.model_registry = json.loads((artifacts_dir / "model_registry.json").read_text())
        self.climatology = json.loads((artifacts_dir / "climatology.json").read_text())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, model_name: str):
        cfg = config.MODEL_CONFIGS[model_name]
        num_features = len(config.FEATURE_COLUMNS)
        if model_name == "lstm":
            model = LSTMModel(num_features, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
        elif model_name == "gru":
            model = GRUModel(num_features, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
        elif model_name == "cnn_lstm":
            model = CNNLSTMModel(
                num_features,
                cfg["conv_channels"],
                cfg["kernel_size"],
                cfg["hidden_size"],
                cfg["num_layers"],
                cfg["dropout"],
            )
        elif model_name == "transformer":
            model = TransformerModel(num_features, cfg["d_model"], cfg["nhead"], cfg["num_layers"], cfg["dropout"])
        elif model_name == "tcn":
            model = TCNModel(num_features, cfg["num_channels"], cfg["kernel_size"], cfg["dropout"])
        elif model_name == "ann":
            model = FeedForwardModel(num_features * config.LOOKBACK_HOURS, cfg["hidden_sizes"], cfg["dropout"])
        else:
            raise ValueError(f"Unknown model {model_name}")

        model_path = self.model_dir / self.model_registry["models"][model_name]
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _get_climatology(self, dt: datetime) -> Dict[str, float]:
        key = f"{dt.month:02d}-{dt.hour:02d}"
        return self.climatology.get(key)

    def _build_sequence(self, target_dt: datetime, overrides: Dict[str, float]) -> pd.DataFrame:
        sequence = []
        for offset in range(config.LOOKBACK_HOURS, 0, -1):
            seq_dt = target_dt - timedelta(hours=offset)
            baseline = self._get_climatology(seq_dt)
            if not baseline:
                baseline = {
                    "temperature_c": 28.0,
                    "humidity_pct": 75.0,
                    "wind_speed_mps": 2.5,
                    "cloud_cover_pct": 45.0,
                    "rainfall_mm": 0.0,
                    "pressure_hpa": 1012.0,
                }
            sequence.append({"datetime": seq_dt, **baseline})

        target_baseline = self._get_climatology(target_dt) or {
            "temperature_c": 28.0,
            "humidity_pct": 75.0,
            "wind_speed_mps": 2.5,
            "cloud_cover_pct": 45.0,
            "rainfall_mm": 0.0,
            "pressure_hpa": 1012.0,
        }
        target_baseline.update({k: v for k, v in overrides.items() if v is not None})
        sequence.append({"datetime": target_dt, **target_baseline})

        df = pd.DataFrame(sequence)
        df["hour"] = df["datetime"].dt.hour
        df["day"] = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df = features.add_time_features(df)
        return df

    def predict(self, target_dt: datetime, overrides: Dict[str, float], model_name: Optional[str] = None) -> float:
        model_name = model_name or self.model_registry["best_model"]
        model = self._build_model(model_name)

        seq_df = self._build_sequence(target_dt, overrides)
        X = seq_df[config.FEATURE_COLUMNS].to_numpy()
        X_scaled = self.feature_scaler.transform(X)
        X_scaled = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])

        if model_name == "ann":
            X_scaled = X_scaled.reshape(1, -1)

        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32).to(self.device)).cpu().numpy()

        prediction = self.target_scaler.inverse_transform(pred_scaled)[0, 0]
        return float(max(0.0, prediction))

    def predict_daily_profile(self, date: datetime, overrides: Dict[str, float], model_name: Optional[str] = None):
        profile = []
        for hour in range(24):
            dt = datetime(date.year, date.month, date.day, hour)
            profile.append({"hour": hour, "prediction": self.predict(dt, overrides, model_name=model_name)})
        return profile
