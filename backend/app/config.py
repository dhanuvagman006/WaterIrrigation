from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
MODEL_DIR = PROJECT_ROOT / "backend" / "model_weights"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

DATASET_FILENAME = "solar_dk_synthetic.csv"
DATA_PATH = DATA_DIR / DATASET_FILENAME
CLIMATOLOGY_PATH = ARTIFACTS_DIR / "climatology.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.pkl"

SEED = 42
LOOKBACK_HOURS = 24
FORECAST_HORIZON = 1
TRAIN_YEARS = 8

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "cloud_cover_pct",
    "rainfall_mm",
    "pressure_hpa",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "is_monsoon",
]

TARGET_COLUMN = "solar_radiation_wm2"

MODEL_CONFIGS = {
    "lstm": {
        "hidden_size": 96,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 25,
    },
    "gru": {
        "hidden_size": 96,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 25,
    },
    "cnn_lstm": {
        "conv_channels": 48,
        "kernel_size": 3,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.25,
        "learning_rate": 1.2e-3,
        "batch_size": 128,
        "epochs": 30,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.15,
        "learning_rate": 1.5e-3,
        "batch_size": 128,
        "epochs": 30,
    },
    "tcn": {
        "num_channels": [64, 64, 64],
        "kernel_size": 3,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 30,
    },
    "ann": {
        "hidden_sizes": [128, 64],
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 20,
    },
}

SEASON_MAPPING = {
    "winter": [12, 1, 2],
    "pre_monsoon": [3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "post_monsoon": [10, 11],
}
