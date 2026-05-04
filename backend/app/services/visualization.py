import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from app import config


def _ensure_dirs():
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_visualizations(artifacts_dir: Path = config.ARTIFACTS_DIR):
    _ensure_dirs()
    predictions = json.loads((artifacts_dir / "predictions.json").read_text())
    histories = json.loads((artifacts_dir / "training_history.json").read_text())
    metrics = json.loads((artifacts_dir / "metrics.json").read_text())

    for model_name, pred in predictions.items():
        y_true = np.array(pred["y_true"])[:500]
        y_pred = np.array(pred["y_pred"])[:500]

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual", linewidth=1)
        plt.plot(y_pred, label="Predicted", linewidth=1)
        plt.title(f"Actual vs Predicted - {model_name.upper()}")
        plt.xlabel("Sample")
        plt.ylabel("Solar Radiation (W/m²)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f"{model_name}_actual_vs_pred.png")
        plt.close()

        residuals = y_true - y_pred
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_pred, y=residuals, s=10)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Residuals - {model_name.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f"{model_name}_residuals.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30)
        plt.title(f"Error Distribution - {model_name.upper()}")
        plt.xlabel("Residual")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f"{model_name}_error_dist.png")
        plt.close()

    for model_name, history in histories.items():
        plt.figure(figsize=(6, 4))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")
        plt.title(f"Loss Curve - {model_name.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f"{model_name}_loss_curve.png")
        plt.close()

    metric_names = list(next(iter(metrics.values()))["metrics"].keys())
    bar_data = {metric: [] for metric in metric_names}
    for model_name, payload in metrics.items():
        for metric in metric_names:
            bar_data[metric].append(payload["metrics"][metric])

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.12
    for i, metric in enumerate(metric_names):
        plt.bar(x + i * width, bar_data[metric], width, label=metric)
    plt.xticks(x + width * (len(metric_names) / 2), [m.upper() for m in metrics.keys()], rotation=45)
    plt.title("Model Metrics Comparison")
    plt.tight_layout()
    plt.legend(fontsize=7)
    plt.savefig(config.PLOTS_DIR / "metrics_comparison.png")
    plt.close()
