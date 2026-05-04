from app import config
from app.services.training import train_all_models
from app.services.visualization import generate_visualizations


def main():
    train_all_models(config.DATA_PATH)
    generate_visualizations(config.ARTIFACTS_DIR)
    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
