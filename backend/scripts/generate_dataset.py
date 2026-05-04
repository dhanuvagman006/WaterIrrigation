from app import config
from app.services.data_generation import build_climatology, generate_synthetic_dataset


def main():
    df = generate_synthetic_dataset(config.DATA_PATH, years=config.TRAIN_YEARS)
    build_climatology(df, config.CLIMATOLOGY_PATH)
    print(f"Dataset saved to {config.DATA_PATH}")
    print(f"Climatology saved to {config.CLIMATOLOGY_PATH}")


if __name__ == "__main__":
    main()
