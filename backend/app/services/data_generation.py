import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app import config


@dataclass
class ClimateParams:
    base_temp: float = 27.5
    temp_season_amp: float = 4.5
    temp_diurnal_amp: float = 5.0
    humidity_base: float = 75.0
    humidity_monsoon_boost: float = 12.0
    wind_base: float = 2.5
    wind_monsoon_boost: float = 1.8
    pressure_base: float = 1012.0
    pressure_monsoon_drop: float = 6.0


def _assign_season(month: int) -> str:
    for season, months in config.SEASON_MAPPING.items():
        if month in months:
            return season
    return "unknown"


def _daylight_factor(hour: int) -> float:
    solar_angle = np.pi * (hour - 6) / 12
    return max(0.0, np.sin(solar_angle))


def _seasonal_factor(day_of_year: int) -> float:
    return 0.75 + 0.25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)


def _monsoon_factor(month: int) -> float:
    return 1.0 if month in config.SEASON_MAPPING["monsoon"] else 0.0


def generate_synthetic_dataset(output_path: Path, years: int = config.TRAIN_YEARS, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2015-01-01 00:00:00")
    end_date = start_date + pd.DateOffset(years=years) - pd.Timedelta(hours=1)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    params = ClimateParams()
    df = pd.DataFrame({"datetime": timestamps})
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["season"] = df["month"].apply(_assign_season)
    df["is_monsoon"] = df["month"].isin(config.SEASON_MAPPING["monsoon"]).astype(int)

    seasonal = _seasonal_factor(df["day_of_year"].to_numpy())
    diurnal = np.sin(2 * np.pi * (df["hour"].to_numpy() - 6) / 24)

    df["temperature_c"] = (
        params.base_temp
        + params.temp_season_amp * (seasonal - 0.75)
        + params.temp_diurnal_amp * diurnal
        + rng.normal(0, 1.2, len(df))
    )

    df["humidity_pct"] = (
        params.humidity_base
        + params.humidity_monsoon_boost * df["is_monsoon"]
        + rng.normal(0, 6, len(df))
    )
    df["humidity_pct"] = df["humidity_pct"].clip(45, 98)

    df["wind_speed_mps"] = (
        params.wind_base
        + params.wind_monsoon_boost * df["is_monsoon"]
        + rng.normal(0, 0.8, len(df))
    )
    df["wind_speed_mps"] = df["wind_speed_mps"].clip(0.2, 9.0)

    base_cloud = 45 + 25 * df["is_monsoon"] + 10 * rng.normal(size=len(df))
    df["cloud_cover_pct"] = np.clip(base_cloud, 5, 98)

    rainfall = rng.gamma(shape=1.8, scale=5.0, size=len(df)) * df["is_monsoon"]
    rainfall += rng.gamma(shape=1.4, scale=2.0, size=len(df)) * (1 - df["is_monsoon"]) * (rng.random(len(df)) < 0.1)
    df["rainfall_mm"] = rainfall.clip(0, 120)

    df["pressure_hpa"] = (
        params.pressure_base
        - params.pressure_monsoon_drop * df["is_monsoon"]
        + rng.normal(0, 1.4, len(df))
    )

    daylight = np.array([_daylight_factor(h) for h in df["hour"]])
    clear_sky = 1100 * seasonal * daylight
    attenuation = (
        1
        - 0.65 * (df["cloud_cover_pct"] / 100)
        - 0.15 * (df["humidity_pct"] / 100)
        - 0.1 * np.tanh(df["rainfall_mm"] / 20)
    )
    attenuation = attenuation.clip(0.1, 1.0)

    df["solar_radiation_wm2"] = (
        clear_sky * attenuation
        + rng.normal(0, 35, len(df))
        + 25 * np.sin(2 * np.pi * df["day_of_year"] / 365)
    )
    df["solar_radiation_wm2"] = df["solar_radiation_wm2"].clip(0, 1200)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_climatology(df: pd.DataFrame, output_path: Path) -> dict:
    grouped = df.groupby(["month", "hour"]).agg(
        {
            "temperature_c": "mean",
            "humidity_pct": "mean",
            "wind_speed_mps": "mean",
            "cloud_cover_pct": "mean",
            "rainfall_mm": "mean",
            "pressure_hpa": "mean",
        }
    )
    climatology = {
        f"{int(month):02d}-{int(hour):02d}": row.to_dict()
        for (month, hour), row in grouped.iterrows()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(climatology, indent=2))
    return climatology
