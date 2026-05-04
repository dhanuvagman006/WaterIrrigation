from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    datetime: datetime
    temperature_c: Optional[float] = Field(None, description="Ambient temperature (°C)")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity (%)")
    wind_speed_mps: Optional[float] = Field(None, description="Wind speed (m/s)")
    cloud_cover_pct: Optional[float] = Field(None, description="Cloud cover (%)")
    rainfall_mm: Optional[float] = Field(None, description="Rainfall (mm)")
    pressure_hpa: Optional[float] = Field(None, description="Pressure (hPa)")
    panel_area_m2: float = Field(10.0, description="Panel area in m²")
    panel_efficiency: float = Field(0.18, description="Panel efficiency (0-1)")
    tilt_factor: float = Field(1.0, description="Tilt factor adjustment")
    model_name: Optional[str] = Field(None, description="Model name or leave empty for best model")


class PredictionResponse(BaseModel):
    model_used: str
    predicted_radiation_wm2: float
    hourly_energy_kwh: float
    daily_energy_kwh: Optional[float] = None


class DailyProfileResponse(BaseModel):
    model_used: str
    date: datetime
    hourly_predictions: List[dict]
    daily_energy_kwh: float


class ModelInfo(BaseModel):
    best_model: str
    models: List[str]


class MetricsResponse(BaseModel):
    metrics: dict
    best_model: str
