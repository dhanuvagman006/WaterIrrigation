from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.schemas import DailyProfileResponse, MetricsResponse, ModelInfo, PredictionRequest, PredictionResponse
from app.services.data_generation import build_climatology, generate_synthetic_dataset
from app.services.inference import InferenceEngine
from app.services.training import train_all_models
from app.services.visualization import generate_visualizations

app = FastAPI(title="Dakshina Kannada Solar Radiation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)

engine: Optional[InferenceEngine] = None


@app.on_event("startup")
async def startup_event():
    global engine
    if config.ARTIFACTS_DIR.exists() and (config.ARTIFACTS_DIR / "model_registry.json").exists():
        engine = InferenceEngine()

    if config.PLOTS_DIR.exists():
        app.mount("/plots", StaticFiles(directory=config.PLOTS_DIR), name="plots")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/models", response_model=ModelInfo)
async def list_models():
    if not config.MODEL_REGISTRY_PATH.exists():
        raise HTTPException(status_code=404, detail="Model registry not found. Train models first.")
    registry = config.MODEL_REGISTRY_PATH.read_text()
    payload = __import__("json").loads(registry)
    return ModelInfo(best_model=payload["best_model"], models=list(payload["models"].keys()))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train models first.")
    metrics_payload = __import__("json").loads(config.METRICS_PATH.read_text())
    best_model = __import__("json").loads(config.MODEL_REGISTRY_PATH.read_text())["best_model"]
    return MetricsResponse(metrics=metrics_payload, best_model=best_model)


def _estimate_energy_kwh(radiation_wm2: float, area: float, efficiency: float, tilt: float) -> float:
    power_w = radiation_wm2 * area * efficiency * tilt
    return power_w / 1000


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    overrides = {
        "temperature_c": request.temperature_c,
        "humidity_pct": request.humidity_pct,
        "wind_speed_mps": request.wind_speed_mps,
        "cloud_cover_pct": request.cloud_cover_pct,
        "rainfall_mm": request.rainfall_mm,
        "pressure_hpa": request.pressure_hpa,
    }
    radiation = engine.predict(request.datetime, overrides, model_name=request.model_name)
    registry = __import__("json").loads(config.MODEL_REGISTRY_PATH.read_text())
    model_used = request.model_name or registry["best_model"]
    hourly_energy = _estimate_energy_kwh(
        radiation,
        request.panel_area_m2,
        request.panel_efficiency,
        request.tilt_factor,
    )
    return PredictionResponse(
        model_used=model_used,
        predicted_radiation_wm2=radiation,
        hourly_energy_kwh=hourly_energy,
    )


@app.post("/predict/daily", response_model=DailyProfileResponse)
async def predict_daily(request: PredictionRequest):
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    overrides = {
        "temperature_c": request.temperature_c,
        "humidity_pct": request.humidity_pct,
        "wind_speed_mps": request.wind_speed_mps,
        "cloud_cover_pct": request.cloud_cover_pct,
        "rainfall_mm": request.rainfall_mm,
        "pressure_hpa": request.pressure_hpa,
    }
    profile = engine.predict_daily_profile(request.datetime, overrides, model_name=request.model_name)
    registry = __import__("json").loads(config.MODEL_REGISTRY_PATH.read_text())
    model_used = request.model_name or registry["best_model"]
    daily_energy = sum(
        _estimate_energy_kwh(p["prediction"], request.panel_area_m2, request.panel_efficiency, request.tilt_factor)
        for p in profile
    )
    return DailyProfileResponse(
        model_used=model_used,
        date=request.datetime,
        hourly_predictions=profile,
        daily_energy_kwh=daily_energy,
    )


@app.post("/admin/generate-dataset")
async def generate_dataset(background_tasks: BackgroundTasks):
    def _job():
        df = generate_synthetic_dataset(config.DATA_PATH, years=config.TRAIN_YEARS)
        build_climatology(df, config.CLIMATOLOGY_PATH)

    background_tasks.add_task(_job)
    return {"status": "Dataset generation started"}


@app.post("/admin/train-models")
async def train_models(background_tasks: BackgroundTasks):
    def _job():
        train_all_models(config.DATA_PATH)
        generate_visualizations(config.ARTIFACTS_DIR)

    background_tasks.add_task(_job)
    return {"status": "Model training started"}
