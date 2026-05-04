# Dakshina Kannada Solar Radiation Prediction System

Research-grade, end-to-end solar radiation forecasting for Dakshina Kannada (Karnataka, India). This project synthesizes coastal climate data and compares six deep learning architectures with a full model-serving API and visualization dashboard.

## ✨ Features

- Hourly synthetic dataset (5–10 years) reflecting monsoon-heavy coastal climate
- Six deep learning models: LSTM, GRU, CNN-LSTM, Transformer, TCN, Feedforward ANN
- Evaluation metrics: RMSE, MAE, R², MAPE, NSE, Explained Variance
- FastAPI backend for dataset generation, training, and predictions
- React dashboard for model comparison and power estimation

## 📁 Project Structure

```
WaterIrrigation/
├─ backend/
│  ├─ app/
│  │  ├─ main.py
│  │  ├─ config.py
│  │  ├─ schemas.py
│  │  └─ services/
│  ├─ data/
│  │  ├─ schema.md
│  │  └─ solar_dk_synthetic.csv
│  ├─ scripts/
│  └─ requirements.txt
├─ frontend/
└─ README.md
```

## ✅ Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_dataset.py
python scripts/train_models.py
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set the API base URL if needed:

```
VITE_API_BASE_URL=http://localhost:8000
```

## 🔬 Dataset

The dataset (`backend/data/solar_dk_synthetic.csv`) contains hourly records for Dakshina Kannada with monsoon effects, humidity-driven attenuation, and seasonal solar variability. See `backend/data/schema.md` for full schema details.

## 🧠 Models & Hyperparameters

Hyperparameters for each of the six models are defined in `backend/app/config.py` and include tuned values for hidden sizes, dropout, learning rates, and epochs.

## 📊 Visualizations

After training, plots and metrics are saved under `backend/artifacts/plots`. The frontend dashboard consumes these assets and shows:

- Actual vs predicted curves
- Training vs validation loss
- Residual plots and error distributions
- Comparative metrics table and bar chart

## ⚡ Solar Power Estimation

Predicted irradiance is converted to energy using:

```
Power (W) = Irradiance (W/m²) × Panel Area (m²) × Efficiency × Tilt Factor
Energy (kWh) = Power / 1000
```

## 🧪 API Endpoints (FastAPI)

- `GET /models` – list available models and best model
- `GET /metrics` – evaluation metrics per model
- `POST /predict` – single-hour prediction + energy estimate
- `POST /predict/daily` – 24-hour profile + daily energy estimate
- `POST /admin/generate-dataset` – regenerate dataset
- `POST /admin/train-models` – retrain models

## 📌 Notes

- Run training once to generate model weights, metrics, and plots.
- Generated artifacts are excluded from git by default.
- Adjust `TRAIN_YEARS` in `backend/app/config.py` for longer datasets.
