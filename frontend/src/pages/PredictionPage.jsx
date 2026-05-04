import { useEffect, useState } from 'react'
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { fetchModels, predictDaily, predictRadiation } from '../services/api'

const emptyWeather = {
  temperature_c: '',
  humidity_pct: '',
  wind_speed_mps: '',
  cloud_cover_pct: '',
  rainfall_mm: '',
  pressure_hpa: '',
}

function PredictionPage() {
  const [models, setModels] = useState({ models: [], best_model: '' })
  const [formData, setFormData] = useState({
    datetime: new Date().toISOString().slice(0, 16),
    panel_area_m2: 10,
    panel_efficiency: 0.18,
    tilt_factor: 1.0,
    model_name: '',
    autoWeather: true,
    ...emptyWeather,
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [daily, setDaily] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data))
      .catch(() => setModels({ models: [], best_model: '' }))
  }, [])

  const handleChange = (event) => {
    const { name, value, type, checked } = event.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
  }

  const payload = () => {
    const weatherPayload = formData.autoWeather
      ? {}
      : {
          temperature_c: formData.temperature_c ? Number(formData.temperature_c) : null,
          humidity_pct: formData.humidity_pct ? Number(formData.humidity_pct) : null,
          wind_speed_mps: formData.wind_speed_mps ? Number(formData.wind_speed_mps) : null,
          cloud_cover_pct: formData.cloud_cover_pct ? Number(formData.cloud_cover_pct) : null,
          rainfall_mm: formData.rainfall_mm ? Number(formData.rainfall_mm) : null,
          pressure_hpa: formData.pressure_hpa ? Number(formData.pressure_hpa) : null,
        }

    return {
      datetime: new Date(formData.datetime).toISOString(),
      panel_area_m2: Number(formData.panel_area_m2),
      panel_efficiency: Number(formData.panel_efficiency),
      tilt_factor: Number(formData.tilt_factor),
      model_name: formData.model_name || null,
      ...weatherPayload,
    }
  }

  const handlePredict = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const response = await predictRadiation(payload())
      setResult(response)
      setDaily(null)
    } catch {
      setError('Prediction failed. Train models and ensure API is running.')
    } finally {
      setLoading(false)
    }
  }

  const handleDaily = async () => {
    setLoading(true)
    setError('')
    try {
      const response = await predictDaily(payload())
      setDaily(response)
      setResult(null)
    } catch {
      setError('Daily prediction failed. Train models and ensure API is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page">
      <div className="card">
        <h2>Real-time & Future Solar Radiation Prediction</h2>
        <p>
          Use auto-generated coastal climate inputs or override with custom weather parameters for Dakshina Kannada.
        </p>
        <form className="grid" onSubmit={handlePredict}>
          <label>
            Date & Time
            <input type="datetime-local" name="datetime" value={formData.datetime} onChange={handleChange} required />
          </label>
          <label>
            Model Selection
            <select name="model_name" value={formData.model_name} onChange={handleChange}>
              <option value="">Best Model (Auto)</option>
              {models.models.map((model) => (
                <option key={model} value={model}>
                  {model.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          <label className="checkbox">
            <input type="checkbox" name="autoWeather" checked={formData.autoWeather} onChange={handleChange} />
            Auto-generate typical weather
          </label>
          <label>
            Panel Area (m²)
            <input type="number" name="panel_area_m2" value={formData.panel_area_m2} onChange={handleChange} step="0.1" />
          </label>
          <label>
            Panel Efficiency (0-1)
            <input type="number" name="panel_efficiency" value={formData.panel_efficiency} onChange={handleChange} step="0.01" />
          </label>
          <label>
            Tilt Factor
            <input type="number" name="tilt_factor" value={formData.tilt_factor} onChange={handleChange} step="0.05" />
          </label>

          {!formData.autoWeather && (
            <>
              <label>
                Temperature (°C)
                <input type="number" name="temperature_c" value={formData.temperature_c} onChange={handleChange} />
              </label>
              <label>
                Humidity (%)
                <input type="number" name="humidity_pct" value={formData.humidity_pct} onChange={handleChange} />
              </label>
              <label>
                Wind Speed (m/s)
                <input type="number" name="wind_speed_mps" value={formData.wind_speed_mps} onChange={handleChange} />
              </label>
              <label>
                Cloud Cover (%)
                <input type="number" name="cloud_cover_pct" value={formData.cloud_cover_pct} onChange={handleChange} />
              </label>
              <label>
                Rainfall (mm)
                <input type="number" name="rainfall_mm" value={formData.rainfall_mm} onChange={handleChange} />
              </label>
              <label>
                Pressure (hPa)
                <input type="number" name="pressure_hpa" value={formData.pressure_hpa} onChange={handleChange} />
              </label>
            </>
          )}
        </form>
        <div className="actions">
          <button className="primary" onClick={handlePredict} disabled={loading}>
            {loading ? 'Running...' : 'Predict Single Hour'}
          </button>
          <button className="ghost" onClick={handleDaily} disabled={loading}>
            {loading ? 'Running...' : 'Predict Daily Profile'}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>

      {result && (
        <div className="card metrics">
          <h3>Prediction Output</h3>
          <div className="stat-grid">
            <div>
              <p className="label">Model Used</p>
              <p className="value">{result.model_used.toUpperCase()}</p>
            </div>
            <div>
              <p className="label">Radiation</p>
              <p className="value">{result.predicted_radiation_wm2.toFixed(2)} W/m²</p>
            </div>
            <div>
              <p className="label">Hourly Energy</p>
              <p className="value">{result.hourly_energy_kwh.toFixed(2)} kWh</p>
            </div>
          </div>
        </div>
      )}

      {daily && (
        <div className="card">
          <h3>Daily Radiation Profile</h3>
          <p>Estimated daily energy: {daily.daily_energy_kwh.toFixed(2)} kWh</p>
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <LineChart data={daily.hourly_predictions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="prediction" stroke="#ff7a59" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </section>
  )
}

export default PredictionPage
