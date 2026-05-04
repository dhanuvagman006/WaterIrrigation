import { useEffect, useMemo, useState } from 'react'
import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { fetchMetrics } from '../services/api'

function DashboardPage() {
  const [metrics, setMetrics] = useState(null)
  const [selectedModel, setSelectedModel] = useState('')

  useEffect(() => {
    fetchMetrics().then((data) => {
      setMetrics(data)
      setSelectedModel(data.best_model)
    })
  }, [])

  const chartData = useMemo(() => {
    if (!metrics) return []
    return Object.entries(metrics.metrics).map(([model, payload]) => ({
      model: model.toUpperCase(),
      ...payload.metrics,
    }))
  }, [metrics])

  if (!metrics) {
    return (
      <section className="page">
        <div className="card">Waiting for metrics. Train models and ensure the API is running.</div>
      </section>
    )
  }

  const modelList = Object.keys(metrics.metrics)
  const bestModel = metrics.best_model

  return (
    <section className="page">
      <div className="card">
        <div className="header-row">
          <h2>Model Comparison Dashboard</h2>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {modelList.map((model) => (
              <option key={model} value={model}>
                {model.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
        <p>Best-performing model: {bestModel.toUpperCase()}</p>
      </div>

      <div className="card">
        <h3>Metrics Bar Chart (All Models)</h3>
        <div style={{ width: '100%', height: 360 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 20, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="RMSE" fill="#ff7a59" />
              <Bar dataKey="MAE" fill="#5b8def" />
              <Bar dataKey="R2" fill="#2ab07f" />
              <Bar dataKey="MAPE" fill="#9d7bd8" />
              <Bar dataKey="NSE" fill="#f5a623" />
              <Bar dataKey="ExplainedVariance" fill="#334155" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3>Evaluation Metrics Table</h3>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>R²</th>
                <th>MAPE</th>
                <th>NSE</th>
                <th>Explained Variance</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.metrics).map(([model, payload]) => {
                const isBest = model === bestModel
                return (
                  <tr key={model} className={isBest ? 'highlight' : ''}>
                    <td>{model.toUpperCase()}</td>
                    <td>{payload.metrics.RMSE.toFixed(2)}</td>
                    <td>{payload.metrics.MAE.toFixed(2)}</td>
                    <td>{payload.metrics.R2.toFixed(3)}</td>
                    <td>{payload.metrics.MAPE.toFixed(2)}</td>
                    <td>{payload.metrics.NSE.toFixed(3)}</td>
                    <td>{payload.metrics.ExplainedVariance.toFixed(3)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-two">
        <div className="card">
          <h3>Actual vs Predicted</h3>
          <img src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/plots/${selectedModel}_actual_vs_pred.png`} alt="Actual vs Predicted" />
        </div>
        <div className="card">
          <h3>Loss Curve</h3>
          <img src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/plots/${selectedModel}_loss_curve.png`} alt="Loss Curve" />
        </div>
      </div>

      <div className="grid-two">
        <div className="card">
          <h3>Residual Plot</h3>
          <img src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/plots/${selectedModel}_residuals.png`} alt="Residuals" />
        </div>
        <div className="card">
          <h3>Error Distribution</h3>
          <img src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/plots/${selectedModel}_error_dist.png`} alt="Error Distribution" />
        </div>
      </div>
    </section>
  )
}

export default DashboardPage
