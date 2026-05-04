function AboutPage() {
  return (
    <section className="page">
      <div className="card">
        <h2>Methodology & Assumptions</h2>
        <p>
          This system synthesizes hourly coastal climate data for Dakshina Kannada, capturing monsoon-driven cloud cover,
          humidity, rainfall, and seasonal solar variability. The dataset spans multiple years and is shaped by
          NASA/IMD-inspired distributions.
        </p>
        <div className="grid-two">
          <div>
            <h3>Model Suite (6)</h3>
            <ul>
              <li>LSTM</li>
              <li>GRU</li>
              <li>CNN-LSTM Hybrid</li>
              <li>Transformer Encoder</li>
              <li>Temporal Convolutional Network</li>
              <li>Feedforward ANN Baseline</li>
            </ul>
          </div>
          <div>
            <h3>Evaluation Metrics</h3>
            <ul>
              <li>RMSE, MAE, R²</li>
              <li>MAPE</li>
              <li>NSE (Nash-Sutcliffe Efficiency)</li>
              <li>Explained Variance</li>
            </ul>
          </div>
        </div>
        <h3>Power Estimation</h3>
        <p>
          Daily energy is estimated by converting predicted irradiance to power using user-defined panel area,
          efficiency, and tilt adjustment. Outputs are in kWh for easy planning.
        </p>
      </div>
    </section>
  )
}

export default AboutPage
