import { NavLink } from 'react-router-dom'

function Layout({ children }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Dakshina Kannada Research Lab</p>
          <h1>Solar Radiation Intelligence System</h1>
        </div>
        <nav>
          <NavLink to="/" end>
            Prediction
          </NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
          <NavLink to="/about">Methodology</NavLink>
        </nav>
      </header>
      <main>{children}</main>
      <footer className="app-footer">
        <p>Research-grade synthetic data and deep learning model suite for coastal Karnataka.</p>
      </footer>
    </div>
  )
}

export default Layout
