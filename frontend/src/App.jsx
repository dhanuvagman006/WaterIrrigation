import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout.jsx'
import PredictionPage from './pages/PredictionPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import AboutPage from './pages/AboutPage.jsx'
import './App.css'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<PredictionPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Layout>
  )
}

export default App
