import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE,
})

export async function fetchModels() {
  const { data } = await api.get('/models')
  return data
}

export async function fetchMetrics() {
  const { data } = await api.get('/metrics')
  return data
}

export async function predictRadiation(payload) {
  const { data } = await api.post('/predict', payload)
  return data
}

export async function predictDaily(payload) {
  const { data } = await api.post('/predict/daily', payload)
  return data
}
