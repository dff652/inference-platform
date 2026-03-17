import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

// Tasks
export const taskApi = {
  list(params) {
    return api.get('/inference/tasks', { params })
  },
  get(id) {
    return api.get(`/inference/tasks/${id}`)
  },
  create(data) {
    return api.post('/inference/tasks', data)
  },
  update(id, data) {
    return api.put(`/inference/tasks/${id}`, data)
  },
  submit(id) {
    return api.post(`/inference/tasks/${id}/submit`)
  },
  cancel(id) {
    return api.post(`/inference/tasks/${id}/cancel`)
  },
  retry(id) {
    return api.post(`/inference/tasks/${id}/retry`)
  },
  stats() {
    return api.get('/inference/tasks/stats')
  },
  results(id) {
    return api.get(`/inference/tasks/${id}/results`)
  },
  logs(id) {
    return api.get(`/inference/tasks/${id}/logs`)
  },
  chartData(taskId, resultId) {
    return api.get(`/inference/tasks/${taskId}/results/${resultId}/chart-data`)
  },
}

// Models
export const modelApi = {
  list(params) {
    return api.get('/models', { params })
  },
  get(id) {
    return api.get(`/models/${id}`)
  },
  create(data) {
    return api.post('/models', data)
  },
  update(id, data) {
    return api.put(`/models/${id}`, data)
  },
  activate(id) {
    return api.post(`/models/${id}/activate`)
  },
  archive(id) {
    return api.post(`/models/${id}/archive`)
  },
}

// Uploads
export const uploadApi = {
  upload(formData) {
    return api.post('/uploads', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
  },
}

// Configs
export const configApi = {
  algorithms() {
    return api.get('/inference/configs/algorithms')
  },
  chattsPrompts() {
    return api.get('/inference/configs/chatts-prompts')
  },
}

// GPU service status
export const gpuApi = {
  status() {
    return api.get('/models/vllm/status')
  },
}

export default api
