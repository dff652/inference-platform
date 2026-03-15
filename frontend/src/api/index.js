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

// Configs
export const configApi = {
  algorithms() {
    return api.get('/inference/configs/algorithms')
  },
}

export default api
