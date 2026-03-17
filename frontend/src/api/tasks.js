import api from './client'

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
