import api from './client'

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

export const gpuApi = {
  status() {
    return api.get('/models/vllm/status')
  },
}
