import api from './client'

export const uploadApi = {
  upload(formData) {
    return api.post('/uploads', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
  },
}
