import api from './client'

export const configApi = {
  algorithms() {
    return api.get('/inference/configs/algorithms')
  },
  chattsPrompts() {
    return api.get('/inference/configs/chatts-prompts')
  },
}
