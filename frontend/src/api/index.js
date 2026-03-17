/**
 * API re-export hub.
 *
 * Each domain has its own file to reduce merge conflicts in multi-person dev.
 * Views should import from the specific module:
 *   import { taskApi } from '../api/tasks'
 *
 * This file re-exports everything for backward compatibility.
 */
export { default as api } from './client'
export { taskApi } from './tasks'
export { modelApi, gpuApi } from './models'
export { configApi } from './configs'
export { uploadApi } from './uploads'
