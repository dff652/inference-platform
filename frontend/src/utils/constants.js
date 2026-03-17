/**
 * Constants re-export hub.
 *
 * Each domain has its own file to reduce merge conflicts in multi-person dev.
 * Views should import from the specific module:
 *   import { statusColors } from '../utils/task-constants'
 *
 * This file re-exports everything for backward compatibility.
 */
export {
  statusColors,
  statusLabels,
  cancellableStatuses,
  retryableStatuses,
  gpuAlgorithms,
} from './task-constants'
