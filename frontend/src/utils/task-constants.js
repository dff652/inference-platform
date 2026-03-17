// Task status → Element Plus tag type mapping
export const statusColors = {
  draft: 'info',
  pending: 'warning',
  queued: 'warning',
  running: '',
  completed: 'success',
  failed: 'danger',
  cancelled: 'info',
  timeout: 'danger',
}

// Task status display labels
export const statusLabels = {
  draft: 'Draft',
  pending: 'Pending',
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  timeout: 'Timeout',
}

// Statuses that allow cancellation
export const cancellableStatuses = ['pending', 'queued', 'running']

// Statuses that allow retry
export const retryableStatuses = ['failed', 'timeout']

// GPU algorithm names
export const gpuAlgorithms = ['chatts', 'qwen']
