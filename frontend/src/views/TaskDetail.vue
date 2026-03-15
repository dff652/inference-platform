<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { taskApi } from '../api'

const route = useRoute()
const router = useRouter()
const task = ref(null)
const results = ref(null)
const logs = ref('')
const activeTab = ref('info')
const loading = ref(true)

let pollTimer = null

const taskId = computed(() => route.params.id)

const statusColors = {
  draft: 'info',
  pending: 'warning',
  queued: 'warning',
  running: '',
  completed: 'success',
  failed: 'danger',
  cancelled: 'info',
  timeout: 'danger',
}

const duration = computed(() => {
  if (!task.value?.started_at) return null
  const start = new Date(task.value.started_at)
  const end = task.value.completed_at ? new Date(task.value.completed_at) : new Date()
  return Math.round((end - start) / 1000)
})

async function fetchTask() {
  try {
    const res = await taskApi.get(taskId.value)
    task.value = res.data
  } finally {
    loading.value = false
  }
}

async function fetchResults() {
  try {
    const res = await taskApi.results(taskId.value)
    results.value = res.data
  } catch {
    results.value = null
  }
}

async function fetchLogs() {
  try {
    const res = await taskApi.logs(taskId.value)
    logs.value = res.data
  } catch {
    logs.value = '(No logs available)'
  }
}

async function handleTabChange(tab) {
  if (tab === 'results' && !results.value) fetchResults()
  if (tab === 'logs' && !logs.value) fetchLogs()
}

async function handleSubmit() {
  await taskApi.submit(taskId.value)
  ElMessage.success('Task submitted')
  fetchTask()
}

async function handleCancel() {
  await taskApi.cancel(taskId.value)
  ElMessage.success('Task cancelled')
  fetchTask()
}

async function handleRetry() {
  await taskApi.retry(taskId.value)
  ElMessage.success('Task resubmitted')
  fetchTask()
}

const isRunning = computed(() =>
  task.value && ['pending', 'queued', 'running'].includes(task.value.status)
)

onMounted(() => {
  fetchTask()
  pollTimer = setInterval(() => {
    if (isRunning.value) fetchTask()
  }, 3000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <div v-loading="loading">
    <template v-if="task">
      <!-- Header -->
      <el-card style="margin-bottom: 16px">
        <div style="display: flex; justify-content: space-between; align-items: center">
          <div>
            <h2 style="margin: 0 0 8px">{{ task.task_name }}</h2>
            <el-space>
              <el-tag :type="statusColors[task.status]" size="large">{{ task.status }}</el-tag>
              <span style="color: #909399">ID: {{ task.id }}</span>
              <span v-if="task.algorithm_name" style="color: #909399">Algorithm: {{ task.algorithm_name }}</span>
              <span v-if="duration !== null" style="color: #909399">Duration: {{ duration }}s</span>
            </el-space>
          </div>
          <el-space>
            <el-button v-if="task.status === 'draft'" type="primary" @click="handleSubmit">Submit</el-button>
            <el-button v-if="isRunning" type="warning" @click="handleCancel">Cancel</el-button>
            <el-button v-if="['failed','timeout'].includes(task.status)" type="info" @click="handleRetry">Retry</el-button>
            <el-button @click="router.push('/tasks')">Back</el-button>
          </el-space>
        </div>
      </el-card>

      <!-- Error Message -->
      <el-alert
        v-if="task.error_message"
        :title="task.error_message"
        type="error"
        show-icon
        style="margin-bottom: 16px"
      />

      <!-- Tabs -->
      <el-card>
        <el-tabs v-model="activeTab" @tab-change="handleTabChange">
          <!-- Info Tab -->
          <el-tab-pane label="Info" name="info">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="Task Name">{{ task.task_name }}</el-descriptions-item>
              <el-descriptions-item label="Status">
                <el-tag :type="statusColors[task.status]">{{ task.status }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="Algorithm">{{ task.algorithm_name || '-' }}</el-descriptions-item>
              <el-descriptions-item label="Executor">{{ task.executor_type }}</el-descriptions-item>
              <el-descriptions-item label="Submitter">{{ task.submitter || '-' }}</el-descriptions-item>
              <el-descriptions-item label="Priority">{{ task.priority }}</el-descriptions-item>
              <el-descriptions-item label="Created">{{ new Date(task.created_at).toLocaleString() }}</el-descriptions-item>
              <el-descriptions-item label="Started">{{ task.started_at ? new Date(task.started_at).toLocaleString() : '-' }}</el-descriptions-item>
              <el-descriptions-item label="Completed">{{ task.completed_at ? new Date(task.completed_at).toLocaleString() : '-' }}</el-descriptions-item>
              <el-descriptions-item label="Celery Task ID">{{ task.celery_task_id || '-' }}</el-descriptions-item>
            </el-descriptions>

            <el-divider>Input Snapshot</el-divider>
            <el-descriptions :column="1" border v-if="task.input_snapshot">
              <el-descriptions-item label="Files">
                <div v-for="f in (task.input_snapshot.files || [])" :key="f">{{ f }}</div>
              </el-descriptions-item>
            </el-descriptions>

            <el-divider>Parameters</el-divider>
            <pre style="background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 13px">{{ JSON.stringify(task.parameter_snapshot, null, 2) }}</pre>
          </el-tab-pane>

          <!-- Results Tab -->
          <el-tab-pane label="Results" name="results">
            <template v-if="results && results.results && results.results.length">
              <el-table :data="results.results" stripe>
                <el-table-column prop="point_name" label="Point" width="150" />
                <el-table-column prop="method" label="Method" width="130" />
                <el-table-column prop="score_avg" label="Score Avg" width="110">
                  <template #default="{ row }">{{ row.score_avg?.toFixed(2) }}</template>
                </el-table-column>
                <el-table-column prop="score_max" label="Score Max" width="110">
                  <template #default="{ row }">{{ row.score_max?.toFixed(2) }}</template>
                </el-table-column>
                <el-table-column prop="segment_count" label="Segments" width="100" />
              </el-table>

              <template v-for="r in results.results" :key="r.id">
                <el-divider v-if="r.segments">Anomaly Segments - {{ r.point_name }}</el-divider>
                <el-table v-if="r.segments" :data="r.segments" stripe size="small">
                  <el-table-column prop="start" label="Start" width="100" />
                  <el-table-column prop="end" label="End" width="100" />
                  <el-table-column prop="length" label="Length" width="100" />
                  <el-table-column prop="score" label="Score" width="100">
                    <template #default="{ row }">{{ row.score?.toFixed(2) }}</template>
                  </el-table-column>
                </el-table>
              </template>
            </template>
            <el-empty v-else description="No results available" />
          </el-tab-pane>

          <!-- Logs Tab -->
          <el-tab-pane label="Logs" name="logs">
            <pre style="background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 4px; max-height: 600px; overflow: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all">{{ logs || '(No logs available)' }}</pre>
          </el-tab-pane>
        </el-tabs>
      </el-card>
    </template>
  </div>
</template>
