<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { taskApi } from '../api'
import { statusColors, statusLabels, cancellableStatuses, retryableStatuses } from '../utils/constants'

const router = useRouter()
const tasks = ref([])
const total = ref(0)
const loading = ref(false)
const stats = ref({})
const filters = ref({
  status: '',
  offset: 0,
  limit: 20,
})

let pollTimer = null

async function fetchTasks() {
  loading.value = true
  try {
    const params = { ...filters.value }
    if (!params.status) delete params.status
    const [taskRes, statsRes] = await Promise.all([
      taskApi.list(params),
      taskApi.stats(),
    ])
    tasks.value = taskRes.data.items
    total.value = taskRes.data.total
    stats.value = statsRes.data
  } finally {
    loading.value = false
  }
}

function handlePageChange(page) {
  filters.value.offset = (page - 1) * filters.value.limit
  fetchTasks()
}

function filterByStatus(status) {
  filters.value.status = filters.value.status === status ? '' : status
  filters.value.offset = 0
  fetchTasks()
}

async function handleSubmit(id) {
  try {
    await taskApi.submit(id)
    ElMessage.success('Task submitted')
    fetchTasks()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || 'Failed to submit task')
  }
}

async function handleCancel(id) {
  try {
    await taskApi.cancel(id)
    ElMessage.success('Task cancelled')
    fetchTasks()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || 'Failed to cancel task')
  }
}

async function handleRetry(id) {
  try {
    await taskApi.retry(id)
    ElMessage.success('Task resubmitted')
    fetchTasks()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || 'Failed to retry task')
  }
}

onMounted(() => {
  fetchTasks()
  pollTimer = setInterval(fetchTasks, 5000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <div>
    <!-- Stats Cards -->
    <el-row :gutter="16" style="margin-bottom: 20px">
      <el-col :span="3" v-for="(label, key) in statusLabels" :key="key">
        <el-card
          shadow="hover"
          :class="{ 'is-active': filters.status === key }"
          style="cursor: pointer; text-align: center"
          @click="filterByStatus(key)"
        >
          <div style="font-size: 24px; font-weight: bold">{{ stats[key] || 0 }}</div>
          <div style="font-size: 12px; color: #909399; margin-top: 4px">{{ label }}</div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Action Bar -->
    <div style="margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center">
      <div>
        <el-tag v-if="filters.status" closable @close="filters.status = ''; fetchTasks()">
          Status: {{ filters.status }}
        </el-tag>
      </div>
      <el-button type="primary" @click="router.push('/tasks/create')">
        <el-icon><Plus /></el-icon> New Task
      </el-button>
    </div>

    <!-- Task Table -->
    <el-table :data="tasks" v-loading="loading" stripe style="width: 100%">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="task_name" label="Task Name" min-width="180">
        <template #default="{ row }">
          <el-link type="primary" @click="router.push(`/tasks/${row.id}`)">{{ row.task_name }}</el-link>
        </template>
      </el-table-column>
      <el-table-column prop="algorithm_name" label="Algorithm" width="130" />
      <el-table-column prop="status" label="Status" width="110">
        <template #default="{ row }">
          <el-tag :type="statusColors[row.status]" size="small">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="submitter" label="Submitter" width="100" />
      <el-table-column label="Created" width="170">
        <template #default="{ row }">
          {{ new Date(row.created_at).toLocaleString() }}
        </template>
      </el-table-column>
      <el-table-column label="Duration" width="100">
        <template #default="{ row }">
          <span v-if="row.started_at && row.completed_at">
            {{ Math.round((new Date(row.completed_at) - new Date(row.started_at)) / 1000) }}s
          </span>
          <span v-else-if="row.started_at">Running...</span>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column label="Actions" width="200" fixed="right">
        <template #default="{ row }">
          <el-button v-if="row.status === 'draft'" type="primary" size="small" @click="handleSubmit(row.id)">Submit</el-button>
          <el-button v-if="cancellableStatuses.includes(row.status)" type="warning" size="small" @click="handleCancel(row.id)">Cancel</el-button>
          <el-button v-if="retryableStatuses.includes(row.status)" type="info" size="small" @click="handleRetry(row.id)">Retry</el-button>
          <el-button size="small" @click="router.push(`/tasks/${row.id}`)">Detail</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- Pagination -->
    <div style="margin-top: 16px; display: flex; justify-content: flex-end">
      <el-pagination
        :current-page="Math.floor(filters.offset / filters.limit) + 1"
        :page-size="filters.limit"
        :total="total"
        layout="total, prev, pager, next"
        @current-change="handlePageChange"
      />
    </div>
  </div>
</template>

<style scoped>
.is-active {
  border-color: #409eff;
}
</style>
