<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { modelApi } from '../api'

const models = ref([])
const loading = ref(false)
const dialogVisible = ref(false)
const editingModel = ref(null)

const form = ref({
  name: '',
  family: '',
  runtime_type: '',
  version: '',
  artifact_uri: '',
  description: '',
})

const statusColors = {
  active: 'success',
  archived: 'info',
  disabled: 'danger',
}

const runtimeTypes = ['transformers', 'transformers+lora', 'sklearn', 'scipy', 'vllm']

async function fetchModels() {
  loading.value = true
  try {
    const res = await modelApi.list()
    models.value = res.data.items || res.data
  } finally {
    loading.value = false
  }
}

function openCreate() {
  editingModel.value = null
  form.value = { name: '', family: '', runtime_type: '', version: '', artifact_uri: '', description: '' }
  dialogVisible.value = true
}

function openEdit(model) {
  editingModel.value = model
  form.value = {
    name: model.name,
    family: model.family,
    runtime_type: model.runtime_type,
    version: model.version,
    artifact_uri: model.artifact_uri || '',
    description: model.description || '',
  }
  dialogVisible.value = true
}

async function handleSave() {
  if (!form.value.name || !form.value.family || !form.value.runtime_type || !form.value.version) {
    ElMessage.warning('Name, family, runtime type, and version are required')
    return
  }
  try {
    const payload = { ...form.value }
    if (!payload.artifact_uri) payload.artifact_uri = null
    if (!payload.description) payload.description = null

    if (editingModel.value) {
      await modelApi.update(editingModel.value.id, payload)
      ElMessage.success('Model updated')
    } else {
      await modelApi.create(payload)
      ElMessage.success('Model created')
    }
    dialogVisible.value = false
    fetchModels()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || 'Failed to save model')
  }
}

async function handleActivate(id) {
  await modelApi.activate(id)
  ElMessage.success('Model activated')
  fetchModels()
}

async function handleArchive(id) {
  await ElMessageBox.confirm('Archive this model?', 'Confirm')
  await modelApi.archive(id)
  ElMessage.success('Model archived')
  fetchModels()
}

onMounted(fetchModels)
</script>

<template>
  <div>
    <div style="margin-bottom: 16px; display: flex; justify-content: space-between">
      <h3 style="margin: 0">Model Registry</h3>
      <el-button type="primary" @click="openCreate">
        <el-icon><Plus /></el-icon> Register Model
      </el-button>
    </div>

    <el-table :data="models" v-loading="loading" stripe>
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="name" label="Name" min-width="160" />
      <el-table-column prop="family" label="Family" width="130" />
      <el-table-column prop="runtime_type" label="Runtime" width="140" />
      <el-table-column prop="version" label="Version" width="80" />
      <el-table-column prop="status" label="Status" width="90">
        <template #default="{ row }">
          <el-tag :type="statusColors[row.status]" size="small">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="Tags" min-width="180">
        <template #default="{ row }">
          <el-tag v-for="t in (row.tags || [])" :key="t" size="small" style="margin-right: 4px" type="info">{{ t }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="artifact_uri" label="Artifact URI" min-width="200" show-overflow-tooltip />
      <el-table-column label="Actions" width="220" fixed="right">
        <template #default="{ row }">
          <el-button size="small" @click="openEdit(row)">Edit</el-button>
          <el-button v-if="row.status !== 'active'" type="success" size="small" @click="handleActivate(row.id)">Activate</el-button>
          <el-button v-if="row.status === 'active'" type="warning" size="small" @click="handleArchive(row.id)">Archive</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- Create/Edit Dialog -->
    <el-dialog
      v-model="dialogVisible"
      :title="editingModel ? 'Edit Model' : 'Register Model'"
      width="560px"
    >
      <el-form :model="form" label-width="120px">
        <el-form-item label="Name" required>
          <el-input v-model="form.name" placeholder="e.g. ChatTS-8B" />
        </el-form-item>
        <el-form-item label="Family" required>
          <el-select v-model="form.family" placeholder="Select" style="width: 100%">
            <el-option label="ChatTS" value="chatts" />
            <el-option label="Qwen" value="qwen" />
            <el-option label="ADTK-HBOS" value="adtk_hbos" />
            <el-option label="Ensemble" value="ensemble" />
            <el-option label="Wavelet" value="wavelet" />
            <el-option label="Isolation Forest" value="isolation_forest" />
          </el-select>
        </el-form-item>
        <el-form-item label="Runtime Type" required>
          <el-select v-model="form.runtime_type" placeholder="Select" style="width: 100%">
            <el-option v-for="rt in runtimeTypes" :key="rt" :label="rt" :value="rt" />
          </el-select>
        </el-form-item>
        <el-form-item label="Version" required>
          <el-input v-model="form.version" placeholder="e.g. v1.0" />
        </el-form-item>
        <el-form-item label="Artifact URI">
          <el-input v-model="form.artifact_uri" placeholder="/home/share/llm_models/..." />
        </el-form-item>
        <el-form-item label="Description">
          <el-input v-model="form.description" type="textarea" :rows="3" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="handleSave">Save</el-button>
      </template>
    </el-dialog>
  </div>
</template>
