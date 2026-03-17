<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { taskApi, configApi, uploadApi } from '../api'
import { gpuAlgorithms } from '../utils/constants'
import { UploadFilled } from '@element-plus/icons-vue'

const router = useRouter()
const algorithms = ref([])
const chattsPrompts = ref([])
const submitting = ref(false)

const form = ref({
  task_name: '',
  algorithm_name: '',
  submitter: '',
  input_files: '',
  n_downsample: 5000,
  prompt_template: 'default',
  max_tokens: 2048,
  submit_now: true,
})

const isGpu = computed(() => gpuAlgorithms.includes(form.value.algorithm_name))
const isChatts = computed(() => form.value.algorithm_name === 'chatts')

const uploadedFiles = ref([])
const inputMode = ref('upload') // 'upload' or 'manual'

async function handleUpload(options) {
  const formData = new FormData()
  formData.append('files', options.file)
  try {
    const res = await uploadApi.upload(formData)
    const files = res.data.files || []
    for (const f of files) {
      uploadedFiles.value.push({ name: f.filename, path: f.path, size: f.size })
    }
    options.onSuccess(res.data)
  } catch (e) {
    options.onError(e)
    ElMessage.error(e.response?.data?.detail || 'Upload failed')
  }
}

function handleRemoveFile(file) {
  const idx = uploadedFiles.value.findIndex(f => f.name === file.name)
  if (idx >= 0) uploadedFiles.value.splice(idx, 1)
}

onMounted(async () => {
  try {
    const res = await configApi.algorithms()
    algorithms.value = (res.data.algorithms || res.data).map(a => ({
      name: a.name,
      label: a.display_name || a.name,
      resource: a.resource || (['chatts', 'qwen'].includes(a.name) ? 'gpu' : 'cpu'),
    }))
  } catch {
    algorithms.value = [
      { name: 'chatts', label: 'ChatTS-8B', resource: 'gpu' },
      { name: 'qwen', label: 'Qwen-3-VL', resource: 'gpu' },
      { name: 'adtk_hbos', label: 'ADTK-HBOS', resource: 'cpu' },
      { name: 'ensemble', label: 'Ensemble', resource: 'cpu' },
      { name: 'wavelet', label: 'Wavelet', resource: 'cpu' },
      { name: 'isolation_forest', label: 'Isolation Forest', resource: 'cpu' },
      { name: 'stl_wavelet', label: 'STL-Wavelet', resource: 'cpu' },
    ]
  }
  // Load ChatTS prompt templates
  try {
    const res = await configApi.chattsPrompts()
    chattsPrompts.value = res.data || []
  } catch {
    chattsPrompts.value = [{ key: 'default', name: '默认精简版', description: '精简JSON格式' }]
  }
})

async function handleSubmit() {
  const hasUploadedFiles = inputMode.value === 'upload' && uploadedFiles.value.length > 0
  const hasManualFiles = inputMode.value === 'manual' && form.value.input_files.trim()

  if (!form.value.task_name || !form.value.algorithm_name || (!hasUploadedFiles && !hasManualFiles)) {
    ElMessage.warning('Please fill in required fields (including input files)')
    return
  }

  submitting.value = true
  try {
    const files = inputMode.value === 'upload'
      ? uploadedFiles.value.map(f => f.path)
      : form.value.input_files.split('\n').map(f => f.trim()).filter(Boolean)

    const params = {
      method: form.value.algorithm_name,
      n_downsample: form.value.n_downsample,
    }
    if (isGpu.value) {
      params.max_tokens = form.value.max_tokens
    }
    if (isChatts.value && form.value.prompt_template !== 'default') {
      params.prompt_template = form.value.prompt_template
    }

    const payload = {
      task_name: form.value.task_name,
      algorithm_name: form.value.algorithm_name,
      submitter: form.value.submitter || undefined,
      input_snapshot: { files },
      parameter_snapshot: params,
    }

    const res = await taskApi.create(payload)
    const taskId = res.data.id

    if (form.value.submit_now) {
      await taskApi.submit(taskId)
      ElMessage.success('Task created and submitted')
    } else {
      ElMessage.success('Task created as draft')
    }

    router.push(`/tasks/${taskId}`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || 'Failed to create task')
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <el-card>
    <template #header>
      <span style="font-weight: bold">Create Inference Task</span>
    </template>

    <el-form :model="form" label-width="160px" style="max-width: 700px">
      <el-form-item label="Task Name" required>
        <el-input v-model="form.task_name" placeholder="e.g. PI_20412 anomaly detection" />
      </el-form-item>

      <el-form-item label="Algorithm" required>
        <el-select v-model="form.algorithm_name" placeholder="Select algorithm" style="width: 100%">
          <el-option
            v-for="alg in algorithms"
            :key="alg.name"
            :label="`${alg.label || alg.name} (${alg.resource || 'cpu'})`"
            :value="alg.name"
          />
        </el-select>
      </el-form-item>

      <el-form-item label="Input Files" required>
        <el-radio-group v-model="inputMode" style="margin-bottom: 12px">
          <el-radio-button value="upload">Upload Files</el-radio-button>
          <el-radio-button value="manual">Manual Path</el-radio-button>
        </el-radio-group>

        <el-upload
          v-if="inputMode === 'upload'"
          drag
          multiple
          :http-request="handleUpload"
          :on-remove="handleRemoveFile"
          accept=".csv,.txt,.xlsx,.xls"
          style="width: 100%"
        >
          <el-icon :size="40" style="color: #909399; margin-bottom: 8px"><UploadFilled /></el-icon>
          <div>Drop CSV files here or <em>click to browse</em></div>
          <template #tip>
            <div style="color: #909399; font-size: 12px">Supports .csv, .txt, .xlsx, .xls (max 100MB each)</div>
          </template>
        </el-upload>

        <el-input
          v-else
          v-model="form.input_files"
          type="textarea"
          :rows="3"
          placeholder="One file path per line, e.g.&#10;/home/share/data/PI_20412.PV.csv"
        />
      </el-form-item>

      <el-form-item label="Downsample">
        <el-input-number v-model="form.n_downsample" :min="100" :max="50000" :step="1000" />
      </el-form-item>

      <el-form-item label="Submitter">
        <el-input v-model="form.submitter" placeholder="Your name (optional)" />
      </el-form-item>

      <!-- GPU model options -->
      <template v-if="isGpu">
        <el-divider>GPU Model Options</el-divider>

        <el-form-item v-if="isChatts" label="Prompt Template">
          <el-select v-model="form.prompt_template" style="width: 100%">
            <el-option
              v-for="p in chattsPrompts"
              :key="p.key"
              :label="p.name"
              :value="p.key"
            >
              <span>{{ p.name }}</span>
              <span style="color: #909399; font-size: 12px; margin-left: 8px">{{ p.description }}</span>
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="Max Tokens">
          <el-input-number v-model="form.max_tokens" :min="256" :max="8192" :step="256" />
        </el-form-item>
      </template>

      <el-divider />

      <el-form-item label="Submit Now">
        <el-switch v-model="form.submit_now" />
        <span style="margin-left: 8px; color: #909399; font-size: 12px">
          {{ form.submit_now ? 'Will submit for execution immediately' : 'Will save as draft' }}
        </span>
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">
          {{ form.submit_now ? 'Create & Submit' : 'Save Draft' }}
        </el-button>
        <el-button @click="router.back()">Cancel</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>
