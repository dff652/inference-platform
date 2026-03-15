<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { taskApi, configApi } from '../api'

const router = useRouter()
const algorithms = ref([])
const submitting = ref(false)

const form = ref({
  task_name: '',
  algorithm_name: '',
  submitter: '',
  input_files: '',
  n_downsample: 5000,
  model_path: '',
  lora_adapter_path: '',
  load_in_4bit: false,
  submit_now: true,
})

const gpuAlgorithms = ['chatts', 'qwen']

onMounted(async () => {
  try {
    const res = await configApi.algorithms()
    // API returns [{id, name, display_name, description}, ...]
    algorithms.value = (res.data.algorithms || res.data).map(a => ({
      name: a.name,
      label: a.display_name || a.name,
      resource: ['chatts', 'qwen'].includes(a.name) ? 'gpu' : 'cpu',
    }))
  } catch {
    algorithms.value = [
      { name: 'chatts', label: 'ChatTS-8B', resource: 'gpu' },
      { name: 'qwen', label: 'Qwen-3-VL', resource: 'gpu' },
      { name: 'adtk_hbos', label: 'ADTK-HBOS', resource: 'cpu' },
      { name: 'ensemble', label: 'Ensemble', resource: 'cpu' },
      { name: 'wavelet', label: 'Wavelet', resource: 'cpu' },
      { name: 'isolation_forest', label: 'Isolation Forest', resource: 'cpu' },
    ]
  }
})

async function handleSubmit() {
  if (!form.value.task_name || !form.value.algorithm_name || !form.value.input_files) {
    ElMessage.warning('Please fill in required fields')
    return
  }

  submitting.value = true
  try {
    const files = form.value.input_files.split('\n').map(f => f.trim()).filter(Boolean)

    const params = {
      method: form.value.algorithm_name,
      n_downsample: form.value.n_downsample,
    }
    if (form.value.model_path) params.model_path = form.value.model_path
    if (form.value.lora_adapter_path) params.lora_adapter_path = form.value.lora_adapter_path
    if (form.value.load_in_4bit) params.load_in_4bit = true

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
        <el-input
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
      <template v-if="gpuAlgorithms.includes(form.algorithm_name)">
        <el-divider>GPU Model Options</el-divider>

        <el-form-item label="Model Path">
          <el-input v-model="form.model_path" placeholder="/home/share/llm_models/..." />
        </el-form-item>

        <el-form-item label="LoRA Adapter">
          <el-input v-model="form.lora_adapter_path" placeholder="(optional)" />
        </el-form-item>

        <el-form-item label="4-bit Quantize">
          <el-switch v-model="form.load_in_4bit" />
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
