<script setup>
import { useRoute } from 'vue-router'
const route = useRoute()

const info = {
  '/data-sources': { title: '数据池', desc: '数据源管理、文件列表、连通性测试', icon: 'Folder' },
  '/annotate': { title: '数据标注', desc: 'D3.js 交互标注、异常标签编辑、标注导出', icon: 'EditPen' },
  '/evaluation': { title: '模型评估', desc: '选择推理结果 + 上传标注 → 多指标对比分析', icon: 'TrophyBase' },
  '/training': { title: '模型微调', desc: 'LLaMA-Factory 集成、微调任务管理', icon: 'Cpu' },
  '/users': { title: '用户管理', desc: 'JWT 认证、API Key 管理、权限分配', icon: 'User' },
  '/settings': { title: '系统设置', desc: '算法参数模板、平台配置', icon: 'Setting' },
}

const page = info[route.path] || { title: '开发中', desc: '该功能模块正在规划中' }
</script>

<template>
  <el-card style="max-width: 600px; margin: 80px auto; text-align: center">
    <div style="font-size: 48px; color: #909399; margin-bottom: 20px">🚧</div>
    <h2 style="margin-bottom: 8px">{{ page.title }}</h2>
    <p style="color: #909399; margin-bottom: 24px">{{ page.desc }}</p>
    <el-tag type="info" size="large">功能开发中</el-tag>
  </el-card>
</template>
