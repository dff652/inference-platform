<script setup>
import { useRouter } from 'vue-router'
import {
  List, Plus, Box, Folder, EditPen, TrophyBase,
  Cpu, Setting, User,
} from '@element-plus/icons-vue'

const router = useRouter()
</script>

<template>
  <el-container style="min-height: 100vh">
    <el-aside width="210px" style="background: #304156">
      <div style="padding: 20px; text-align: center; color: #fff; font-size: 15px; font-weight: bold">
        推理平台
      </div>
      <el-menu
        :default-active="$route.path"
        router
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
      >
        <!-- 任务中心 -->
        <el-sub-menu index="task-group">
          <template #title>
            <el-icon><List /></el-icon>
            <span>任务中心</span>
          </template>
          <el-menu-item index="/tasks">任务看板</el-menu-item>
          <el-menu-item index="/tasks/create">创建推理任务</el-menu-item>
        </el-sub-menu>

        <!-- 数据管理 -->
        <el-sub-menu index="data-group">
          <template #title>
            <el-icon><Folder /></el-icon>
            <span>数据管理</span>
          </template>
          <el-menu-item index="/data-sources">数据池</el-menu-item>
          <el-menu-item index="/annotate">数据标注</el-menu-item>
        </el-sub-menu>

        <!-- 模型管理 -->
        <el-sub-menu index="model-group">
          <template #title>
            <el-icon><Cpu /></el-icon>
            <span>模型管理</span>
          </template>
          <el-menu-item index="/models">模型中心</el-menu-item>
          <el-menu-item index="/evaluation">模型评估</el-menu-item>
          <el-menu-item index="/training">模型微调</el-menu-item>
        </el-sub-menu>

        <!-- 系统管理 -->
        <el-sub-menu index="sys-group">
          <template #title>
            <el-icon><Setting /></el-icon>
            <span>系统管理</span>
          </template>
          <el-menu-item index="/users">用户管理</el-menu-item>
          <el-menu-item index="/settings">系统设置</el-menu-item>
        </el-sub-menu>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header style="border-bottom: 1px solid #e6e6e6; display: flex; align-items: center; justify-content: space-between">
        <el-breadcrumb separator="/">
          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
          <el-breadcrumb-item>{{ $route.meta.title || $route.name }}</el-breadcrumb-item>
        </el-breadcrumb>
      </el-header>

      <el-main style="background: #f0f2f5; padding: 20px">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style>
.el-aside .el-menu {
  border-right: none;
}
.el-sub-menu .el-menu-item {
  padding-left: 50px !important;
}
</style>
