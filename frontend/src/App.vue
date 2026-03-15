<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<template>
  <el-container style="min-height: 100vh">
    <el-aside width="200px" style="background: #304156">
      <div style="padding: 20px; text-align: center; color: #fff; font-size: 16px; font-weight: bold">
        Inference Platform
      </div>
      <el-menu
        :default-active="$route.path"
        router
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
      >
        <el-menu-item index="/tasks">
          <el-icon><List /></el-icon>
          <span>Task Monitor</span>
        </el-menu-item>
        <el-menu-item index="/tasks/create">
          <el-icon><Plus /></el-icon>
          <span>Create Task</span>
        </el-menu-item>
        <el-menu-item index="/models">
          <el-icon><Box /></el-icon>
          <span>Model Center</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header style="border-bottom: 1px solid #e6e6e6; display: flex; align-items: center; justify-content: space-between">
        <el-breadcrumb separator="/">
          <el-breadcrumb-item :to="{ path: '/' }">Home</el-breadcrumb-item>
          <el-breadcrumb-item>{{ $route.name }}</el-breadcrumb-item>
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
</style>
