import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/tasks',
  },
  // 任务中心
  {
    path: '/tasks',
    name: 'TaskList',
    meta: { title: '任务看板' },
    component: () => import('../views/TaskList.vue'),
  },
  {
    path: '/tasks/create',
    name: 'TaskCreate',
    meta: { title: '创建推理任务' },
    component: () => import('../views/TaskCreate.vue'),
  },
  {
    path: '/tasks/:id',
    name: 'TaskDetail',
    meta: { title: '任务详情' },
    component: () => import('../views/TaskDetail.vue'),
    props: true,
  },
  // 数据管理
  {
    path: '/data-sources',
    name: 'DataSources',
    meta: { title: '数据池' },
    component: () => import('../views/Placeholder.vue'),
  },
  {
    path: '/annotate',
    name: 'Annotator',
    meta: { title: '数据标注' },
    component: () => import('../views/Placeholder.vue'),
  },
  // 模型管理
  {
    path: '/models',
    name: 'ModelCenter',
    meta: { title: '模型中心' },
    component: () => import('../views/ModelCenter.vue'),
  },
  {
    path: '/evaluation',
    name: 'Evaluation',
    meta: { title: '模型评估' },
    component: () => import('../views/Placeholder.vue'),
  },
  {
    path: '/training',
    name: 'Training',
    meta: { title: '模型微调' },
    component: () => import('../views/Placeholder.vue'),
  },
  // 系统管理
  {
    path: '/users',
    name: 'UserManagement',
    meta: { title: '用户管理' },
    component: () => import('../views/Placeholder.vue'),
  },
  {
    path: '/settings',
    name: 'SystemSettings',
    meta: { title: '系统设置' },
    component: () => import('../views/Placeholder.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
