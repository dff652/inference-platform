import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/tasks',
  },
  {
    path: '/tasks',
    name: 'TaskList',
    component: () => import('../views/TaskList.vue'),
  },
  {
    path: '/tasks/create',
    name: 'TaskCreate',
    component: () => import('../views/TaskCreate.vue'),
  },
  {
    path: '/tasks/:id',
    name: 'TaskDetail',
    component: () => import('../views/TaskDetail.vue'),
    props: true,
  },
  {
    path: '/models',
    name: 'ModelCenter',
    component: () => import('../views/ModelCenter.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
