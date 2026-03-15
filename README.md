# Inference Platform

时序异常检测推理平台，提供 ML 模型推理任务的统一管理、调度和监控。

## 架构

- **后端**: FastAPI + SQLAlchemy 2.0 (async) + Celery + Redis
- **前端**: Vue 3 + Vite (开发中)
- **部署**: Docker Compose

## 支持算法

| 算法 | 类型 | 资源 |
|------|------|------|
| ChatTS-8B | 大模型时序分析 | GPU |
| Qwen-3-VL | 多模态视觉语言模型 | GPU |
| ADTK-HBOS | 统计方法 | CPU |
| Ensemble | 多方法投票 | CPU |
| Wavelet | 小波分解 | CPU |
| Isolation Forest | 树模型 | CPU |

## 快速启动

```bash
# 后端开发环境
conda activate inference-platform
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload

# Docker 部署
docker compose up -d
```

## 项目结构

```
├── backend/          # FastAPI 后端
│   ├── app/
│   │   ├── api/          # 路由 (tasks, models, configs, data_sources)
│   │   ├── models/       # SQLAlchemy ORM 模型
│   │   ├── schemas/      # Pydantic 请求/响应模型
│   │   ├── services/     # 业务逻辑层
│   │   ├── adapters/     # 执行适配器 (CLI subprocess)
│   │   └── core/         # 配置、数据库、Celery
│   └── requirements.txt
├── frontend/         # Vue 3 前端 (开发中)
├── docker/           # Dockerfile
├── docs/             # 设计文档
├── configs/          # 配置文件
├── scripts/          # 运维脚本
└── data/             # 数据目录 (git ignored)
```

## 文档

- [可行性评估](docs/01-可行性评估.md)
- [技术选型决策](docs/02-技术选型决策.md)
- [项目结构说明](docs/03-项目结构说明.md)
- [联调记录与环境配置](docs/04-联调记录与环境配置.md)
- [开发进度与待办](docs/05-开发进度与待办.md)
