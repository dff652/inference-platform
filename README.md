# Inference Platform

时序异常检测推理平台，基于 FastAPI + Celery + Vue 3，统一管理 CPU 算法和 GPU 大模型推理任务。

## 当前状态

- 截至 2026-03-16，MVP 主链路已完成，可用于内部联调和灰度试用
- CPU 方法已集成到平台内直接执行
- GPU 方法当前使用 `transformers` 常驻服务，对外暴露 OpenAI 兼容 HTTP 接口
- 真正切换到 vLLM 仍属于后续规划，当前代码里保留了兼容 vLLM 接口的适配层

## 当前真实架构

```text
Vue 3 Frontend (:5173)
        |
        v
FastAPI API (:8100)
        |
        v
Celery Worker + Redis
   |             |
   |             +--> 任务状态 / 结果索引 / 日志
   |
   +--> CPU 任务 -> 平台内 dispatcher 直接执行
   |
   +--> GPU 任务 -> HTTP 调 GPU 服务
                    |- ChatTS service (:8001, transformers)
                    |- Qwen service   (:8002, transformers)
   |
   +--> fallback -> 旧项目 subprocess
```

## 部署模式

### 1. 开发联调模式

适合当前仓库默认用法。

- 宿主机启动 Redis
- 宿主机启动 GPU 服务：`services/chatts_serve.py` / `services/qwen_serve.py`
- 宿主机启动 FastAPI / Celery / 前端

### 2. 混合部署模式

当前 `docker-compose.yml` 只启动：

- `backend`
- `celery_worker`
- `redis`

注意：

- Compose **不会**启动 ChatTS/Qwen GPU 服务
- `celery_worker` 依赖宿主机挂载的旧项目目录、模型目录、conda 环境
- 若 GPU 服务跑在宿主机，需确保容器内配置的 `VLLM_*_ENDPOINT` 可访问到这些服务

## 快速开始

### 前置条件

- Conda
- Redis
- Node.js 18+
- NVIDIA GPU + CUDA（GPU 方法可选）
- 模型权重目录
- 旧项目 `ts-iteration-loop` 目录

### 1. 安装后端环境

```bash
conda create -n inference-platform python=3.11 -y
conda activate inference-platform
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 安装 GPU 环境

```bash
conda create -n chatts python=3.12 -y
conda activate chatts
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install 'transformers>=5.0' accelerate bitsandbytes
pip install apache-iotdb statsmodels joblib kneed adtk pyod \
    tsdownsample PyWavelets psutil pynvml ruptures seaborn \
    pandas numpy scipy scikit-learn matplotlib Pillow tqdm \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 安装 CPU 算法环境

```bash
conda create -n ts python=3.8 -y
conda activate ts
pip install pandas numpy scipy scikit-learn statsmodels \
    adtk kneed pyod apache-iotdb tsdownsample \
    PyWavelets psutil pynvml ruptures seaborn joblib tqdm \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 前端

```bash
cd frontend
npm install
```

### 5. 配置

复制模板：

```bash
cp .env.example .env
```

按实际路径修改 `.env`。

### 6. 启动

```bash
# Redis
redis-server --daemonize yes

# GPU 服务
./services/start_vllm.sh all

# 后端 API
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8100

# Celery Worker
cd backend
celery -A app.core.celery_app worker --loglevel=info

# 前端
cd frontend
npm run dev

# 种子数据（首次）
cd ..
python scripts/seed_data.py
```

## Docker Compose

```bash
cp .env.example .env
docker compose up -d
```

当前 Compose 额外要求：

- `.env` 中配置 `OLD_PROJECT_PATH`
- `.env` 中配置 `MODEL_DIR`
- `.env` 中配置 `CONDA_PREFIX`
- GPU 服务仍需在宿主机单独启动，或自行扩展 Compose

## 部署后验证

```bash
curl http://localhost:8100/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8100/api/v1/models/vllm/status
curl http://localhost:8100/api/v1/inference/configs/algorithms
```

## API

主要接口：

- `GET /health`
- `POST/GET /api/v1/inference/tasks`
- `POST /api/v1/inference/tasks/{id}/submit`
- `POST /api/v1/inference/tasks/{id}/cancel`
- `POST /api/v1/inference/tasks/{id}/retry`
- `GET /api/v1/inference/tasks/{id}/results`
- `GET /api/v1/inference/tasks/{id}/logs`
- `POST /api/v1/predict`
- `POST /api/v1/uploads`
- `GET /api/v1/models/vllm/status`

## 文档索引

- [docs/03-项目结构说明.md](docs/03-项目结构说明.md)
- [docs/04-联调记录与环境配置.md](docs/04-联调记录与环境配置.md)
- [docs/05-开发进度与待办.md](docs/05-开发进度与待办.md)
- [docs/06-推理框架选型.md](docs/06-推理框架选型.md)
- [docs/07-部署操作手册.md](docs/07-部署操作手册.md)
- [docs/08-vllm-validation-plan.md](docs/08-vllm-validation-plan.md)
