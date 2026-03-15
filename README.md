# Inference Platform - 时序异常检测推理平台

基于 FastAPI + Celery + Vue 3 的时序异常检测推理平台，支持多种 GPU/CPU 算法的统一调度与管理。

## 架构

```
┌─────────────┐    ┌───────────┐    ┌──────────────┐    ┌─────────────────┐
│  Vue 3 前端  │───▶│ FastAPI   │───▶│ Celery Worker│───▶│ 推理执行适配器    │
│  :5173      │    │ :8100     │    │              │    │ (subprocess)    │
└─────────────┘    └─────┬─────┘    └──────┬───────┘    └────────┬────────┘
                         │                 │                     │
                    ┌────▼────┐      ┌─────▼─────┐    ┌─────────▼─────────┐
                    │ SQLite  │      │   Redis   │    │ ts-iteration-loop │
                    │   DB    │      │  Broker   │    │    run.py         │
                    └─────────┘      └───────────┘    └───────────────────┘
```

- **后端**: FastAPI + SQLAlchemy 2.0 (async) + Celery + Redis
- **前端**: Vue 3 + Vite + Element Plus
- **部署**: Docker Compose (GPU 支持)

## 支持的算法

| 算法 | 类型 | 模型 | 硬件要求 | 验证状态 |
|------|------|------|----------|----------|
| ChatTS | GPU (LLM) | ChatTS-8B | 8GB+ VRAM (4-bit) | ✅ |
| Qwen-VL | GPU (多模态) | Qwen3-VL-8B-Instruct | 18GB+ VRAM (bf16) | ✅ |
| Qwen-VL | GPU (多模态) | Qwen3.5-27B | 48GB+ VRAM (4-bit) | 代码就绪 |
| ADTK-HBOS | CPU (统计) | 无 | CPU only | ✅ |
| Ensemble | CPU (投票) | 无 | CPU only | 待验证 |
| Wavelet | CPU (小波) | 无 | CPU only | 待验证 |
| Isolation Forest | CPU (树模型) | 无 | CPU only | 待验证 |

## 快速开始

### 前置条件

- Redis
- conda (Miniconda / Anaconda)
- Node.js 18+ (前端)
- NVIDIA GPU + CUDA（GPU 算法可选）

### 1. 平台后端

```bash
conda create -n inference-platform python=3.11 -y
conda activate inference-platform
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. GPU 推理环境（可选）

```bash
conda create -n chatts python=3.12 -y
conda activate chatts
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install 'transformers>=5.0' accelerate bitsandbytes
pip install apache-iotdb statsmodels joblib kneed adtk pyod \
    tsdownsample PyWavelets psutil pynvml ruptures seaborn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. CPU 推理环境

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
cd frontend && npm install
```

### 5. 配置

编辑 `backend/.env` 或直接修改 `backend/app/core/config.py`：

```env
# 旧项目推理脚本
OLD_PROJECT_PATH=/path/to/ts-iteration-loop
OLD_PYTHON_PATH=/path/to/conda/envs/ts/bin/python
GPU_PYTHON_PATH=/path/to/conda/envs/chatts/bin/python

# Redis
REDIS_URL=redis://localhost:6379/0

# 数据
DATA_INFERENCE_DIR=/path/to/inference/output
```

### 6. 启动

```bash
# Redis
redis-server --daemonize yes

# 后端
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8100

# Celery Worker（新终端）
cd backend && celery -A app.core.celery_app worker --loglevel=info

# 前端（新终端）
cd frontend && npm run dev

# 种子数据（首次）
python scripts/seed_data.py
```

## Docker 部署

```bash
# 创建 .env 文件配置宿主机路径
cat > .env <<EOF
OLD_PROJECT_PATH=/home/dff652/TS-anomaly-detection/ts-iteration-loop
MODEL_DIR=/home/data1/llm_models
CONDA_PREFIX=/home/dff652/miniconda3
EOF

# 启动（GPU 需要 nvidia-container-toolkit）
docker compose up -d
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/inference/tasks` | POST/GET | 任务创建/列表 |
| `/api/v1/inference/tasks/{id}` | GET/PUT | 任务详情/更新 |
| `/api/v1/inference/tasks/{id}/submit` | POST | 提交执行 |
| `/api/v1/inference/tasks/{id}/cancel` | POST | 取消任务 |
| `/api/v1/inference/tasks/{id}/retry` | POST | 重试任务 |
| `/api/v1/inference/tasks/{id}/results` | GET | 推理结果 |
| `/api/v1/inference/tasks/{id}/logs` | GET | 执行日志 |
| `/api/v1/inference/tasks/stats` | GET | 状态统计 |
| `/api/v1/models` | POST/GET | 模型管理 |
| `/api/v1/inference/configs` | POST/GET | 配置模板 |

## 项目结构

```
inference-platform/
├── backend/                 # FastAPI 后端
│   ├── app/
│   │   ├── api/            # API 路由 (tasks, models, configs, data_sources)
│   │   ├── core/           # 配置、数据库、Celery
│   │   ├── models/         # SQLAlchemy ORM 模型
│   │   ├── schemas/        # Pydantic 请求/响应模型
│   │   ├── services/       # 业务逻辑 + 状态机
│   │   └── adapters/       # 执行适配器 (CLI subprocess)
│   ├── data/               # SQLite DB + 推理结果
│   └── requirements.txt
├── frontend/               # Vue 3 + Vite + Element Plus
│   └── src/
│       ├── views/          # 任务列表/创建/详情/模型中心
│       └── api/            # Axios API 客户端
├── docker/                 # Dockerfile
├── docs/                   # 设计文档（5 份）
├── scripts/                # 种子数据等工具脚本
├── data/                   # 测试数据 (git ignored)
└── docker-compose.yml      # Docker 编排 (含 GPU 支持)
```

## 文档

- [01-可行性评估](docs/01-可行性评估.md)
- [02-技术选型决策](docs/02-技术选型决策.md)
- [03-项目结构说明](docs/03-项目结构说明.md)
- [04-联调记录与环境配置](docs/04-联调记录与环境配置.md) — 环境安装、路径配置、验证记录
- [05-开发进度与待办](docs/05-开发进度与待办.md) — 里程碑与进度跟踪

## 三套 conda 环境说明

| 环境 | Python | 职责 | 关键依赖 |
|------|--------|------|----------|
| `inference-platform` | 3.11 | 平台后端 + Celery | FastAPI, SQLAlchemy, Celery |
| `chatts` | 3.12 | GPU 推理 (ChatTS/Qwen) | transformers 5.3, bitsandbytes, torch |
| `ts` | 3.8 | CPU 推理 (ADTK/Ensemble) | adtk, statsmodels, scikit-learn |

执行适配器根据方法自动切换 Python 解释器，无需手动指定。
