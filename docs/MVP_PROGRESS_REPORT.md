# 推理平台 — 阶段性汇报与工作日志

**项目**: inference-platform（时序异常检测推理平台）
**时间范围**: 2026-03-15 至 2026-03-17
**当前状态**: 一期核心功能开发完成，Qwen3-VL 的 real vLLM 验证通过

---

## 1. 项目整体进度汇报

本项目已完成一期 MVP 主体开发，并在 **2026-03-17** 完成 `Qwen3-VL + real vLLM` 的单机受限配置验证。当前已达到内部灰度试用标准，且 GPU 推理层已确认可在不改平台接口的情况下逐步切换到真实 `vllm serve`。

### 核心功能模块

*   **后端 API（FastAPI + SQLAlchemy 2.0）**:
    *   20+ RESTful 接口，覆盖任务、模型、配置模板、数据源四组 CRUD。
    *   5 个核心数据模型（InferenceTask / ModelEntity / ConfigTemplate / DataSource / ResultIndex）。
    *   8 状态任务状态机（DRAFT → PENDING → QUEUED → RUNNING → COMPLETED / FAILED / CANCELLED / TIMEOUT），含转换校验。
*   **异步执行引擎（Celery + Redis）**:
    *   CLI 执行适配器（CLIExecutorAdapter），通过 subprocess 调用旧项目 `run.py`，实现零侵入集成。
    *   GPU/CPU Python 环境自动切换（chatts conda env ↔ ts env）。
    *   任务取消支持（SIGTERM/SIGKILL + Celery revoke）。
    *   结果自动收集：解析 `metrics.json` / `segments.json`，回写 `InferenceResultIndex`。
*   **前端管理界面（Vue 3 + Element Plus）**:
    *   任务监控：8 状态卡片 + 筛选表格 + 5s 自动轮询 + 分页。
    *   任务创建：算法选择 + GPU 模型参数 + 立即提交/草稿模式。
    *   任务详情：Info / Results / Logs 三标签 + 异常段表格展示。
    *   模型中心：模型 CRUD + 激活/归档 + 算法类型筛选。
*   **GPU 算法验证**:
    *   ChatTS-8B 验证通过（4-bit 量化，单卡 ~7.6GB）。
    *   Qwen3-VL-8B-Instruct 验证通过（bf16，单卡 ~18GB）。
    *   Qwen3-VL-8B-Instruct real `vllm serve` 验证通过（`vllm 0.11.0`，单机受限配置）。
    *   Qwen3.5-27B 代码就绪（需 48GB+ 显存设备）。
*   **CPU 算法支持**:
    *   ADTK-HBOS 端到端验证通过。
    *   Ensemble / Isolation Forest 方法名映射完成（`ensemble → piecewise_linear`, `isolation_forest → iforest`）。

### 技术架构

*   **后端**: FastAPI + SQLAlchemy 2.0 (async) + SQLite (WAL) + Celery + Redis
*   **前端**: Vite + Vue 3 + Element Plus + Vue Router + Pinia + Axios
*   **部署**: Docker Compose（backend + celery_worker + redis），支持 NVIDIA GPU runtime
*   **硬件**: 2× NVIDIA RTX 2080 Ti (22GB)，GPU/CPU 任务自动路由

---

## 2. 每日开发工作详情

### 开发日历 (2026.03)

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | | | | **15**<br>项目搭建<br>全链路打通<br>前端完成 |
| **16**<br>GPU验证<br>生产加固<br>CPU算法映射 | **17**<br>vLLM验证<br>Qwen3-VL跑通<br>平台联调 | | | | | |

---

### 3月15日（项目创建 — 后端 + 全链路 + 前端一日完成）

> **阶段目标**: 从零搭建推理平台，打通 API → Celery → 执行器 → 结果回收全链路，并交付可用前端。

#### 22:31 — 项目脚手架搭建（commit: `b112d66`）

*   **后端框架搭建**:
    *   FastAPI + SQLAlchemy 2.0 异步架构，20+ API 接口。
    *   5 个核心数据模型设计与实现。
    *   8 状态任务状态机 + 状态转换校验。
    *   CLI 执行适配器（`executor_adapter.py`），subprocess 调用旧项目 `run.py`。
*   **基础设施**:
    *   Celery + Redis 任务队列配置。
    *   Docker Compose 编排（backend + worker + redis）。
    *   Dockerfile 与环境变量模板。
*   **验证**: 端到端测试通过（adtk_hbos 算法）。
*   **文档**: 5 份设计文档（可行性评估 / 技术选型 / 项目结构 / 环境配置 / 进度待办）。
*   **产出**: 41 个文件，2061 行新增代码。

#### 22:44 — Celery 全链路修复（commit: `8f8c63d`）

*   **状态机修复**: PENDING → QUEUED → RUNNING 正确转换。
*   **竞态条件修复**: 先 commit PENDING 状态再 dispatch Celery 任务。
*   **脱离 session 修复**: 在 session 上下文内读取 task snapshots。
*   **新增接口**:
    *   `GET /tasks/{id}/results` — 返回 metrics、segments、结果文件列表。
    *   `GET /tasks/{id}/logs` — 返回执行日志。
*   **结果收集**: 从 `metrics.json` 提取指标写入 `InferenceResultIndex`。
*   **产出**: 3 个文件，183 行新增 / 29 行修改。

#### 23:15 — Vue 3 前端完整交付（commit: `cb07bc9`）

*   **技术栈**: Vite + Vue 3 + Element Plus + Vue Router + Pinia + Axios。
*   **页面开发**:
    *   **TaskList** — 状态卡片 + 筛选表格 + 5s 自动轮询 + 分页。
    *   **TaskCreate** — 算法选择 + GPU 模型可选参数 + 提交/草稿模式。
    *   **TaskDetail** — Info / Results / Logs 三标签 + 异常段展示 + 自动刷新。
    *   **ModelCenter** — 模型 CRUD + 激活/归档 + 算法类型选择。
*   **API 代理**: Vite 配置反向代理到后端 `localhost:8100`。
*   **产出**: 18 个文件，2731 行新增代码。

#### 23:18 — 种子数据 + 前端字段修复（commit: `895f444`）

*   **种子数据脚本** (`scripts/seed_data.py`):
    *   注册 7 个模型（ChatTS-8B / ChatTS-8B-LoRA / Qwen-3-VL / ADTK-HBOS / Ensemble / Wavelet / Isolation Forest）。
    *   注册 7 个配置模板（含 GPU/CPU 资源标记和默认参数）。
*   **前端修复**:
    *   ModelCenter.vue: 适配后端 schema（`family` / `runtime_type` 字段）。
    *   TaskCreate.vue: 修正算法 API 响应到组件格式的映射。
*   **产出**: 4 个文件，307 行新增。

---

### 3月16日（GPU 验证 + 生产加固 + CPU 算法映射）

> **阶段目标**: 完成 GPU 大模型推理验证，加固生产稳定性，补齐 CPU 算法映射。

#### 01:24 — GPU 算法支持验证（commit: `f15777d`）

*   **环境适配**:
    *   新增 `GPU_PYTHON_PATH` 配置，指向 chatts conda 环境（Python 3.12）。
    *   执行适配器 GPU/CPU Python 自动切换（`GPU_METHODS` + `_get_python_path`）。
*   **参数修复**: `load_in_4bit` 全链路统一为 str 类型（auto / true / false / force）。
*   **接口补充**: retry 端点正确分发 Celery 任务。
*   **验证结果**:
    *   ✅ ChatTS-8B — 4-bit force 量化，单卡 ~7.6GB（cuda:0 / cuda:1 均可）。
    *   ✅ Qwen3-VL-8B — bf16 全精度，单卡 ~18GB（cuda:1）。
    *   🔧 Qwen3.5-27B — 代码就绪，需 48GB+ 显存设备。
*   **种子数据扩充**: 新增 Qwen3-VL-8B-Instruct 和 Qwen3.5-27B 模型注册。
*   **产出**: 6 个文件，103 行新增 / 28 行修改。

#### 01:29 — 环境与部署文档完善（commit: `95bd24f`）

*   **README 重写**: 架构图 + 完整部署指南 + 3 个 conda 环境说明 + API 参考。
*   **环境文档更新**: GPU 环境详情 + 所有 conda 环境 + 模型路径 + 验证记录。
*   **Docker 增强**: GPU 支持（nvidia runtime）+ 模型/conda 卷挂载。
*   **环境变量**: 更新 `.env.example`，新增 GPU 相关配置。
*   **产出**: 4 个文件，395 行新增 / 137 行修改。

#### 09:56 — 生产加固（commit: `f94171a`）

*   **前端错误处理**:
    *   为全部 8 个 action handler 添加 try-catch（submit / cancel / retry / activate / archive 等）。
    *   API 调用后显示正确的成功/错误提示。
*   **任务取消**:
    *   后端实现 subprocess SIGTERM / SIGKILL 终止 + Celery revoke。
    *   新增 cancel API 端点。
*   **Celery 加固**:
    *   PENDING → RUNNING 状态转换原子化（单次 commit）。
    *   处理 `SoftTimeLimitExceeded` / `Terminated` 异常。
    *   修复废弃的 `datetime.utcnow()` 调用。
*   **安全加固**: CORS 收紧为 `localhost:5173` / `localhost:8100`（替代通配符 `*`）。
*   **产出**: 7 个文件，119 行新增 / 36 行修改。

#### 10:20 — CPU 算法方法名映射（commit: `53f418c`）

*   **方法映射**: 在执行适配器中添加 `METHOD_NAME_MAP`，将平台算法名翻译为旧项目 CLI 方法名：
    *   `ensemble` → `piecewise_linear`
    *   `isolation_forest` → `iforest`
*   **产出**: 1 个文件，8 行新增。

---

### 3月17日（Qwen3-VL real vLLM 验证）

> **阶段目标**: 验证 `Qwen3-VL` 能否脱离当前 `transformers` 服务，直接跑在真实 `vllm serve` 上，并确认平台是否无需改造即可接入。

#### 运行时重建与兼容性收敛

*   **新建干净环境**: `qwen-vllm011-clean`。
*   **对齐运行时矩阵**:
    *   `torch==2.8.0`
    *   `torchaudio==2.8.0`
    *   `torchvision==0.23.0`
    *   `xformers==0.0.32.post1`
    *   `vllm==0.11.0`
*   **修复依赖漂移**:
    *   初次安装拉到了 `transformers 5.3.0`，导致 `Qwen2Tokenizer` 与 vLLM tokenizer 缓存逻辑不兼容。
    *   最终锁定为 `transformers 4.57.6` + `huggingface-hub 0.36.2` 后恢复正常。

#### 启动条件验证

*   **失败结论明确**:
    *   `vllm 0.8.5` 不可用，先后卡在 bf16 硬件限制和 `Qwen3VLConfig` 兼容问题。
    *   `vllm 0.11.0` 默认参数在 RTX 2080 Ti 22GB 上显存不足，无法直接启动。
*   **找到可工作配置**:
    *   `dtype=half`
    *   `--enforce-eager`
    *   `--gpu-memory-utilization 0.97`
    *   `--max-model-len 1024`
    *   `--max-num-seqs 1`
    *   `--max-num-batched-tokens 1024`
    *   `--limit-mm-per-prompt {"image":1,"video":0}`
    *   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
*   **脚本固化**:
    *   `services/start_qwen_vllm.sh` 已支持上述参数化配置。

#### 接口与平台联调结果

*   **OpenAI 兼容接口通过**:
    *   `/health`
    *   `/v1/models`
    *   文本 `chat/completions`
    *   图像输入 `chat/completions`
*   **平台 smoke test 通过**:
    *   `scripts/validate_qwen_vllm.py` 对 `PI_20412.PV.csv` 返回结构化异常结果
    *   临时后端实例指向 `http://localhost:8003/v1` 后：
        *   `GET /api/v1/models/vllm/status` 通过
        *   `POST /api/v1/predict` 通过
*   **联调顺手修复**:
    *   `/predict` 返回 `segments` 时的 Pydantic 序列化告警已修复，统一转为 `SegmentResult`

#### 阶段结论

*   `Qwen3-VL + real vLLM` 已验证可用。
*   当前可用范围是 **单机、低并发、受限上下文**，更适合验证和轻量内部使用。
*   平台后端与适配层 **不需要重写**，后续工作重点是部署固化、Celery 链路验证和生产化评估。

---

## 3. 里程碑总览

| 里程碑 | 完成日期 | 内容 |
|--------|----------|------|
| M1: 后端 API 可用 | ✅ 2026-03-15 | FastAPI + 20+ API + 执行适配器 + 端到端联调 |
| M2: Celery 全链路 | ✅ 2026-03-15 | Redis + submit → Celery → 执行器 → 结果回收 |
| M3: 前端 P0 页面 | ✅ 2026-03-15 | Vue 3 四大页面（任务创建/监控/详情 + 模型中心）|
| M3.5: GPU 算法验证 | ✅ 2026-03-16 | ChatTS-8B + Qwen3-VL-8B 全链路通过 |
| M3.6: Qwen3-VL real vLLM 验证 | ✅ 2026-03-17 | `vllm serve` + OpenAI 接口 + `/predict` 联调通过 |
| M4: 一期灰度上线 | 待定 | 内部小范围使用 |
| M5: CPU 算法全量验证 | 待定 | Ensemble + Wavelet + Isolation Forest 验证 |
| M6: 多机多卡部署 | 待定 | PostgreSQL + 多 worker + GPU 调度 |

---

## 4. 代码统计

| 指标 | 数量 |
|------|------|
| 总提交数 | 8 |
| 新增/修改文件 | 69 个（去重后约 45 个独立文件） |
| 代码新增行数 | ~5,800+ 行 |
| 后端 Python 文件 | 18 个 |
| 前端 Vue/JS 文件 | 12 个 |
| 文档 | 6 份设计文档 + README |
| 配置/Docker | 6 个 |

---

## 5. 已知技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SQLite 并发写入限制 | Celery 多 worker 并发写 DB 可能冲突 | 一期单 worker；二期切 PostgreSQL |
| Qwen3.5-27B 显存不足 | 2×22GB 不够加载 | 代码已支持多卡和 4-bit，换 A100/4090 后可用 |
| ChatTS-8B 4-bit 质量 | force 量化可能降低检测精度 | 后续在大显存设备上使用 bf16 全精度 |
| Qwen3-VL vLLM 吞吐受限 | 当前需 eager + 单并发 + 较短上下文 | 先用于验证/轻量场景，后续评估更大显存设备或继续调参 |
| ChatTS 暂不适合切 real vLLM | 当前模型/插件路径不稳定 | 先保留 transformers 服务，等待上游支持或单独做 PoC |
| 旧项目 run.py 输出不稳定 | 不同方法的 JSON 结构可能有差异 | CPU/GPU 新路径已逐步去 subprocess，结果解析也做了容错 |
