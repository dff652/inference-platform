# vLLM Validation Plan

## Last Updated

2026-03-17

## 1. Goal

Validate whether the current GPU serving layer can move from `transformers` services to real `vllm serve` with minimal platform changes.

This branch is for validation only.

Out of scope for the first round:

- production rollout
- full multi-node deployment
- replacing CPU paths

## 2. Current Baseline

Current production-like baseline in this repo:

- ChatTS service: `services/chatts_serve.py`
- Qwen service: `services/qwen_serve.py`
- platform adapter: `backend/app/adapters/vllm_backend.py`
- dispatcher route: `backend/app/algorithms/dispatcher.py`

The platform already talks to GPU services through OpenAI-compatible HTTP endpoints. This means the main validation target is the GPU service implementation, not the platform API shape.

## 3. Validation Order

Priority update on 2026-03-16:

- ChatTS is deferred for now
- Qwen3-VL is the only in-scope model for the first validation round

### Phase 1: Qwen3-VL on real vLLM

Reason:

- lower integration risk
- standard multimodal model
- official vLLM docs already expose a Qwen3-VL serving path

Success criteria:

- `vllm serve` starts successfully
- `/v1/models` and `/v1/chat/completions` are reachable
- platform `check_vllm_health()` passes
- one end-to-end task succeeds from Celery through result files

### Phase 2: ChatTS on real vLLM

Current status:

- deferred
- do not spend engineering time here before Qwen3-VL passes

Reason:

- highest value if successful
- highest risk because of custom time series multimodal path

Success criteria:

- ChatTS custom plugin registers successfully
- time series multimodal request is accepted
- returned JSON can be parsed by existing platform code
- one end-to-end task completes without falling back to `transformers`

### Phase 3: Compare with current transformers services

Compare:

- cold start time
- steady-state latency
- GPU memory usage
- output stability

## 4. Decision Gates

### Promote Qwen3-VL to vLLM if:

- startup is stable
- inference output format is compatible
- latency or operational simplicity is better than current service

### Keep ChatTS on transformers if:

- plugin registration is unstable
- multimodal time series path is incompatible
- output quality or stability regresses

## 5. Fallback Strategy

- Keep current `transformers` services intact
- Only switch one model family at a time
- Preserve OpenAI-compatible endpoint shape
- Do not remove current health checks or parser logic during validation

## 6. First Experiments

1. Run Qwen3-VL with real `vllm serve`
2. Point `VLLM_QWEN_ENDPOINT` to the new service
3. Run `/api/v1/models/vllm/status`
4. Run `/api/v1/predict`
5. Submit one Celery task
6. Record latency, memory, and output structure

Helper files added on this branch:

- `services/start_qwen_vllm.sh`
- `scripts/validate_qwen_vllm.py`

## 7. Risks

- ChatTS plugin may depend on a specific vLLM version
- OpenAI multimodal payload details may differ from current adapter assumptions
- container network and host network endpoints may differ during mixed deployment
- Qwen3.5-27B still depends on larger VRAM capacity even if vLLM works

## 8. Expected Outputs of This Branch

- validation notes
- minimal config changes for real vLLM testing
- optional alternate service launcher scripts
- no irreversible architecture change before validation passes

## 9. Current Findings

### 2026-03-16: Qwen3-VL with vLLM 0.8.5 on RTX 2080 Ti

Test setup:

- GPU: RTX 2080 Ti, compute capability 7.5
- model: `/home/data1/llm_models/Qwen/Qwen3-VL-8B-Instruct`
- command path: `services/start_qwen_vllm.sh`

Observed result:

1. Default startup failed because vLLM attempted `bfloat16`
2. Forcing `--dtype=half` fixed the bf16 hardware issue
3. Model load still failed in vLLM 0.8.5 with:
   - fallback to Transformers backend
   - `AttributeError: 'Qwen3VLConfig' object has no attribute 'vocab_size'`

Interim conclusion:

- Qwen3-VL validation is not blocked by platform API shape
- it is currently blocked by the vLLM 0.8.5 and local model/config compatibility path
- next useful step is version-matrix validation, not more platform-side refactor

### 2026-03-16: vLLM 0.11.0 isolated upgrade probe

Test setup:

- isolated env: `chatts-vllm011`
- installation strategy: upgrade `vllm` wheel first, avoid touching online `chatts` env

Observed result:

1. `vllm 0.11.0` wheel installs successfully in the isolated env
2. CLI import fails before serving starts
3. failure is:
   - `ImportError: ... vllm/_C.abi3.so: undefined symbol ... ConstantString::create`

Interim conclusion:

- this is a native extension / ABI mismatch
- the immediate blocker is no longer Qwen3-VL model config
- the isolated env still carries an older runtime stack than `vllm 0.11.0` expects
- the next required step is a full matching runtime matrix upgrade:
  - `torch==2.8.0`
  - `torchaudio==2.8.0`
  - `torchvision==0.23.0`
  - `xformers==0.0.32.post1`
  - plus pinned companion packages required by `vllm 0.11.0`

### 2026-03-17: clean env rebuild and Hugging Face stack alignment

Test setup:

- clean env: `qwen-vllm011-clean`
- runtime matrix:
  - `torch==2.8.0`
  - `torchaudio==2.8.0`
  - `torchvision==0.23.0`
  - `xformers==0.0.32.post1`
- `vllm==0.11.0`

Observed result:

1. `vllm` CLI imports successfully in the clean env
2. initial install pulled `transformers==5.3.0`, which caused tokenizer incompatibility:
   - `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
3. pinning the HF stack to:
   - `transformers==4.57.6`
   - `huggingface-hub==0.36.2`
   restores `Qwen2TokenizerFast` behavior and removes that blocker

Interim conclusion:

- the ABI issue is solved by rebuilding the runtime cleanly
- `vllm 0.11.0` on this host still needs a pinned HF stack
- the blocker is now reduced to serving-time memory tuning

### 2026-03-17: Qwen3-VL starts successfully on RTX 2080 Ti with constrained settings

Working serve profile:

- env: `qwen-vllm011-clean`
- dtype: `half`
- `--enforce-eager`
- `--gpu-memory-utilization 0.97`
- `--max-model-len 1024`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 1024`
- `--limit-mm-per-prompt '{"image":1,"video":0}'`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Observed result:

1. default `vllm serve` settings still fail on this 22 GB card
2. with the constrained profile above, service startup succeeds
3. `/health` and `/v1/models` are reachable
4. text-only OpenAI chat works
5. image input also works in the OpenAI-compatible path
6. `scripts/validate_qwen_vllm.py` succeeds against `http://localhost:8003/v1`

Representative validation output:

- endpoint: `http://localhost:8003/v1`
- model: `qwen`
- sample result from `PI_20412.PV.csv`:
  - one anomaly interval around `[450, 460]` to `[450, 470]`, depending on generation

Interim conclusion:

- `Qwen3-VL + vllm 0.11.0` is viable on this machine
- it is not viable with default settings
- the limiting factor is VRAM, not platform compatibility
- current working mode favors single-request validation and light internal usage over throughput

### 2026-03-17: platform integration smoke test passes

Test setup:

- temporary backend instance on `http://localhost:8101`
- env override:
  - `VLLM_QWEN_ENDPOINT=http://localhost:8003/v1`
- existing platform code unchanged

Observed result:

1. `GET /api/v1/models/vllm/status` reports:
   - `qwen` healthy on `http://localhost:8003/v1`
2. `POST /api/v1/predict` with method `qwen` succeeds
3. result files, metrics JSON, and segments JSON are generated
4. no platform-side adapter change was required

Conclusion:

- the current platform integration path is compatible with real `vllm serve`
- rollout work is now mostly:
  - packaging the validated runtime
  - encoding the working serve flags
  - deciding whether this low-concurrency profile is acceptable for production use
