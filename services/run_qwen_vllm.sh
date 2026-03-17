#!/bin/bash
# Wrapper script for systemd to start Qwen3-VL via vLLM.
# Systemd has trouble with JSON arguments, so we use this wrapper.

exec /home/dff652/miniconda3/envs/qwen-vllm011-clean/bin/vllm serve \
    /home/data1/llm_models/Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8002 \
    --served-model-name qwen \
    --dtype half \
    --gpu-memory-utilization 0.97 \
    --max-model-len 1024 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --enforce-eager
