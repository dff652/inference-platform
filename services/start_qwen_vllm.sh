#!/bin/bash
# Start a real vLLM serve process for Qwen3-VL validation.
#
# Usage:
#   ./services/start_qwen_vllm.sh
#
# Optional env vars:
#   QWEN_VLLM_MODEL_PATH
#   QWEN_VLLM_PORT
#   QWEN_VLLM_HOST
#   QWEN_VLLM_GPU
#   QWEN_VLLM_ENV
#   QWEN_VLLM_SERVED_NAME
#   QWEN_VLLM_MAX_MODEL_LEN
#   QWEN_VLLM_GPU_MEMORY_UTILIZATION
#   QWEN_VLLM_MAX_NUM_SEQS
#   QWEN_VLLM_MAX_NUM_BATCHED_TOKENS
#   QWEN_VLLM_LIMIT_MM_PER_PROMPT
#   QWEN_VLLM_ENFORCE_EAGER
#   PYTORCH_CUDA_ALLOC_CONF

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

QWEN_VLLM_MODEL_PATH="${QWEN_VLLM_MODEL_PATH:-/home/data1/llm_models/Qwen/Qwen3-VL-8B-Instruct}"
QWEN_VLLM_PORT="${QWEN_VLLM_PORT:-8002}"
QWEN_VLLM_HOST="${QWEN_VLLM_HOST:-0.0.0.0}"
QWEN_VLLM_GPU="${QWEN_VLLM_GPU:-1}"
QWEN_VLLM_ENV="${QWEN_VLLM_ENV:-chatts}"
QWEN_VLLM_SERVED_NAME="${QWEN_VLLM_SERVED_NAME:-qwen}"
QWEN_VLLM_MAX_MODEL_LEN="${QWEN_VLLM_MAX_MODEL_LEN:-1024}"
QWEN_VLLM_GPU_MEMORY_UTILIZATION="${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.97}"
QWEN_VLLM_MAX_NUM_SEQS="${QWEN_VLLM_MAX_NUM_SEQS:-1}"
QWEN_VLLM_MAX_NUM_BATCHED_TOKENS="${QWEN_VLLM_MAX_NUM_BATCHED_TOKENS:-1024}"
QWEN_VLLM_LIMIT_MM_PER_PROMPT="${QWEN_VLLM_LIMIT_MM_PER_PROMPT:-{\"image\":1,\"video\":0}}"
QWEN_VLLM_ENFORCE_EAGER="${QWEN_VLLM_ENFORCE_EAGER:-1}"
QWEN_VLLM_DTYPE="${QWEN_VLLM_DTYPE:-half}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Starting Qwen3-VL with vLLM"
echo "  model path: $QWEN_VLLM_MODEL_PATH"
echo "  host:       $QWEN_VLLM_HOST"
echo "  port:       $QWEN_VLLM_PORT"
echo "  gpu:        $QWEN_VLLM_GPU"
echo "  env:        $QWEN_VLLM_ENV"
echo "  served as:  $QWEN_VLLM_SERVED_NAME"
echo "  dtype:      $QWEN_VLLM_DTYPE"
echo "  max len:    $QWEN_VLLM_MAX_MODEL_LEN"
echo "  max seqs:   $QWEN_VLLM_MAX_NUM_SEQS"
echo "  max batch:  $QWEN_VLLM_MAX_NUM_BATCHED_TOKENS"
echo "  mem util:   $QWEN_VLLM_GPU_MEMORY_UTILIZATION"
echo "  eager:      $QWEN_VLLM_ENFORCE_EAGER"
echo "  mm limit:   $QWEN_VLLM_LIMIT_MM_PER_PROMPT"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$QWEN_VLLM_ENV"

cmd=(
  vllm serve "$QWEN_VLLM_MODEL_PATH"
  --host "$QWEN_VLLM_HOST"
  --port "$QWEN_VLLM_PORT"
  --served-model-name "$QWEN_VLLM_SERVED_NAME"
  --dtype "$QWEN_VLLM_DTYPE"
  --gpu-memory-utilization "$QWEN_VLLM_GPU_MEMORY_UTILIZATION"
  --max-model-len "$QWEN_VLLM_MAX_MODEL_LEN"
  --max-num-seqs "$QWEN_VLLM_MAX_NUM_SEQS"
  --max-num-batched-tokens "$QWEN_VLLM_MAX_NUM_BATCHED_TOKENS"
  --limit-mm-per-prompt "$QWEN_VLLM_LIMIT_MM_PER_PROMPT"
)

if [ "$QWEN_VLLM_ENFORCE_EAGER" = "1" ]; then
  cmd+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES="$QWEN_VLLM_GPU" \
PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
"${cmd[@]}"
