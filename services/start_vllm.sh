#!/bin/bash
# Start GPU model services for the inference platform.
# Usage: ./services/start_vllm.sh [chatts|qwen|all]
#
# ChatTS: transformers-based serve on GPU 0 (port 8001)
# Qwen-VL: transformers-based serve on GPU 1 (port 8002)
# Both expose OpenAI-compatible /v1/chat/completions endpoints.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BASE="$HOME/miniconda3"

# ========== Configuration ==========
# ChatTS (transformers backend, GPU 0)
CHATTS_MODEL_PATH="${CHATTS_MODEL_PATH:-/home/data1/llm_models/bytedance-research/ChatTS-8B}"
CHATTS_PORT=8001
CHATTS_GPU=0
CHATTS_ENV="chatts"

# Qwen-VL (transformers backend, GPU 1)
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/data1/llm_models/Qwen/Qwen3-VL-8B-Instruct}"
QWEN_PORT=8002
QWEN_GPU=1
QWEN_ENV="chatts"

# ========== Functions ==========

start_chatts() {
    echo "Starting ChatTS serve on port $CHATTS_PORT (GPU $CHATTS_GPU)..."
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CHATTS_ENV"

    PYTHONPATH=/home/dff652/TS-anomaly-detection/ChatTS \
    CUDA_VISIBLE_DEVICES=$CHATTS_GPU \
    python "$SCRIPT_DIR/chatts_serve.py" \
        --model-path "$CHATTS_MODEL_PATH" \
        --port "$CHATTS_PORT" \
        --gpu-device 0 \
        &

    echo "ChatTS PID: $!"
}

start_qwen() {
    echo "Starting Qwen-VL serve on port $QWEN_PORT (GPU $QWEN_GPU)..."
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$QWEN_ENV"

    CUDA_VISIBLE_DEVICES=$QWEN_GPU \
    python "$SCRIPT_DIR/qwen_serve.py" \
        --model-path "$QWEN_MODEL_PATH" \
        --port "$QWEN_PORT" \
        --gpu-device 0 \
        &

    echo "Qwen-VL PID: $!"
}

# ========== Main ==========
case "${1:-all}" in
    chatts)
        start_chatts
        ;;
    qwen)
        start_qwen
        ;;
    all)
        start_chatts
        start_qwen
        ;;
    *)
        echo "Usage: $0 [chatts|qwen|all]"
        exit 1
        ;;
esac

echo ""
echo "GPU model services starting. Check health:"
echo "  ChatTS: curl http://localhost:$CHATTS_PORT/health"
echo "  Qwen-VL: curl http://localhost:$QWEN_PORT/health"
echo "  Platform status: curl http://localhost:8100/api/v1/models/vllm/status"
