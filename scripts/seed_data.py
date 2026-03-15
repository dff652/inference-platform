#!/usr/bin/env python3
"""Seed data script: register models and config templates via API."""

import sys
import httpx

BASE_URL = "http://localhost:8100/api/v1"

MODELS = [
    {
        "name": "ChatTS-8B",
        "family": "chatts",
        "runtime_type": "transformers",
        "version": "v1.0",
        "artifact_uri": "/home/data1/llm_models/bytedance-research/ChatTS-8B",
        "base_model": "Qwen2-7B",
        "compatibility": {
            "python": "3.8+",
            "gpu": "required",
            "min_vram_gb": 16,
            "cuda_device": "cuda:1",
        },
        "tags": ["gpu", "llm", "time-series", "anomaly-detection"],
        "description": "ByteDance ChatTS-8B model for time-series anomaly detection via natural language interaction. Based on Qwen2-7B with LoRA fine-tuning.",
    },
    {
        "name": "ChatTS-8B-LoRA",
        "family": "chatts",
        "runtime_type": "transformers+lora",
        "version": "v1.0",
        "artifact_uri": "/home/data1/llm_models/bytedance-research/ChatTS-8B",
        "base_model": "Qwen2-7B",
        "compatibility": {
            "python": "3.8+",
            "gpu": "required",
            "min_vram_gb": 12,
            "cuda_device": "cuda:1",
            "lora_adapter": "/home/douff/ts/ChatTS-Training/saves/chatts-8b",
        },
        "tags": ["gpu", "llm", "time-series", "lora"],
        "description": "ChatTS-8B with custom LoRA adapter for domain-specific anomaly detection.",
    },
    {
        "name": "Qwen3-VL-8B-Instruct",
        "family": "qwen",
        "runtime_type": "transformers",
        "version": "v3.0",
        "artifact_uri": "/home/data1/llm_models/Qwen/Qwen3-VL-8B-Instruct",
        "base_model": "Qwen3-VL-8B",
        "compatibility": {
            "python": "3.12+",
            "gpu": "required",
            "min_vram_gb": 18,
            "cuda_device": "cuda:1",
        },
        "tags": ["gpu", "multimodal", "vision-language", "anomaly-detection"],
        "description": "Qwen3-VL-8B 视觉语言模型，通过时序图表图像分析进行异常检测。bf16 单卡 18GB。",
    },
    {
        "name": "Qwen3.5-27B",
        "family": "qwen",
        "runtime_type": "transformers",
        "version": "v3.5",
        "artifact_uri": "/home/data1/llm_models/Qwen/Qwen3___5-27B",
        "base_model": "Qwen3.5-27B",
        "compatibility": {
            "python": "3.12+",
            "gpu": "required",
            "min_vram_gb": 48,
            "note": "27B 模型需要 >=48GB 显存或多卡，4-bit 量化约 15GB",
        },
        "tags": ["gpu", "multimodal", "vision-language", "anomaly-detection", "large"],
        "description": "Qwen3.5-27B 多模态模型，支持图文理解。需要大显存设备（48GB+）或 4-bit 量化多卡部署。",
    },
    {
        "name": "ADTK-HBOS",
        "family": "adtk_hbos",
        "runtime_type": "sklearn",
        "version": "v1.0",
        "artifact_uri": None,
        "compatibility": {
            "python": "3.8+",
            "gpu": "not_required",
        },
        "tags": ["cpu", "statistical", "fast"],
        "description": "ADTK + Histogram-Based Outlier Score (HBOS) statistical anomaly detection. Lightweight, no GPU required.",
    },
    {
        "name": "Ensemble",
        "family": "ensemble",
        "runtime_type": "sklearn",
        "version": "v1.0",
        "artifact_uri": None,
        "compatibility": {
            "python": "3.8+",
            "gpu": "not_required",
        },
        "tags": ["cpu", "ensemble", "voting"],
        "description": "Multi-method voting ensemble combining statistical and ML-based anomaly detectors.",
    },
    {
        "name": "Wavelet",
        "family": "wavelet",
        "runtime_type": "scipy",
        "version": "v1.0",
        "artifact_uri": None,
        "compatibility": {
            "python": "3.8+",
            "gpu": "not_required",
        },
        "tags": ["cpu", "signal-processing"],
        "description": "Wavelet decomposition-based anomaly detection for frequency-domain analysis.",
    },
    {
        "name": "Isolation Forest",
        "family": "isolation_forest",
        "runtime_type": "sklearn",
        "version": "v1.0",
        "artifact_uri": None,
        "compatibility": {
            "python": "3.8+",
            "gpu": "not_required",
        },
        "tags": ["cpu", "tree-based", "unsupervised"],
        "description": "Isolation Forest tree-based unsupervised anomaly detection.",
    },
]

CONFIG_TEMPLATES = [
    {
        "name": "ChatTS Default",
        "algorithm_id": 1,
        "algorithm_name": "chatts",
        "default_params": {
            "method": "chatts",
            "n_downsample": 5000,
            "chatts_device": "cuda:1",
            "chatts_max_new_tokens": 4096,
            "chatts_prompt_template": "default",
            "chatts_load_in_4bit": "auto",
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/ts/bin/python",
            "cuda_visible_devices": "1",
        },
        "resource_profile": "gpu",
    },
    {
        "name": "ChatTS 4-bit Quantized",
        "algorithm_id": 1,
        "algorithm_name": "chatts",
        "default_params": {
            "method": "chatts",
            "n_downsample": 5000,
            "chatts_device": "cuda:1",
            "chatts_load_in_4bit": True,
            "chatts_max_new_tokens": 4096,
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/ts/bin/python",
            "cuda_visible_devices": "1",
        },
        "resource_profile": "gpu",
    },
    {
        "name": "Qwen3-VL-8B Default",
        "algorithm_id": 2,
        "algorithm_name": "qwen",
        "default_params": {
            "method": "qwen",
            "n_downsample": 5000,
            "chatts_device": "cuda:1",
            "chatts_load_in_4bit": "false",
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/chatts/bin/python",
            "cuda_visible_devices": "1",
        },
        "resource_profile": "gpu",
    },
    {
        "name": "ADTK-HBOS Default",
        "algorithm_id": 3,
        "algorithm_name": "adtk_hbos",
        "default_params": {
            "method": "adtk_hbos",
            "n_downsample": 5000,
            "bin_nums": 20,
            "use_clustering": True,
            "use_trend": True,
            "use_resid": True,
            "decompose": True,
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/ts/bin/python",
        },
        "resource_profile": "cpu",
    },
    {
        "name": "Ensemble Default",
        "algorithm_id": 4,
        "algorithm_name": "ensemble",
        "default_params": {
            "method": "ensemble",
            "n_downsample": 5000,
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/ts/bin/python",
        },
        "resource_profile": "cpu",
    },
    {
        "name": "Wavelet Default",
        "algorithm_id": 5,
        "algorithm_name": "wavelet",
        "default_params": {
            "method": "wavelet",
            "n_downsample": 5000,
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/ts/bin/python",
        },
        "resource_profile": "cpu",
    },
    {
        "name": "Isolation Forest Default",
        "algorithm_id": 6,
        "algorithm_name": "isolation_forest",
        "default_params": {
            "method": "isolation_forest",
            "n_downsample": 5000,
            "ratio": 0.1,
        },
        "env_profile": {
            "python_path": "/home/dff652/miniconda3/envs/ts/bin/python",
        },
        "resource_profile": "cpu",
    },
]


def seed():
    client = httpx.Client(base_url=BASE_URL, timeout=10)

    # Check API is up
    try:
        r = client.get("/../../health")
        r.raise_for_status()
    except Exception:
        print("ERROR: Backend not reachable at", BASE_URL)
        sys.exit(1)

    print("=== Seeding Models ===")
    for m in MODELS:
        try:
            r = client.post("/models", json=m)
            if r.status_code == 201:
                data = r.json()
                print(f"  [OK] {m['name']} (id={data['id']})")
            else:
                print(f"  [SKIP] {m['name']}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            print(f"  [ERR] {m['name']}: {e}")

    print("\n=== Seeding Config Templates ===")
    for c in CONFIG_TEMPLATES:
        try:
            r = client.post("/inference/configs", json=c)
            if r.status_code == 201:
                data = r.json()
                print(f"  [OK] {c['name']} (id={data['id']})")
            else:
                print(f"  [SKIP] {c['name']}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            print(f"  [ERR] {c['name']}: {e}")

    # Verify
    print("\n=== Verification ===")
    models = client.get("/models").json()
    configs = client.get("/inference/configs").json()
    print(f"  Models: {models.get('total', len(models))} registered")
    print(f"  Config Templates: {configs.get('total', len(configs))} registered")


if __name__ == "__main__":
    seed()
