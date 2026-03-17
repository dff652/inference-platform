"""
vLLM Backend Adapter — calls vLLM OpenAI-compatible API for GPU model inference.

Supports:
- Qwen-VL: time series → matplotlib image → base64 → vLLM vision API
- ChatTS: time series array → multi_modal_data → vLLM multimodal API

Both vLLM servers expose OpenAI-compatible chat/completions endpoints.
"""
import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field

import httpx
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# vLLM endpoint mapping
VLLM_ENDPOINTS = {
    "chatts": settings.VLLM_CHATTS_ENDPOINT,
    "qwen": settings.VLLM_QWEN_ENDPOINT,
}

# Default prompt templates
QWEN_PROMPT = (
    "分析图中的时间序列数据，基于信号特征识别异常区域。\n"
    '输出必须是标准JSON格式：{"detected_anomalies":[{"interval":[start,end],"type":"类型","reason":"原因"}]}；'
    '若无异常：{"detected_anomalies":[]}。\n'
    "异常区域必须以连续索引区间 [start, end] 表示，且满足 end - start + 1 > 5。\n"
    "请精确标注异常区间的起止索引。"
)

CHATTS_PROMPT_DEFAULT = (
    "我有一个长度为 {ts_len} 的时间序列：<ts><ts/>。请识别该时间序列中所有异常或异常片段。\n\n"
    "从全局视角找出具有统计显著性的异常（如极值、突变），忽略正常周期波动。\n\n"
    "【输出要求】\n"
    "仅输出一个名为 anomalies 的 JSON 数组，不要输出任何其他文字或代码块标记。\n"
    "每个元素必须严格包含以下4个字段：\n"
    "- range: [起始索引, 结束索引]，整数\n"
    "- amp: 异常幅度，保留两位小数\n"
    '- label: 简短标签（不超过10字），如"向下尖峰"、"突增"、"趋势漂移"\n'
    "- detail: 简短描述（不超过30字）\n\n"
    "【格式示例】\n"
    "anomalies = [\n"
    '    {"range": [137, 139], "amp": 1.91, "label": "向下尖峰", "detail": "数值从1.91骤降至0后恢复"},\n'
    '    {"range": [500, 520], "amp": 5.23, "label": "趋势漂移", "detail": "整体水平逐渐上升约5.23"}\n'
    "]\n\n"
    "注意：必须使用双引号，每个对象的字段顺序为 range、amp、label、detail。"
)


@dataclass
class VLLMResult:
    """Result from a vLLM inference call."""
    success: bool
    raw_text: str = ""
    anomalies: list = field(default_factory=list)
    error: str | None = None


# ---------- Image rendering for Qwen-VL ----------

def render_timeseries_image(values: np.ndarray, dpi: int = 100) -> str:
    """Render time series as PNG and return base64-encoded string."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 4), dpi=dpi)
    ax.plot(values, color="black", linewidth=1)
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------- Output parsing ----------

def parse_qwen_output(text: str) -> list[dict]:
    """Parse Qwen-VL output: {"detected_anomalies": [...]}."""
    text = text.strip()
    # Strip markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "detected_anomalies" in data:
            return data["detected_anomalies"]
    except json.JSONDecodeError:
        pass

    # Fix trailing commas then retry
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        data = json.loads(fixed)
        if isinstance(data, dict) and "detected_anomalies" in data:
            return data["detected_anomalies"]
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract [start, end] intervals
    anomalies = []
    for pattern in [r"\[(\d+)\s*,\s*(\d+)\]", r"\((\d+)\s*,\s*(\d+)\)"]:
        for start, end in re.findall(pattern, text):
            anomalies.append({
                "interval": [int(start), int(end)],
                "type": "unknown",
                "reason": "regex_extracted",
            })
    return anomalies


def parse_chatts_output(text: str) -> list[dict]:
    """Parse ChatTS output: anomalies = [{"range": [...], ...}]."""
    m = re.search(r"anomalies\s*=\s*\[", text)
    if not m:
        # Fallback: try finding a JSON array directly
        try:
            arr = json.loads(text.strip())
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
        return []

    start = m.end() - 1  # points to '['

    # Scan for matching ']'
    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    raw = text[start : end + 1] if end is not None else text[start:]

    # Handle truncated output
    if end is None:
        last_brace = raw.rfind("}")
        if last_brace == -1:
            return []
        raw = raw[: last_brace + 1] + "]"

    try:
        anomalies = json.loads(raw)
    except json.JSONDecodeError:
        raw_fixed = re.sub(r",\s*]", "]", raw)
        try:
            anomalies = json.loads(raw_fixed)
        except json.JSONDecodeError:
            return []

    return anomalies if isinstance(anomalies, list) else []


# ---------- vLLM API calls ----------

async def call_qwen_vllm(
    values: np.ndarray,
    prompt: str | None = None,
    model_name: str = "qwen",
    max_tokens: int | None = None,
) -> VLLMResult:
    """
    Call Qwen-VL via vLLM OpenAI API with a rendered time series image.

    Args:
        values: 1D numpy array of time series values
        prompt: override prompt text (defaults to QWEN_PROMPT)
        model_name: model name for vLLM (sent in request body)
        max_tokens: max output tokens
    """
    endpoint = VLLM_ENDPOINTS.get("qwen", settings.VLLM_QWEN_ENDPOINT)
    prompt = prompt or QWEN_PROMPT
    max_tokens = max_tokens or settings.VLLM_MAX_TOKENS

    # Render image
    try:
        image_b64 = render_timeseries_image(values)
    except Exception as e:
        return VLLMResult(success=False, error=f"Image rendering failed: {e}")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    return await _call_vllm_api(endpoint, payload, parse_qwen_output)


async def call_chatts_vllm(
    values: np.ndarray,
    prompt_template: str | None = None,
    model_name: str = "chatts",
    max_tokens: int | None = None,
) -> VLLMResult:
    """
    Call ChatTS via vLLM OpenAI API with time series multimodal data.

    Args:
        values: 1D numpy array of time series values
        prompt_template: prompt template with {ts_len} placeholder
        model_name: model name for vLLM
        max_tokens: max output tokens
    """
    endpoint = VLLM_ENDPOINTS.get("chatts", settings.VLLM_CHATTS_ENDPOINT)
    template = prompt_template or CHATTS_PROMPT_DEFAULT
    max_tokens = max_tokens or settings.VLLM_MAX_TOKENS

    ts_len = len(values)
    user_prompt = template.replace("{ts_len}", str(ts_len))

    # ChatTS uses custom multimodal data via extra fields
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        # vLLM custom multimodal data for ChatTS plugin
        "mm_data": {"timeseries": [values.tolist()]},
    }

    return await _call_vllm_api(endpoint, payload, parse_chatts_output)


async def _call_vllm_api(endpoint: str, payload: dict, parser) -> VLLMResult:
    """Send request to vLLM OpenAI API and parse response."""
    url = f"{endpoint}/chat/completions"
    timeout = settings.VLLM_REQUEST_TIMEOUT

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            return VLLMResult(
                success=False,
                error=f"vLLM API returned {resp.status_code}: {resp.text[:500]}",
            )

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        anomalies = parser(raw_text)

        return VLLMResult(success=True, raw_text=raw_text, anomalies=anomalies)

    except httpx.ConnectError:
        return VLLMResult(
            success=False,
            error=f"Cannot connect to vLLM server at {endpoint}. Is the server running?",
        )
    except httpx.TimeoutException:
        return VLLMResult(
            success=False,
            error=f"vLLM request timed out after {timeout}s",
        )
    except Exception as e:
        return VLLMResult(success=False, error=f"vLLM API call failed: {e}")


async def check_vllm_health(method: str) -> dict:
    """Check if a vLLM server is reachable and list available models."""
    endpoint = VLLM_ENDPOINTS.get(method)
    if not endpoint:
        return {"healthy": False, "error": f"No endpoint configured for '{method}'"}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{endpoint}/models")
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            return {
                "healthy": True,
                "endpoint": endpoint,
                "models": [m.get("id") for m in models],
            }
        return {"healthy": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"healthy": False, "endpoint": endpoint, "error": str(e)}
