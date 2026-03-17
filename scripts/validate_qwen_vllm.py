"""
Minimal smoke test for a Qwen3-VL vLLM endpoint.

This script:
1. reads a CSV with a Time column and one value column
2. renders the time series to a PNG in memory
3. calls the OpenAI-compatible /v1/chat/completions API
4. prints the raw output and parsed anomalies

Example:
  python scripts/validate_qwen_vllm.py \
    --csv /home/dff652/dff_project/inference-platform/data/PI_20412.PV.csv \
    --endpoint http://localhost:8002/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd

from app.adapters.vllm_backend import QWEN_PROMPT, parse_qwen_output, render_timeseries_image


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen3-VL vLLM endpoint")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--endpoint", default="http://localhost:8002/v1", help="vLLM base endpoint")
    parser.add_argument("--model", default="qwen", help="served model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="max output tokens")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Time"])
    value_cols = [c for c in df.columns if c != "Time"]
    if not value_cols:
        raise SystemExit("CSV must contain a Time column and one value column")

    values = df[value_cols[0]].astype(float).to_numpy()
    image_b64 = render_timeseries_image(values)

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QWEN_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        models_resp = await client.get(f"{args.endpoint}/models")
        print("== /v1/models ==")
        print(models_resp.status_code)
        print(models_resp.text[:1000])
        print()

        resp = await client.post(f"{args.endpoint}/chat/completions", json=payload)
        print("== /v1/chat/completions ==")
        print(resp.status_code)

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        print(content[:4000])
        print()

        anomalies = parse_qwen_output(content)
        print("== parsed anomalies ==")
        print(json.dumps(anomalies, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
