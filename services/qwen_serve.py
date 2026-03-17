"""
Qwen3-VL Serving Adapter.

Thin FastAPI service that wraps HuggingFace transformers for Qwen3-VL,
exposing an OpenAI-compatible HTTP endpoint with image input support.

For time series anomaly detection, the platform renders time series data
as matplotlib PNG images and sends them via the standard image_url field.

Run in the chatts conda environment:
    conda activate chatts
    CUDA_VISIBLE_DEVICES=1 python services/qwen_serve.py \
        --model-path /home/data1/llm_models/Qwen/Qwen3-VL-8B-Instruct

Architecture:
    Platform Backend (:8100) → HTTP → Qwen Serve (:8002) → transformers
"""
import argparse
import base64
import logging
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen-serve")

# Global model state
_model = None
_processor = None
_model_id = "qwen"
_device = None
_lock = threading.Lock()


# --- Pydantic models (OpenAI-compatible) ---

class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = "qwen"
    messages: list[ChatMessage] = []
    max_tokens: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.1, ge=0, le=2)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = "qwen-0"
    object: str = "chat.completion"
    model: str = "qwen"
    choices: list[ChatCompletionChoice]
    usage: dict = {}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def _extract_images_and_text(messages: list[ChatMessage]):
    """Convert OpenAI-style messages to Qwen3-VL processor format."""
    processed_messages = []
    images = []

    for msg in messages:
        if isinstance(msg.content, str):
            processed_messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            # Multimodal content with text and images
            parts = []
            for part in msg.content:
                if part.type == "text":
                    parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url" and part.image_url:
                    url = part.image_url.get("url", "")
                    if url.startswith("data:image"):
                        # Extract base64 data
                        header, b64data = url.split(",", 1)
                        img_bytes = base64.b64decode(b64data)
                        img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        images.append(img)
                        parts.append({"type": "image", "image": img})
                    else:
                        parts.append({"type": "image", "image": url})
            processed_messages.append({"role": msg.role, "content": parts})

    return processed_messages, images


def create_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _model, _processor, _model_id, _device
        logger.info(f"Loading Qwen3-VL model from {args.model_path}...")
        t0 = time.time()

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        _device = f"cuda:{args.gpu_device}"

        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.float16,
            device_map=_device,
            attn_implementation="eager",
        )
        _model.eval()

        _processor = AutoProcessor.from_pretrained(args.model_path)

        _model_id = args.model_name

        logger.info(
            f"Qwen3-VL model loaded in {time.time() - t0:.1f}s "
            f"(GPU mem: {torch.cuda.memory_allocated() / 1024**3:.1f} GB)"
        )
        yield
        logger.info("Shutting down Qwen serve")

    app = FastAPI(title="Qwen3-VL Serve", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok" if _model is not None else "loading"}

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(data=[ModelInfo(id=_model_id)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        processed_messages, images = _extract_images_and_text(req.messages)

        try:
            with _lock:
                # Apply chat template
                text_prompt = _processor.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Process inputs
                if images:
                    inputs = _processor(
                        text=[text_prompt],
                        images=images,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = _processor(
                        text=[text_prompt],
                        padding=True,
                        return_tensors="pt",
                    )

                inputs = {k: v.to(_device) for k, v in inputs.items()}
                prompt_tokens = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = _model.generate(
                        **inputs,
                        max_new_tokens=req.max_tokens,
                        temperature=max(req.temperature, 0.01),
                        do_sample=req.temperature > 0,
                    )

                generated_tokens = outputs[0][prompt_tokens:]
                generated_text = _processor.decode(
                    generated_tokens, skip_special_tokens=True
                )
                completion_tokens = len(generated_tokens)

            return ChatCompletionResponse(
                model=_model_id,
                choices=[
                    ChatCompletionChoice(
                        message={"role": "assistant", "content": generated_text},
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Serving Adapter")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3-VL model")
    parser.add_argument("--model-name", default="qwen", help="Model name in API responses")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--default-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    import uvicorn
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
