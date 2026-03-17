"""
ChatTS Serving Adapter.

Thin FastAPI service that wraps HuggingFace transformers for ChatTS,
exposing an OpenAI-compatible HTTP endpoint with time series
multimodal data support.

Run in the chatts conda environment:
    conda activate chatts
    PYTHONPATH=/home/dff652/TS-anomaly-detection/ChatTS \
    python services/chatts_serve.py --model-path /home/data1/llm_models/bytedance-research/ChatTS-8B

Architecture:
    Platform Backend (:8100) → HTTP → ChatTS Serve (:8001) → transformers
"""
import argparse
import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chatts-serve")

# Global model state
_model = None
_tokenizer = None
_processor = None
_model_id = "chatts"
_device = None
_lock = threading.Lock()  # serialize generation (single GPU)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "chatts"
    messages: list[ChatMessage] = []
    max_tokens: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.1, ge=0, le=2)
    # Custom field for time series data
    mm_data: dict | None = Field(None, description="Multimodal data: {timeseries: [[values]]}")


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = "chatts-0"
    object: str = "chat.completion"
    model: str = "chatts"
    choices: list[ChatCompletionChoice]
    usage: dict = {}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def create_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _model, _tokenizer, _processor, _model_id, _device
        logger.info(f"Loading ChatTS model from {args.model_path}...")
        t0 = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        _device = f"cuda:{args.gpu_device}"

        _model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map=_device,
            attn_implementation="eager",
        )
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

        _processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            tokenizer=_tokenizer,
        )

        _model_id = args.model_name

        logger.info(
            f"ChatTS model loaded in {time.time() - t0:.1f}s "
            f"(GPU mem: {torch.cuda.memory_allocated() / 1024**3:.1f} GB)"
        )
        yield
        logger.info("Shutting down ChatTS serve")

    app = FastAPI(title="ChatTS Serve", lifespan=lifespan)

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

        # Build prompt from messages (ChatML format)
        prompt_parts = []
        for msg in req.messages:
            if msg.role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        # Prepare timeseries data
        timeseries_list = []
        if req.mm_data and "timeseries" in req.mm_data:
            for ts in req.mm_data["timeseries"]:
                if isinstance(ts, np.ndarray):
                    timeseries_list.append(ts.astype(np.float64))
                else:
                    timeseries_list.append(np.array(ts, dtype=np.float64))

        try:
            with _lock:
                if timeseries_list:
                    inputs = _processor(
                        text=[prompt],
                        timeseries=timeseries_list,
                        padding=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = _tokenizer(prompt, return_tensors="pt")

                inputs = {k: v.to(_device) for k, v in inputs.items()}

                with torch.no_grad():
                    if timeseries_list:
                        # Pre-merge timeseries embeddings to avoid cache_position
                        # mismatch in generate() (sequence length changes after merge)
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs["attention_mask"]
                        timeseries = inputs["timeseries"]

                        inputs_embeds = _model.get_input_embeddings()(input_ids)
                        ts_features, patch_cnt = _model.ts_encoder(timeseries)
                        inputs_embeds = inputs_embeds.to(ts_features.dtype)

                        inputs_embeds, attention_mask, position_ids, _, _ = (
                            _model._merge_input_ids_with_time_series_features(
                                ts_features, inputs_embeds, input_ids,
                                attention_mask, None, patch_cnt,
                            )
                        )

                        prompt_tokens = int(attention_mask.sum().item())

                        outputs = _model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            max_new_tokens=req.max_tokens,
                            temperature=max(req.temperature, 0.01),
                            do_sample=req.temperature > 0,
                        )
                        # With inputs_embeds, generate() returns only new tokens
                        generated_tokens = outputs[0]
                    else:
                        prompt_tokens = int(inputs["attention_mask"].sum().item())
                        outputs = _model.generate(
                            **inputs,
                            max_new_tokens=req.max_tokens,
                            temperature=max(req.temperature, 0.01),
                            do_sample=req.temperature > 0,
                        )
                        generated_tokens = outputs[0][prompt_tokens:]

                    generated_text = _tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    completion_tokens = len(generated_tokens)

            return ChatCompletionResponse(
                model=_model_id,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=generated_text),
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
    parser = argparse.ArgumentParser(description="ChatTS Serving Adapter")
    parser.add_argument("--model-path", required=True, help="Path to ChatTS model")
    parser.add_argument("--model-name", default="chatts", help="Model name in API responses")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--default-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    import uvicorn
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
