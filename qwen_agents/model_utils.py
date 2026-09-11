"""
model_utils.py
─────────────────────────────────────────────────────────────────
Shared helper that wraps a local Qwen3-VL checkpoint (loaded via
unsloth's FastVisionModel, full precision) as a pydantic-ai `Model`
so both agents can plug it into `Agent(model=...)`.

This has nothing to do with MCP or A2A — it is pure local inference.
Both forensic_agent.py and profiler_agent.py import `make_model`
from here instead of duplicating the huggingface_hub/unsloth
loading code.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from unsloth import FastVisionModel

from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _download(repo_id: str) -> str:
    """
    Explicitly pull a model repo down via huggingface_hub before loading
    it, rather than relying on `from_pretrained`'s implicit download.
    Returns the local snapshot directory. Safe to call repeatedly —
    huggingface_hub no-ops (just verifies the cache) if already present.
    """
    return snapshot_download(repo_id=repo_id, token=os.getenv("HF_TOKEN"))


def _load_with_unsloth(local_path: str, processor_path: Optional[str]):
    """
    Load via unsloth's FastVisionModel — full precision (no 4-bit
    quantization), which is faster/lighter to run inference through
    than plain transformers thanks to unsloth's fused kernels.
    """
    model, processor = FastVisionModel.from_pretrained(
        local_path,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)  # enable unsloth's fast inference path

    if processor_path:
        processor = AutoProcessor.from_pretrained(processor_path)

    return model, processor


@lru_cache(maxsize=4)
def _load(model_id: str, adapter_path: Optional[str], processor_path: Optional[str]):
    """Load (and cache) the model + processor for a given model_id, via unsloth."""
    local_path = _download(model_id)
    model, processor = _load_with_unsloth(local_path, processor_path)

    if adapter_path:
        model.load_adapter(adapter_path)

    return model, processor


def _extract_chat_messages(messages: list[ModelMessage]) -> list[dict]:
    """Turn pydantic-ai's message history into Qwen chat-template messages."""
    chat: list[dict] = []

    for msg in messages:
        if not isinstance(msg, ModelRequest):
            # Skip prior ModelResponse turns for this simple single-shot use case.
            continue

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                chat.append({"role": "system", "content": part.content})

            elif isinstance(part, UserPromptPart):
                content_items: list[dict] = []
                raw = part.content
                pieces = raw if isinstance(raw, list) else [raw]
                for piece in pieces:
                    if isinstance(piece, str):
                        content_items.append({"type": "text", "text": piece})
                    elif isinstance(piece, ImageUrl):
                        content_items.append({"type": "image", "image": piece.url})
                    elif isinstance(piece, BinaryContent) and piece.is_image:
                        content_items.append(
                            {"type": "image", "image": piece.data}
                        )
                chat.append({"role": "user", "content": content_items})

    return chat


def make_model(
    model_id: str,
    *,
    adapter_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
) -> FunctionModel:
    """
    Build a pydantic-ai `Model` backed by a local Qwen3-VL checkpoint,
    loaded via unsloth's FastVisionModel (full precision, no 4-bit
    quantization).

    Returns a `FunctionModel` (a model driven by a plain Python callable)
    so `Agent(model=make_model(...))` works exactly like it would with a
    hosted-API model, but everything runs on-device with no network call
    and no MCP/tool involvement.
    """

    def _run(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        model, processor = _load(model_id, adapter_path, processor_path)
        chat = _extract_chat_messages(messages)

        # If the agent declared an output schema, ask the model to reply
        # with JSON matching it — the model's profile declares
        # `supports_json_object_output=True`, so pydantic-ai will validate
        # the returned text against the output type directly.
        if agent_info.output_tools:
            schema_hint = json.dumps(
                [t.parameters_json_schema for t in agent_info.output_tools]
            )
            chat.append(
                {
                    "role": "system",
                    "content": (
                        "Respond with ONLY a single JSON object matching this "
                        f"schema, no prose, no markdown fences: {schema_hint}"
                    ),
                }
            )

        inputs = processor.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return ModelResponse(parts=[TextPart(content=output_text)])

    return FunctionModel(_run, model_name=model_id)
