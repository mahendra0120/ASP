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

Tool calling: since this is a raw local-generation backend (not a
hosted API with native function-calling), tool calls are implemented
via a simple text protocol — the model is told to emit
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` when it
wants to invoke a tool (e.g. the Profiler agent's MCP filesystem
`read_file`). We detect that block in the generated text, parse it,
and return a real pydantic-ai `ToolCallPart` so the framework
actually executes the tool and loops generation with the result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from unsloth import FastVisionModel

from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("model_utils")

hf_home = os.getenv("HF_HOME")
if not hf_home:
    raise RuntimeError("HF_HOME is not set")

log.info(f"Using HF_HOME: {hf_home}")

# Regex for detecting a tool-call block in the model's raw generated text.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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
    """
    Turn pydantic-ai's message history into Qwen chat-template messages.

    Handles four things:
      - System prompts (as list-of-content-parts, required by the
        multimodal processor's apply_chat_template).
      - User prompts — text + image pieces, merged into ONE user turn
        per ModelRequest (fasta2a's bridge splits text/file into
        separate UserPromptParts, but the chat template needs them
        combined so the model actually associates the prompt with the
        image).
      - Prior assistant turns (ModelResponse) — plain text answers, or
        a tool call the assistant previously made.
      - Tool results (ToolReturnPart, inside a later ModelRequest) —
        folded into a user-role text message describing what the tool
        returned, so the model can see it and continue reasoning.
    """
    chat: list[dict] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            user_content_items: list[dict] = []

            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    chat.append(
                        {"role": "system", "content": [{"type": "text", "text": part.content}]}
                    )

                elif isinstance(part, UserPromptPart):
                    raw = part.content
                    pieces = raw if isinstance(raw, list) else [raw]
                    for piece in pieces:
                        if isinstance(piece, str):
                            user_content_items.append({"type": "text", "text": piece})
                        elif isinstance(piece, ImageUrl):
                            user_content_items.append({"type": "image", "image": piece.url})
                        elif isinstance(piece, BinaryContent) and piece.is_image:
                            user_content_items.append({"type": "image", "image": piece.data})
                        elif isinstance(piece, DocumentUrl):
                            # fasta2a's bridge misclassifies our image file parts as
                            # generic DocumentUrl (media_type detection gap on its
                            # side) — every file we send through ImagePart is
                            # actually an image, so treat DocumentUrl as one here.
                            user_content_items.append({"type": "image", "image": piece.url})

                elif isinstance(part, ToolReturnPart):
                    tool_text = (
                        f"Tool '{part.tool_name}' returned:\n"
                        f"{part.model_response_str()}"
                    )
                    user_content_items.append({"type": "text", "text": tool_text})

            if user_content_items:
                chat.append({"role": "user", "content": user_content_items})

        elif isinstance(msg, ModelResponse):
            # A prior turn from this same model — either it answered with
            # plain text, or it made a tool call. Represent both as an
            # assistant turn so the model has continuity across the
            # tool-call -> tool-result -> follow-up-generation loop.
            text_parts = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    text_parts.append(part.content)
                elif isinstance(part, ToolCallPart):
                    call_repr = json.dumps({"name": part.tool_name, "arguments": part.args})
                    text_parts.append(f"<tool_call>{call_repr}</tool_call>")
            if text_parts:
                chat.append(
                    {"role": "assistant", "content": [{"type": "text", "text": "\n".join(text_parts)}]}
                )

    return chat


def _build_tool_instructions(agent_info: AgentInfo) -> Optional[str]:
    """
    Describe the agent's available MCP/function tools to the model in
    plain text, along with the exact text protocol it must use to call
    one. Returns None if the agent has no tools at all.
    """
    tools = list(agent_info.function_tools or [])
    if not tools:
        return None

    tool_lines = []
    for t in tools:
        schema = json.dumps(t.parameters_json_schema)
        tool_lines.append(f"- {t.name}: {t.description}\n  Arguments schema: {schema}")

    return (
        "You have access to the following tools:\n\n"
        + "\n".join(tool_lines)
        + "\n\nTo call a tool, respond with ONLY this and nothing else "
        "(no prose, no explanation, exact format):\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {<matching the schema above>}}</tool_call>\n\n'
        "After a tool result is shown to you, continue and give your real "
        "answer as plain text (no tool_call tag) once you have what you need."
    )


def _build_chat_and_inputs(
    model,
    processor,
    model_id: str,
    messages: list[ModelMessage],
    agent_info: AgentInfo,
    reference_images: Optional[list],
):
    """
    Shared preamble for both the blocking (`_run`) and streaming
    (`_stream_run`) generation paths: build the Qwen chat-template
    turns (reference images, tool instructions, output-schema hint)
    and tokenize them. Returns (inputs, tokenizer).
    """
    chat = _extract_chat_messages(messages)

    is_first_turn = not any(isinstance(m, ModelResponse) for m in messages)
    if reference_images and is_first_turn:
        ref_content = [
            {"type": "text", "text": "Reference material (read this before analyzing the case):"}
        ]
        for img_path in reference_images:
            ref_content.append({"type": "image", "image": str(img_path)})
        chat.insert(0, {"role": "user", "content": ref_content})
        log.info(f"[{model_id}] Attached {len(reference_images)} reference image(s)")

    has_image = any(
        isinstance(m.get("content"), list)
        and any(c.get("type") == "image" for c in m["content"])
        for m in chat
        if m["role"] == "user"
    )
    log.info(f"[{model_id}] Built chat with {len(chat)} turns (image attached: {has_image})")

    tool_instructions = _build_tool_instructions(agent_info)
    if tool_instructions:
        chat.append(
            {"role": "system", "content": [{"type": "text", "text": tool_instructions}]}
        )
        log.info(
            f"[{model_id}] {len(agent_info.function_tools)} tool(s) available: "
            f"{[t.name for t in agent_info.function_tools]}"
        )

    if agent_info.output_tools:
        schema_hint = json.dumps([t.parameters_json_schema for t in agent_info.output_tools])
        chat.append(
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "Respond with ONLY a single JSON object matching this "
                        f"schema, no prose, no markdown fences: {schema_hint}"
                    ),
                }],
            }
        )

    inputs = processor.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    tokenizer = getattr(processor, "tokenizer", processor)
    return inputs, tokenizer


def _strip_thinking(raw_text: str) -> str:
    """
    Remove a <think>...</think> reasoning block, returning only what
    comes after it. Falls back to stripping from an unclosed <think>
    onward (rather than leaking raw chain-of-thought) if the model
    ran out of budget before emitting the closing tag.
    """
    if "</think>" in raw_text:
        return raw_text.split("</think>", 1)[1]
    if "<think>" in raw_text:
        # Unclosed — the model was still "thinking" when generation
        # stopped. Better to show nothing than leak raw reasoning.
        return raw_text.split("<think>", 1)[0]
    return raw_text


def _strip_special_tokens(text: str, tokenizer) -> str:
    """For a COMPLETE final text: remove special tokens and trim whitespace."""
    return _replace_special_tokens(text, tokenizer).strip()


def _replace_special_tokens(text: str, tokenizer) -> str:
    """
    For a streaming DELTA: remove special tokens only, WITHOUT
    trimming whitespace — a delta chunk's leading/trailing space is
    real inter-word spacing once concatenated with neighboring
    chunks, not padding to discard (unlike the final-text case).
    """
    for special_tok in getattr(tokenizer, "all_special_tokens", []):
        text = text.replace(special_tok, "")
    return text


def make_model(
    model_id: str,
    *,
    adapter_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    temperature: float = 0.15,
    max_new_tokens: int = 4096,
    reference_images: Optional[list] = None,
) -> FunctionModel:
    """
    Build a pydantic-ai `Model` backed by a local Qwen3-VL checkpoint,
    loaded via unsloth's FastVisionModel (full precision, no 4-bit
    quantization).

    Returns a `FunctionModel` (a model driven by plain Python
    callables) so `Agent(model=make_model(...))` works exactly like it
    would with a hosted-API model, but everything runs on-device with
    no network call. Both a blocking path (`_run`, used by
    `agent.run()`) and a streaming path (`_stream_run`, used by
    `agent.run_stream()`) are provided — see `_stream_run`'s docstring
    for what "streaming" does and does not cover in this repo's
    architecture (the model can genuinely stream; the A2A hop between
    main.py and the agent subprocess currently cannot, since the
    pinned fasta2a==0.6.1 has no SSE/`message/stream` endpoint).

    Both paths strip a <think>...</think> reasoning block so only the
    final answer is ever returned to the caller — see `_strip_thinking`.

    Supports real MCP/tool execution via a simple text-based tool-call
    protocol (see `_build_tool_instructions`/`_TOOL_CALL_RE`) — if the
    agent has tools attached (via `toolsets=[...]`), the model is told
    how to invoke them, and a detected tool-call is turned into a real
    `ToolCallPart` so pydantic-ai's graph actually runs the tool and
    loops generation with its result.
    """

    def _run(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        model, processor = _load(model_id, adapter_path, processor_path)
        inputs, tokenizer = _build_chat_and_inputs(
            model, processor, model_id, messages, agent_info, reference_images
        )

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            repetition_penalty=1.15,
            # Stop generation the moment a complete tool-call block is
            # emitted, instead of burning the rest of the token budget
            # re-generating near-identical repeats of the same call.
            stop_strings=["</tool_call>"],
            tokenizer=tokenizer,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode WITH special tokens first — if skip_special_tokens=True is
        # used here, tokenizers that register </think> as an actual special
        # token will silently strip the tag itself during decoding, which
        # means a string search for it below never finds it and the raw
        # chain-of-thought leaks straight into the final answer. Instead we
        # decode with tags intact, split on the visible tag, and only THEN
        # strip out any remaining special-token text from what's left.
        raw_text = processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        output_text = _strip_special_tokens(_strip_thinking(raw_text), tokenizer)

        # Detect a tool-call block. If present, return a real ToolCallPart
        # so pydantic-ai actually executes the corresponding tool (e.g. the
        # Profiler agent's MCP read_file) instead of just leaving it as
        # inert text the model happened to type.
        match = _TOOL_CALL_RE.search(output_text)
        if match:
            try:
                call_data = json.loads(match.group(1))
                tool_name = call_data["name"]
                tool_args = call_data.get("arguments", {})
                tool_call_id = str(uuid.uuid4())
                log.info(f"[{model_id}] Tool call detected -> {tool_name}({tool_args})")
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=tool_call_id)]
                )
            except (json.JSONDecodeError, KeyError) as e:
                log.info(f"[{model_id}] Malformed tool_call block, treating as plain text: {e}")

        log.info(f"[{model_id}] Final text answer ({len(output_text)} chars), no tool call")
        return ModelResponse(parts=[TextPart(content=output_text)])

    async def _stream_run(messages: list[ModelMessage], agent_info: AgentInfo):
        """
        Streaming counterpart to `_run`. Yields plain `str` deltas
        (new text only, not the accumulated total — matching
        FunctionModel's `handle_text_delta` protocol) representing
        ONLY the final-answer portion of the generation.

        Thinking-token handling: everything from the start of
        generation up through the closing `</think>` tag is consumed
        and DISCARDED here, never yielded — this repo's requirement is
        "only the final answer in markdown", not a live view into the
        model's reasoning. We buffer silently until we're past
        `</think>`, then start yielding real deltas.

        Tool-call handling: pydantic-ai's FunctionModel forbids mixing
        plain `str` deltas with `DeltaToolCalls` in a single streamed
        response (see fasta2a's stream_function docstring). Since our
        tool-call protocol requires the model to emit ONLY a
        `<tool_call>...</tool_call>` block with no other prose when
        calling a tool, we can safely decide which "mode" this
        response is in from its first few post-</think> characters:
        if they look like the start of a tool call, we suppress all
        text streaming and, once generation finishes, yield a single
        complete `DeltaToolCalls` item instead; otherwise we stream
        real text deltas as they arrive.

        NOTE ON WHAT THIS DOES NOT DO: this streams the *model's*
        generation, in-process. It is not currently wired across the
        A2A hop between main.py and this agent's subprocess — the
        pinned fasta2a==0.6.1 has no SSE/`message/stream` endpoint, so
        main.py still receives one complete result over plain
        JSON-RPC. This function is the correct foundation for true
        end-to-end streaming if that's added later (or for testing
        agents directly, in-process, without going through A2A).
        """
        import threading
        from transformers import TextIteratorStreamer

        model, processor = _load(model_id, adapter_path, processor_path)
        inputs, tokenizer = _build_chat_and_inputs(
            model, processor, model_id, messages, agent_info, reference_images
        )

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            repetition_penalty=1.15,
            stop_strings=["</tool_call>"],
            tokenizer=tokenizer,
            streamer=streamer,
        )
        gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        gen_thread.start()

        buffer = ""            # raw text so far (special tokens intact)
        past_think = False     # have we passed the </think> boundary yet?
        mode: Optional[str] = None   # "text" or "tool_call", decided once, right after </think>
        already_yielded_len = 0      # how much of the post-</think> text we've already emitted (text mode only)
        TOOL_CALL_PREFIX = "<tool_call>"
        sentinel = object()

        while True:
            chunk = await asyncio.to_thread(next, streamer, sentinel)
            if chunk is sentinel:
                break
            buffer += chunk

            if not past_think:
                if "</think>" not in buffer:
                    continue  # still inside <think>...</think> — suppress entirely
                past_think = True

            after = buffer.split("</think>", 1)[1]

            if mode is None:
                stripped = after.lstrip()
                if not stripped:
                    continue  # nothing meaningful yet, keep waiting
                if stripped.startswith(TOOL_CALL_PREFIX):
                    mode = "tool_call"
                    continue  # suppress; handled once the loop ends, below
                if TOOL_CALL_PREFIX.startswith(stripped):
                    continue  # ambiguous prefix (e.g. just "<" or "<tool_c") — keep waiting
                mode = "text"
                already_yielded_len = len(after)
                visible = _replace_special_tokens(after, tokenizer)
                if visible:
                    yield visible
                continue

            if mode == "text":
                new_part = after[already_yielded_len:]
                already_yielded_len = len(after)
                visible = _replace_special_tokens(new_part, tokenizer)
                if visible:
                    yield visible
            # mode == "tool_call": suppress streaming, just keep buffering;
            # handled once the loop ends, below.

        gen_thread.join()

        if mode == "tool_call" or (mode is None and "<tool_call>" in buffer):
            final_text = _strip_special_tokens(_strip_thinking(buffer), tokenizer)
            match = _TOOL_CALL_RE.search(final_text)
            if match:
                try:
                    call_data = json.loads(match.group(1))
                    tool_call_id = str(uuid.uuid4())
                    from pydantic_ai.models.function import DeltaToolCall
                    log.info(f"[{model_id}] (stream) Tool call detected -> {call_data.get('name')}")
                    yield {
                        0: DeltaToolCall(
                            name=call_data["name"],
                            json_args=json.dumps(call_data.get("arguments", {})),
                            tool_call_id=tool_call_id,
                        )
                    }
                    return
                except (json.JSONDecodeError, KeyError) as e:
                    log.info(f"[{model_id}] (stream) Malformed tool_call block, falling back to text: {e}")
            # Malformed / didn't actually resolve into a tool call after
            # all — since we suppressed all along in this branch, we've
            # yielded nothing yet, so it's still safe to fall back to a
            # single text item without violating the no-mixing rule.
            if final_text:
                yield final_text

        log.info(f"[{model_id}] (stream) Done ({len(buffer)} raw chars)")

    return FunctionModel(_run, stream_function=_stream_run, model_name=model_id)
