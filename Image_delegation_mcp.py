"""
mcp_server.py
─────────────────────────────────────────────────────────────────
FastMCP vision-tools server shared by all three agents.
Runs as an HTTP SSE server on port 9000.

CORRECTED VERSION — the A2A delegation tools now build requests
matching the real A2A wire schema:
  - JSON-RPC method is "message/send" (not "tasks/send")
  - Part discriminator key is "kind" (not "type")
  - Message requires "message_id"
  - Images go through a FilePart {kind:"file", file:{uri, mime_type}},
    there is no "image_url" part type
  - Structured agent results come back as DataPart artifacts:
    {"kind": "data", "data": {"result": <the pydantic model dump>}}

Skill files are read via the standard MCP filesystem server
(@modelcontextprotocol/server-filesystem), NOT a tool here.
Agents call read_file() on that server for their instructions.

Tools:
  delegate_image_to_synthesis  — Agent 1 → Agent 2 via A2A
  delegate_to_expert_agent     — any agent → Agent 3 via A2A
  fetch_image_base64           — URL → base64 bytes
  store_result                 — write to shared in-memory store
  get_result                   — read from shared in-memory store
  compute_image_hash           — SHA-256 fingerprint
  utc_now                      — current UTC ISO timestamp
  log_event                    — structured log line
"""

import asyncio
import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

MCP_PORT       = int(os.getenv("MCP_SERVER_PORT",      "9000"))
SYNTHESIS_PORT = int(os.getenv("SYNTHESIS_AGENT_PORT", "8002"))
EXPERT_PORT    = int(os.getenv("EXPERT_AGENT_PORT",    "8003"))

mcp = FastMCP(
    name="QwenVL-Vision-Tools",
    instructions=(
        "Vision utilities and A2A delegation tools for Qwen VL agents. "
        "Use the filesystem MCP server (read_file) for skill files."
    ),
)

_store: dict[str, Any] = {}   # shared result cache (use Redis in prod)


# ════════════════════════════════════════════════════════════════
#  A2A wire helper — matches the real message/send schema
# ════════════════════════════════════════════════════════════════

async def _a2a_send_message(
    url: str,
    parts: list[dict],
    metadata: dict,
) -> dict:
    """
    Fire a real A2A message/send JSON-RPC request and return the
    raw JSON-RPC response body.

    parts: list of TextPart/FilePart dicts, each with "kind" key.
    metadata: extension metadata attached to the Message (not the
              task — A2A tasks get their own server-generated id).
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      str(uuid.uuid4()),
        "method":  "message/send",
        "params":  {
            "message": {
                "role":       "user",
                "kind":       "message",
                "message_id": str(uuid.uuid4()),
                "parts":      parts,
                "metadata":   metadata,
            },
            "configuration": {"blocking": True},
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload,
                                 headers={"Content-Type": "application/json"})
        resp.raise_for_status()
    return resp.json()


def _extract_data_or_text(result: dict) -> str:
    """
    Pull the agent's structured (DataPart) or plain (TextPart) output
    out of a Task result and return it as a JSON string for the caller.
    """
    artifacts = result.get("artifacts", [])
    for art in artifacts:
        for part in art.get("parts", []):
            if part.get("kind") == "data":
                data = part.get("data", {})
                return json.dumps(data.get("result", data))
    # Fall back to concatenated text parts
    texts = [
        part.get("text", "")
        for art in artifacts
        for part in art.get("parts", [])
        if part.get("kind") == "text"
    ]
    return "\n".join(texts)


# ════════════════════════════════════════════════════════════════
#  Tool 1 — Agent 1 → Agent 2 (Synthesis)
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def delegate_image_to_synthesis(
    task_id: str,
    image_url: str,
    visual_analysis: str,
    reason: str,
) -> dict:
    """
    Forward image + VisualAnalysis JSON to the Synthesis Agent via A2A.

    Call this after visual analysis when the skill file's delegation
    rules say to proceed. The MCP server makes the HTTP call — the
    model never constructs requests itself.

    Args:
        task_id:         Correlation ID (passed via metadata; A2A
                          generates its own task id server-side).
        image_url:        Original image URL (not base64).
        visual_analysis: Agent 1's VisualAnalysis as a JSON string.
        reason:          One sentence explaining why delegating.

    Returns:
        delegated, synthesis_result (JSON string), a2a_task_id, or error.
    """
    parts = [
        {
            "kind": "text",
            "text": (
                f"[Auto-delegated by Visual Analyzer — {reason}]\n\n"
                f"Original image URL: {image_url}\n\n"
                f"VisualAnalysis JSON:\n{visual_analysis}"
            ),
        },
        {"kind": "file", "file": {"uri": image_url}},
    ]

    try:
        body = await _a2a_send_message(
            url=f"http://localhost:{SYNTHESIS_PORT}",
            parts=parts,
            metadata={"delegated_by": "visual_analyzer", "parent_task_id": task_id},
        )
    except httpx.HTTPError as exc:
        return {"delegated": False, "error": str(exc)}

    if "error" in body:
        return {"delegated": False, "error": str(body["error"])}

    result = body.get("result", {})
    return {
        "delegated":        True,
        "a2a_task_id":      result.get("id", ""),
        "synthesis_result": _extract_data_or_text(result),
        "state":            result.get("status", {}).get("state", "unknown"),
    }


# ════════════════════════════════════════════════════════════════
#  Tool 2 — Any agent → Agent 3 (Domain Expert)
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def delegate_to_expert_agent(
    task_id: str,
    image_url: str,
    context_json: str,
    reason: str,
) -> dict:
    """
    Forward an image + prior context to the Domain Expert Agent via A2A.

    Call this when domain-specific deep analysis is needed (e.g. low
    confidence, domain-relevant context_category, explicit user request).

    Args:
        task_id:      Correlation ID (passed via metadata).
        image_url:    Original image URL (not base64).
        context_json: Prior VisualAnalysis / SynthesisResult JSON, or "{}".
        reason:       One sentence explaining why delegating.

    Returns:
        delegated, expert_result (JSON string), a2a_task_id, or error.
    """
    parts = [
        {
            "kind": "text",
            "text": (
                f"[Delegated to Domain Expert — {reason}]\n\n"
                f"image_url={image_url}\n\n"
                f"Prior context:\n{context_json}"
            ),
        },
        {"kind": "file", "file": {"uri": image_url}},
    ]

    try:
        body = await _a2a_send_message(
            url=f"http://localhost:{EXPERT_PORT}",
            parts=parts,
            metadata={"delegated_by": "orchestrator", "parent_task_id": task_id},
        )
    except httpx.HTTPError as exc:
        return {"delegated": False, "error": str(exc)}

    if "error" in body:
        return {"delegated": False, "error": str(body["error"])}

    result = body.get("result", {})
    return {
        "delegated":     True,
        "a2a_task_id":   result.get("id", ""),
        "expert_result": _extract_data_or_text(result),
        "state":         result.get("status", {}).get("state", "unknown"),
    }


# ════════════════════════════════════════════════════════════════
#  Tool 3 — Fetch image as base64
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def fetch_image_base64(image_url: str) -> dict:
    """
    Fetch an image from a URL and return it as base64.

    Returns:
        dict with data (base64 string), mime_type, size_bytes, url.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(image_url, follow_redirects=True)
        r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
    return {
        "data":       base64.b64encode(r.content).decode(),
        "mime_type":  mime,
        "size_bytes": len(r.content),
        "url":        image_url,
    }


# ════════════════════════════════════════════════════════════════
#  Tools 4 / 5 — Shared result store
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def store_result(task_id: str, agent_id: str, result: str) -> dict:
    """Persist an agent result under task_id:agent_id for later retrieval."""
    key = f"{task_id}:{agent_id}"
    _store[key] = {
        "result":    result,
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "agent_id":  agent_id,
    }
    return {"stored": True, "key": key}


@mcp.tool()
async def get_result(task_id: str, agent_id: str) -> dict:
    """Retrieve a stored agent result by task_id + agent_id."""
    entry = _store.get(f"{task_id}:{agent_id}")
    return {"found": entry is not None, **(entry or {})}


# ════════════════════════════════════════════════════════════════
#  Tools 6 / 7 / 8 — Utilities
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def compute_image_hash(image_url: str) -> dict:
    """SHA-256 fingerprint of an image for deduplication / caching."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(image_url, follow_redirects=True)
        r.raise_for_status()
    return {"sha256": hashlib.sha256(r.content).hexdigest(), "url": image_url}


@mcp.tool()
async def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
async def log_event(level: str, source: str, message: str) -> dict:
    """Emit a structured log entry to stdout."""
    entry = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "level":   level.upper(),
        "source":  source,
        "message": message,
    }
    print(f"[MCP-LOG] {json.dumps(entry)}")
    return entry


# ════════════════════════════════════════════════════════════════
#  Entrypoint
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"MCP Vision-Tools server  :{MCP_PORT}")
    mcp.run(transport="sse", port=MCP_PORT)


# ════════════════════════════════════════════════════════════════
#  Tool 9 — Keyword trigger scanner
#
#  Used by the Profiler Agent (= domain_expert_agent — it already
#  profiles visual content) to decide whether its own result text
#  warrants escalation to the RAG Agent for grounded knowledge.
# ════════════════════════════════════════════════════════════════

import re as _re

DEFAULT_TRIGGER_KEYWORDS = ["cause", "manner"]


@mcp.tool()
async def check_keyword_triggers(
    text: str,
    keywords: list[str] | None = None,
) -> dict:
    """
    Scan text for trigger keywords (case-insensitive, whole-word match).

    Call this on your own result text (e.g. overall_assessment +
    expert_notes + findings descriptions joined together) right before
    returning, to decide whether to escalate to the RAG Agent.

    Args:
        text:     Text to scan, e.g. your draft overall_assessment.
        keywords: Override the default trigger list. Defaults to
                  ["cause", "manner"] if omitted.

    Returns:
        dict:
          triggered        — True if any keyword matched
          matched_keywords  — list of keywords that matched
          checked_keywords — the full keyword list that was checked
    """
    kws = keywords or DEFAULT_TRIGGER_KEYWORDS
    text_lower = text.lower()
    matched = [
        kw for kw in kws
        if _re.search(rf"\b{_re.escape(kw.lower())}\b", text_lower)
    ]
    return {
        "triggered":        bool(matched),
        "matched_keywords":  matched,
        "checked_keywords": kws,
    }


# ════════════════════════════════════════════════════════════════
#  Tool 10 — Local RAG retrieval
#  Real embedding-based search: Octen embedding model (HuggingFace
#  transformers, via sentence-transformers) + LangChain FAISS index.
#  See octen_rag.py for the embedding/indexing implementation.
#  This replaces the earlier stdlib TF-IDF version — same tool name
#  and same return shape, so no caller-side changes needed.
# ════════════════════════════════════════════════════════════════

import octen_rag as _octen_rag


@mcp.tool()
async def rag_search(search_query: str, top_k: int = 5) -> dict:
    """
    Search the local knowledge base using Octen embeddings + FAISS
    cosine similarity (real semantic search, not keyword matching).

    First call lazily builds (or loads a cached) FAISS index from
    skills/rag_corpus.json using the Octen embedding model. Replace
    that file with your real domain knowledge base (forensic
    references, manuals, case law, SOPs, etc.) to make this useful
    in production, then call rebuild_octen_index() once to re-embed.

    Args:
        search_query: Natural-language query (typically the matched
                      trigger sentence(s) from the Profiler Agent's
                      result).
        top_k:        Max number of results to return.

    Returns:
        dict:
          results     — list of {text, source, score}, sorted desc
          corpus_size — number of indexed chunks in the knowledge base
    """
    # Runs the (blocking) embedding model call off the event loop.
    return await asyncio.to_thread(_octen_rag.query, search_query, top_k)


@mcp.tool()
async def rebuild_octen_index() -> dict:
    """
    Re-embed and re-index skills/rag_corpus.json from scratch.

    Call this after editing rag_corpus.json — the FAISS index is
    cached on disk and won't pick up corpus changes automatically.
    """
    await asyncio.to_thread(_octen_rag.rebuild_index)
    store = await asyncio.to_thread(_octen_rag.get_vectorstore)
    size = store.index.ntotal if hasattr(store, "index") else 0
    return {"rebuilt": True, "corpus_size": size}


# ════════════════════════════════════════════════════════════════
#  Tool 11 — Any agent → RAG Agent via A2A
#  (triggered after keyword match, not called unconditionally)
# ════════════════════════════════════════════════════════════════

RAG_PORT = int(os.getenv("RAG_AGENT_PORT", "8004"))


@mcp.tool()
async def delegate_to_rag_agent(
    task_id: str,
    query: str,
    context_json: str,
    reason: str,
) -> dict:
    """
    Forward a query + context to the RAG Agent via A2A.

    Call this ONLY after check_keyword_triggers returned triggered=true.
    Do not call unconditionally — the RAG Agent should only run when
    the Profiler Agent's own text raised a flagged term.

    Args:
        task_id:      Correlation ID.
        query:         The matched sentence(s) or a focused question
                       derived from them — what to retrieve evidence for.
        context_json:  Profiler Agent's full result JSON, or "{}".
        reason:        Which keyword(s) triggered this, e.g.
                       "Matched keyword: 'cause' in overall_assessment".

    Returns:
        delegated, rag_result (JSON string), a2a_task_id, or error.
    """
    parts = [{
        "kind": "text",
        "text": (
            f"[Triggered by Profiler Agent keyword match — {reason}]\n\n"
            f"Query: {query}\n\n"
            f"Profiler context:\n{context_json}"
        ),
    }]

    try:
        body = await _a2a_send_message(
            url=f"http://localhost:{RAG_PORT}",
            parts=parts,
            metadata={"delegated_by": "profiler_agent", "parent_task_id": task_id,
                      "trigger_reason": reason},
        )
    except httpx.HTTPError as exc:
        return {"delegated": False, "error": str(exc)}

    if "error" in body:
        return {"delegated": False, "error": str(body["error"])}

    result = body.get("result", {})
    return {
        "delegated":   True,
        "a2a_task_id": result.get("id", ""),
        "rag_result":  _extract_data_or_text(result),
        "state":       result.get("status", {}).get("state", "unknown"),
    }
