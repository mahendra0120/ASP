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
  delegate_to_profiler_agent   — any agent → Profiler agent via A2A
  delegate_to_forensic_agent   — any agent → Forensic agent via A2A
  fetch_image_base64           — URL → base64 bytes
  store_result                 — write to the PostgreSQL result store
  get_result                   — read from the PostgreSQL result store
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

import httpx
from mcp.server.mcpserver import MCPServer

import db

MCP_PORT      = int(os.getenv("MCP_SERVER_PORT",      "9000"))
PROFILER_PORT = int(os.getenv('PROFILER_AGENT_PORT', '8011'))
FORENSIC_PORT = int(os.getenv('FORENSIC_AGENT_PORT', '8002'))

mcp = MCPServer(
    name="QwenVL-Vision-Tools",
    instructions=(
        "Vision utilities and A2A delegation tools for Qwen VL agents. "
        "Use the filesystem MCP server (read_file) for skill files."
    ),
)


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
#  Tool 1 — any agent → Profiler agent
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def delegate_to_profiler_agent(
    task_id: str,
    image_url: str,
    visual_analysis: str,
    reason: str,
) -> dict:
    """
    Forward image + analysis to the Profiler agent via A2A.

    The MCP server makes the HTTP call — the model never constructs
    requests itself. Note: the standard pipeline (main.py) already
    drives Forensic → Profiler directly via A2AClient, so this tool
    is only needed for an agent that isn't already receiving the
    Profiler's result through that orchestration.

    Args:
        task_id:         Correlation ID (passed via metadata; A2A
                          generates its own task id server-side).
        image_url:        Original image URL (not base64).
        visual_analysis: Prior analysis JSON/text to forward.
        reason:          One sentence explaining why delegating.

    Returns:
        delegated, profiler_result (JSON string), a2a_task_id, or error.
    """
    parts = [
        {
            "kind": "text",
            "text": (
                f"[Auto-delegated — {reason}]\n\n"
                f"Original image URL: {image_url}\n\n"
                f"Prior analysis:\n{visual_analysis}"
            ),
        },
        {"kind": "file", "file": {"uri": image_url}},
    ]

    try:
        body = await _a2a_send_message(
            url=f"http://localhost:{PROFILER_PORT}",
            parts=parts,
            metadata={"delegated_by": "mcp_tool", "parent_task_id": task_id},
        )
    except httpx.HTTPError as exc:
        return {"delegated": False, "error": str(exc)}

    if "error" in body:
        return {"delegated": False, "error": str(body["error"])}

    result = body.get("result", {})
    return {
        "delegated":       True,
        "a2a_task_id":     result.get("id", ""),
        "profiler_result": _extract_data_or_text(result),
        "state":           result.get("status", {}).get("state", "unknown"),
    }


# ════════════════════════════════════════════════════════════════
#  Tool 2 — any agent → Forensic agent
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def delegate_to_forensic_agent(
    task_id: str,
    image_url: str,
    context_json: str,
    reason: str,
) -> dict:
    """
    Forward an image + prior context to the Forensic agent via A2A.

    Note: the standard pipeline (main.py) already sends case images to
    the Forensic agent directly via A2AClient. This tool exists for
    an agent that needs to delegate to the Forensic agent outside
    that orchestration (e.g. a follow-up request mid-conversation).

    Args:
        task_id:      Correlation ID (passed via metadata).
        image_url:    Original image URL (not base64).
        context_json: Prior analysis JSON, or "{}".
        reason:       One sentence explaining why delegating.

    Returns:
        delegated, forensic_result (JSON string), a2a_task_id, or error.
    """
    parts = [
        {
            "kind": "text",
            "text": (
                f"[Delegated to Forensic agent — {reason}]\n\n"
                f"image_url={image_url}\n\n"
                f"Prior context:\n{context_json}"
            ),
        },
        {"kind": "file", "file": {"uri": image_url}},
    ]

    try:
        body = await _a2a_send_message(
            url=f"http://localhost:{FORENSIC_PORT}",
            parts=parts,
            metadata={"delegated_by": "mcp_tool", "parent_task_id": task_id},
        )
    except httpx.HTTPError as exc:
        return {"delegated": False, "error": str(exc)}

    if "error" in body:
        return {"delegated": False, "error": str(body["error"])}

    result = body.get("result", {})
    return {
        "delegated":      True,
        "a2a_task_id":    result.get("id", ""),
        "forensic_result": _extract_data_or_text(result),
        "state":          result.get("status", {}).get("state", "unknown"),
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
#  Tools 4 / 5 — Shared result store (PostgreSQL-backed, see db.py)
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def store_result(task_id: str, agent_id: str, result: str) -> dict:
    """Persist an agent result under task_id:agent_id for later retrieval."""
    return await db.store_result(task_id, agent_id, result)


@mcp.tool()
async def get_result(task_id: str, agent_id: str) -> dict:
    """Retrieve a stored agent result by task_id + agent_id."""
    return await db.get_result(task_id, agent_id)


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
    # Invalidate any cached FAISS index from a previous run by default —
    # see octen_rag.clear_index_cache()'s docstring for why: a stale
    # cache built with older chunking logic (or an older corpus) was
    # otherwise silently reused instead of picked up. This just deletes
    # the cache; the first real rag_search call after startup rebuilds
    # it lazily (a few seconds for the current small corpus). Set
    # RAG_REBUILD_ON_STARTUP=false once you're not actively iterating
    # on forensic_knowledge/*.md or octen_rag.py's chunking logic.
    if os.getenv("RAG_REBUILD_ON_STARTUP", "true").strip().lower() not in ("false", "0", "no"):
        if _octen_rag.clear_index_cache():
            print("[MCP] Cleared cached RAG index (.octen_faiss_index/) — will rebuild on first rag_search call.")
        else:
            print("[MCP] No cached RAG index found — will build fresh on first rag_search call.")
    try:
        mcp.run(transport="sse", port=MCP_PORT)
    finally:
        asyncio.run(db.close_pool())


# ════════════════════════════════════════════════════════════════
#  Tool 9 — Keyword trigger scanner
#
#  Used by the Profiler Agent to decide whether its own result text
#  warrants a rag_search call for grounded knowledge.
# ════════════════════════════════════════════════════════════════

import octen_rag as _octen_rag  # shared keyword-trigger + RAG logic

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
    returning, to decide whether to call rag_search.

    Args:
        text:     Text to scan, e.g. your draft overall_assessment.
        keywords: Override the default trigger list. Defaults to
                  ["cause", "manner"] if omitted. Pass
                  octen_rag.TRAUMA_TRIGGER_KEYWORDS to check for
                  evidence-of-trauma language instead (this is what the
                  Forensic→RAG orchestration in main.py checks for,
                  since the Forensic agent itself has no MCP tools).

    Returns:
        dict:
          triggered        — True if any keyword matched
          matched_keywords  — list of keywords that matched
          checked_keywords — the full keyword list that was checked
    """
    return _octen_rag.check_keyword_trigger(text, keywords or DEFAULT_TRIGGER_KEYWORDS)


# ════════════════════════════════════════════════════════════════
#  Tool 10 — Local RAG retrieval
#  Real embedding-based search: Octen embedding model (HuggingFace
#  transformers, via sentence-transformers) + LangChain FAISS index.
#  See octen_rag.py for the embedding/indexing implementation.
#  This replaces the earlier stdlib TF-IDF version — same tool name
#  and same return shape, so no caller-side changes needed.
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def rag_search(search_query: str, top_k: int = 5) -> dict:
    """
    Search the local knowledge base using Octen embeddings + FAISS
    cosine similarity (real semantic search, not keyword matching).

    First call lazily builds (or loads a cached) FAISS index from
    every *.md file in forensic_knowledge/ (gunshot wounds, postmortem
    changes, thermal injuries, asphyxiation, sharp force injury, etc.)
    using the Octen embedding model. Call rebuild_octen_index() after
    editing any file in that folder to re-embed.

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
    Re-embed and re-index the forensic_knowledge/*.md corpus from scratch.

    Call this after editing/adding a file in forensic_knowledge/ — the
    FAISS index is cached on disk and won't pick up corpus changes
    automatically.
    """
    await asyncio.to_thread(_octen_rag.rebuild_index)
    store = await asyncio.to_thread(_octen_rag.get_vectorstore)
    size = store.index.ntotal if hasattr(store, "index") else 0
    return {"rebuilt": True, "corpus_size": size}
