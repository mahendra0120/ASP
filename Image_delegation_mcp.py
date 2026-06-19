import base64
import hashlib
import uuid
import json
import os
from typing import Any
from datetime import datetime, timezone
import httpx
import asyncio
from mcp.server.fastmcp import FastMCP

MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "9000"))
FORENSIC_AGENT = int(os.getenv("FORENSIC_AGENT_PORT", "8002"))

mcp = FastMCP(
    name = "Image-Delegation",
    instructions=(
        "Vision utilities and A2A delegation tools for Vision agents. "
        "Use the filesystem MCP server (read_file) for skill files."
    ),
    port = MCP_PORT,
)

_store: dict[str, Any] = {}

#A2A Delegation Helpers
async def a2a_post(url: str, task_id: str, parts: list[dict], metadata: dict) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "methods": "tasks/send",
        "params": {
            "id": task_id,
            "message": {"role": "user", "parts": parts},
            "metadata": metadata
        }
    }
    async with httpx.AsyncClient(timeout = 120.0) as client:
        resp = await client.post(url, json = payload, headers = {"Content-Type": "application/json"})
        resp.raise_for_status()
    return resp.json()

#Image Delegation Tool
@mcp.tool()
async def delegate(task_id: str, image_url: str, reason: str) -> dict:
    sub_id = f"{task_id}-forensic-{str(uuid.uuid4())[:6]}"
    try:
        body = await a2a_post(
            url = f"http://127.0.0.1:{FORENSIC_AGENT}/jsonrpc",
            task_id = sub_id,
            parts = [
                {
                    "type": "text",
                    "text": (
                        f"[Auto-delegated by Visual Analyzer — {reason}]\n\n"
                        f"Original Image URL: {image_url}\n\n"
                    )
                }
            ],
            metadata = {"parent_task_id": task_id}
        )
    except httpx.HTTPError as exc:
        return {"delegated": False, "error": str(exc)}

    if "error" in body:
        return {"delegated": False, "error": str(body["error"])}

    result = body.get("result", {})
    text = "\n".join(
        a.get("text", "") for a in result.get("artifacts", [])
        if a.get("type") == "text"
        )
    return {
        "delegated": True, 
        "sub_task_id": sub_id, 
        "forensic_analysis": text, 
        "state": result.get("status", {}).get("state", "unknown")}

#Image Base64 Conversion
@mcp.tool()
async def image_base64(image_url: str) -> dict:
    async with httpx.AsyncClient(timeout = 15.0) as client:
        r = await client.get(image_url, follow_redirects = True)
        r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
    return {
        "data": base64.b64encode(r.content).decode(),
        "mime-type": mime,
        "size_bytes": len(r.content),
        "url": image_url
    }

#Shared result store

@mcp.tool()
async def store_result(task_id: str, agent_id: str, result: str) -> dict:
    key = f"{task_id}: {agent_id}"
    _store[key] = {
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id
    }
    return {"stored": True, "key": key}

@mcp.tool()
async def get_result(task_id: str, agent_id: str) -> dict:
    entry = _store.get(f"{task_id}: {agent_id}")
    return {"found": entry is not None, **(entry or {})}

# Utilities

@mcp.tool()
async def compute_image_hash(image_url: str) -> dict:
    async with httpx.AsyncClient(timeout = 15.0) as client:
        r = await client.get(image_url, follow_redirects = True)
        r.raise_for_status()
    return {"sha256": hashlib.sha256(r.content).hexdigest(), "url": image_url}

@mcp.tool()
async def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

@mcp.tool()
async def log_event(level: str, source: str, message: str) -> dict:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "source": source,
        "message": message
    }
    print(f"[MCP-LOG] {json.dumps(entry)}")
    return entry

#Entrypoint

if __name__ == "__main__":
    print(f"MCP Image Delegation Server: {MCP_PORT}")
    mcp.run(transport = "sse")