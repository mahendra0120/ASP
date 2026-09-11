"""
A2A_image_delegation_client.py
─────────────────────────────────────────────────────────────────
Minimal JSON-RPC client for talking to fasta2a-served agents.

NOTE: this was originally written against an older draft of the A2A
spec. The installed fasta2a implements the current spec, where:
  - the send method is `message/send`, not `tasks/send`
  - the client sends a `Message` (role/parts/messageId/kind), not a
    task with a client-chosen id — the server assigns the task id
  - `Part` is a `kind`-discriminated union: `{"kind": "text", ...}`,
    `{"kind": "file", "file": {"uri": ..., "mimeType": ...}}`, or
    `{"kind": "data", "data": ...}` — not a flat `{"url": ...}`
  - `Message` itself needs `"kind": "message"`
  - a completed task's state is `"completed"`, not `"success"`
  - artifacts carry their content in `artifact["parts"]`, each of
    which may have a `text` field — not directly on the artifact
`tasks/get` and `tasks/cancel` are unchanged from the older draft.
"""

import os
import json
import httpx
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

#Message Types

@dataclass
class TextPart:
    text: str

    def to_dict(self) -> dict:
        return {"kind": "text", "text": self.text}

@dataclass
class ImagePart:
    url: str
    media_type: Optional[str] = None

    def to_dict(self) -> dict:
        file_obj = {"uri": self.url}
        if self.media_type:
            file_obj["mimeType"] = self.media_type
        return {"kind": "file", "file": file_obj}

@dataclass
class A2AMessage:
    parts: list[TextPart | ImagePart]
    role: str = "user"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "kind": "message",
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "messageId": self.message_id,
        }

#Task Result

@dataclass
class A2ATask:
    id: str
    state: str
    artifacts: list[dict] = field(default_factory = list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.state == "completed"

    @property
    def failed(self) -> bool:
        return self.state in ("failed", "rejected")

    def output(self) -> str:
        texts = []
        for artifact in self.artifacts:
            for part in artifact.get("parts", []):
                if "text" in part:
                    texts.append(part["text"])
        return "\n".join(texts)

    def json_output(self) -> Any:
        text = self.output()
        return json.loads(text) if text else {}


# A2A HTTP Client

class A2AClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
    
    async def ag_card(self) -> dict:
        async with httpx.AsyncClient(timeout = 10.0) as c:
            r = await c.get(f"{self.base_url}/.well-known/agent.json")
            r.raise_for_status()
            return r.json()

    async def send_task(
        self,
        message: A2AMessage,
        metadata: Optional[dict] = None) -> A2ATask:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": message.to_dict(),
                **({"metadata": metadata} if metadata else {})
            }
        }

        async with httpx.AsyncClient(timeout = self._timeout) as c:
            r = await c.post(self.base_url, json = payload, headers = {"Content-Type": "application/json"})
            r.raise_for_status()
            body = r.json()

        if "error" in body:
            return A2ATask(id = message.message_id, state = "failed", error = str(body["error"]))

        res = body.get("result", {})
        return A2ATask(
            id = res.get("id", message.message_id),
            state = res.get("status", {}).get("state", "unknown"),
            artifacts = res.get("artifacts", [])
        )
    
    async def get_task(self, task_id: str) -> A2ATask:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id}}
        async with httpx.AsyncClient(timeout = 30.0) as c:
            r = await c.post(self.base_url, json = payload)
            r.raise_for_status()
            res = r.json().get("result", {})
        return A2ATask(
            id = res.get("id", task_id),
            state = res.get("status", {}).get("state", "unknown"),
            artifacts = res.get("artifacts", [])
        )

    async def cancel_task(self, task_id: str) -> bool:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/cancel",
            "params": {"id": task_id}
        }
        async with httpx.AsyncClient(timeout = 10.0) as c:
            r = await c.post(self.base_url, json = payload)
            return r.status_code == 200



PROFILER_AGENT = A2AClient(f"http://localhost:{os.getenv('PROFILER_AGENT_PORT', '8001')}")
FORENSIC_AGENT = A2AClient(f"http://localhost:{os.getenv('FORENSIC_AGENT_PORT', '8002')}")