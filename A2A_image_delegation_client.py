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
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}

@dataclass
class ImagePart:
    url: str
    type: str = "url"

    def to_dict(self) -> dict:
        return {"type": self.type, "url": self.url}

@dataclass
class A2AMessage:
    parts: list[TextPart | ImagePart]
    role: str = "user"

    def to_dict(self) -> dict:
        return {"role": self.role, "parts": [p.to_dict() for p in self.parts]}

#Task Result

@dataclass
class A2ATask:
    id: str
    state: str
    artifacts: list[dict] = field(default_factory = list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.state == "success"
    
    @property
    def failed(self) -> bool:
        return self.state == "failed"

    def output(self) -> str:
        return "\n".join(a.get("text", "") for a in self.artifacts if a.get("type") == "text")

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
        task_id: Optional[str] = None,
        metadata: Optional[dict] = None) -> A2ATask:
        task_id = task_id or str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": message.to_dict(),
                **({"metadata": metadata} if metadata else {})
            }
        }

        async with httpx.AsyncClient(timeout = self._timeout) as c:
            r = await c.post(self.base_url, json = payload, headers = {"Content-Type": "application/json"})
            r.raise_for_status()
            body = r.json()

        if "error" in body:
            return A2ATask(id = task_id, state = "failed", error = str(body["error"]))

        res = body.get("result", {})
        return A2ATask(
            id = res.get("id", task_id),
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