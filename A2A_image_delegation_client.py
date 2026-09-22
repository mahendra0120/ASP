"""
A2A_image_delegation_client.py
─────────────────────────────────────────────────────────────────
Minimal JSON-RPC client for talking to fasta2a-served agents.

NOTE: the wire format below was verified directly against the
installed fasta2a package (pydantic_ai bridge + fasta2a.schema) on
2026-09-22 — do not trust an older comment here over that. As of
fasta2a 2.0.1 (the version `fasta2a[pydantic-ai]>=0.6.1` in
pyproject.toml currently resolves to, since that constraint has no
upper bound):
  - the send method is `message/send`, params = {"message": ...}
  - `Message` = {"role", "parts", "messageId", ...} — camelCase via
    an alias_generator, and there is NO "kind" field on Message.
  - `Part` is a FLAT, untagged dict — fields are mutually exclusive
    by which key is present, there is NO "kind" discriminator and
    NO nested "file" object:
        {"text": "..."}                          # text
        {"url": "...", "mediaType": "image/png"}  # file by URL
        {"raw": "<base64>", "mediaType": "..."}   # file by bytes
    `media_type` is aliased to `mediaType` on the wire (same
    to_camel() aliasing applies to `message_id` -> `messageId`).
  - a completed task's state is `"completed"`, not `"success"`
  - artifacts carry their content in `artifact["parts"]`, each of
    which may have a `text` field — not directly on the artifact
`tasks/get` and `tasks/cancel` are unchanged from the older draft.

fasta2a has changed this wire format more than once across recent
releases (see pyproject.toml's fasta2a pin comment) — if agents ever
start silently not receiving images/files again after a dependency
update, re-verify this against the actually-installed version's
`fasta2a/schema.py` (`Part`/`Message` TypedDicts) rather than
assuming this comment is still accurate.
"""

import os
import json
import httpx
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Optional

#Message Types

@dataclass
class TextPart:
    text: str

    def to_dict(self) -> dict:
        return {"text": self.text}


@dataclass
class ImagePart:
    url: str
    media_type: Optional[str] = None

    def to_dict(self) -> dict:
        part: dict = {"url": self.url}
        if self.media_type:
            part["mediaType"] = self.media_type
        return part

@dataclass
class A2AMessage:
    parts: list[TextPart | ImagePart]
    role: str = "user"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
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
    def failed(self) -> bool:
        return self.state in ("failed", "rejected")

    def output(self) -> str:
        texts = []
        for artifact in self.artifacts:
            for part in artifact.get("parts", []):
                if "text" in part:
                    texts.append(part["text"])
        return "\n".join(texts)


# A2A HTTP Client

class A2AClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def send_task(
        self,
        message: A2AMessage,
        metadata: Optional[dict] = None,
        poll_interval: float = 2.0,
        max_wait: float = 1800.0,
    ) -> A2ATask:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": message.to_dict(),
                **({"metadata": metadata} if metadata else {})
            }
        }

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(self.base_url, json=payload, headers={"Content-Type": "application/json"})
            r.raise_for_status()
            body = r.json()

        if "error" in body:
            print(f"[A2A:{self.base_url}] send_task error: {body['error']}")
            return A2ATask(id=message.message_id, state="failed", error=str(body["error"]))

        res = body.get("result", {})
        task_id = res.get("id", message.message_id)
        state = res.get("status", {}).get("state", "unknown")
        artifacts = res.get("artifacts", [])

        print(f"[A2A:{self.base_url}] task {task_id} -> state={state} (elapsed=0.0s)")

        # message/send only enqueues the task; poll tasks/get until it
        # reaches a terminal state (completed / failed / rejected / canceled).
        elapsed = 0.0
        terminal_states = ("completed", "failed", "rejected", "canceled")
        while state not in terminal_states and elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            task = await self.get_task(task_id)
            if task.state != state:
                print(f"[A2A:{self.base_url}] task {task_id} -> state={task.state} (elapsed={elapsed:.1f}s)")
            state = task.state
            artifacts = task.artifacts

        if state not in terminal_states:
            print(f"[A2A:{self.base_url}] task {task_id} TIMED OUT after {max_wait}s (last state: {state})")
            return A2ATask(
                id=task_id, state="failed", artifacts=artifacts,
                error=f"Task did not complete within {max_wait}s (last state: {state})"
            )

        print(f"[A2A:{self.base_url}] task {task_id} finished: state={state}, artifacts={len(artifacts)}")
        return A2ATask(id=task_id, state=state, artifacts=artifacts)
    
    async def get_task(self, task_id: str) -> A2ATask:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id}}
        async with httpx.AsyncClient(timeout = 30.0) as c:
            r = await c.post(self.base_url, json = payload)
            r.raise_for_status()
            full = r.json()
            res = full.get("result", {})

        state = res.get("status", {}).get("state", "unknown")
        if state in ("failed", "rejected"):
            print(f"[A2A:{self.base_url}] RAW FAILED TASK RESPONSE:")
            print(json.dumps(full, indent=2))

        status_message = res.get("status", {}).get("message")
        error_text = None
        if status_message:
            error_text = json.dumps(status_message)

        return A2ATask(
            id = res.get("id", task_id),
            state = state,
            artifacts = res.get("artifacts", []),
            error = error_text,
        )


PROFILER_AGENT = A2AClient(f"http://localhost:{os.getenv('PROFILER_AGENT_PORT', '8011')}")
FORENSIC_AGENT = A2AClient(f"http://localhost:{os.getenv('FORENSIC_AGENT_PORT', '8002')}")
