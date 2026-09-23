"""
A2A_image_delegation_client.py
─────────────────────────────────────────────────────────────────
Minimal JSON-RPC client for talking to fasta2a-served agents.

WIRE FORMAT — this was a genuinely confusing one across several
verification passes (see git history / conversation log if this
needs revisiting), so here is the FULL evidence trail as of
2026-09-22, all against fasta2a==2.0.1 (pinned in pyproject.toml,
confirmed to be what's ACTUALLY installed in this repo's .venv via
`importlib.metadata.version('fasta2a')`, not just what a lockfile
claims is resolved):
  1. Direct source inspection of `fasta2a.schema.Message` and `Part`
     shows both are FLAT TypedDicts with camelCase aliasing — NO
     "kind" field exists on either one.
  2. Direct round-trip testing against `fasta2a.schema.a2a_request_ta`
     — the exact TypeAdapter `applications.py` uses to validate every
     incoming JSON-RPC request — confirms: a flat, un-tagged part
     like `{"url": ..., "mediaType": ...}` validates correctly and
     keeps its data; a "kind"-tagged/nested part like
     `{"kind": "file", "file": {"uri": ...}}` does NOT raise an
     error, but silently validates down to an EMPTY `{}` (every field
     on it is an unrecognized extra key), which then fails later
     inside the pydantic_ai bridge with "Unsupported part" — this is
     why sending the kind-tagged format can look like it's "working"
     (no request-level error) right up until images silently vanish.
  So: for 2.0.1, the correct, verified wire shape is:
  - the send method is `message/send`, params = {"message": ...}
  - `Message` = {"role", "parts", "messageId", ...} — camelCase via
    an alias_generator (message_id -> messageId), NO "kind" field.
  - `Part` is FLAT and untagged — fields are mutually exclusive by
    which key is present, NOT by a "kind" discriminator:
        {"text": "..."}                          # text
        {"url": "...", "mediaType": "image/png"}  # file by URL
        {"raw": "<base64>", "mediaType": "..."}   # file by bytes
    (`media_type` is aliased to `mediaType` on the wire.)
  - a completed task's state is `"completed"`, not `"success"`
  - artifacts carry their content in `artifact["parts"]`, each of
    which may have a `text` field — not directly on the artifact
`tasks/get` and `tasks/cancel` are unchanged from the older draft.

fasta2a's A2A schema has genuinely changed shape more than once
across its release history (0.6.x used a "kind"-discriminated,
nested-file shape; 2.0.1 — what's actually pinned and installed
here — uses this flat shape instead). If this ever needs revisiting
after a dependency bump, do NOT trust this comment blindly — repeat
the verification steps above (metadata.version, inspect.getsource on
Message/Part, and an actual a2a_request_ta.validate_json() round
trip) against whatever is ACTUALLY importable in the target .venv at
that time, since a lockfile/pyproject.toml constraint and what's
truly installed can drift out of sync.
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
