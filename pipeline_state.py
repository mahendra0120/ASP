"""
pipeline_state.py
─────────────────────────────────────────────────────────────────
Crash-recovery layer for the Forensic -> RAG -> Profiler pipeline.

Every step's output is checkpointed to PostgreSQL (see db.py) while it
is produced, so nothing is lost when an agent server, the UI process or
the whole pod dies. A re-run of the same case skips finished steps and
continues partial ones. Agent servers stay stateless: all progress lives
in the database, keyed by a deterministic case_id.

Design rules:
  * The database is an OPTIONAL safety net. If Postgres is unreachable
    the pipeline still runs; it just says it is not saving.
  * case_id = hash(image bytes + extra context), so re-running the same
    inputs finds the same saved work without any session state.
  * A partial report is continued by re-prompting the agent with the
    text already written (cut back to its last complete line).
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import db

log = logging.getLogger("pipeline_state")

_GRADIO_FILE_MARK = "gradio_api/file="
_STORE_TIMEOUT_S = 20.0
_PUBLISHED_STEPS = ("forensic", "profiler", "final")   # also mirrored into agent_results (MCP get_result)


# ───────────────────────── case identity ─────────────────────────

def _local_path(url: str) -> Optional[Path]:
    """Local file behind an uploaded-image URL, or None for external URLs."""
    if _GRADIO_FILE_MARK in url:
        return Path(url.split(_GRADIO_FILE_MARK, 1)[1])
    return None


def _fingerprint(url: str) -> str:
    p = _local_path(url)
    if p is not None:
        try:
            return "file:" + hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            pass
    return "url:" + url


def case_id_for(image_urls: list[str], extra_context: str = "") -> str:
    """Deterministic id: the same images (same bytes, same order) + context => same case."""
    h = hashlib.sha256()
    for u in image_urls:
        h.update(_fingerprint(u).encode())
        h.update(b"\0")
    h.update((extra_context or "").strip().encode())
    return h.hexdigest()[:12]


def image_available(url: str) -> bool:
    p = _local_path(url)
    return p.exists() if p is not None else True


def meta_json(image_urls: list[str], extra_context: str) -> str:
    return json.dumps({
        "image_urls": image_urls,
        "extra_context": extra_context or "",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def loads(text: str) -> Any:
    return json.loads(text)


# ───────────────────── continuing partial output ─────────────────────

def trim_partial(text: str) -> str:
    """Cut a streamed partial back to its last COMPLETE line, so the seam is never mid-word."""
    t = (text or "").rstrip()
    cut = t.rfind("\n")
    return t[:cut].rstrip() if cut > 0 else ""


def continuation_block(kind: str, already_written: str) -> str:
    return (
        f"\n\n=== RESUMING AN INTERRUPTED {kind.upper()} ===\n"
        "An earlier attempt at this task was cut off part-way. The text it had already "
        "written is below, between the markers. Do NOT repeat or rewrite it. Continue from "
        "exactly where it stops: finish any unfinished section first, then write every "
        "remaining required section in the required order. Output ONLY the continuation "
        "(no preamble, no restating earlier sections).\n"
        "<<<ALREADY WRITTEN>>>\n"
        f"{already_written}\n"
        "<<<END OF ALREADY WRITTEN>>>"
    )


_LISTISH = re.compile(r"\s*([-*+]\s|\d+[.)]\s|\|)")


def join_continuation(prefix: str, new: str) -> str:
    new = (new or "").lstrip("\n")
    if not prefix:
        return new
    base = prefix.rstrip()
    last_line = base.splitlines()[-1] if base else ""
    if _LISTISH.match(new) and _LISTISH.match(last_line):
        return base + "\n" + new          # keep list items / table rows contiguous
    return base + "\n\n" + new


# ───────────────────────── the safe store ─────────────────────────

class CaseStore:
    """
    All database access for one case. Never raises: on the first database
    failure it switches itself off for the rest of the run (so a down
    Postgres cannot slow every streamed chunk) and `ok` becomes False.
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.ok = True
        self.error: Optional[str] = None

    async def _run(self, coro, default=None):
        if not self.ok:
            coro.close()
            return default
        try:
            return await asyncio.wait_for(coro, timeout=_STORE_TIMEOUT_S)
        except Exception as e:                      # noqa: BLE001 - DB must never break the pipeline
            self.ok = False
            self.error = f"{type(e).__name__}: {e}"
            log.warning(f"[case {self.case_id}] Postgres unavailable, continuing WITHOUT saving progress: {self.error}")
            return default

    def status_suffix(self) -> str:
        return "" if self.ok else "  ⚠️ (database unavailable — progress is NOT being saved)"

    async def load_all(self) -> dict:
        return await self._run(db.load_steps(self.case_id), default={}) or {}

    async def delete_all(self) -> None:
        await self._run(db.delete_case(self.case_id))

    async def save(self, step: str, content: str, status: str) -> None:
        wrote = await self._run(db.save_step(self.case_id, step, content, status), default=False)
        if wrote and status == "complete" and step in _PUBLISHED_STEPS:
            await self._run(db.store_result(self.case_id, step, content))   # visible to MCP get_result

    async def invalidate(self, steps: list[str]) -> None:
        """Forget downstream steps that were computed from an older upstream result."""
        await self._run(db.delete_steps(self.case_id, steps))

    async def load_meta(self) -> Optional[dict]:
        steps = await self.load_all()
        m = steps.get("meta")
        if not m:
            return None
        try:
            return json.loads(m["content"])
        except ValueError:
            return None


async def checkpointed(gen: AsyncIterator[str], store: CaseStore, step: str,
                       prefix: str = "", min_interval: float = 10.0) -> AsyncIterator[str]:
    """
    Wrap a text-streaming async generator: save progress at most every
    `min_interval` seconds, save the latest text if the stream dies or is
    cancelled, and mark the step complete when it ends normally. With a
    `prefix` (already-written text being continued) every yielded/saved
    value is prefix + continuation.
    """
    joined = prefix
    last_saved = time.monotonic()
    finished = False
    try:
        async with aclosing(gen) as g:
            async for text in g:
                joined = join_continuation(prefix, text) if prefix else text
                yield joined
                if time.monotonic() - last_saved >= min_interval:
                    await store.save(step, joined, "partial")
                    last_saved = time.monotonic()
        finished = True
    finally:
        if finished:
            await store.save(step, joined, "complete")
        elif joined.strip():
            await store.save(step, joined, "partial")


# ───────────────────────── UI helpers ─────────────────────────

def describe_case(row: dict) -> str:
    steps = dict(s.split(":", 1) for s in (row.get("steps") or "").split(",") if ":" in s)
    marks = {"complete": "✅", "partial": "⏸"}
    parts = [f"{name[:4]} {marks.get(steps[name], '·')}" if name in steps else f"{name[:4]} ·"
             for name in ("forensic", "rag", "profiler")]
    n_img = ""
    try:
        n_img = f" · {len(json.loads(row['meta'])['image_urls'])} img"
    except Exception:                                # noqa: BLE001
        pass
    when = (row.get("updated_at") or "")[:16].replace("T", " ")
    return f"{row['case_id']} — {' '.join(parts)}{n_img} — {when} UTC"


async def list_cases_safe(limit: int = 25) -> list[dict]:
    try:
        return await asyncio.wait_for(db.list_cases(limit), timeout=_STORE_TIMEOUT_S)
    except Exception as e:                            # noqa: BLE001
        log.warning(f"Could not list saved cases (Postgres unavailable): {e}")
        return []


# ───────────────────── memory management (UI delete buttons) ─────────────────────

async def run_safe(coro, timeout: float = _STORE_TIMEOUT_S) -> tuple[bool, Any]:
    """Await a database call; return (True, result) or (False, "ErrorType: message"). Never raises."""
    try:
        return True, await asyncio.wait_for(coro, timeout=timeout)
    except Exception as e:                            # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
