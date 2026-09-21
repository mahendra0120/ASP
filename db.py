"""
db.py
─────────────────────────────────────────────────────────────────
PostgreSQL-backed persistence for agent results.

Replaces the old in-memory `_store: dict[str, Any] = {}` placeholder
that used to live in Image_delegation_mcp.py. Results are keyed by
(task_id, agent_id) — task_id is the uuid4 generated per-run in
main.py — and are now durable across process restarts and shared
correctly between the MCP server and A2A agent processes (the old
dict was process-local, so it silently failed to share state once
the agents ran as separate subprocesses).

Configure via the DATABASE_URL env var, e.g.:
    postgresql://asp_user:asp_password@localhost:5432/asp

A single asyncpg pool is created lazily on first use and reused for
the lifetime of the process.
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://asp_user:asp_password@localhost:5432/asp",
)

_pool: Optional[asyncpg.Pool] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_results (
    task_id     TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    result      TEXT NOT NULL,
    stored_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (task_id, agent_id)
);
"""


async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool, creating it (and the schema) on first call."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute(_SCHEMA)
    return _pool


async def close_pool() -> None:
    """Close the pool cleanly, e.g. on server shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def store_result(task_id: str, agent_id: str, result: str) -> dict[str, Any]:
    """Persist (or overwrite) an agent result under (task_id, agent_id)."""
    pool = await get_pool()
    stored_at = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_results (task_id, agent_id, result, stored_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (task_id, agent_id)
            DO UPDATE SET result = EXCLUDED.result, stored_at = EXCLUDED.stored_at
            """,
            task_id, agent_id, result, stored_at,
        )
    return {"stored": True, "key": f"{task_id}:{agent_id}"}


async def get_result(task_id: str, agent_id: str) -> dict[str, Any]:
    """Retrieve a stored agent result by task_id + agent_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT result, stored_at, agent_id
            FROM agent_results
            WHERE task_id = $1 AND agent_id = $2
            """,
            task_id, agent_id,
        )
    if row is None:
        return {"found": False}
    return {
        "found": True,
        "result": row["result"],
        "stored_at": row["stored_at"].isoformat(),
        "agent_id": row["agent_id"],
    }
