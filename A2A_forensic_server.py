"""
A2A_forensic_server.py
─────────────────────────────────────────────────────────────────
Serves ONLY the Forensic agent over A2A, in its own OS process.
Counterpart to A2A_profiler_server.py — see that file's docstring for
why Profiler and Forensic are split into separate processes rather
than run together via asyncio.gather() in one.
"""

import asyncio
import logging
import os
import traceback

import uvicorn
from fasta2a.pydantic_ai import _bridge
from fasta2a.pydantic_ai import agent_to_a2a

# Import ONLY the Forensic agent module directly — see
# A2A_profiler_server.py's comment and qwen_agents/__init__.py for why
# this no longer drags the Profiler agent's model in too.
from qwen_agents.Forensic_agent.forensic_agent import forensic_agent

FORENSIC_PORT = int(os.getenv("FORENSIC_AGENT_PORT", "8002"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_original_run_task = _bridge.AgentWorker.run_task


async def _debug_run_task(self, params):
    try:
        return await _original_run_task(self, params)
    except Exception:
        print("=" * 80)
        print("FORENSIC AGENT TASK EXCEPTION (captured before fasta2a silently swallows it):")
        traceback.print_exc()
        print("=" * 80)
        raise


_bridge.AgentWorker.run_task = _debug_run_task

forensic_app = agent_to_a2a(
    forensic_agent,
    name="Forensic agent",
    description=(
        "Analyzes the case image(s) it's given (e.g. images of a body) and "
        "returns structured forensic findings. No MCP — there is no "
        "forensic skill file, so this agent has no tools."
    ),
    version="1.0.0",
    url=f"http://localhost:{FORENSIC_PORT}",
)


async def run_forensic_server() -> None:
    await uvicorn.Server(uvicorn.Config(
        app=forensic_app,
        host="0.0.0.0",
        port=FORENSIC_PORT,
        log_level="debug",
    )).serve()


if __name__ == "__main__":
    asyncio.run(run_forensic_server())
