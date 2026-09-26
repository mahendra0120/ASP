"""
A2A_profiler_server.py
─────────────────────────────────────────────────────────────────
Serves ONLY the Profiler agent over A2A, in its own OS process.

This used to be half of A2A_image_delegation_server.py, which ran
Profiler and Forensic in the same process via asyncio.gather(). That
had two problems:
  1. Both agents' ~8B-parameter checkpoints load at *import* time,
     sequentially, before either uvicorn server binds its port — so
     the combined process routinely took longer than main.py's 60s
     `_wait_for_ports` timeout to report ready on EITHER port, even
     though nothing was actually broken.
  2. Because they shared one process (and one GPU's VRAM), a crash
     while loading or running one agent (e.g. a CUDA OOM) could take
     the other down with it, and there was no way to restart one
     without the other.

Splitting them into separate processes fixes both: each agent's model
load now happens in parallel with the other (roughly half the wall
-clock startup time of loading sequentially), each gets its own
per-process memory accounting, and a crash in this process no longer
affects A2A_forensic_server.py (or vice versa). main.py's
launch_servers() starts this alongside A2A_forensic_server.py as two
separate subprocesses.

A2A_image_delegation_server.py (the original combined script) is left
in place for anyone who deliberately wants both agents in a single
process — e.g. a single-GPU box where the OS-level process split
doesn't help because it's the same physical VRAM pool either way.
"""

import asyncio
import logging
import os
import traceback

import uvicorn
from fasta2a.pydantic_ai import _bridge
from fasta2a.pydantic_ai import agent_to_a2a

# Import ONLY the Profiler agent module directly — deliberately not via
# `qwen_agents` package's __init__.py (which eagerly imports the
# Forensic agent too and would load both models in this process,
# defeating the point of splitting them).
from qwen_agents.Profiler_agent.profiler_agent import profile_agent as profiler_agent

PROFILER_PORT = int(os.getenv("PROFILER_AGENT_PORT", "8011"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Same fasta2a-exception-swallowing workaround as the combined script —
# duplicated here (rather than imported) since each server is now its
# own process and needs its own copy of the patch applied.
_original_run_task = _bridge.AgentWorker.run_task


async def _debug_run_task(self, params):
    try:
        return await _original_run_task(self, params)
    except Exception:
        print("=" * 80)
        print("PROFILER AGENT TASK EXCEPTION (captured before fasta2a silently swallows it):")
        traceback.print_exc()
        print("=" * 80)
        raise


_bridge.AgentWorker.run_task = _debug_run_task

profiler_app = agent_to_a2a(
    profiler_agent,
    name="Profiler agent",
    description=(
        "Builds a behavioral profile from the Forensic agent's findings "
        "(received via A2A) plus case images. The only agent with MCP "
        "access, scoped to its own criminal-behavioral-analysis skill file."
    ),
    version="1.0.0",
    url=f"http://localhost:{PROFILER_PORT}",
)


async def run_profiler_server() -> None:
    await uvicorn.Server(uvicorn.Config(
        app=profiler_app,
        host="0.0.0.0",
        port=PROFILER_PORT,
        log_level="debug",
    )).serve()


if __name__ == "__main__":
    asyncio.run(run_profiler_server())
