# NOTE: main.py's launch_servers() no longer launches THIS script by
# default — it now launches A2A_profiler_server.py and
# A2A_forensic_server.py as two separate processes instead, so a crash
# in one agent can't take the other down, and their model loads happen
# in parallel rather than sequentially in one process (see those files'
# docstrings for the full reasoning). This combined script is kept
# around for anyone who deliberately wants both agents in a single
# process (e.g. simpler process management when VRAM isn't a concern).

import os
import asyncio
import uvicorn
import logging
import traceback
from fasta2a.pydantic_ai import _bridge
from fasta2a.pydantic_ai import agent_to_a2a
from qwen_agents import profiler_agent, forensic_agent

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
        print("AGENT TASK EXCEPTION (captured before fasta2a silently swallows it):")
        traceback.print_exc()
        print("=" * 80)
        raise

_bridge.AgentWorker.run_task = _debug_run_task

PROFILER_PORT = int(os.getenv('PROFILER_AGENT_PORT', '8011'))
FORENSIC_PORT = int(os.getenv('FORENSIC_AGENT_PORT', '8002'))

profiler_app = agent_to_a2a(
    profiler_agent,
    name = "Profiler agent",
    description = (
        "Builds a behavioral profile from the Forensic agent's findings "
        "(received via A2A) plus case images. The only agent with MCP "
        "access, scoped to its own criminal-behavioral-analysis skill file."
    ),
    version = "1.0.0",
    url=f"http://localhost:{PROFILER_PORT}"
)


forensic_app = agent_to_a2a(
    forensic_agent,
    name = "Forensic agent",
    description = (
        "Analyzes the case image(s) it's given (e.g. images of a body) and "
        "returns structured forensic findings. No MCP — there is no "
        "forensic skill file, so this agent has no tools."
    ),
    version = "1.0.0",
    url=f"http://localhost:{FORENSIC_PORT}"
)

async def run_profiler_server() -> None:
    await uvicorn.Server(uvicorn.Config(
        app = profiler_app, 
        host = "0.0.0.0",
        port = PROFILER_PORT,
        log_level = "debug" )).serve()

async def run_forensic_server() -> None:
    await uvicorn.Server(uvicorn.Config(
        app = forensic_app,
        host = "0.0.0.0",
        port = FORENSIC_PORT,
        log_level = "debug")).serve()

async def run_both() -> None:
    await asyncio.gather(
        run_profiler_server(),
        run_forensic_server()
        )

if __name__ == "__main__":
    asyncio.run(run_both())
