"""
a2a_server.py — serve ONE agent over A2A, in its own OS process.

    python a2a_server.py forensic     # port FORENSIC_AGENT_PORT (default 8002)
    python a2a_server.py profiler     # port PROFILER_AGENT_PORT (default 8011)

main.py starts one process per agent. Separate processes load their ~8B-parameter
checkpoints in parallel, give each its own memory accounting (and optionally its own GPU),
and keep a crash in one from taking the other down. Only the selected agent's module is
imported, so only its model is ever loaded here.
"""
import env_defaults  # noqa: F401  (HF_HOME / HF_HUB_OFFLINE / ports; must run before huggingface_hub is imported)

import asyncio
import faulthandler
import logging
import os
import sys
import traceback

faulthandler.enable()

import uvicorn
from fasta2a.pydantic_ai import _bridge, agent_to_a2a

AGENTS = {
    "forensic": {
        "port": ("FORENSIC_AGENT_PORT", "8002"),
        "name": "Forensic agent",
        "description": (
            "Analyzes the case image(s) it's given (e.g. images of a body) and "
            "returns structured forensic findings. No MCP — there is no "
            "forensic skill file, so this agent has no tools."
        ),
    },
    "profiler": {
        "port": ("PROFILER_AGENT_PORT", "8011"),
        "name": "Profiler agent",
        "description": (
            "Builds a behavioral profile from the Forensic agent's findings "
            "(received via A2A) plus case images. The only agent with MCP "
            "access, scoped to its own criminal-behavioral-analysis skill file."
        ),
    },
}


def load_agent(kind: str):
    if kind == "forensic":
        from qwen_agents.Forensic_agent.forensic_agent import forensic_agent
        return forensic_agent
    from qwen_agents.Profiler_agent.profiler_agent import profiler_agent
    return profiler_agent


def main(kind: str) -> None:
    spec = AGENTS[kind]
    port = int(os.getenv(spec["port"][0], spec["port"][1]))

    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # fasta2a swallows exceptions raised inside a task; print them to this process's log.
    original_run_task = _bridge.AgentWorker.run_task

    async def debug_run_task(self, params):
        try:
            return await original_run_task(self, params)
        except Exception:
            print("=" * 80)
            print(f"{kind.upper()} AGENT TASK EXCEPTION (captured before fasta2a silently swallows it):")
            traceback.print_exc()
            print("=" * 80)
            raise

    _bridge.AgentWorker.run_task = debug_run_task

    app = agent_to_a2a(
        load_agent(kind),
        name=spec["name"],
        description=spec["description"],
        version="1.0.0",
        url=f"http://localhost:{port}",
    )
    asyncio.run(uvicorn.Server(uvicorn.Config(
        app=app, host="0.0.0.0", port=port, log_level="debug",
    )).serve())


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in AGENTS:
        sys.exit(f"usage: python a2a_server.py {'|'.join(AGENTS)}")
    main(sys.argv[1])
