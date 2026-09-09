import os
import asyncio
import uvicorn
from fasta2a.pydantic_ai import agent_to_a2a
from qwen_agents import profiler_agent, forensic_agent

PROFILER_PORT = int(os.getenv('PROFILER_AGENT_PORT', '8001'))
FORENSIC_PORT = int(os.getenv('FORENSIC_AGENT_PORT', '8002'))

profiler_app = agent_to_a2a(
    profiler_agent,
    name = "Profiler agent",
    description = ("Building a serial killer profile from crime scene images"),
    version = "1.0.0",
    url=f"http://localhost:{PROFILER_PORT}"
)



Forensic_app = agent_to_a2a(
    forensic_agent,
    name = "Forensic agent",
    description = ("Analysing crime scene images"),
    version = "1.0.0",
    url=f"http://localhost:{FORENSIC_PORT}"
)

async def run_profiler_server() -> None:
    await uvicorn.Server(uvicorn.Config(
        app = profiler_app, 
        host = "0.0.0.0",
        port = PROFILER_PORT,
        log_level = "info" )).serve()

async def run_forensic_server() -> None:
    await uvicorn.Server(uvicorn.Config(
        app = Forensic_app,
        host = "0.0.0.0",
        port = FORENSIC_PORT,
        log_level = "info")).serve()

async def run_both() -> None:
    await asyncio.gather(
        run_profiler_server(),
        run_forensic_server()
        )

if __name__ == "__main__":
    asyncio.run(run_both())