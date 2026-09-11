"""
profiler_agent.py
─────────────────────────────────────────────────────────────────
Agent 1 in the pipeline — takes the ForensicResult produced by
Agent 2 (received over A2A) plus the original case images/notes and
builds the behavioral profile.

This is the ONLY agent in the pipeline that gets an MCP server, and
it is scoped to exactly one thing: read-only filesystem access to
THIS agent's own skills markdown folder
(`.agents/skills/criminal-behavioral-analysis/`). The Forensic agent
has no skill file, so it gets no MCP at all (see forensic_agent.py).
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import login
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport

from qwen_agents.model_utils import make_model

load_dotenv()

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# make_model pulls this down via huggingface_hub.snapshot_download the
# first time it's loaded.
PROFILER_MODEL_ID = os.getenv(
    "PROFILER_MODEL_ID", "Kizzington/Qwen3-VL-8B-Thinking-heretic"
)

# Sandbox root: only this agent's skills folder, nothing else on disk.
SKILLS_DIR = Path(__file__).parent / ".agents" / "skills"
PROFILER_SKILL = SKILLS_DIR / "criminal-behavioral-analysis" / "SKILL.md"

# Standard filesystem MCP server — read_file, list_directory, get_file_info.
# Sandboxed to SKILLS_DIR so the Profiler agent can only ever read its own
# skill markdown, never the Forensic agent's files (it has none) or
# anything else on the filesystem. Requires Node.js >= 18 for npx.
#
# `init_timeout` is set generously because the FIRST time this runs on a
# fresh pod/volume, `npx -y @modelcontextprotocol/server-filesystem` has
# to download that package from the npm registry before it can respond
# to the MCP handshake at all — the default timeout is too short for a
# cold npx install and causes a spurious "Failed to initialize server
# session" error on first run. Subsequent runs are fast since npm caches
# the package (put the npm cache on the network volume too, e.g. via
# `npm config set cache /workspace/.npm-cache`, so this stays fast across
# pod restarts).
#
# NOTE: this repo was originally written against `MCPServerStdio` /
# `MCPServerSSE`, which pydantic-ai removed in 2.0 in favor of a single
# `MCPToolset` built on FastMCP's `Client` — passing a pre-built `Client`
# (rather than a bare transport) is what lets us set `init_timeout`.
# `fasta2a[pydantic-ai]` (used to serve these agents over A2A — see
# A2A_image_delegation_server.py) hard-requires pydantic-ai-slim>=2.40.0,
# so this repo needs post-2.0 pydantic-ai either way.
skills_client = FastMCPClient(
    StdioTransport(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(SKILLS_DIR)],
        env={**os.environ},
    ),
    init_timeout=60,
)
skills_toolset = MCPToolset(skills_client)


class BoundingBox(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    position: str = Field(description="Rough position in frame: top-left, center …")


class ProfileAnalysis(BaseModel):
    """Output of Agent 1 — Profiler Agent."""

    task_id: str
    scene_description: str
    objects: list[BoundingBox] = Field(default_factory=list)
    text_content: Optional[str] = None
    dominant_colors: list[str] = Field(default_factory=list)
    context_category: str = Field(
        description="nature | urban | product | document | person | other"
    )
    mood: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    forensic_result: Optional[str] = Field(
        None,
        description=(
            "The ForensicResult JSON received from the Forensic agent over "
            "A2A, which this profile is built on top of."
        ),
    )


profile_agent: Agent[None, ProfileAnalysis] = Agent(
    model=make_model(PROFILER_MODEL_ID),
    output_type=ProfileAnalysis,
    system_prompt=(
        "You are a criminal behavioral profiler.\n\n"
        f"BEFORE doing anything else, call read_file('{PROFILER_SKILL}') "
        "to load your operating instructions, then follow every step exactly.\n\n"
        "You will be given the ForensicResult produced by the Forensic agent "
        "(received via A2A) plus the case images/notes. Use the forensic "
        "findings as evidence for your profile — do not re-derive forensic "
        "conclusions yourself.\n\n"
        "Tools available:\n"
        "  • Filesystem (sandboxed to your own skills folder) — read_file, "
        "list_directory, get_file_info"
    ),
    toolsets=[skills_toolset],
    retries=2,
)


async def run_profiler(task_id: str, prompt: str, forensic_result_json: str) -> ProfileAnalysis:
    """
    Convenience entry point for direct (non-A2A) use.

    `forensic_result_json` is the ForensicResult produced by the Forensic
    agent (already returned via A2A by the time this is called).
    """
    full_prompt = (
        f"[task_id={task_id}]\n\n"
        f"Forensic agent result:\n{forensic_result_json}\n\n"
        f"{prompt}"
    )
    result = await profile_agent.run(full_prompt)
    return result.output
