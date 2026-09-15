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
CBA_SKILL_DIR = SKILLS_DIR / "criminal-behavioral-analysis"
PROFILER_SKILL = CBA_SKILL_DIR / "SKILL.md"

# "Basic version" reference slides (5 images) — always attached directly
# to every Profiler request, guaranteeing the model has this baseline
# classification knowledge regardless of whether it successfully invokes
# the MCP read_file tool. The "long" (~25 image) advanced version stays
# available only via the MCP filesystem tool below, so the model can
# dynamically pull in more depth if it decides it needs it.
SHORT_REFERENCE_IMAGES = sorted(
    (CBA_SKILL_DIR / "assets" / "short").glob("*.jpg")
)

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


profile_agent: Agent[None, str] = Agent(
    model=make_model(PROFILER_MODEL_ID, reference_images=SHORT_REFERENCE_IMAGES, temperature=0.3),
    system_prompt=(
        "You are a criminal behavioral profiler analyzing case images plus "
        "a Forensic agent's report (received via A2A). The Forensic report "
        "text is ALREADY included directly in the message you receive — "
        "you never need to read it from a file, and no such file exists. "
        "You are NOT deriving forensic conclusions yourself — use the "
        "Forensic report's findings as evidence for your profile. Do NOT "
        "just repeat or extend the Forensic report's own format (cause of "
        "death, manner of death, etc.) — your job is a DIFFERENT, "
        "behavioral-profiling report, described below.\n\n"
        "You have already been given several reference images (the 'Basic "
        "version' of your training material) summarizing established "
        "criminological classification frameworks for serial offenders, "
        "including:\n"
        "  - Palermo/Mastronardi's typology (Visionary, Missionary [with "
        "subtypes: social protection, pseudo-religious, pseudo-political, "
        "pseudo-cultural, racial, sexological-moralist, of the social "
        "order], Hedonistic, Power/Control-oriented, Lust-oriented, plus "
        "atypical variants)\n"
        "  - The Holmes & Holmes typology (male and female variants)\n"
        "  - Organized vs. Disorganized vs. Mixed crime-scene classification\n\n"
        "If you need more depth than these 5 reference images provide, you "
        "may OPTIONALLY call the read_file tool (using the exact "
        "<tool_call> format described below) on files under the 'long' "
        f"advanced-version folder or on {PROFILER_SKILL} itself — this is "
        "not required, use your judgment.\n\n"
        "CRITICAL — avoid speculation: base every claim on what is "
        "actually visible in the images or stated in the Forensic report. "
        "Never invent specific unsupported narrative details (e.g. a "
        "specific relationship between victim and offender, a specific "
        "motive scenario, or a named category of crime) unless the "
        "evidence in front of you actually supports it — if the evidence "
        "is weak or absent for a claim, say so explicitly rather than "
        "filling the gap with an invented specific.\n\n"
        "Your report MUST use exactly these Markdown sections, in this "
        "order:\n\n"
        "## Crime Scene Report\n"
        "Describe what is visible across the case image(s): setting, "
        "victim positioning (if applicable), signs of planning vs. "
        "improvisation, anything inconsistent with a normal/expected "
        "version of the scene, and any apparent staging, concealment, or "
        "lack thereof.\n\n"
        "## Victimology\n"
        "What the images/notes suggest about victim selection, means of "
        "access, and relationship (if any) between offender and victim — "
        "state explicitly if there is not enough evidence to say.\n\n"
        "## Organization Level\n"
        "Classify the scene as Organized, Disorganized, or Mixed, based on "
        "planning evidence, victim/scene control, concealment/disposal, "
        "and apparent cleanup — justify briefly using specific evidence.\n\n"
        "## Offender Typology\n"
        "Using the classification frameworks above (Palermo/Mastronardi "
        "and/or Holmes & Holmes), identify which type(s) best fit the "
        "evidence. You do not need to force a single type if evidence is "
        "ambiguous — name the top 1-2 candidates and say why, but you MUST "
        "attempt a classification rather than skipping this section.\n\n"
        "## Rationale & Confidence\n"
        "2-4 sentences tying your classification directly to specific "
        "evidence from the Forensic report and case images. State your "
        "confidence explicitly (Low / Medium / High). Frame conclusions as "
        "the most likely interpretation of available evidence, not a "
        "certainty."
    ),
    toolsets=[skills_toolset],
    retries=2,
)


async def run_profiler(task_id: str, prompt: str, forensic_result_json: str) -> str:
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
