"""
profiler_agent.py
─────────────────────────────────────────────────────────────────
The Profiler agent: takes the Forensic agent's report (received over A2A) plus the
original case images and builds the behavioral profile.

It is the ONLY agent with an MCP server, scoped to exactly one thing: read-only access to
its own skills folder (`.agents/skills/criminal-behavioral-analysis/`). The Forensic agent
has no skill file, so it gets no MCP at all.
"""

import os
from pathlib import Path

from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from qwen_agents.model_utils import make_model

# make_model downloads this via huggingface_hub on first load.
PROFILER_MODEL_ID = os.getenv(
    "PROFILER_MODEL_ID", "Kizzington/Qwen3-VL-8B-Thinking-heretic"
)

# Same reasoning as FORENSIC_LOAD_IN_4BIT in forensic_agent.py.
PROFILER_LOAD_IN_4BIT = os.getenv("PROFILER_LOAD_IN_4BIT", "false").strip().lower() not in (
    "false", "0", "no",
)

# Sandbox root: only this agent's skills folder, nothing else on disk.
SKILLS_DIR = Path(__file__).parent / ".agents" / "skills"
CBA_SKILL_DIR = SKILLS_DIR / "criminal-behavioral-analysis"
PROFILER_SKILL = CBA_SKILL_DIR / "SKILL.md"

# "Basic version" reference slides (5 images), attached to every Profiler request so the
# model always has the baseline classification frameworks even if it never calls a tool.
# The "long" (~25 image) advanced version is reachable only through the MCP filesystem tool.
SHORT_REFERENCE_IMAGES = sorted(
    (CBA_SKILL_DIR / "assets" / "short").glob("*.jpg")
)

# Standard filesystem MCP server (read_file, list_directory, ...), sandboxed to SKILLS_DIR so
# the agent can only read its own skill markdown. Requires Node.js >= 18 for npx.
#
# `init_timeout` is generous because on a fresh pod `npx -y @modelcontextprotocol/server-filesystem`
# first has to download the package before it can answer the MCP handshake (setup_env.sh
# pre-installs it and keeps the npm cache on the network volume, so later runs are fast).
#
# pydantic-ai 2.x replaced MCPServerStdio/MCPServerSSE with `MCPToolset` built on FastMCP's
# `Client`; passing a pre-built Client (not a bare transport) is what allows `init_timeout`.
skills_client = FastMCPClient(
    StdioTransport(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(SKILLS_DIR)],
        env={**os.environ},
    ),
    init_timeout=60,
)
_ALLOWED_SKILL_TOOLS = {"read_file", "list_directory"}
skills_toolset = MCPToolset(skills_client).filtered(
    lambda ctx, tool_def: tool_def.name in _ALLOWED_SKILL_TOOLS
)


def _load_skill_text(path: Path) -> str:
    """Read a skill file, failing loudly if it is missing. SKILL.md and OUTPUT.md are loaded
    straight into the system prompt (below) so the agent ALWAYS has them, instead of
    depending on the model choosing to call `read_file` — which it sometimes skipped."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Profiler agent's required skill file is missing: {path} "
            "— this file defines the agent's core workflow and output "
            "format; the agent cannot run correctly without it."
        ) from e


_SKILL_MD = _load_skill_text(PROFILER_SKILL)
_OUTPUT_MD = _load_skill_text(CBA_SKILL_DIR / "references" / "OUTPUT.md")


profiler_agent: Agent[None, str] = Agent(
    model=make_model(
        PROFILER_MODEL_ID,
        processor_path="Qwen/Qwen3-VL-8B-Thinking",
        reference_images=SHORT_REFERENCE_IMAGES,
        temperature=0.3,
        # The system prompt embeds the full skill file + output spec, and the model reasons in
        # <think> over it, so the default budget was exhausted before any answer was written.
        max_new_tokens=12288,
        load_in_4bit=PROFILER_LOAD_IN_4BIT,
    ),
    system_prompt=(
        "You are a criminal behavioral profiler analyzing case images plus "
        "a Forensic agent's report (received via A2A).\n\n"
        "You are NOT deriving forensic conclusions yourself — use the "
        "Forensic report's findings as evidence for your profile. Do NOT "
        "just repeat or extend the Forensic report's own format (cause of "
        "death, manner of death, etc.) — your job is a DIFFERENT, "
        "behavioral-profiling report, as described in your skill file "
        "below.\n\n"
        "Note: some runs of this pipeline supply the Forensic report "
        "concurrently with RAG grounding lookup rather than waiting for "
        "it, so grounded reference material from forensic_knowledge/ may "
        "or may not be present below — proceed with whatever you're given "
        "and don't assume its absence means none exists.\n\n"
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
        "--- Your skill file (criminal-behavioral-analysis/SKILL.md) ---\n"
        "This defines your workflow. It is included here directly so you "
        "always have it regardless of whether you call any tool — you may "
        "still optionally use read_file to go deeper into references/ or "
        "assets/long/ as it describes below.\n\n"
        f"TOOL RULES: your only tools are read_file and list_directory, and they can ONLY "
        f"read inside {SKILLS_DIR}. Use absolute paths under that folder, for example "
        f"{CBA_SKILL_DIR}/references/REFERENCE.md. The case images are ALREADY attached "
        "to this message: never try to open, locate or re-read them with a tool, and "
        "never invent file paths such as /data/.... If a tool returns an error, do NOT "
        "repeat the same call; continue with what you already have.\n\n"
        f"{_SKILL_MD}\n\n"
        "--- Your required output format (references/OUTPUT.md), verbatim ---\n"
        "You MUST use exactly these Markdown sections, in this order:\n\n"
        f"{_OUTPUT_MD}\n\n"
        "---\n\n"
        "CRITICAL — avoid speculation: base every claim on what is "
        "actually visible in the images or stated in the Forensic report. "
        "Never invent specific unsupported narrative details (e.g. a "
        "specific relationship between victim and offender, a specific "
        "motive scenario, or a named category of crime) unless the "
        "evidence in front of you actually supports it — if the evidence "
        "is weak or absent for a claim, say so explicitly rather than "
        "filling the gap with an invented specific."
    ),
    toolsets=[skills_toolset],
    retries=4,
)
