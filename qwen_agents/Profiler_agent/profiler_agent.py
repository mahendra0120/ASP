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
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport

from qwen_agents.model_utils import make_model

load_dotenv()

if os.getenv("HF_TOKEN") and not os.getenv("HF_HUB_OFFLINE"):
    login(token=os.getenv("HF_TOKEN"))
elif os.getenv("HF_TOKEN") and os.getenv("HF_HUB_OFFLINE"):
    # See forensic_agent.py for why this is skipped rather than
    # calling login() (which always raises OfflineModeIsEnabled here).
    print(
        "HF_HUB_OFFLINE is set — skipping huggingface_hub login() "
        "(relying on locally cached model weights)."
    )

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


def _load_skill_text(path: Path) -> str:
    """
    Read a skill markdown file from disk, failing loudly if it's
    missing rather than silently running the agent without it.

    Why this exists: the Profiler's actual workflow/output-format
    instructions live in SKILL.md and references/OUTPUT.md — this
    function loads their live content directly into the system
    prompt at agent-construction time (below), so the agent is
    GUARANTEED to have them on every single run. Previously this
    agent's system_prompt was a hand-copied duplicate of that
    content that could silently drift out of sync with the actual
    skill files, and the *live* files were only ever offered to the
    model as an optional `read_file` MCP call — which the model is
    free to skip, and evidently sometimes does skip in practice.
    Reading them here removes that failure mode entirely: there is
    no tool-call dependency for the mandatory instructions anymore.
    """
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


profile_agent: Agent[None, str] = Agent(
    model=make_model(
        PROFILER_MODEL_ID,
        reference_images=SHORT_REFERENCE_IMAGES,
        temperature=0.3,
        # This agent's system prompt embeds the full skill file +
        # output-format spec + criminological typology background —
        # thousands of tokens before generation even starts — plus it
        # does real <think> reasoning over that material. The
        # FunctionModel default max_new_tokens (see model_utils.py's
        # make_model) was getting exhausted while still inside <think>,
        # which model_utils.py's _stream_run now handles gracefully
        # either way, but a bigger budget is the actual fix: it lets
        # this agent reliably finish reasoning AND write its answer.
        max_new_tokens=12288,
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
    retries=2,
)
