"""
forensic_agent.py
─────────────────────────────────────────────────────────────────
Agent 2 in the pipeline — receives the image(s) relevant to the case
(e.g. the shot(s) showing the body) directly over A2A and produces a
a markdown forensic report.

IMPORTANT — no MCP here on purpose:
There is no forensic-skill markdown file for this agent to read, so
it is NOT given the filesystem MCP server, and it does not need the
vision-tools/A2A-delegation MCP server either — the caller (main.py,
via the A2A client) is what drives the pipeline and forwards this
agent's result on to the Profiler agent afterwards. This agent's job
is only: image(s) in -> markdown report out.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import login
from pydantic_ai import Agent

from qwen_agents.model_utils import make_model

load_dotenv()

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# Fine-tuned forensic model. `make_model` pulls this down via
# huggingface_hub.snapshot_download the first time it's loaded.
FORENSIC_MODEL_ID = os.getenv(
    "FORENSIC_MODEL_ID",
    "mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12",
)


# No toolsets — no MCP servers of any kind are attached to this agent.
forensic_agent: Agent[None, str] = Agent(
    model=make_model(FORENSIC_MODEL_ID),
    system_prompt=(
        "You are a forensic image-analysis agent. You are given one or more "
        "images relevant to a case (for example, images showing a body) and "
        "any accompanying notes.\n\n"
        "You have no tools and no external skill file — analyze only what is "
        "visible in the image(s) you were given and write your findings as "
        "a clear Markdown report. Be precise and factual.\n\n"
        "CRITICAL — avoid template padding: describe ONLY injuries/findings "
        "you can actually see. Do NOT assume a symmetric or repeated "
        "pattern across body regions (e.g. do not report an injury on "
        "every region just because you found one on some regions). "
        "Examine each area independently: if a region shows no visible "
        "injury or finding, either omit it or explicitly say 'no visible "
        "injury' — never invent one to match a pattern from other "
        "regions. If a detail isn't clearly visible (exact measurements, "
        "wound depth, weapon type, etc.), say so explicitly rather than "
        "guessing a specific-sounding number or detail.\n\n"
        "MANDATORY FINAL SECTION — end every report with exactly this "
        "heading, verbatim:\n"
        "## Evidence of Trauma\n"
        "The first word of this section MUST be either 'Yes' or 'No'.\n"
        "  - If No: write 'No.' followed by one sentence confirming no "
        "traumatic findings were observed.\n"
        "  - If Yes: write 'Yes.' followed by a plain-language list of "
        "each traumatic finding you described above (e.g. gunshot "
        "wound, ligature mark, stab wound, contusion, burn) so it can "
        "be used as a search query against a forensic reference "
        "knowledge base — do not just say 'see above', restate the "
        "findings here in your own words."
    ),
    toolsets=[],
    retries=2,
)
