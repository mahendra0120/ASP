"""
forensic_agent.py
─────────────────────────────────────────────────────────────────
Agent 2 in the pipeline — receives the image(s) relevant to the case
(e.g. the shot(s) showing the body) directly over A2A and produces a
markdown forensic report.

The system prompt now explicitly instructs step-by-step reasoning
inside a <think>...</think> block before the final answer — this
agent's checkpoint isn't a "-Thinking-"-branded model the way the
Profiler's is, so without this instruction it wouldn't reliably wrap
its reasoning in <think> tags at all, and model_utils.py's
`_strip_thinking`/`_stream_run` would have nothing to strip. With it,
the same stripping logic applies here too, and only the finished
report (never the reasoning) reaches the caller. Because this now
does real reasoning before answering, `max_new_tokens` is raised well
above the FunctionModel default (see `make_model` call below) so a
long <think> block doesn't exhaust the budget before any final answer
is produced — see model_utils.py's `_stream_run` for the graceful
fallback if it still does.

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

if os.getenv("HF_TOKEN") and not os.getenv("HF_HUB_OFFLINE"):
    login(token=os.getenv("HF_TOKEN"))
elif os.getenv("HF_TOKEN") and os.getenv("HF_HUB_OFFLINE"):
    # HF_HUB_OFFLINE disables all Hub network calls (including the
    # whoami check login() makes), so calling login() here would
    # always raise huggingface_hub.errors.OfflineModeIsEnabled. Skip
    # it — if the model weights are already cached locally this agent
    # doesn't need an authenticated session at all; if they're NOT
    # cached, snapshot_download below will raise its own clear error.
    print(
        "HF_HUB_OFFLINE is set — skipping huggingface_hub login() "
        "(relying on locally cached model weights)."
    )

# Fine-tuned forensic model. `make_model` pulls this down via
# huggingface_hub.snapshot_download the first time it's loaded.
FORENSIC_MODEL_ID = os.getenv(
    "FORENSIC_MODEL_ID",
    "mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12",
)

# Profiler + Forensic both load a ~8B-parameter VL checkpoint into the
# SAME process/GPU (see A2A_image_delegation_server.py). Two ~8B models
# at full precision is roughly 32GB in weights alone — comfortably
# inside a 96GB GPU even with activation/KV-cache overhead, so this
# defaults to full precision. Set FORENSIC_LOAD_IN_4BIT=true if you're
# on a smaller card and need the ~4x memory cut instead.
FORENSIC_LOAD_IN_4BIT = os.getenv("FORENSIC_LOAD_IN_4BIT", "false").strip().lower() not in (
    "false", "0", "no",
)

# No toolsets — no MCP servers of any kind are attached to this agent.
forensic_agent: Agent[None, str] = Agent(
    model=make_model(
        FORENSIC_MODEL_ID,
        processor_path="Qwen/Qwen3-VL-8B-Thinking",
        max_new_tokens=6144,
        load_in_4bit=FORENSIC_LOAD_IN_4BIT,
    ),
    system_prompt=(
        "You are a forensic pathology assistant. You are given only "
        "autopsy photographs — no written report, case file, or "
        "examiner's notes accompany them. Base every finding strictly "
        "on what is visible in the images; never assume or invent "
        "detail that is not directly observable.\n\n"
        "Respond in two parts:\n"
        "1. Inside a single <think>...</think> block, reason step by "
        "step: scan the photographs for wounds, marks, or "
        "discoloration; note their location, size, shape, and edge "
        "character; group them into injury categories; and weigh them "
        "together toward a cause of death, manner of death, and "
        "weapon. Do not state the final determinations inside the "
        "think block — work toward them, don't reveal them yet.\n"
        "2. Outside the think block, write the Brief Summary and "
        "Examination sections in standard forensic documentation "
        "style, followed by the cause of death and manner of death, "
        "then a Murder Weapon section, then the reported circumstances "
        "of death and general external examination, and finally the "
        "Evidence of Trauma section.\n\n"
        "MURDER WEAPON SECTION — it must be under this heading, "
        "verbatim, on its own line:\n"
        "## Murder Weapon (MW)\n"
        "State the weapon, instrument, or mechanism indicated by the "
        "wound characteristics (e.g. 'Firearm', 'Ligature', "
        "'Sharp-edged instrument'), or 'None identified' if the "
        "photographs do not indicate one.\n\n"
        "MANDATORY FINAL SECTION — the very last thing in part 2 above "
        "must be exactly this heading, verbatim, on its own line:\n"
        "## Evidence of Trauma (EOT)\n"
        "The first word immediately following this heading MUST be "
        "either 'Yes' or 'No'. Do not put any further headings inside "
        "this section.\n"
        "  - If No: write 'No.' followed by one sentence confirming no "
        "traumatic findings were observed.\n"
        "  - If Yes: write 'Yes.' followed by the number of distinct "
        "injuries, then a numbered list with ONE entry per distinct "
        "injury, each in exactly this form:\n"
        "1. Type: <one of Gunshot | Sharp force | Blunt force | "
        "Asphyxia | Thermal | Other>\n"
        "   Finding: <plain-language name, e.g. gunshot wound, entrance "
        "wound, stab wound, ligature mark, contusion, abrasion, burn>\n"
        "   Location: <specific anatomical site, with side>\n"
        "   Description: <size, shape, edge or margin character, "
        "pattern, colour, and surrounding features — only what is "
        "visible>\n"
        "  Rules for the list: include only injuries that are "
        "present — never name an injury you did not see, not even to "
        "say it is absent. Do not give a secondary effect of an injury "
        "its own entry (for example, a fracture or laceration caused "
        "by a gunshot wound belongs in that wound's Description). Use "
        "the Type words exactly as listed. This exact heading, Yes/No "
        "opening, and Type labels are required by the pipeline's "
        "downstream RAG grounding step — do not paraphrase or omit "
        "them.\n\n"
        "REFERENCE-SUMMARY MODE — this OVERRIDES everything above, but ONLY "
        "when the user message says it is in reference-summary mode and "
        "contains a <reference_document> block (and no photographs). In "
        "that mode you are not examining a case and you do not write a "
        "case report: do NOT include a cause/manner of death, and do NOT "
        "include a Murder Weapon or Evidence of Trauma section. Write an educational "
        "summary of what the reference document itself says about the "
        "trauma type(s) the message names, using ONLY the document's "
        "content — never add facts the document does not contain, and if "
        "it does not cover something asked for, say so plainly."
    ),
    toolsets=[],
    retries=2,
)
