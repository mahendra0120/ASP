"""
forensic_agent.py
─────────────────────────────────────────────────────────────────
The Forensic agent: receives the case image(s) over A2A and produces a markdown forensic
report. Its system prompt makes it reason inside a <think>...</think> block before
answering (this checkpoint isn't a "-Thinking-" model, so it wouldn't reliably do so
unprompted); model_utils strips that block, so only the finished report reaches the caller.
`max_new_tokens` is raised well above the default so a long <think> doesn't exhaust the
budget before the answer.

No MCP on purpose: there is no forensic skill file for it to read. main.py drives the
pipeline and forwards this agent's report to the Profiler itself.
"""

import os

from pydantic_ai import Agent

from qwen_agents.model_utils import make_model

# Fine-tuned forensic model; make_model downloads it via huggingface_hub on first load.
FORENSIC_MODEL_ID = os.getenv(
    "FORENSIC_MODEL_ID",
    "mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12",
)

# Defaults to full precision (two ~8B models are ~32GB of weights, fine on a 96GB GPU).
# Set FORENSIC_LOAD_IN_4BIT=true on a smaller card for a ~4x memory cut.
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
