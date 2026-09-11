"""
forensic_agent.py
─────────────────────────────────────────────────────────────────
Agent 2 in the pipeline — receives the image(s) relevant to the case
(e.g. the shot(s) showing the body) directly over A2A and produces a
structured ForensicResult.

IMPORTANT — no MCP here on purpose:
There is no forensic-skill markdown file for this agent to read, so
it is NOT given the filesystem MCP server, and it does not need the
vision-tools/A2A-delegation MCP server either — the caller (main.py,
via the A2A client) is what drives the pipeline and forwards this
agent's result on to the Profiler agent afterwards. This agent's job
is only: image(s) in -> ForensicResult out.
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from huggingface_hub import login
from pydantic import BaseModel, Field
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


class ForensicResult(BaseModel):
    """Output of Agent 2 — Forensic Agent."""

    task_id: str
    executive_summary: str
    key_insights: list[str] = Field(min_length=1)
    recommendations: list[str] = Field(min_length=1)
    seo_tags: list[str] = Field(default_factory=list)
    accessibility_description: str
    quality_score: float = Field(ge=0.0, le=10.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# No toolsets — no MCP servers of any kind are attached to this agent.
forensic_agent: Agent[None, ForensicResult] = Agent(
    model=make_model(FORENSIC_MODEL_ID),
    output_type=ForensicResult,
    system_prompt=(
        "You are a forensic image-analysis agent. You are given one or more "
        "images relevant to a case (for example, images showing a body) and "
        "any accompanying notes.\n\n"
        "You have no tools and no external skill file — analyze only what is "
        "visible in the image(s) you were given and produce your findings "
        "directly as the required structured output. Be precise, factual, "
        "and avoid speculation beyond what the image evidence supports."
    ),
    toolsets=[],
    retries=2,
)


async def run_forensic_analysis(task_id: str, prompt: str) -> ForensicResult:
    """
    Convenience entry point for direct (non-A2A) use, e.g. from tests.

    `prompt` should already describe/embed the image(s) to analyze —
    when served over A2A (see A2A_image_delegation_server.py), the
    image parts and text sent by the caller are what the agent sees.
    """
    result = await forensic_agent.run(f"[task_id={task_id}]\n\n{prompt}")
    return result.output
