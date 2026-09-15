"""
main_gradio.py - Enhanced with Image Upload + Extra Context
"""

import os
import asyncio
import json
import logging
import mimetypes
import subprocess
import sys
import time
import uuid
from pathlib import Path
import tempfile
import shutil

import gradio as gr
from dotenv import load_dotenv
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

from A2A_image_delegation_client import (
    A2AMessage, TextPart, ImagePart,
    PROFILER_AGENT, FORENSIC_AGENT,
)

load_dotenv()
console = Console()

# Global variables
mcp_proc = None
a2a_proc = None
TEMP_IMAGE_DIR = Path("temp_uploads")
TEMP_IMAGE_DIR.mkdir(exist_ok=True)


def guess_image_media_type(url_or_path: str) -> str:
    """
    Best-effort MIME type detection from a filename/URL extension.
    Falls back to image/jpeg if nothing recognizable is found, since
    the A2A file part needs a concrete media_type for the receiving
    agent to correctly classify it as an image.
    """
    mime_type, _ = mimetypes.guess_type(url_or_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "image/jpeg"


# =============================================
# Server Management (unchanged)
# =============================================

def launch_servers():
    global mcp_proc, a2a_proc
    try:
        mcp_proc = subprocess.Popen([sys.executable, "Image_delegation_mcp.py"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        a2a_proc = subprocess.Popen([sys.executable, "A2A_image_delegation_server.py"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return f"✅ Servers started (MCP:9000 | Profiler:{os.getenv('PROFILER_AGENT_PORT', '8001')} | Forensic:{os.getenv('FORENSIC_AGENT_PORT', '8002')})"
    except Exception as e:
        return f"❌ Server launch failed: {e}"


def stop_servers():
    global mcp_proc, a2a_proc
    for proc in (mcp_proc, a2a_proc):
        if proc:
            proc.terminate()
            try: proc.wait(5)
            except: proc.kill()
    return "🛑 Servers stopped."


# =============================================
# Helper: Save uploaded image and return URL
# =============================================

MAX_IMAGES = 15


def save_uploaded_images(images) -> list[str]:
    """
    Save one or more uploaded images and return their local URLs.
    `images` is whatever gr.File(file_count="multiple") hands back —
    a list of tempfile-like objects (or plain path strings depending
    on Gradio version). Returns [] if nothing was uploaded.
    """
    if not images:
        return []

    urls = []
    for image in images:
        temp_path = TEMP_IMAGE_DIR / f"{uuid.uuid4()}_{Path(image.name).name if hasattr(image, 'name') else 'image.jpg'}"
        shutil.copy(image.name if hasattr(image, 'name') else image, temp_path)
        urls.append(f"http://127.0.0.1:7860/gradio_api/file={temp_path}")
        log.info(f"Image received and saved -> {temp_path.name}")

    return urls


# =============================================
# Pipeline
# =============================================

async def run_pipeline_async(image_urls: list[str], prompt: str, extra_context: str):
    try:
        task_id = str(uuid.uuid4())[:8]

        full_prompt = prompt
        if extra_context and extra_context.strip():
            full_prompt += f"\n\nAdditional Context:\n{extra_context}"

        # 1) The case image(s) (e.g. showing the body) go straight to the
        #    Forensic agent over A2A. It has no MCP/skill access — it just
        #    analyzes what's in front of it.
        forensic_data = await step_forensic(image_urls, full_prompt)

        # 2) The Forensic agent's result is then forwarded over A2A to the
        #    Profiler agent, which is the only agent with MCP access
        #    (to its own criminal-behavioral-analysis skill markdown).
        profiler_data = await step_profiler(image_urls, full_prompt, forensic_data)

        log.info(f"Combining Forensic + Profiler results into final profiling report (task_id={task_id})")
        final_markdown = (
            f"# Case {task_id}\n\n"
            f"## Forensic Analysis\n\n{forensic_data}\n\n"
            f"## Profiler Analysis\n\n{profiler_data}\n"
        )

        Path("pipeline_result.md").write_text(final_markdown)

        return (
            forensic_data,
            profiler_data,
            final_markdown,
            "✅ Pipeline completed successfully!",
        )
    except Exception as e:
        error = f"❌ {str(e)}"
        return error, error, error, error


def run_pipeline(images, image_url, prompt, extra_context):
    # Prioritize uploaded images over URL; combine if both given.
    image_urls: list[str] = []

    if images:
        if len(images) > MAX_IMAGES:
            return [f"❌ Too many images ({len(images)}). Max is {MAX_IMAGES}."] * 4
        image_urls.extend(save_uploaded_images(images))

    if image_url and image_url.strip():
        image_urls.append(image_url.strip())

    if not image_urls:
        return ["❌ Please provide at least one image upload or image URL"] * 4

    if len(image_urls) > MAX_IMAGES:
        return [f"❌ Too many images ({len(image_urls)}). Max is {MAX_IMAGES}."] * 4

    log.info(f"Pipeline starting with {len(image_urls)} image(s)")

    try:
        return asyncio.run(run_pipeline_async(image_urls, prompt, extra_context))
    except Exception as e:
        error = f"❌ Critical error: {e}"
        return [error] * 4


async def step_forensic(image_urls: list[str], prompt: str):
    """Send the relevant case image(s) straight to the Forensic agent via A2A."""
    log.info(f"[1/4] Sending {len(image_urls)} image(s) to Forensic agent")
    image_parts = [
        ImagePart(url=url, media_type=guess_image_media_type(url)) for url in image_urls
    ]
    start = time.monotonic()
    task = await FORENSIC_AGENT.send_task(
        A2AMessage(parts=[TextPart(prompt), *image_parts])
    )
    elapsed = time.monotonic() - start
    if task.failed:
        log.info(f"[1/4] Forensic agent FAILED after {elapsed:.1f}s: {task.error}")
        raise RuntimeError(f"Forensic agent failed: {task.error}")
    result = task.output()
    log.info(f"[2/4] Forensic agent completed in {elapsed:.1f}s ({len(result)} chars) -- forwarding to Profiler via A2A")
    return result


async def step_profiler(image_urls: list[str], prompt: str, forensic_report: str):
    """Forward the Forensic agent's result (+ original image/notes) to the Profiler agent."""
    log.info(f"[3/4] Sending {len(image_urls)} image(s) + Forensic result to Profiler agent")
    combined_prompt = (
        f"{prompt}\n\nForensic agent report (from A2A):\n"
        f"{forensic_report}"
    )
    image_parts = [
        ImagePart(url=url, media_type=guess_image_media_type(url)) for url in image_urls
    ]
    start = time.monotonic()
    task = await PROFILER_AGENT.send_task(
        A2AMessage(parts=[TextPart(combined_prompt), *image_parts])
    )
    elapsed = time.monotonic() - start
    if task.failed:
        log.info(f"[3/4] Profiler agent FAILED after {elapsed:.1f}s: {task.error}")
        raise RuntimeError(f"Profiler agent failed: {task.error}")
    result = task.output()
    log.info(f"[4/4] Profiler agent completed in {elapsed:.1f}s ({len(result)} chars) -- combining into final profile")
    return result

# =============================================
# Gradio UI
# =============================================

with gr.Blocks(title="A2A Multi-Agent Pipeline") as demo:
    gr.Markdown("# 🧠 A2A 2-Agent Vision Pipeline\nForensic → Profiler | Image Upload + Extra Context")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Server Control")
            server_status = gr.Textbox(label="Status", value="Not started", lines=2, interactive=False)
            with gr.Row():
                gr.Button("🚀 Start Servers", variant="primary").click(launch_servers, outputs=server_status)
                gr.Button("⏹ Stop Servers").click(stop_servers, outputs=server_status)

        with gr.Column(scale=2):
            gr.Markdown("### Input")
            with gr.Tabs():
                with gr.Tab("📤 Upload Images"):
                    image_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Photos (max 15)",
                    )
                with gr.Tab("🔗 Image URL"):
                    url_input = gr.Textbox(label="Image URL", placeholder="https://...", lines=1)

            prompt = gr.Textbox(
                label="Main Prompt",
                lines=3,
                value="Perform a comprehensive visual analysis..."
            )
            
            extra_context = gr.Textbox(
                label="Additional Context (Optional)",
                placeholder="This photo was taken in a forest during autumn. The person is a biologist studying mushrooms.",
                lines=3
            )
            
    run_btn = gr.Button("▶ Run Pipeline", variant="primary", size="large")

    with gr.Row():
        with gr.Column():
            forensic_out = gr.Markdown(label="🔬 Forensic Agent (no MCP)")
        with gr.Column():
            profiler_out = gr.Markdown(label="🔍 Profiler Agent (MCP: skills/ only)")

    with gr.Accordion("📄 Full Result", open=False):
        full_out = gr.Markdown(label="Complete Report")

    status_out = gr.Textbox(label="Pipeline Status", interactive=False)

    # Run button
    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, url_input, prompt, extra_context],
        outputs=[forensic_out, profiler_out, full_out, status_out],
    )

    gr.Markdown("**Tip:** You can upload an image **and** add extra context text.")

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(TEMP_IMAGE_DIR.resolve())],
    )
