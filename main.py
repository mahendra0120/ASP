"""
main_gradio.py - Enhanced with Image Upload + Extra Context
"""

import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path
import tempfile
import shutil

import gradio as gr
from dotenv import load_dotenv
from rich.console import Console

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
        return "✅ Servers started (MCP:9000 | Profiler:8001 | Forensic:8002)"
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

def save_uploaded_image(image) -> str:
    """Save uploaded image and return a local URL that the agents can access"""
    if image is None:
        return None
    
    # Create unique filename
    temp_path = TEMP_IMAGE_DIR / f"{uuid.uuid4()}_{Path(image.name).name if hasattr(image, 'name') else 'image.jpg'}"
    shutil.copy(image.name if hasattr(image, 'name') else image, temp_path)
    
    # Return local URL (Gradio serves static files from /file=...)
    return f"http://127.0.0.1:7860/file={temp_path}"


# =============================================
# Pipeline
# =============================================

async def run_pipeline_async(image_url: str, prompt: str, extra_context: str):
    try:
        task_id = str(uuid.uuid4())[:8]

        full_prompt = prompt
        if extra_context and extra_context.strip():
            full_prompt += f"\n\nAdditional Context:\n{extra_context}"

        # 1) The case image(s) (e.g. showing the body) go straight to the
        #    Forensic agent over A2A. It has no MCP/skill access — it just
        #    analyzes what's in front of it.
        forensic_data = await step_forensic(image_url, full_prompt)

        # 2) The Forensic agent's result is then forwarded over A2A to the
        #    Profiler agent, which is the only agent with MCP access
        #    (to its own criminal-behavioral-analysis skill markdown).
        profiler_data = await step_profiler(image_url, full_prompt, forensic_data)

        final_result = {
            "task_id": task_id,
            "forensic": forensic_data,
            "profiler": profiler_data,
        }

        Path("pipeline_result.json").write_text(json.dumps(final_result, indent=2))

        return (
            json.dumps(forensic_data, indent=2),
            json.dumps(profiler_data, indent=2),
            json.dumps(final_result, indent=2),
            "✅ Pipeline completed successfully!",
        )
    except Exception as e:
        error = f"❌ {str(e)}"
        return error, error, error, error


def run_pipeline(image, image_url, prompt, extra_context):
    # Prioritize uploaded image over URL
    final_url = None
    if image is not None:
        final_url = save_uploaded_image(image)
    elif image_url:
        final_url = image_url.strip()

    if not final_url:
        return ["❌ Please provide either an image upload or image URL"] * 4

    try:
        return asyncio.run(run_pipeline_async(final_url, prompt, extra_context))
    except Exception as e:
        error = f"❌ Critical error: {e}"
        return [error] * 4


async def step_forensic(image_url: str, prompt: str):
    """Send the relevant case image(s) straight to the Forensic agent via A2A."""
    task = await FORENSIC_AGENT.send_task(
        A2AMessage(parts=[TextPart(prompt), ImagePart(url=image_url)])
    )
    if task.failed:
        raise RuntimeError(f"Forensic agent failed: {task.error}")
    return task.json_output()


async def step_profiler(image_url: str, prompt: str, forensic_json: dict):
    """Forward the Forensic agent's result (+ original image/notes) to the Profiler agent."""
    combined_prompt = (
        f"{prompt}\n\nForensic agent result (from A2A):\n"
        f"{json.dumps(forensic_json, indent=2)}"
    )
    task = await PROFILER_AGENT.send_task(
        A2AMessage(parts=[TextPart(combined_prompt), ImagePart(url=image_url)])
    )
    if task.failed:
        raise RuntimeError(f"Profiler agent failed: {task.error}")
    return task.json_output()

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
                with gr.Tab("📤 Upload Image"):
                    image_input = gr.Image(type="filepath", label="Upload Photo")
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
            forensic_out = gr.Code(label="🔬 Forensic Agent (no MCP)", language="json", lines=12)
        with gr.Column():
            profiler_out = gr.Code(label="🔍 Profiler Agent (MCP: skills/ only)", language="json", lines=12)

    with gr.Accordion("📄 Full Result", open=False):
        full_out = gr.Code(label="Complete JSON", language="json", lines=15)

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
    )