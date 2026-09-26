"""
main_gradio.py - Enhanced with Image Upload + Extra Context
"""

import os
import asyncio
import logging
import mimetypes
import re
import socket
import subprocess
import sys
import time
import uuid
from contextlib import aclosing
from pathlib import Path
import shutil

import gradio as gr
from dotenv import load_dotenv

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
import octen_rag

load_dotenv()

# Global variables
mcp_proc = None
profiler_proc = None
forensic_proc = None
REPO_ROOT = Path(__file__).resolve().parent
TEMP_IMAGE_DIR = Path("temp_uploads")
TEMP_IMAGE_DIR.mkdir(exist_ok=True)

MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "9000"))
PROFILER_PORT = int(os.getenv("PROFILER_AGENT_PORT", "8011"))
FORENSIC_PORT = int(os.getenv("FORENSIC_AGENT_PORT", "8002"))

# Sent as the per-request user-turn text alongside the case images. The
# actual task instructions (autopsy-only framing, the <think> block, the
# required output sections) now live in forensic_agent.py's system_prompt
# — this is just the minimal turn needed to hand the images over, since
# the "Main Prompt" textbox has been removed from the UI below.
FORENSIC_USER_MESSAGE = "Analyze the attached autopsy photograph(s) and produce your forensic report."


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

def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Quick TCP connect check — used to confirm a subprocess is actually
    listening rather than trusting that Popen() returning means it's up."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_ports(ports: dict[str, int], timeout: float = 60.0, interval: float = 1.0) -> dict[str, bool]:
    """Poll a set of {label: port} until each is open or `timeout` elapses.
    Returns {label: reached_or_not}."""
    remaining = dict(ports)
    deadline = time.monotonic() + timeout
    ready: dict[str, bool] = {}
    while remaining and time.monotonic() < deadline:
        for label, port in list(remaining.items()):
            if _port_open(port):
                ready[label] = True
                del remaining[label]
        if remaining:
            time.sleep(interval)
    for label in remaining:
        ready[label] = False
    return ready


def _process_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


# Loading a ~8B-parameter vision-language checkpoint (even at 4-bit —
# see qwen_agents/model_utils.py) through unsloth is a lot slower than a
# typical web-service startup, especially on a cold cache or modest
# GPU/CPU. 60s was tuned for "did the process fail to bind a socket",
# not "is a multi-billion-parameter model still loading" — those are
# different questions with different acceptable timeouts. Configurable
# via env var so it can be tuned per-machine without editing code.
SERVER_READY_TIMEOUT = float(os.getenv("SERVER_READY_TIMEOUT_SECONDS", "300"))


def launch_servers():
    """
    Start the MCP tools server, the Profiler A2A server, and the
    Forensic A2A server as three SEPARATE subprocesses, then actually
    verify they came up before reporting success.

    Profiler and Forensic used to be started together as a single
    combined process (A2A_image_delegation_server.py, run via
    asyncio.gather). That process loads both agents' ~8B-parameter
    checkpoints at import time, sequentially, before either uvicorn
    server binds its port — so:
      - Startup routinely exceeded the old 60s timeout for BOTH ports
        at once, even when nothing was actually broken (just still
        loading).
      - A crash loading or running either model (e.g. a CUDA OOM from
        two full-precision ~8B models sharing one GPU) took the other
        agent down with it, since they were one OS process.
    Running A2A_profiler_server.py and A2A_forensic_server.py as two
    separate processes fixes both: their model loads now happen in
    parallel instead of one after the other, and a crash in one is
    isolated to that one port/process — the other keeps serving, and
    the failure is reported against the specific process that died
    rather than a shared, ambiguous one.

    Other fixes over the original version, which just called Popen()
    and immediately reported "started" regardless of what actually
    happened:
      - Uses absolute paths (REPO_ROOT / "<file>.py") and sets `cwd`
        explicitly, so this no longer depends on whatever directory
        the Gradio app process happens to have been launched from.
      - Refuses to double-launch if a server from a previous click is
        still alive.
      - Captures each subprocess's stdout/stderr to its own log file
        (rather than an unread PIPE, which can eventually deadlock a
        subprocess once its output buffer fills) and polls the actual
        ports until they're open (or SERVER_READY_TIMEOUT elapses),
        reporting per-service success/failure instead of a blanket
        "started".
    """
    global mcp_proc, profiler_proc, forensic_proc

    if _process_alive(mcp_proc) and _process_alive(profiler_proc) and _process_alive(forensic_proc):
        return "ℹ️ Servers already running — stop them first if you want to restart."

    mcp_script = REPO_ROOT / "Image_delegation_mcp.py"
    profiler_script = REPO_ROOT / "A2A_profiler_server.py"
    forensic_script = REPO_ROOT / "A2A_forensic_server.py"
    missing = [p.name for p in (mcp_script, profiler_script, forensic_script) if not p.exists()]
    if missing:
        return f"❌ Server launch failed: missing file(s) in {REPO_ROOT}: {', '.join(missing)}"

    log_dir = REPO_ROOT / "server_logs"
    log_dir.mkdir(exist_ok=True)
    mcp_logfile = log_dir / "mcp_server.log"
    profiler_logfile = log_dir / "profiler_server.log"
    forensic_logfile = log_dir / "forensic_server.log"
    mcp_log = open(mcp_logfile, "a")
    profiler_log = open(profiler_logfile, "a")
    forensic_log = open(forensic_logfile, "a")

    try:
        mcp_proc = subprocess.Popen(
            [sys.executable, str(mcp_script)],
            stdout=mcp_log, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        profiler_proc = subprocess.Popen(
            [sys.executable, str(profiler_script)],
            stdout=profiler_log, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        forensic_proc = subprocess.Popen(
            [sys.executable, str(forensic_script)],
            stdout=forensic_log, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
    except Exception as e:
        return f"❌ Server launch failed: {e}"

    ready = _wait_for_ports(
        {"MCP": MCP_PORT, "Profiler": PROFILER_PORT, "Forensic": FORENSIC_PORT},
        timeout=SERVER_READY_TIMEOUT,
    )

    # A process that already exited tells us more than "port never opened"
    # — surface that explicitly, and point at the log file to read why.
    # Each service now has its own process and its own log file, so this
    # correctly attributes a Forensic crash to Forensic alone instead of
    # implicating Profiler too.
    lines = []
    for label, port, proc, logfile in (
        ("MCP", MCP_PORT, mcp_proc, mcp_logfile),
        ("Profiler", PROFILER_PORT, profiler_proc, profiler_logfile),
        ("Forensic", FORENSIC_PORT, forensic_proc, forensic_logfile),
    ):
        if ready.get(label):
            lines.append(f"✅ {label}:{port}")
        elif not _process_alive(proc):
            lines.append(f"❌ {label}:{port} — process exited early (see {logfile})")
        else:
            lines.append(f"⚠️ {label}:{port} — not responding after {int(SERVER_READY_TIMEOUT)}s (see {logfile})")

    return "\n".join(lines)


def stop_servers():
    global mcp_proc, profiler_proc, forensic_proc
    stopped = []
    for name, proc in (("MCP", mcp_proc), ("Profiler", profiler_proc), ("Forensic", forensic_proc)):
        if proc:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
            stopped.append(name)
    mcp_proc = None
    profiler_proc = None
    forensic_proc = None
    return f"🛑 Stopped: {', '.join(stopped)}" if stopped else "ℹ️ No servers were running."


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

async def run_pipeline_async(image_urls: list[str], extra_context: str):
    forensic_text = ""
    profiler_text = ""
    rag_text = ""
    final_text = ""

    try:
        task_id = str(uuid.uuid4())[:8]

        forensic_message = FORENSIC_USER_MESSAGE
        if extra_context and extra_context.strip():
            forensic_message += f"\n\nAdditional Context:\n{extra_context}"

        yield forensic_text, profiler_text, rag_text, final_text, "⏳ Running Forensic agent..."

        # 1) The case image(s) go straight to the Forensic agent over A2A.
        #    It has no MCP/skill access — it just analyzes what's in front
        #    of it. This is GENUINE streaming: each `forensic_text` below
        #    is the real, growing output of the remote model's generation
        #    (via fasta2a==2.0.1's `message/stream` SSE endpoint), not a
        #    replay of an already-finished string. Thinking tokens never
        #    reach here at all — stripped before the model layer ever
        #    yields a delta (model_utils.py's `_stream_run`) and,
        #    redundantly, fasta2a's bridge only ever forwards TextPart
        #    deltas in the first place.
        #
        #    This step has to happen alone, first: both of the next two
        #    steps need its finished text as input.
        async for forensic_text in step_forensic_stream(image_urls, forensic_message):
            yield forensic_text, profiler_text, rag_text, final_text, "⏳ Running Forensic agent..."
        forensic_data = forensic_text
        yield forensic_text, profiler_text, rag_text, final_text, (
            "⏳ Forensic agent complete — running Profiler agent and RAG grounding concurrently..."
        )

        # 2) RAG and the Profiler agent now run AT THE SAME TIME, both
        #    driven off the just-finished Forensic report:
        #      - RAG (a FAISS/embedding similarity search over
        #        forensic_knowledge/, CPU-bound, no model generation) reads
        #        the Forensic report's own "## Evidence of Trauma" section
        #        to decide whether to fire, then fetches grounding passages.
        #      - The Profiler agent (GPU-bound generation, the slow part)
        #        receives the Forensic report + case images and starts
        #        generating immediately — it does NOT wait for RAG this
        #        time, since RAG is normally so much faster than a full
        #        model generation that it finishes well before the
        #        Profiler does anyway. If the Profiler's report happens to
        #        need the RAG passages verbatim quoted, this is the
        #        trade-off: it no longer blocks on them first. If you'd
        #        rather have the Profiler wait and receive RAG grounding
        #        merged into its own prompt, that's the previous
        #        strictly-sequential design and is easy to restore — just
        #        say so.
        #    RAG is wrapped as a single-item async generator so it can be
        #    fanned in alongside the Profiler's real multi-chunk stream —
        #    whichever finishes a step first updates its own panel
        #    immediately rather than waiting on the other.
        rag_task_gen = _one_shot_stream(step_forensic_rag(task_id, forensic_data))
        profiler_gen = step_profiler_stream(image_urls, forensic_data)

        async with aclosing(_run_concurrently({"rag": rag_task_gen, "profiler": profiler_gen})) as stream:
            async for latest, errors in stream:
                if errors:
                    raise RuntimeError(
                        "; ".join(f"{k} step: {e}" for k, e in errors.items())
                    )
                if latest["rag"] != "":
                    rag_data = latest["rag"]
                    rag_text = (
                        format_rag_section(rag_data) if rag_data["triggered"]
                        else "_No evidence-of-trauma language detected in the Forensic report — RAG pipeline not triggered._"
                    )
                if latest["profiler"] != "":
                    profiler_text = latest["profiler"]
                yield forensic_text, profiler_text, rag_text, final_text, (
                    "⏳ Running Profiler agent and RAG grounding concurrently..."
                )
        profiler_data = profiler_text
        yield forensic_text, profiler_text, rag_text, final_text, "⏳ Finalizing report..."

        log.info(f"Combining Forensic + RAG + Profiler results into final profiling report (task_id={task_id})")
        sections = [
            f"# Case {task_id}",
            f"## Forensic Analysis\n\n{forensic_data}",
        ]
        if rag_data["triggered"]:
            sections.append(f"## Forensic Knowledge Base Grounding (RAG)\n\n{rag_text}")
        sections.append(f"## Profiler Analysis\n\n{profiler_data}")
        final_markdown = "\n\n".join(sections) + "\n"

        Path("pipeline_result.md").write_text(final_markdown)

        yield forensic_text, profiler_text, rag_text, final_markdown, "✅ Pipeline completed successfully!"
    except Exception as e:
        error = f"❌ {str(e)}"
        yield error, error, error, error, error


async def _one_shot_stream(coro):
    """Wrap a plain coroutine as a single-item async generator, so a
    one-shot awaitable (like the RAG step) can be fanned in alongside a
    real multi-chunk stream (like the Profiler agent) via
    `_run_concurrently`."""
    yield await coro


async def _run_concurrently(gens: dict):
    """
    Fan-in helper: runs several async generators at once and yields
    (latest, errors) as soon as ANY of them produces a new value, so the
    UI can update whichever panel just changed without waiting on the
    others.

    `gens` maps a short key (e.g. "profiler") to an async generator.
    `latest[key]` starts as `""` and is replaced with whatever that
    generator yields (a growing string for a real stream, or a single
    dict for a one-shot step wrapped via `_one_shot_stream`) — callers
    distinguish "not yet available" from "available" by checking for
    that initial `""` sentinel. Stops once every generator is exhausted;
    if any raises, its exception is reported via `errors` on the same
    yield instead of propagating immediately, so the caller can decide
    whether to keep going or abort.
    """
    queue: asyncio.Queue = asyncio.Queue()
    latest: dict = {k: "" for k in gens}
    pending = set(gens)

    async def _pump(key, gen):
        try:
            async for chunk in gen:
                await queue.put((key, "chunk", chunk))
        except Exception as e:
            await queue.put((key, "error", e))
        finally:
            await queue.put((key, "done", None))

    tasks = [asyncio.create_task(_pump(k, g)) for k, g in gens.items()]
    errors: dict[str, Exception] = {}

    try:
        while pending:
            key, kind, payload = await queue.get()
            if kind == "chunk":
                latest[key] = payload
                yield dict(latest), dict(errors)
            elif kind == "error":
                errors[key] = payload
                pending.discard(key)
                yield dict(latest), dict(errors)
            elif kind == "done":
                pending.discard(key)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


async def run_pipeline(images, image_url, extra_context):
    # Prioritize uploaded images over URL; combine if both given.
    image_urls: list[str] = []

    if images:
        if len(images) > MAX_IMAGES:
            yield tuple([f"❌ Too many images ({len(images)}). Max is {MAX_IMAGES}."] * 5)
            return
        image_urls.extend(save_uploaded_images(images))

    if image_url and image_url.strip():
        image_urls.append(image_url.strip())

    if not image_urls:
        yield tuple(["❌ Please provide at least one image upload or image URL"] * 5)
        return

    if len(image_urls) > MAX_IMAGES:
        yield tuple([f"❌ Too many images ({len(image_urls)}). Max is {MAX_IMAGES}."] * 5)
        return

    log.info(f"Pipeline starting with {len(image_urls)} image(s)")

    try:
        async for update in run_pipeline_async(image_urls, extra_context):
            yield update
    except Exception as e:
        error = f"❌ Critical error: {e}"
        yield tuple([error] * 5)


async def step_forensic_stream(image_urls: list[str], prompt: str):
    """
    Stream the Forensic agent's markdown report as it's generated, via
    fasta2a==2.0.1's `message/stream` SSE endpoint (A2AClient.stream_task).

    Yields the ACCUMULATED text so far on every delta (not just the new
    piece), since that's what the Gradio Markdown box needs to redraw
    each update — callers that want just the tail should diff against
    the previous yield themselves.
    """
    log.info(f"[1/4] Streaming {len(image_urls)} image(s) to Forensic agent")
    image_parts = [
        ImagePart(url=url, media_type=guess_image_media_type(url)) for url in image_urls
    ]
    start = time.monotonic()
    text = ""
    try:
        async for delta in FORENSIC_AGENT.stream_task(
            A2AMessage(parts=[TextPart(prompt), *image_parts])
        ):
            text += delta
            yield text
    except RuntimeError as e:
        elapsed = time.monotonic() - start
        log.info(f"[1/4] Forensic agent stream FAILED after {elapsed:.1f}s: {e}")
        raise RuntimeError(f"Forensic agent failed: {e}") from e
    elapsed = time.monotonic() - start
    log.info(f"[2/4] Forensic agent stream finished in {elapsed:.1f}s ({len(text)} chars) -- forwarding to Profiler via A2A")
    if not text:
        raise RuntimeError("Forensic agent returned an empty result")


async def step_forensic_rag(task_id: str, forensic_report: str) -> dict:
    """
    Check the Forensic agent's report for evidence of trauma and, if
    found, run the Octen/FAISS RAG pipeline against forensic_knowledge/
    to fetch grounding reference material.

    This is where the "RAG pipeline starts up" trigger lives: the Forensic
    agent is deliberately tool-less (see forensic_agent.py), so this check
    happens here in the orchestrator, right after its report comes back
    and before it's forwarded to the Profiler agent.

    The trigger itself is just reading the report's own mandatory
    "## Evidence of Trauma" section (Yes/No) — see
    octen_rag.check_trauma_and_retrieve for the fallback keyword scan
    used only if that section is missing.

    Returns the dict shape from octen_rag.check_trauma_and_retrieve:
        {triggered, source, matched_keywords, query, results, corpus_size}
    """
    log.info(f"[RAG] Reading Forensic report's 'Evidence of Trauma' section (task_id={task_id})")
    # Embedding + FAISS calls are blocking (sentence-transformers), so run
    # off the event loop rather than stalling the async pipeline.
    rag_data = await asyncio.to_thread(octen_rag.check_trauma_and_retrieve, forensic_report)

    if rag_data["triggered"]:
        log.info(
            f"[RAG] Triggered via {rag_data['source']} -> "
            f"fetched {len(rag_data['results'])} chunk(s) from forensic_knowledge/ "
            f"(corpus_size={rag_data['corpus_size']}, task_id={task_id})"
        )
    else:
        log.info(f"[RAG] No evidence of trauma reported — RAG pipeline not triggered (task_id={task_id})")

    return rag_data


_SENTENCE_END_RE = re.compile(r'[.!?"\')\]]\s*$')
_SENTENCE_START_RE = re.compile(r'^[A-Z0-9"\'(#]')


def _mark_excerpt_boundaries(text: str) -> str:
    """
    Prefix/suffix a chunk with an ellipsis if it doesn't land on a
    clean sentence boundary, so a genuine mid-sentence excerpt reads
    as "this continues before/after" rather than looking like the
    text randomly stops. RecursiveCharacterTextSplitter now prefers
    sentence-ending separators (see octen_rag.py) so this should be
    the exception rather than the rule, but long technical sentences
    can still occasionally exceed chunk_size.
    """
    text = text.strip()
    if not text:
        return text
    if not _SENTENCE_START_RE.match(text):
        text = "…" + text
    if not _SENTENCE_END_RE.search(text):
        text = text + "…"
    return text


def format_rag_section(rag_data: dict) -> str:
    """
    Render RAG grounding results as clean Markdown, grouped by source
    file and — within each file — restored to the reference material's
    own original order rather than left in FAISS's raw similarity-score
    order.

    Score order is right for RANKING which chunks to keep, but wrong
    for READING them: top_k=5 chunks can come from several different
    forensic_knowledge/*.md files and land in an arbitrary interleaved
    order (e.g. a gunshot-wounds passage, then an asphyxia passage,
    then back to gunshot-wounds), which reads as disjointed even though
    each individual chunk is a legitimate match. Grouping by source
    (each group heading ordered by that source's best-scoring chunk,
    so the most relevant topic still leads) and sorting each group's
    chunks by their original position in the file fixes that, without
    losing the relevance ranking that got them retrieved in the first
    place.
    """
    if not rag_data["triggered"]:
        return "_Not triggered — Forensic report's 'Evidence of Trauma' section said No._"

    by_source: dict[str, list[dict]] = {}
    source_order: list[str] = []
    for r in rag_data["results"]:
        if not r.get("text", "").strip():
            continue
        source = r.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
            source_order.append(source)  # first appearance = best score for this source
        by_source[source].append(r)

    if not source_order:
        return "_No matching reference material found._"

    sections = []
    for source in source_order:
        chunks = sorted(by_source[source], key=lambda r: r.get("chunk_index", 0))
        title = Path(source).stem.replace("_", " ").replace("-", " ").strip().title()
        body = "\n\n".join(_mark_excerpt_boundaries(c["text"]) for c in chunks)
        sections.append(f"### {title}\n*(source: {source})*\n\n{body}")

    return "\n\n---\n\n".join(sections)


async def step_profiler_stream(image_urls: list[str], forensic_report: str, rag_data: dict | None = None):
    """
    Forward the Forensic agent's result (+ original images, + any RAG
    grounding) to the Profiler agent, streaming its markdown report as
    it's generated (same genuine SSE mechanism as `step_forensic_stream`
    — see that docstring for how thinking tokens and tool-call protocol
    text are kept out of what's yielded here).
    """
    log.info(f"[3/4] Streaming {len(image_urls)} image(s) + Forensic result to Profiler agent")
    combined_prompt = f"Forensic agent report (from A2A):\n{forensic_report}"
    if rag_data and rag_data.get("triggered"):
        trigger_desc = (
            "its Evidence of Trauma section" if rag_data.get("source") == "evidence_of_trauma_section"
            else f"keyword fallback match: {', '.join(rag_data['matched_keywords'])}"
        )
        combined_prompt += (
            f"\n\nGrounded reference material (from forensic_knowledge/, retrieved because "
            f"the Forensic report's {trigger_desc}):\n"
            f"{format_rag_section(rag_data)}"
        )
    image_parts = [
        ImagePart(url=url, media_type=guess_image_media_type(url)) for url in image_urls
    ]
    start = time.monotonic()
    text = ""
    try:
        async for delta in PROFILER_AGENT.stream_task(
            A2AMessage(parts=[TextPart(combined_prompt), *image_parts])
        ):
            text += delta
            yield text
    except RuntimeError as e:
        elapsed = time.monotonic() - start
        log.info(f"[3/4] Profiler agent stream FAILED after {elapsed:.1f}s: {e}")
        raise RuntimeError(f"Profiler agent failed: {e}") from e
    elapsed = time.monotonic() - start
    log.info(f"[4/4] Profiler agent stream finished in {elapsed:.1f}s ({len(text)} chars) -- combining into final profile")
    if not text:
        raise RuntimeError("Profiler agent returned an empty result")

# =============================================
# Gradio UI
# =============================================

with gr.Blocks(title="A2A Multi-Agent Pipeline") as demo:
    gr.Markdown("# 🧠 A2A 2-Agent Vision Pipeline\nForensic → Profiler | Image Upload + Extra Context")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Server Control")
            server_status = gr.Textbox(label="Status", value="Not started", lines=4, interactive=False)
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
            profiler_out = gr.Markdown(label="🔍 Profiler Agent (MCP: skills/ only) — runs concurrently with RAG")

    gr.Markdown(
        "### 📚 RAG Grounding (triggered by the Forensic report's own 'Evidence of "
        "Trauma' verdict — runs concurrently with the Profiler agent, not before it)"
    )
    rag_out = gr.Markdown(label="Forensic Knowledge Base Grounding")

    with gr.Accordion("📄 Full Result", open=False):
        full_out = gr.Markdown(label="Complete Report")

    status_out = gr.Textbox(label="Pipeline Status", interactive=False)

    # Run button
    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, url_input, extra_context],
        outputs=[forensic_out, profiler_out, rag_out, full_out, status_out],
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
