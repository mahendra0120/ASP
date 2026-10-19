"""
main.py — Gradio UI + orchestration for the Forensic -> (RAG || Profiler) pipeline.

Each agent runs as its own A2A server subprocess (a2a_server.py). Every pipeline step is
checkpointed to Postgres (pipeline_state.py / db.py) so a crashed run can be resumed.
"""
import env_defaults  # noqa: F401  (HF_HOME / HF_HUB_OFFLINE / ports; must run before huggingface_hub is imported)

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import aclosing
from pathlib import Path

import gradio as gr
import httpx

import example_images
import octen_rag
import pipeline_state as pstate
from a2a_client import (
    A2AMessage, TextPart, ImagePart,
    PROFILER_AGENT, FORENSIC_AGENT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "server_logs"
TEMP_IMAGE_DIR = Path("temp_uploads")
TEMP_IMAGE_DIR.mkdir(exist_ok=True)

PROFILER_PORT = int(os.getenv("PROFILER_AGENT_PORT", "8011"))
FORENSIC_PORT = int(os.getenv("FORENSIC_AGENT_PORT", "8002"))
AGENT_PORTS = {"Profiler": PROFILER_PORT, "Forensic": FORENSIC_PORT}

# Loading an ~8B-parameter checkpoint is much slower than a typical web-service start,
# especially on a cold cache, so wait long for the ports. Tunable per machine.
SERVER_READY_TIMEOUT = float(os.getenv("SERVER_READY_TIMEOUT_SECONDS", "300"))

MAX_IMAGES = 15

# Per-request user text sent with the case images; the real task instructions live in
# forensic_agent.py's system prompt.
FORENSIC_USER_MESSAGE = "Analyze the attached autopsy photograph(s) and produce your forensic report."


# =============================================
# GPU placement
# =============================================
# Each agent server loads its model with device_map="cuda", i.e. onto whichever GPU that
# PROCESS can see, so placing an agent = setting CUDA_VISIBLE_DEVICES for its subprocess:
#   * 2+ GPUs -> Forensic on the first, Profiler on the second (true parallelism).
#   * 1 GPU / none detected -> nothing set; both share it.
# Overrides: FORENSIC_GPU / PROFILER_GPU pin an agent to a GPU index or UUID (set both);
# ASP_MULTI_GPU=false forces the shared-GPU layout.

def _detect_gpus() -> list[str]:
    """IDs (valid CUDA_VISIBLE_DEVICES tokens) of the GPUs this app may use.
    Honours an existing CUDA_VISIBLE_DEVICES, else asks nvidia-smi, else torch."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [t.strip() for t in visible.split(",") if t.strip()]
        return [] if not ids or ids[0] == "-1" else ids
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        ids = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
        if ids:
            return ids
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:                      # noqa: BLE001 - no torch / no CUDA
        return []


def plan_gpu_placement() -> dict[str, str | None]:
    """{"Forensic": gpu_or_None, "Profiler": gpu_or_None}. None = leave CUDA_VISIBLE_DEVICES alone."""
    forensic = os.getenv("FORENSIC_GPU", "").strip() or None
    profiler = os.getenv("PROFILER_GPU", "").strip() or None
    multi_ok = os.getenv("ASP_MULTI_GPU", "true").strip().lower() not in ("false", "0", "no")
    if forensic is None and profiler is None and multi_ok:
        gpus = _detect_gpus()
        if len(gpus) >= 2:
            forensic, profiler = gpus[0], gpus[1]
    return {"Forensic": forensic, "Profiler": profiler}


_gpu_placement: dict[str, str | None] | None = None


def _agent_env(name: str) -> dict | None:
    """Subprocess environment for an agent server (None = inherit as-is)."""
    global _gpu_placement
    if _gpu_placement is None:
        _gpu_placement = plan_gpu_placement()
    gpu = _gpu_placement.get(name)
    return {**os.environ, "CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else None


def _gpu_tag(name: str) -> str:
    gpu = (_gpu_placement or {}).get(name)
    return f" (GPU {gpu})" if gpu is not None else ""


# =============================================
# Server management
# =============================================

agent_procs: dict[str, subprocess.Popen] = {}


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """TCP connect check: confirms a subprocess is actually listening, not just spawned."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_ports(ports: dict[str, int], timeout: float = 60.0, interval: float = 1.0) -> dict[str, bool]:
    """Poll {label: port} until each is open or `timeout` elapses. Returns {label: reached}."""
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


def _watch_exit(name: str, proc: subprocess.Popen) -> None:
    """Record how an agent server exited (e.g. SIGKILL from the OOM killer) in server_logs/exit_codes.log."""
    def _wait():
        rc = proc.wait()
        try:
            sig = signal.Signals(-rc).name if rc < 0 else ""
        except ValueError:
            sig = ""
        msg = (f"[server-watch] {name} exited: returncode={rc} {sig} "
               f"at {time.strftime('%H:%M:%S')} (expected if you pressed Stop)")
        print(msg, flush=True)
        with open(LOG_DIR / "exit_codes.log", "a") as f:
            f.write(msg + "\n")

    threading.Thread(target=_wait, daemon=True).start()


def _spawn_agent(name: str) -> Path:
    """Start one agent server subprocess. Its output goes to server_logs/<name>_server.log;
    the previous run's log is kept as .prev (it may hold the crash trace). Returns the log path."""
    LOG_DIR.mkdir(exist_ok=True)
    logfile = LOG_DIR / f"{name.lower()}_server.log"
    if logfile.exists():
        os.replace(logfile, logfile.with_name(logfile.name + ".prev"))
    agent_procs[name] = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "a2a_server.py"), name.lower()],
        stdout=open(logfile, "w"), stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT), env=_agent_env(name),
    )
    _watch_exit(name, agent_procs[name])
    return logfile


def launch_servers():
    """
    Start the Profiler and Forensic A2A servers as two SEPARATE subprocesses, then verify
    they came up (per-service success/failure, not a blanket "started").

    Separate processes load their models in parallel and isolate crashes (e.g. a CUDA OOM
    in one leaves the other serving). Output is captured to per-service log files rather
    than an unread PIPE, which can eventually deadlock a subprocess once its buffer fills.
    """
    global _gpu_placement

    if len(agent_procs) == len(AGENT_PORTS) and all(_process_alive(p) for p in agent_procs.values()):
        return "ℹ️ Servers already running — stop them first if you want to restart."

    # Invalidate the cached RAG index so the first query after startup reflects the current
    # corpus and chunking logic. Set RAG_REBUILD_ON_STARTUP=false to keep it across restarts.
    if os.getenv("RAG_REBUILD_ON_STARTUP", "true").strip().lower() not in ("false", "0", "no"):
        if octen_rag.clear_index_cache():
            log.info("Cleared cached RAG index (.octen_faiss_index/) - it will rebuild on first use.")

    _gpu_placement = plan_gpu_placement()
    if _gpu_placement["Forensic"] is not None or _gpu_placement["Profiler"] is not None:
        log.info(
            f"GPU placement: Forensic -> GPU {_gpu_placement['Forensic']}, "
            f"Profiler -> GPU {_gpu_placement['Profiler']}"
        )
    else:
        log.info("GPU placement: not split (one GPU, none detected, or ASP_MULTI_GPU=false) - agents share the default GPU")

    try:
        logfiles = {name: _spawn_agent(name) for name in AGENT_PORTS}
    except Exception as e:
        return f"❌ Server launch failed: {e}"

    ready = _wait_for_ports(AGENT_PORTS, timeout=SERVER_READY_TIMEOUT)

    # A process that already exited tells us more than "port never opened": say so and
    # point at the log that explains why.
    lines = []
    for label, port in AGENT_PORTS.items():
        if ready.get(label):
            lines.append(f"✅ {label}:{port}{_gpu_tag(label)}")
        elif not _process_alive(agent_procs[label]):
            lines.append(f"❌ {label}:{port} — process exited early (see {logfiles[label]})")
        else:
            lines.append(f"⚠️ {label}:{port} — not responding after {int(SERVER_READY_TIMEOUT)}s (see {logfiles[label]})")

    return "\n".join(lines)


def stop_servers():
    stopped = []
    for name, proc in agent_procs.items():
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stopped.append(name)
    agent_procs.clear()
    return f"🛑 Stopped: {', '.join(stopped)}" if stopped else "ℹ️ No servers were running."


async def _ensure_agent_up(name: str) -> None:
    """Crash recovery: restart ONLY the named agent server if it isn't listening
    (launch_servers() would relaunch both and orphan the healthy one)."""
    port = AGENT_PORTS[name]
    if _port_open(port):
        return
    log.warning(f"{name} agent server is not listening on :{port} - restarting just that server")
    logfile = await asyncio.to_thread(_spawn_agent, name)
    ready = await asyncio.to_thread(_wait_for_ports, {name: port}, SERVER_READY_TIMEOUT)
    log.info(f"✅ {name}:{port} restarted" if ready.get(name) else f"⚠️ {name}:{port} did not come up (see {logfile})")


# =============================================
# Uploaded images
# =============================================

def guess_image_media_type(url_or_path: str) -> str:
    """MIME type from the file extension, defaulting to image/jpeg: the A2A file part needs
    a concrete media_type for the receiving agent to classify it as an image."""
    mime_type, _ = mimetypes.guess_type(url_or_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "image/jpeg"


def save_uploaded_images(images) -> list[str]:
    """Copy uploaded images (whatever gr.File hands back: tempfile-likes or path strings)
    into temp_uploads/ and return their local URLs. [] if nothing was uploaded."""
    urls = []
    for image in images or []:
        temp_path = TEMP_IMAGE_DIR / f"{uuid.uuid4()}_{Path(image.name).name if hasattr(image, 'name') else 'image.jpg'}"
        shutil.copy(image.name if hasattr(image, 'name') else image, temp_path)
        urls.append(f"http://127.0.0.1:7860/gradio_api/file={temp_path}")
        log.info(f"Image received and saved -> {temp_path.name}")
    return urls


def _image_parts(image_urls: list[str]) -> list[ImagePart]:
    return [ImagePart(url=url, media_type=guess_image_media_type(url)) for url in image_urls]


# =============================================
# Pipeline
# =============================================

async def run_pipeline_async(image_urls: list[str], extra_context: str, case_id: str, resume_mode: str = "continue"):
    """
    Forensic -> (RAG || Profiler) with every step checkpointed to Postgres
    (see pipeline_state.py). resume_mode:
      "continue" - reuse finished steps, continue partial reports where they stopped
      "asis"     - reuse finished steps, accept partial reports as they are
      "fresh"    - discard saved progress for this case and start over
    Yields (forensic, profiler, reference summaries, final report, status) for the UI.
    """
    forensic_text = profiler_text = rag_text = edu_text = ""
    store = pstate.CaseStore(case_id)

    def panels(msg: str, final: str = ""):
        """The 5 UI outputs. The reference-summaries panel shows the summaries once they exist."""
        return forensic_text, profiler_text, edu_text or rag_text, final, msg + store.status_suffix()

    try:
        forensic_message = FORENSIC_USER_MESSAGE
        if extra_context and extra_context.strip():
            forensic_message += f"\n\nAdditional Context:\n{extra_context}"

        saved: dict = {}
        if resume_mode == "fresh":
            await store.delete_all()
        else:
            saved = await store.load_all()
        await store.save("meta", pstate.meta_json(image_urls, extra_context), "complete")

        summary = ", ".join(f"{k}={v['status']}" for k, v in saved.items() if k != "meta")
        if summary:
            log.info(f"[case {case_id}] resuming with saved progress: {summary} (mode={resume_mode})")
        yield panels(
            f"♻️ Case {case_id}: found saved progress ({summary}) - resuming..." if summary
            else f"⏳ Case {case_id}: starting..."
        )

        # 1) Forensic. Finished -> reused; partial -> continued (or accepted as-is);
        #    otherwise generated. Streams the agent's real output, checkpointed as it goes.
        sf = saved.get("forensic")
        forensic_done = False
        prefix = ""
        if sf and sf["status"] == "complete":
            forensic_text, forensic_done = sf["content"], True
            yield panels("♻️ Forensic report restored from the database - skipping that step")
        elif sf and sf["status"] == "partial":
            if resume_mode == "asis":
                forensic_text, forensic_done = sf["content"], True
                await store.save("forensic", forensic_text, "complete")
                await store.invalidate(["rag", "education", "profiler", "final"])
                yield panels("⏭ Using the saved partial Forensic report as-is")
            else:
                prefix = pstate.trim_partial(sf["content"])
                forensic_text = prefix
        forensic_reused = bool(sf and sf["status"] == "complete")

        if not forensic_done:
            # Anything computed from an older Forensic result is now stale.
            await store.invalidate(["rag", "education", "profiler", "final"])
            if not _port_open(FORENSIC_PORT):
                yield panels("🔧 Forensic server is down - restarting it (the model reloads on first use)...")
                await _ensure_agent_up("Forensic")
            prompt = forensic_message
            if prefix:
                prompt += pstate.continuation_block("forensic report", prefix)
            running = "⏳ Continuing the interrupted Forensic report..." if prefix else "⏳ Running Forensic agent..."
            yield panels(running)
            async for forensic_text in pstate.checkpointed(
                step_forensic_stream(image_urls, prompt), store, "forensic", prefix=prefix
            ):
                yield panels(running)
        yield panels("⏳ Forensic agent complete - running Profiler agent and reference summaries concurrently...")

        # 2) RAG and the Profiler run AT THE SAME TIME off the finished Forensic report.
        #    Saved RAG/Profiler results are only reused if the Forensic report they were
        #    built from was itself reused unchanged.
        saved_rag = saved.get("rag") if forensic_reused else None
        sp = saved.get("profiler") if forensic_reused else None
        if sp:
            bad = profiler_output_problem(sp["content"])
            if bad:
                log.warning(f"[case {case_id}] saved Profiler result is invalid ({bad}) - discarding it")
                await store.save("profiler_failed", sp["content"], "partial")
                await store.invalidate(["profiler", "final"])
                sp = None
                yield panels(
                    f"♻️ Saved Profiler result was invalid ({bad}) - discarded; "
                    "re-running only the Profiler (Forensic, RAG and summary are reused)..."
                )

        saved_edu = saved.get("education") if forensic_reused else None
        rag_future: asyncio.Future = asyncio.get_running_loop().create_future()

        async def _rag_step():
            try:
                data = None
                if saved_rag and saved_rag["status"] == "complete":
                    try:
                        data = json.loads(saved_rag["content"])
                    except ValueError:
                        pass
                if data is None:
                    data = await step_forensic_rag(case_id, forensic_text)
                    await store.save("rag", pstate.dumps(data), "complete")
            except asyncio.CancelledError:
                rag_future.cancel()
                raise
            except Exception as e:
                rag_future.set_exception(e)
                rag_future.exception()      # mark retrieved; the RAG step's own error is reported by the caller
                raise
            rag_future.set_result(data)
            return data

        async def _edu_step():
            """Forensic agent is idle while the Profiler runs: load the FULL reference
            file for each trauma type in the report's Evidence of Trauma section and
            stream one "Reference Summary" section per type. The files are chosen
            straight from the finished report, so streaming starts immediately; RAG's
            retrieval is awaited only when the report names no recognisable type.
            Optional extra - its failure never aborts the run."""
            try:
                if (saved_edu and saved_edu["status"] == "complete"
                        and "## Reference Summary" in saved_edu["content"]):
                    yield saved_edu["content"]
                    return
                section = octen_rag.extract_evidence_of_trauma_section(forensic_text)
                rag = {"triggered": True, "query": section or ""}
                sources = (
                    octen_rag.select_trauma_sources(section)
                    if section and octen_rag.section_says_yes(section) else []
                )
                if not sources:
                    try:
                        rag = await rag_future
                    except asyncio.CancelledError:
                        if rag_future.cancelled():      # RAG step failed/cancelled: reported by the rag step
                            return
                        raise
                    except Exception:                   # noqa: BLE001
                        return
                    sources = education_sources(rag)
                if not sources:
                    return
                if not _port_open(FORENSIC_PORT):
                    await _ensure_agent_up("Forensic")
                async for text in pstate.checkpointed(
                    step_forensic_education_multi_stream(rag, sources), store, "education"
                ):
                    yield text
            except Exception as e:           # noqa: BLE001
                log.warning(f"[EDU] Reference summaries skipped: {e}")
                yield f"## Reference Summary\n\n_Summary unavailable: {e}_"

        async def _saved_text(t: str):
            yield t

        profiler_source = None
        p_prefix = ""
        if sp and sp["status"] == "complete":
            profiler_source = _saved_text(sp["content"])
        elif sp and sp["status"] == "partial":
            if resume_mode == "asis":
                await store.save("profiler", sp["content"], "complete")
                profiler_source = _saved_text(sp["content"])
            else:
                p_prefix = pstate.trim_partial(sp["content"])
                profiler_text = p_prefix
        if profiler_source is None:
            if not _port_open(PROFILER_PORT):
                yield panels("🔧 Profiler server is down - restarting it (the model reloads on first use)...")
                await _ensure_agent_up("Profiler")
            cont = pstate.continuation_block("behavioral profile", p_prefix) if p_prefix else None
            profiler_source = quarantine_bad_profiler(
                pstate.checkpointed(
                    guard_profiler_output(step_profiler_stream(image_urls, forensic_text, cont)),
                    store, "profiler", prefix=p_prefix,
                ),
                store,
            )

        rag_text = "_Checking the Forensic report's Evidence of Trauma section..._"
        async with aclosing(_run_concurrently(
            {"rag": _one_shot_stream(_rag_step()), "profiler": profiler_source, "edu": _edu_step()}
        )) as stream:
            async for latest, errors in stream:
                if errors:
                    raise RuntimeError(
                        "; ".join(f"{k} step: {e}" for k, e in errors.items())
                    )
                if latest["rag"] != "" and not latest["rag"]["triggered"]:
                    rag_text = "_No evidence of trauma reported in the Forensic report - no reference summaries._"
                if latest["profiler"] != "":
                    profiler_text = latest["profiler"]
                if latest["edu"] != "":
                    edu_text = latest["edu"]
                yield panels("⏳ Running Profiler agent and reference summaries concurrently...")
        yield panels("⏳ Finalizing report...")

        log.info(f"Combining Forensic + reference summaries + Profiler results into final profiling report (task_id={case_id})")
        sections = [
            f"# Case {case_id}",
            f"## Forensic Analysis\n\n{forensic_text}",
        ]
        if edu_text:
            sections.append(edu_text)     # one "## Reference Summary - <Trauma>" section per trauma type
        sections.append(f"## Profiler Analysis\n\n{profiler_text}")
        final_markdown = "\n\n".join(sections) + "\n"

        Path("pipeline_result.md").write_text(final_markdown)
        await store.save("final", final_markdown, "complete")

        yield panels("✅ Pipeline completed successfully!", final_markdown)
    except Exception as e:
        # Keep every panel's text rather than wiping a finished Forensic report.
        # Progress is already in Postgres.
        await asyncio.sleep(0.3)   # cancelled tasks (e.g. a Profiler stopped by a RAG error) finish flushing their partial output
        hint = (
            f" - progress is saved as case {case_id}; press ▶ Run Pipeline again "
            f"(resume mode) to continue from where it stopped."
            if store.ok else " - the database is unavailable, so this progress was NOT saved."
        )
        yield forensic_text, profiler_text, rag_text, "", f"❌ {e}{hint}"


async def _one_shot_stream(coro):
    """Wrap a coroutine as a single-item async generator so `_run_concurrently` can fan it
    in alongside real multi-chunk streams."""
    yield await coro


async def _run_concurrently(gens: dict):
    """
    Run several async generators at once and yield (latest, errors) whenever ANY of them
    produces a value, so the UI can update whichever panel just changed.

    `latest[key]` starts as "" and is replaced by whatever that generator yields (a growing
    string for a stream, a dict for a one-shot step); callers tell "not yet available" by
    that "" sentinel. Stops when every generator is exhausted. A generator's exception is
    reported via `errors` on the next yield instead of propagating, so the caller decides
    whether to abort.
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


async def run_pipeline(images, image_url, extra_context, resume_mode="continue", saved_case=None):
    image_urls: list[str] = []

    if saved_case:
        # Resume a saved case without re-uploading: images + context come from the database.
        meta = await pstate.CaseStore(saved_case).load_meta()
        if not meta or not meta.get("image_urls"):
            yield tuple([f"❌ Saved case {saved_case} was not found in the database."] * 5)
            return
        image_urls = list(meta["image_urls"])
        extra_context = meta.get("extra_context", "")
        missing = [u for u in image_urls if not pstate.image_available(u)]
        if missing:
            yield tuple([
                f"❌ Saved case {saved_case}: {len(missing)} image file(s) no longer exist on disk. "
                f"Re-upload the original images with the same context and run again - the case id "
                f"will match and your saved progress will be used."
            ] * 5)
            return
        case_id = saved_case
    else:
        # Uploaded images and an image URL can be combined.
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
        case_id = pstate.case_id_for(image_urls, extra_context)

    log.info(f"Pipeline starting with {len(image_urls)} image(s) (case {case_id}, mode={resume_mode})")

    try:
        async for update in run_pipeline_async(image_urls, extra_context, case_id, resume_mode or "continue"):
            yield update
    except Exception as e:
        error = f"❌ Critical error: {e}"
        yield tuple([error] * 5)


# =============================================
# Agent steps (each streams the ACCUMULATED text so far)
# =============================================

async def step_forensic_stream(image_urls: list[str], prompt: str):
    """Stream the Forensic agent's markdown report via the A2A `message/stream` endpoint."""
    log.info(f"[1/4] Streaming {len(image_urls)} image(s) to Forensic agent")
    start = time.monotonic()
    text = ""
    try:
        async for delta in FORENSIC_AGENT.stream_task(
            A2AMessage(parts=[TextPart(prompt), *_image_parts(image_urls)])
        ):
            text += delta
            yield text
    except (RuntimeError, httpx.HTTPError) as e:
        elapsed = time.monotonic() - start
        log.info(f"[1/4] Forensic agent stream FAILED after {elapsed:.1f}s: {e}")
        raise RuntimeError(f"Forensic agent failed: {e}") from e
    elapsed = time.monotonic() - start
    log.info(f"[2/4] Forensic agent stream finished in {elapsed:.1f}s ({len(text)} chars) -- forwarding to Profiler via A2A")
    if not text:
        raise RuntimeError("Forensic agent returned an empty result")


async def step_forensic_rag(case_id: str, forensic_report: str) -> dict:
    """
    Check the Forensic report's "Evidence of Trauma" section and, if it says Yes, run the
    Octen/FAISS retrieval over forensic_knowledge/. The Forensic agent is deliberately
    tool-less, so this check lives here in the orchestrator (see
    octen_rag.check_trauma_and_retrieve for the result shape and the keyword fallback).
    """
    log.info(f"[RAG] Reading Forensic report's 'Evidence of Trauma' section (task_id={case_id})")
    # Embedding + FAISS calls block, so run them off the event loop.
    rag_data = await asyncio.to_thread(octen_rag.check_trauma_and_retrieve, forensic_report)

    if rag_data["triggered"]:
        log.info(
            f"[RAG] Triggered via {rag_data['source']} -> "
            f"fetched {len(rag_data['results'])} chunk(s) from forensic_knowledge/ "
            f"(corpus_size={rag_data['corpus_size']}, task_id={case_id})"
        )
    else:
        log.info(f"[RAG] No evidence of trauma reported — RAG pipeline not triggered (task_id={case_id})")

    return rag_data


# Optional cap on how much of a reference file is sent to the Forensic agent (characters).
# 0 = send the COMPLETE file. Gunshot_wounds.md is ~119k chars (~30k tokens); set e.g.
# RAG_FULL_FILE_MAX_CHARS=60000 if the model runs out of memory/context on it.
RAG_FULL_FILE_MAX_CHARS = int(os.getenv("RAG_FULL_FILE_MAX_CHARS", "0"))


def education_sources(rag_data: dict) -> list[str]:
    """The forensic_knowledge/*.md files for the reference summaries: ONE per distinct
    trauma type in the report. If no type was recognised (or the case was saved before
    this existed), the single best similarity match. [] if RAG didn't trigger."""
    if not rag_data.get("triggered"):
        return []
    sources = [
        s for s in dict.fromkeys(rag_data.get("trauma_sources") or [])
        if octen_rag.load_full_source_text(s) is not None
    ]
    if sources:
        return sources
    single = rag_data.get("primary_source") or octen_rag.select_primary_source(
        rag_data.get("results", [])
    )
    return [single] if single else []


def education_title(source: str) -> str:
    return Path(source).stem.replace("_", " ").replace("-", " ").strip().title()


def build_education_prompt(rag_data: dict, source: str, file_text: str, n_files: int = 1) -> str:
    findings = (rag_data.get("query") or "").strip()
    if len(findings) > 2000:
        findings = findings[:2000] + "…"
    truncation_note = ""
    if RAG_FULL_FILE_MAX_CHARS and len(file_text) > RAG_FULL_FILE_MAX_CHARS:
        file_text = file_text[:RAG_FULL_FILE_MAX_CHARS]
        truncation_note = (
            " NOTE: the file below was cut short to fit the model's context, "
            "so mention that your summary covers only the first part of it."
        )
    scope = (
        "the trauma type(s) named above"
        if n_files <= 1 else
        "those of the trauma types above that THIS file covers (the other types "
        "have their own reference files - do not discuss them, just skip them)"
    )
    return (
        "You are now in reference-summary mode (educational). No photographs "
        "are attached and you are NOT examining a case.\n\n"
        "The case report listed these trauma findings (context only - do not "
        f"analyse or re-examine the case):\n{findings}\n\n"
        f'Below is the complete text of the reference file "{source}". Write an '
        f"educational summary of what this file says about {scope}: key definitions, how the injury is produced, its "
        "characteristic features, how it is told apart from look-alikes, and "
        "any caveats or limitations the file mentions. Use short bold labels and "
        "bullet points, and do not use '#' markdown headings. Use only the file's content; if it does not cover one "
        f"of these, say so.{truncation_note}\n\n"
        f'<reference_document source="{source}">\n{file_text}\n</reference_document>'
    )


async def step_forensic_education_stream(rag_data: dict, source: str, n_files: int = 1):
    """
    Load the COMPLETE text of one forensic_knowledge/*.md file into the (now idle)
    Forensic agent and stream back its educational summary of what that file says about
    the reported trauma. Text-only: no images.
    """
    file_text = await asyncio.to_thread(octen_rag.load_full_source_text, source)
    if not file_text or not file_text.strip():
        raise RuntimeError(f"Reference file {source!r} not found in forensic_knowledge/")

    prompt = build_education_prompt(rag_data, source, file_text, n_files)
    log.info(
        f"[EDU] Loading full {source} ({len(file_text):,} chars) into the Forensic agent "
        f"for an educational summary"
    )
    start = time.monotonic()
    text = ""
    try:
        async for delta in FORENSIC_AGENT.stream_task(A2AMessage(parts=[TextPart(prompt)])):
            text += delta
            yield text
    except (RuntimeError, httpx.HTTPError) as e:
        raise RuntimeError(f"Forensic agent (educational summary) failed: {e}") from e
    log.info(f"[EDU] Educational summary finished in {time.monotonic() - start:.1f}s ({len(text)} chars)")
    if not text:
        raise RuntimeError("Forensic agent returned an empty educational summary")


async def step_forensic_education_multi_stream(rag_data: dict, sources: list[str]):
    """One educational summary per trauma file, one after another (the prompt never holds
    more than one file). Yields the combined text so far, one
    "## Reference Summary - <Trauma>" section per file."""
    done = ""
    for source in sources:
        header = f"## Reference Summary - {education_title(source)}\n\n"
        text = ""
        yield done + header + "_Loading the full reference file into the Forensic agent..._"
        async for text in step_forensic_education_stream(rag_data, source, len(sources)):
            yield done + header + text
        done += header + text + "\n\n"


async def step_profiler_stream(image_urls: list[str], forensic_report: str, continuation: str | None = None):
    """Forward the Forensic report (+ the original images) to the Profiler agent and
    stream its markdown report, like `step_forensic_stream`."""
    log.info(f"[3/4] Streaming {len(image_urls)} image(s) + Forensic result to Profiler agent")
    combined_prompt = f"Forensic agent report (from A2A):\n{forensic_report}"
    if continuation:
        combined_prompt += continuation
    start = time.monotonic()
    text = ""
    try:
        async for delta in PROFILER_AGENT.stream_task(
            A2AMessage(parts=[TextPart(combined_prompt), *_image_parts(image_urls)])
        ):
            text += delta
            yield text
    except (RuntimeError, httpx.HTTPError) as e:
        elapsed = time.monotonic() - start
        log.info(f"[3/4] Profiler agent stream FAILED after {elapsed:.1f}s: {e}")
        raise RuntimeError(f"Profiler agent failed: {e}") from e
    elapsed = time.monotonic() - start
    log.info(f"[4/4] Profiler agent stream finished in {elapsed:.1f}s ({len(text)} chars) -- combining into final profile")
    if not text:
        raise RuntimeError("Profiler agent returned an empty result")


class BadProfilerOutput(RuntimeError):
    """The Profiler 'finished' but what it produced is not a report."""
    def __init__(self, text: str, reason: str):
        super().__init__(reason)
        self.text = text


def profiler_output_problem(text: str) -> str | None:
    """
    Why `text` is NOT a valid Profiler report, or None if it looks fine. Catches a final
    answer that is a raw tool call (bare, in <tool_call> tags, or in a code fence) and
    model_utils' "_⚠️ ..." generation-failure placeholders.
    """
    t = (text or "").strip()
    if not t:
        return "empty output"
    if t.startswith("_⚠️"):
        return "the model reported a generation failure instead of a report"
    if t.lower().startswith("<tool_call>"):
        return "the model emitted a raw tool-call block instead of a report"
    inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    if inner and inner[0] in "{[":
        try:
            obj = json.loads(inner)
        except ValueError:
            return None
        calls = obj if isinstance(obj, list) else [obj]
        if calls and all(isinstance(c, dict) and "name" in c and "arguments" in c for c in calls):
            names = ", ".join(str(c["name"]) for c in calls)
            return f"the model emitted a raw tool call ({names}) instead of a report"
    return None


async def guard_profiler_output(gen):
    """Pass the Profiler's stream through, then - BEFORE the checkpoint wrapper can mark the
    step complete - raise BadProfilerOutput if the finished text isn't a real report."""
    last = ""
    async with aclosing(gen) as g:
        async for text in g:
            last = text
            yield text
    problem = profiler_output_problem(last)
    if problem:
        raise BadProfilerOutput(last, problem)


async def quarantine_bad_profiler(gen, store):
    """Outermost wrapper: on BadProfilerOutput, keep the bad text for debugging under the
    step 'profiler_failed', discard the saved 'profiler' (and 'final') so a re-run
    regenerates ONLY the Profiler, and surface an error. Forensic, RAG and education
    results stay saved and are reused."""
    try:
        async with aclosing(gen) as g:
            async for text in g:
                yield text
    except BadProfilerOutput as e:
        await store.save("profiler_failed", e.text, "partial")
        await store.invalidate(["profiler", "final"])
        raise RuntimeError(f"Profiler output was invalid ({e}) and was discarded") from e


# =============================================
# UI handlers
# =============================================

async def list_cases_for_ui():
    rows = await pstate.list_cases_safe()
    return gr.Dropdown(choices=[(pstate.describe_case(r), r["case_id"]) for r in rows], value=None)


async def _delete_reply(msg: str):
    """Standard reply of the delete buttons: status text, refreshed case list, cleared confirm box."""
    return msg, await list_cases_for_ui(), gr.Checkbox(value=False)


_NOT_CONFIRMED = "⚠️ Tick the confirmation box first - nothing was deleted."


async def delete_selected_case_ui(confirm, case_id):
    if not confirm:
        return await _delete_reply(_NOT_CONFIRMED)
    if not case_id:
        return await _delete_reply("⚠️ Pick a saved case in the dropdown above first.")
    ok, res = await pstate.run_safe(pstate.db.forget_case(str(case_id)))
    msg = (f"🗑 Deleted case {case_id} ({res} saved step(s))." if res else f"Nothing was saved under case {case_id}.") if ok \
        else f"❌ Database unavailable - nothing deleted: {res}"
    log.info(f"[memory] {msg}")
    return await _delete_reply(msg)


async def delete_old_cases_ui(confirm, days):
    if not confirm:
        return await _delete_reply(_NOT_CONFIRMED)
    try:
        days = int(days)
        if days < 1:
            raise ValueError
    except (TypeError, ValueError):
        return await _delete_reply("⚠️ Enter a whole number of days (1 or more).")
    ok, res = await pstate.run_safe(pstate.db.delete_cases_older_than(days))
    msg = f"🧹 Deleted {res} case(s) with no activity in the last {days} day(s)." if ok \
        else f"❌ Database unavailable - nothing deleted: {res}"
    log.info(f"[memory] {msg}")
    return await _delete_reply(msg)


async def delete_all_cases_ui(confirm):
    if not confirm:
        return await _delete_reply(_NOT_CONFIRMED)
    ok, res = await pstate.run_safe(pstate.db.delete_all_cases())
    msg = f"🗑 Deleted all {res} saved case(s)." if ok else f"❌ Database unavailable - nothing deleted: {res}"
    log.info(f"[memory] {msg}")
    return await _delete_reply(msg)


def refresh_examples():
    """Thumbnails of photos uploaded before (temp_uploads/), newest first, duplicates removed.
    Set SHOW_UPLOAD_EXAMPLES=0 to hide them (they are real case photos)."""
    if os.getenv("SHOW_UPLOAD_EXAMPLES", "1") == "0":
        return [], []
    items = example_images.list_example_images(TEMP_IMAGE_DIR, limit=24)
    return [(it["path"], it["name"]) for it in items], [it["path"] for it in items]


def add_example(current_files, example_list, evt: gr.SelectData):
    """Clicking a thumbnail adds that photo to the upload box (never twice, never beyond MAX_IMAGES)."""
    files = example_images.normalize_paths(current_files)
    try:
        chosen = (example_list or [])[int(evt.index)]
    except (TypeError, ValueError, IndexError):
        return files or None
    return example_images.add_unique(files, [chosen], MAX_IMAGES, warn=gr.Warning) or None


def add_all_examples(current_files, example_list):
    files = example_images.normalize_paths(current_files)
    return example_images.add_unique(files, list(example_list or []), MAX_IMAGES, warn=gr.Warning) or None


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

            with gr.Accordion("🖼 Examples - photos you uploaded before", open=True):
                gr.Markdown(
                    "Click a thumbnail to add it to **Upload Photos** above. The same photo is never added twice. "
                    "Pick the same photos in the same order as before to find that case's saved progress."
                )
                example_gallery = gr.Gallery(
                    show_label=False, columns=6, height=220, object_fit="cover",
                    allow_preview=False, interactive=False,
                )
                example_paths = gr.State([])
                with gr.Row():
                    add_all_btn = gr.Button("➕ Add all shown (up to 15)")
                    refresh_examples_btn = gr.Button("🔄 Refresh examples")

    run_btn = gr.Button("▶ Run Pipeline", variant="primary", size="large")

    with gr.Row():
        resume_mode = gr.Radio(
            choices=[
                ("♻️ Resume - continue partial reports", "continue"),
                ("⏭ Resume - use partial reports as-is", "asis"),
                ("🆕 Start fresh (discard saved progress)", "fresh"),
            ],
            value="continue",
            label="Progress is saved to Postgres. If this exact case (same images + context) was run before:",
        )
        saved_case = gr.Dropdown(
            choices=[], value=None, allow_custom_value=True,
            label="...or resume a saved case (overrides the inputs above)",
        )
        refresh_cases = gr.Button("🔄 Refresh saved cases")

    with gr.Accordion("🗑 Manage saved progress (Postgres memory)", open=False):
        gr.Markdown(
            "Saved cases accumulate in Postgres until you delete them. Deleting is **permanent**: "
            "tick the box, then press one button. Image files in `temp_uploads/` are not touched. "
            "Avoid deleting a case while it is still running."
        )
        delete_confirm = gr.Checkbox(label="I understand this permanently deletes saved progress", value=False)
        with gr.Row():
            delete_case_btn = gr.Button("🗑 Delete the case selected above")
            older_days = gr.Number(label="Older than (days)", value=7, minimum=1, precision=0)
            delete_old_btn = gr.Button("🧹 Delete cases older than N days")
            delete_all_btn = gr.Button("⚠️ Delete ALL saved cases", variant="stop")
        memory_status = gr.Textbox(label="Result", interactive=False)

    with gr.Row():
        with gr.Column():
            forensic_out = gr.Markdown(label="🔬 Forensic Agent (no MCP)")
        with gr.Column():
            profiler_out = gr.Markdown(label="🔍 Profiler Agent (MCP: skills/ only) — runs concurrently with the reference summaries")

    gr.Markdown(
        "### 📚 Reference Summaries (one per trauma type in the Forensic report's 'Evidence of "
        "Trauma' — written by the Forensic agent from the full reference files, concurrently "
        "with the Profiler agent)"
    )
    rag_out = gr.Markdown(label="Reference Summaries")

    with gr.Accordion("📄 Full Result", open=False):
        full_out = gr.Markdown(label="Complete Report")

    status_out = gr.Textbox(label="Pipeline Status", interactive=False)

    # Run button
    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, url_input, extra_context, resume_mode, saved_case],
        outputs=[forensic_out, profiler_out, rag_out, full_out, status_out],
    ).then(fn=list_cases_for_ui, outputs=saved_case).then(
        fn=refresh_examples, outputs=[example_gallery, example_paths]
    )
    refresh_cases.click(fn=list_cases_for_ui, outputs=saved_case)
    demo.load(fn=list_cases_for_ui, outputs=saved_case)
    demo.load(fn=refresh_examples, outputs=[example_gallery, example_paths])
    refresh_examples_btn.click(fn=refresh_examples, outputs=[example_gallery, example_paths])
    example_gallery.select(fn=add_example, inputs=[image_input, example_paths], outputs=image_input)
    add_all_btn.click(fn=add_all_examples, inputs=[image_input, example_paths], outputs=image_input)
    delete_case_btn.click(fn=delete_selected_case_ui, inputs=[delete_confirm, saved_case],
                          outputs=[memory_status, saved_case, delete_confirm])
    delete_old_btn.click(fn=delete_old_cases_ui, inputs=[delete_confirm, older_days],
                         outputs=[memory_status, saved_case, delete_confirm])
    delete_all_btn.click(fn=delete_all_cases_ui, inputs=[delete_confirm],
                         outputs=[memory_status, saved_case, delete_confirm])

    gr.Markdown("**Tip:** You can upload an image **and** add extra context text.")

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(TEMP_IMAGE_DIR.resolve())],
    )
