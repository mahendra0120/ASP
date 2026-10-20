# ASP — Agentic Serial-crime Profiler

A two-agent Qwen3-VL pipeline (Forensic agent → Profiler agent, wired
together over A2A) that turns case images into a forensic report and
a behavioral profile. An MCP filesystem server backs the Profiler
agent's skill files, results are stored in PostgreSQL, and a local RAG
lookup runs over `forensic_knowledge/`.

## Running on RunPod

**Pod requirements**
- A GPU pod with a **network volume mounted at `/workspace`** (model
  weights, the uv/npm caches and Postgres backups live there so they
  survive pod restarts).
- HTTP port **7860** exposed in the pod settings (that's the Gradio UI;
  you'll reach it through RunPod's proxy URL for that port).
- Any Ubuntu-based image with an NVIDIA driver. Running as root without
  `sudo` is fine — the scripts handle both.

**First run**
```bash
cd /workspace
git clone https://github.com/mahendra0120/ASP.git
cd ASP
cp .env.example .env        # set HF_TOKEN if either model repo is private/gated
bash start.sh
```

**After every pod restart** — just run `bash start.sh` again. It is
idempotent and does, in order:

| Step | Script | What it does |
|---|---|---|
| 1 | `setup_env.sh` | Installs Node/`npx` + `uv` if missing, points the HF / uv / npm caches at `/workspace`, pre-installs the MCP filesystem server, exports the env vars |
| 2 | `setup_postgres.sh` | Installs/starts Postgres, creates the `asp` role + DB, restores the latest backup, starts the background backup loop |
| 3 | `uv sync --frozen` | Installs Python deps (fast after the first run — cached on the volume) |
| 4 | `main.py` | Starts the Gradio UI. Press **🚀 Start Servers** in it to launch the Profiler and Forensic agent servers (`a2a_server.py`) |

**Environment variables** (set by `setup_env.sh`)

| Var | Value |
|---|---|
| `HF_HOME` | `/workspace/hf_cache` |
| `HF_HUB_OFFLINE` | `1` once the cache has models; left unset on an empty cache so the first run can download |
| `PROFILER_AGENT_PORT` | `8011` |
| `FORENSIC_AGENT_PORT` | `8002` |

`HF_TOKEN` is a credential and lives only in `.env` (gitignored) —
never in a script or committed file. It's only needed for the first
download of a private/gated repo; once weights are cached the app runs
offline and the token isn't used.

**Logs** — `server_logs/{profiler,forensic}_server.log`. Each launch
starts a fresh log and keeps the previous one as `*.log.prev`, so a
crash trace survives a restart. If the Profiler fails, its traceback
is in `profiler_server.log`; `exit_codes.log` records how each server
process exited (e.g. SIGKILL from the OOM killer).

**Database backups** — a snapshot is written to
`/workspace/pg-backups/latest.dump` every 5 minutes. Run
`bash backup_postgres.sh` for an immediate one before deliberately
stopping the pod.

`setup_env.sh` must be **sourced** (`source setup_env.sh`), not
executed, if you run it by hand; `start.sh` does that for you.

## Project layout

| Path | Role |
|---|---|
| `main.py` | Gradio UI + pipeline orchestration (Forensic → RAG ‖ Profiler), server start/stop/restart |
| `a2a_server.py` | Serves one agent over A2A: `python a2a_server.py forensic\|profiler` |
| `a2a_client.py` | Minimal streaming A2A client used by `main.py` |
| `qwen_agents/` | The two agents and `model_utils.py` (local Qwen3-VL inference as a pydantic-ai model) |
| `octen_rag.py` | Embedding retrieval + trauma-type selection over `forensic_knowledge/` |
| `pipeline_state.py`, `db.py` | Postgres checkpointing so interrupted runs can resume |
| `example_images.py` | Thumbnail gallery of previously uploaded photos |
| `env_defaults.py` | Loads `.env` and sets cache/port defaults before anything imports `huggingface_hub` |
