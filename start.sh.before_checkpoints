#!/usr/bin/env bash
#
# start.sh — one command to bring the whole stack up on a RunPod pod.
#
#   bash start.sh
#
# Run it from anywhere; it's safe to re-run after every pod restart
# (each step skips work that's already done):
#   1. setup_env.sh      → Node/npx, uv, caches on /workspace, env vars
#   2. setup_postgres.sh → Postgres + role/db, restore latest backup,
#                          start the background backup loop
#   3. uv sync           → Python deps (persisted on the volume, so
#                          only the first run is slow)
#   4. main.py           → Gradio UI on :7860 (expose HTTP port 7860
#                          in your pod settings to reach it)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# shellcheck disable=SC1091
source ./setup_env.sh
bash ./setup_postgres.sh

echo ">> Syncing Python dependencies..."
uv sync --frozen

echo ">> Starting ASP (Gradio on 0.0.0.0:7860)..."
exec uv run python main.py
