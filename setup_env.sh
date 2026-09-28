#!/usr/bin/env bash
#
# setup_env.sh  (RunPod)
# ─────────────────────────────────────────────────────────────────
# Prepares a RunPod pod for this repo. Safe to source repeatedly
# (every step is a no-op if already done), and meant to be SOURCED so
# the exports below land in your shell:
#   source setup_env.sh
#
# What it does:
#   1. Installs Node.js/npm (for `npx` — the Profiler agent's
#      filesystem MCP server, see profiler_agent.py) and `uv`, if
#      they're missing. Pod local disk is wiped on restart, so this
#      re-runs (quickly) each time you start the pod.
#   2. Keeps the big, slow-to-rebuild things on the /workspace network
#      volume so they SURVIVE pod restarts: HF model weights, uv's
#      Python + package cache, and the npm cache.
#   3. Pre-fetches the MCP filesystem server package so the Profiler
#      never has to reach the npm registry mid-request.
#   4. Exports HF_HOME, HF_HUB_OFFLINE, PROFILER_AGENT_PORT,
#      FORENSIC_AGENT_PORT.
#
# HF_TOKEN is intentionally NOT handled here — it's a credential.
# Put it in .env (gitignored; see .env.example).

(return 0 2>/dev/null) || {
    echo "!! setup_env.sh must be SOURCED, not executed:"
    echo "     source setup_env.sh"
    exit 1
}

if [ ! -d /workspace ]; then
    echo "!! /workspace not found — attach a RunPod network volume at /workspace"
    echo "   (model weights, uv cache and DB backups all live there)."
    return 1
fi

# RunPod pods run as root and often have no `sudo`.
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# ── Persistent caches on the network volume ────────────────────────
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.uv/python}"
# Network volumes usually can't hardlink from the cache into .venv;
# copy explicitly instead of warning on every sync.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export npm_config_cache="${npm_config_cache:-/workspace/.npm}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$npm_config_cache"

# ── Base tools needed by the installers below ──────────────────────
if ! command -v curl >/dev/null 2>&1 || ! command -v gpg >/dev/null 2>&1; then
    echo ">> Installing curl/gnupg/ca-certificates..."
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends curl ca-certificates gnupg
fi

# ── Node.js / npm (npx) ────────────────────────────────────────────
# Requires Node >= 18; the distro `nodejs` package is often older, so
# use NodeSource's current LTS.
if ! command -v npx >/dev/null 2>&1; then
    echo ">> Installing Node.js (npx not found on PATH)..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | $SUDO bash -
    $SUDO apt-get install -y nodejs
else
    echo ">> npx already installed ($(npx --version)) — skipping."
fi

# ── uv ──────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo ">> uv already installed ($(uv --version)) — skipping."
fi

# ── Pre-fetch the Profiler's MCP filesystem server ─────────────────
# profiler_agent.py spawns `npx -y @modelcontextprotocol/server-filesystem`
# on demand with a 60s init timeout. Installing it up front means a
# slow/blocked npm registry can't make that first request fail.
if ! npm ls -g --depth=0 @modelcontextprotocol/server-filesystem >/dev/null 2>&1; then
    echo ">> Pre-installing @modelcontextprotocol/server-filesystem..."
    npm install -g @modelcontextprotocol/server-filesystem \
        || echo "!! Pre-install failed — the Profiler will try npx at request time instead."
fi

# ── Ports ───────────────────────────────────────────────────────────
export PROFILER_AGENT_PORT="${PROFILER_AGENT_PORT:-8011}"
export FORENSIC_AGENT_PORT="${FORENSIC_AGENT_PORT:-8002}"

# ── HF offline mode ────────────────────────────────────────────────
# HF_HUB_OFFLINE=1 skips all Hub network calls, which is what you want
# once weights are cached — but on a FRESH volume it would make the
# very first model download fail. So: default to 1 only when the cache
# already contains models; otherwise leave it off so the first run can
# download. An explicit HF_HUB_OFFLINE in your environment always wins.
if [ -z "${HF_HUB_OFFLINE+x}" ]; then
    if ls -d "$HF_HOME"/hub/models--* >/dev/null 2>&1; then
        export HF_HUB_OFFLINE=1
    else
        echo ">> HF cache is empty — leaving HF_HUB_OFFLINE off so weights can download."
        echo "   (Set HF_TOKEN in .env first if either model repo is private/gated.)"
        echo "   Re-source this script after the first run to switch to offline mode."
    fi
fi

if [ ! -f "$(dirname "${BASH_SOURCE[0]}")/.env" ]; then
    echo ">> No .env found — copy .env.example to .env and set HF_TOKEN if needed."
fi

echo ""
echo "✅ Environment ready:"
echo "   HF_HOME=${HF_HOME}"
echo "   HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
echo "   PROFILER_AGENT_PORT=${PROFILER_AGENT_PORT}"
echo "   FORENSIC_AGENT_PORT=${FORENSIC_AGENT_PORT}"
echo "   npx: $(command -v npx || echo 'NOT FOUND')"
echo "   uv:  $(command -v uv || echo 'NOT FOUND')"
echo ""
