"""
env_defaults.py
─────────────────────────────────────────────────────────────────
Import this FIRST (before torch / transformers / huggingface_hub) in every
entry point: main.py, the two A2A servers, model_utils.

It makes the runtime settings independent of HOW the process was started
(`bash start.sh`, `uv run python main.py`, the "Start Servers" button, a fresh
tmux / Jupyter terminal ...), so you no longer have to `export` them by hand.

Precedence for each setting:  real environment variable  >  .env file  >  default below.

  HF_HOME              default /workspace/hf_cache
  PROFILER_AGENT_PORT  default 8011
  FORENSIC_AGENT_PORT  default 8002
  HF_HUB_OFFLINE       default "1" when model weights are already cached under
                       HF_HOME/hub, otherwise left unset (so a fresh volume can
                       still download them). An explicit value always wins.

Check what the programs will actually see:   uv run python env_defaults.py
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DEFAULTS = {
    "HF_HOME": "/workspace/hf_cache",
    "PROFILER_AGENT_PORT": "8011",
    "FORENSIC_AGENT_PORT": "8002",
}
_KEYS = ("HF_HOME", "HF_HUB_OFFLINE", "PROFILER_AGENT_PORT", "FORENSIC_AGENT_PORT")


def apply() -> dict:
    for key in (*_DEFAULTS, "HF_HUB_OFFLINE"):   # an EMPTY variable counts as unset (dotenv would otherwise respect it)
        if os.environ.get(key) == "":
            os.environ.pop(key)
    try:                                     # .env fills gaps only; it never overrides the real environment
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except Exception:                        # noqa: BLE001 - python-dotenv missing/unreadable must not stop startup
        pass

    env = os.environ
    for key, default in _DEFAULTS.items():
        if not env.get(key):                 # unset OR empty
            env[key] = default

    if not env.get("HF_HUB_OFFLINE"):
        hub = Path(env["HF_HOME"]) / "hub"
        if hub.is_dir() and any(hub.glob("models--*")):
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_HUB_OFFLINE", None)  # an empty string would still count as "set" for some checks
    return {k: env.get(k) for k in _KEYS}


SETTINGS = apply()

if __name__ == "__main__":
    for k in _KEYS:
        print(f"   {k}={SETTINGS[k] if SETTINGS[k] is not None else '<unset>'}")
