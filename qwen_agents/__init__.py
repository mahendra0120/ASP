"""
qwen_agents package init.

IMPORTANT: this used to eagerly do
    from .Forensic_agent.forensic_agent import forensic_agent
    from .Profiler_agent.profiler_agent import profile_agent as profiler_agent
at import time. Since importing ANY submodule of a package first runs
the package's __init__.py, that meant even code that only wanted one
agent (e.g. a script importing just
`qwen_agents.Forensic_agent.forensic_agent`) silently triggered loading
BOTH agents' ~8B-parameter checkpoints — which is exactly what
A2A_profiler_server.py / A2A_forensic_server.py need to avoid in order
to run each agent in its own process with an isolated memory budget.

`__getattr__` below makes `forensic_agent` / `profiler_agent` lazy
attributes: the submodule (and its heavy model load) only happens the
first time the attribute is actually accessed, not on package import.
`from qwen_agents import forensic_agent` still works exactly as before
for code (like the original combined A2A_image_delegation_server.py)
that genuinely wants both in one process.
"""

__all__ = ["forensic_agent", "profiler_agent"]


def __getattr__(name):
    if name == "forensic_agent":
        from .Forensic_agent.forensic_agent import forensic_agent
        return forensic_agent
    if name == "profiler_agent":
        from .Profiler_agent.profiler_agent import profile_agent
        return profile_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
