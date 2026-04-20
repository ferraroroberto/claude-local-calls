"""Cross-platform backend dispatcher.

    python -m src.run_backend hub    # run the FastAPI hub on port 8000
    python -m src.run_backend qwen   # run llama-server with the qwen entry
    python -m src.run_backend glm    # run llama-server with the glm entry

Each .bat / .sh launcher is a one-liner that calls this module.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .host_profile import resolve as resolve_host
from .llama_process import VENDOR_LLAMA, build_command, resolve_model_by_id
from .model_registry import enabled_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_hub() -> int:
    # Delegate to the existing FastAPI entrypoint.
    from . import server
    server.main()
    return 0


def _run_backend(model_id: str) -> int:
    host = resolve_host()
    model = resolve_model_by_id(model_id)
    if model is None:
        known = [m.id for m in enabled_models()]
        print(f"model {model_id!r} not enabled on host {host.id}. known: {known}", file=sys.stderr)
        return 2
    if model.backend != "openai":
        print(f"model {model_id!r} is backend={model.backend}; nothing to spawn", file=sys.stderr)
        return 2

    cmd = build_command(model)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if sys.platform == "win32":
        env["PATH"] = str(VENDOR_LLAMA) + os.pathsep + env.get("PATH", "")
    print("-> " + " ".join(cmd))
    # Foreground execution so Ctrl+C works.
    return subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m src.run_backend (hub|<model_id>)", file=sys.stderr)
        return 2
    target = args[0]
    if target == "hub":
        return _run_hub()
    return _run_backend(target)


if __name__ == "__main__":
    sys.exit(main())
