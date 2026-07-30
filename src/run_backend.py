"""Cross-platform backend dispatcher.

    python -m src.run_backend hub    # run the FastAPI hub on port 8000
    python -m src.run_backend qwen   # run llama-server with the qwen entry
    python -m src.run_backend glm    # run llama-server with the glm entry

``launchers/run_model.bat`` / ``run_model.sh`` are the sole entry point for
every per-model launcher (#448 dedup — 22 near-identical hand-rolled scripts
used to each hardcode their own title/port/banner, with nothing to catch a
copy-paste port typo). ``--banner <id>`` below is what the ``.bat`` pulls its
window title + console banner from, live off ``config/models.yaml``, so
there is exactly one place a model's port/name can come from.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

from .backend_process import (
    build_command,
    external_pid as backend_external_pid,
    is_reachable as backend_is_reachable,
    resolve_model_by_id,
    vendor_dir_for,
)
from .host_profile import resolve as resolve_host
from .model_registry import enabled_models, launchable_local_ids
from .server_process import (
    BASE_URL as HUB_BASE_URL,
    external_pid as hub_external_pid,
    is_reachable as hub_is_reachable,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_hub() -> int:
    # Adopt: if the hub is already up (e.g. started by the tray or another
    # `run_hub` window), don't try to bind :8000 a second time — uvicorn
    # would crash with WinError 10048. Print and exit cleanly so the user
    # can see what happened in the launcher's terminal.
    if hub_is_reachable(timeout=0.4):
        ext = hub_external_pid()
        suffix = f" (PID {ext})" if ext else ""
        log.info("hub already running at %s%s — nothing to do.", HUB_BASE_URL, suffix)
        return 0
    from . import server
    server.main()
    return 0


def _run_backend(model_id: str) -> int:
    host = resolve_host()
    model = resolve_model_by_id(model_id)
    if model is None:
        known = [m.id for m in enabled_models()]
        log.error("model %r not enabled on host %s. known: %s", model_id, host.id, known)
        return 2
    if model.backend not in ("openai", "whisper", "tts"):
        log.error("model %r is backend=%s; nothing to spawn", model_id, model.backend)
        return 2
    if model.host_chain and host.id not in model.host_chain:
        log.error(
            "model %r is owned by host(s) %r, not %r — run it there "
            "(this host proxies to it instead of spawning it locally)",
            model_id, model.host_chain, host.id,
        )
        return 2

    # Same adopt-check as the hub: skip if something already answers on
    # this model's port.
    if backend_is_reachable(model, timeout=0.4):
        ext = backend_external_pid(model_id)
        suffix = f" (PID {ext})" if ext else ""
        log.info("%s already running on :%s%s — nothing to do.", model.display_name, model.port, suffix)
        return 0

    cmd = build_command(model)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if sys.platform == "win32":
        env["PATH"] = str(vendor_dir_for(model)) + os.pathsep + env.get("PATH", "")
    log.info("-> %s", " ".join(cmd))
    # Foreground execution so Ctrl+C works.
    return subprocess.call(cmd, env=env, cwd=str(PROJECT_ROOT))


# backend -> (window/box label, endpoint hint printed under the URL line).
# ``None`` hint means the URL line already says everything (llama-server's
# OpenAI-shape base already ends in ``/v1``).
_BANNER_KIND = {"openai": "llama-server", "whisper": "whisper-server", "tts": "tts_server"}
_BANNER_HINT = {
    "openai": None,
    "whisper": "POST WAV to /v1/audio/transcriptions",
    "tts": "POST text to /v1/audio/speech",
}


def _banner_lines(model_id: str) -> list[str]:
    """Window title (line 1) + console-banner body lines (rest), derived
    live from the registry entry so a launcher never hand-retypes a port."""
    model = resolve_model_by_id(model_id)
    if model is None:
        return [f"Local LLM Hub - {model_id}", f"{model_id}: not enabled on this host"]
    kind = _BANNER_KIND.get(model.backend, model.backend)
    if model.port is None:
        url = "(no port configured)"
    elif model.backend == "openai":
        url = model.url or f"http://127.0.0.1:{model.port}/v1"
    else:
        # whisper/tts banners show the bare base — the endpoint hint below
        # spells out the actual POST path (matches the old per-model .bat text).
        url = (model.url or f"http://127.0.0.1:{model.port}").removesuffix("/v1").rstrip("/")
    lines = [f"Local LLM Hub - {model.display_name}", f"{kind}: {model.display_name} on {url}"]
    hint = _BANNER_HINT.get(model.backend)
    if hint:
        lines.append(hint)
    lines.append("Ctrl+C to stop")
    return lines


def _launchable_targets() -> list[str]:
    """The hub plus every model this host can spawn locally — the exact set
    the bulk launchers (run_all.bat / run_all.sh) enumerate so they mirror
    the active host's ``enabled:`` contract instead of a stale hardcoded
    roster. Remote-owned rows (proxied, not run here) and disabled rows are
    absent by construction; each id here is one ``run_backend`` will start.
    """
    return ["hub", *launchable_local_ids()]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        log.error(
            "usage: python -m src.run_backend (hub|<model_id>|--list-launchable|--banner <model_id>)"
        )
        return 2
    target = args[0]
    if target == "--list-launchable":
        # One id per line on stdout so the bulk launchers can enumerate us.
        # Logging goes to stderr, keeping stdout a clean id list for `for /f`
        # (batch) and the `while read` loop (bash).
        for name in _launchable_targets():
            print(name)
        return 0
    if target == "--banner":
        if len(args) < 2:
            log.error("usage: python -m src.run_backend --banner <model_id>")
            return 2
        # Stdout-only, one line per banner line — run_model.bat's `for /f`
        # reads these to set its window title + console box.
        for line in _banner_lines(args[1]):
            print(line)
        return 0
    if target == "hub":
        return _run_hub()
    return _run_backend(target)


if __name__ == "__main__":
    sys.exit(main())
