"""Thin wrapper around the `claude -p` CLI.

Shells out to the Claude Code CLI in headless JSON mode and returns the
parsed envelope. Uses the user's local Claude auth — no API key required.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ClaudeCLIError(RuntimeError):
    pass


def call_claude(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Invoke `claude -p --output-format json` and return the parsed envelope.

    Prompt is fed via stdin to avoid command-line length limits.
    """
    args: List[str] = ["claude", "-p", "--output-format", "json"]
    if model:
        args += ["--model", model]
    if system:
        args += ["--system-prompt", system]

    try:
        proc = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except FileNotFoundError as e:
        raise ClaudeCLIError(
            "`claude` CLI not found on PATH. Install Claude Code first."
        ) from e

    if proc.returncode != 0:
        raise ClaudeCLIError(
            f"claude -p exited {proc.returncode}: {proc.stderr[:500]}"
        )

    raw = (proc.stdout or "").strip()
    if not raw:
        raise ClaudeCLIError("empty stdout from claude -p")

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClaudeCLIError(f"could not parse claude -p JSON: {raw[:200]!r}") from e

    if envelope.get("is_error"):
        raise ClaudeCLIError(
            f"claude -p returned is_error=true: {str(envelope)[:300]}"
        )
    return envelope
