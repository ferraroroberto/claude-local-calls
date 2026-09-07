"""Thin wrapper around the `claude -p` CLI.

Shells out to the Claude Code CLI in headless JSON mode and returns the
parsed envelope. Uses the user's local Claude auth — no API key required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .no_window import NO_WINDOW
from .server_common import safe_span, start_span

logger = logging.getLogger(__name__)


def _argv_hash(args: List[str]) -> str:
    """Stable short hash of an argv vector for span correlation."""
    return hashlib.blake2b(
        " ".join(args).encode("utf-8", errors="replace"), digest_size=6
    ).hexdigest()


class ClaudeCLIError(RuntimeError):
    pass


def call_claude(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    attachments: Optional[Sequence[Path]] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Invoke `claude -p --output-format json` and return the parsed envelope.

    Prompt is fed via stdin to avoid command-line length limits.
    ``attachments`` (images and/or PDF documents) are passed via
    ``--add-dir`` (the temp dir holding them is added to Claude's allowed
    filesystem set) and their absolute paths are prepended to the prompt so
    Claude knows to read them.
    """
    args: List[str] = ["claude", "-p", "--output-format", "json"]
    if model:
        args += ["--model", model]
    if system:
        args += ["--system-prompt", system]

    if attachments:
        # All attachments live under a single per-request temp dir today;
        # pass that one parent dir via --add-dir and reference each file by
        # absolute path in the prompt.
        parents = {Path(p).resolve().parent for p in attachments}
        for d in parents:
            args += ["--add-dir", str(d)]
        refs = "\n".join(f"- {Path(p).resolve()}" for p in attachments)
        prompt = f"Attached files:\n{refs}\n\n{prompt}"

    with start_span("local_llm_hub.claude_cli", "claude_cli.invoke") as span:
        if span is not None and hasattr(span, "set_attribute"):
            with safe_span("claude_cli.invoke"):
                span.set_attribute("claude_cli.argv_hash", _argv_hash(args))
                if model:
                    span.set_attribute("claude_cli.model", model)
                span.set_attribute("claude_cli.attachments", len(attachments or []))
        try:
            # Suppress the Windows Terminal window that would otherwise spawn
            # for every request when the hub itself is running under pythonw
            # (e.g. launched from the tray with CREATE_NO_WINDOW — children
            # don't inherit the parent's no-window state).
            creationflags = NO_WINDOW
            proc = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                check=False,
                shell=False,
                creationflags=creationflags,
            )
        except FileNotFoundError as e:
            raise ClaudeCLIError(
                "`claude` CLI not found on PATH. Install Claude Code first."
            ) from e

        if span is not None and hasattr(span, "set_attribute"):
            with safe_span("claude_cli.invoke"):
                span.set_attribute("claude_cli.exit_code", int(proc.returncode))
                span.set_attribute("claude_cli.stderr_bytes", len(proc.stderr or ""))

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


def call_claude_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    system: Optional[str] = None,
    attachments: Optional[Sequence[Path]] = None,
    timeout: float = 600.0,
) -> Iterator[Dict[str, Any]]:
    """Yield the JSON-lines records emitted by Claude Code's stream mode.

    ``--include-partial-messages`` exposes the nested Anthropic
    ``stream_event`` records that contain text deltas. The caller owns shape
    filtering because Claude Code also writes lifecycle, rate-limit, and final
    result records to the same stdout stream.

    The subprocess is terminated if the downstream iterator closes early, so
    a disconnected HTTP client cannot leave a paid CLI request running. A
    timer enforces the same bounded runtime as :func:`call_claude` while a
    daemon reader drains stderr to prevent a full pipe from deadlocking a
    long response.
    """
    args: List[str] = [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
    ]
    if model:
        args += ["--model", model]
    if system:
        args += ["--system-prompt", system]

    if attachments:
        parents = {Path(p).resolve().parent for p in attachments}
        for directory in parents:
            args += ["--add-dir", str(directory)]
        refs = "\n".join(f"- {Path(p).resolve()}" for p in attachments)
        prompt = f"Attached files:\n{refs}\n\n{prompt}"

    # A tracing context manager cannot span generator yields: Starlette may
    # consume successive chunks in different worker contexts, and detaching an
    # OTel context token from a different context raises. Trace process launch
    # here; the route-level span owns the full streamed request lifetime.
    with start_span("local_llm_hub.claude_cli", "claude_cli.launch") as span:
        if span is not None and hasattr(span, "set_attribute"):
            with safe_span("claude_cli.launch"):
                span.set_attribute("claude_cli.argv_hash", _argv_hash(args))
                if model:
                    span.set_attribute("claude_cli.model", model)
                span.set_attribute("claude_cli.attachments", len(attachments or []))
                span.set_attribute("claude_cli.streaming", True)
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                creationflags=NO_WINDOW,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(
                "`claude` CLI not found on PATH. Install Claude Code first."
            ) from exc

    stderr_parts: List[str] = []
    stderr_chars = 0
    timed_out = threading.Event()

    def _drain_stderr() -> None:
        nonlocal stderr_chars
        if proc.stderr is None:
            return
        for part in proc.stderr:
            if stderr_chars < 4000:
                stderr_parts.append(part)
                stderr_chars += len(part)

    def _kill_on_timeout() -> None:
        if proc.poll() is None:
            timed_out.set()
            proc.kill()

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    timer = threading.Timer(timeout, _kill_on_timeout)
    timer.daemon = True
    timer.start()

    try:
        if proc.stdin is None or proc.stdout is None:
            raise ClaudeCLIError("claude stream pipes were not created")
        proc.stdin.write(prompt)
        proc.stdin.close()
        for line in proc.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ClaudeCLIError(
                    f"could not parse claude stream JSON: {raw[:200]!r}"
                ) from exc
            if isinstance(record, dict):
                yield record
        return_code = proc.wait()
        stderr_thread.join(timeout=1.0)
        stderr = "".join(stderr_parts)
        if timed_out.is_set():
            raise ClaudeCLIError(f"claude -p stream timed out after {timeout:g}s")
        if return_code != 0:
            raise ClaudeCLIError(
                f"claude -p stream exited {return_code}: {stderr[:500]}"
            )
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        stderr_thread.join(timeout=1.0)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
