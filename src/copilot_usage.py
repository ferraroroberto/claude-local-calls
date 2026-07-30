"""Host-side GitHub Copilot usage parser (issue #231).

Two independent local sources feed the shared ``UsageRecord`` shape (tagged
``vendor="copilot"``), so both flow through the same Code-tab summary builder
as Claude and Codex data:

1. **Copilot CLI session logs** — ``~/.copilot/session-state/<uuid>/``. Every
   session gets a ``workspace.yaml`` (cwd/repo metadata); only *clean-shutdown*
   sessions also get an ``events.jsonl`` whose ``session.shutdown`` event
   carries exact per-model credit/token totals. Sessions still in flight (or
   that crashed) have no ``events.jsonl`` and are skipped — no usage data is
   recoverable from ``session.db``/``session-store.db`` without further
   reverse-engineering, left for a future pass.
2. **VS Code Copilot Chat session logs** — the Copilot Chat extension writes
   a per-session ``.jsonl`` event log under
   ``%APPDATA%\\Code\\User\\workspaceStorage\\<hash>\\chatSessions\\<uuid>.jsonl``
   once a session has real exchanges. Each request event carries an exact
   ``copilotCredits`` float (not an estimate) plus ``promptTokens`` /
   ``completionTokens`` and the resolved model id. Older/inactive sessions
   only have an empty ``<uuid>.json`` skeleton (no ``requests``) and are
   skipped.

Design constraints mirror ``codex_usage.py``:
- **Read-only** — never modify the session files.
- **Zero subprocesses** — parse the files directly.
- **Passive** — nothing runs on any request path; the SPA polls on its own
  30 s interval.
- **Mtime cache** — each file is only re-parsed when its mtime changes.

Credit → USD conversion (empirically confirmed against a real session, both
the CLI's ``totalNanoAiu`` and VS Code's ``copilotCredits`` are in *credits*,
and 1 credit = $0.01 per GitHub's AI Credits billing model)::

    CLI:      credits_usd = totalNanoAiu / 1e9 / 100   (nanoAiu -> credits -> USD)
    VS Code:  credits_usd = copilotCredits / 100        (already in credits)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

import yaml

from src.usage_common import (
    FileStats,
    UsageRecord,
    encode_project_key,
    load_cached,
    parse_iso_ts,
    project_pretty,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CLI_SESSION_STATE_DIR: Path = Path.home() / ".copilot" / "session-state"

if sys.platform == "darwin":
    _VSCODE_WORKSPACE_STORAGE_DIR: Path = (
        Path.home()
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "workspaceStorage"
    )
else:
    _VSCODE_WORKSPACE_STORAGE_DIR = (
        Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        / "Code"
        / "User"
        / "workspaceStorage"
    )

# Fields on a VS Code chat request we replay from the jsonl patch stream —
# everything else (message content, response text deltas, tool calls) is
# irrelevant to usage extraction and left unreplayed.
_VSCODE_REPLAY_FIELDS = {"result", "copilotCredits", "promptTokens", "completionTokens"}

# File-level mtime caches (module-level singletons), independent of Claude/Codex.
_cli_file_cache: Dict[str, FileStats] = {}
_vscode_file_cache: Dict[str, FileStats] = {}
_vscode_workspace_cache: Dict[str, Optional[str]] = {}


# ---------------------------------------------------------------------------
# Part A — Copilot CLI session logs
# ---------------------------------------------------------------------------


def _read_workspace_cwd(session_dir: Path) -> Optional[str]:
    try:
        raw = yaml.safe_load((session_dir / "workspace.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if isinstance(raw, dict):
        cwd = raw.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return None


def _parse_cli_events_file(path: Path, cwd: Optional[str]) -> List[UsageRecord]:
    """Parse one ``events.jsonl`` and return one record per model used.

    Session-granular, not per-turn: the CLI only exposes cumulative
    per-model totals at ``session.shutdown``, not a per-request breakdown.
    """
    key = encode_project_key(cwd) if cwd else "copilot"
    project_name = project_pretty(key) if cwd else "(unknown)"

    records: List[UsageRecord] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "session.shutdown":
                    continue

                data = obj.get("data") or {}
                session_id = data.get("sessionId") or path.parent.name
                ts = parse_iso_ts(obj.get("timestamp"))
                model_metrics = data.get("modelMetrics") or {}
                for model, metrics in model_metrics.items():
                    if not isinstance(metrics, dict):
                        continue
                    usage = metrics.get("usage") or {}
                    nano_aiu = metrics.get("totalNanoAiu") or 0
                    records.append(
                        UsageRecord(
                            session_id=session_id,
                            project_key=key,
                            project_name=project_name,
                            model=model,
                            ts=ts,
                            input_tokens=int(usage.get("inputTokens") or 0),
                            output_tokens=int(usage.get("outputTokens") or 0),
                            cache_creation_tokens=int(usage.get("cacheWriteTokens") or 0),
                            cache_read_tokens=int(usage.get("cacheReadTokens") or 0),
                            reasoning_output_tokens=int(usage.get("reasoningTokens") or 0),
                            vendor="copilot",
                            credits_usd=nano_aiu / 1e9 / 100,
                        )
                    )
    except OSError as exc:
        _log.warning("⚠️ copilot_usage: cannot read %s: %s", path, exc)
    return records


def _load_cli_events(path: Path, cwd: Optional[str]) -> List[UsageRecord]:
    return load_cached(path, _cli_file_cache, lambda p: _parse_cli_events_file(p, cwd))


def _cli_records() -> List[UsageRecord]:
    """Scan all Copilot CLI session-state dirs and return every usage record."""
    records: List[UsageRecord] = []

    if not _CLI_SESSION_STATE_DIR.exists():
        return records

    try:
        session_dirs = [p for p in _CLI_SESSION_STATE_DIR.iterdir() if p.is_dir()]
    except OSError as exc:
        _log.warning("⚠️ copilot_usage: cannot list %s: %s", _CLI_SESSION_STATE_DIR, exc)
        return records

    for sdir in session_dirs:
        events_path = sdir / "events.jsonl"
        if not events_path.is_file():
            continue
        cwd = _read_workspace_cwd(sdir)
        records.extend(_load_cli_events(events_path, cwd))

    return records


# ---------------------------------------------------------------------------
# Part C — VS Code Copilot Chat session logs
# ---------------------------------------------------------------------------


def _decode_vscode_project_path(uri: str) -> Optional[str]:
    """Decode a VS Code workspace/folder file URI into a filesystem path.

    ``file:///e%3A/automation/oracle-to-gcp.code-workspace`` -> the *project
    folder* path (the ``.code-workspace`` file's own basename minus suffix,
    matching this user's convention of one ``<repo>.code-workspace`` file
    per repo sitting alongside the repo directory) so the encoded project
    key matches Claude/Codex's key for the same repo.
    """
    try:
        parsed = urlparse(uri)
    except ValueError:
        return None
    if parsed.scheme != "file" or not parsed.path:
        return None
    raw_path = unquote(parsed.path)
    # "/e:/automation/oracle-to-gcp.code-workspace" -> "E:\automation\oracle-to-gcp.code-workspace"
    raw_path = raw_path.lstrip("/")
    if len(raw_path) < 2 or raw_path[1] != ":":
        return None
    win_path = raw_path[0].upper() + raw_path[1:].replace("/", "\\")
    if win_path.lower().endswith(".code-workspace"):
        win_path = win_path[: -len(".code-workspace")]
    return win_path


def _read_workspace_project_key(hash_dir: Path) -> Optional[str]:
    cache_key = str(hash_dir)
    if cache_key in _vscode_workspace_cache:
        return _vscode_workspace_cache[cache_key]

    project_key: Optional[str] = None
    try:
        raw = json.loads((hash_dir / "workspace.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict):
        uri = raw.get("workspace") or raw.get("folder")
        if isinstance(uri, str):
            path = _decode_vscode_project_path(uri)
            if path:
                project_key = encode_project_key(path)

    _vscode_workspace_cache[cache_key] = project_key
    return project_key


def _replay_vscode_requests(lines: List[dict]) -> List[dict]:
    """Minimally replay a chat-session jsonl patch stream into a request list.

    Only tracks the top-level ``requests`` array itself and the handful of
    per-request fields ``_VSCODE_REPLAY_FIELDS`` needs — message/response
    content patches are ignored, this is not a general JSON-patch engine.
    """
    requests: List[dict] = []
    for obj in lines:
        kind = obj.get("kind")
        if kind == 0:
            v = obj.get("v") or {}
            requests = list(v.get("requests") or [])
        elif kind == 2:
            k = obj.get("k")
            if k == ["requests"]:
                idx = obj.get("i")
                v = obj.get("v")
                if isinstance(idx, int) and 0 <= idx <= len(requests):
                    requests.insert(idx, v)
        elif kind == 1:
            k = obj.get("k") or []
            if len(k) == 3 and k[0] == "requests" and k[2] in _VSCODE_REPLAY_FIELDS:
                idx = k[1]
                if isinstance(idx, int) and 0 <= idx < len(requests):
                    requests[idx][k[2]] = obj.get("v")
    return requests


def _resolved_model_from_request(req: dict) -> str:
    result = req.get("result") or {}
    metadata = result.get("metadata") or {}
    model = metadata.get("resolvedModel")
    if isinstance(model, str) and model:
        return model
    # Fall back to parsing "GPT-5 mini • 0.8 credits" out of result.details.
    details = result.get("details")
    if isinstance(details, str) and "•" in details:
        return details.split("•")[0].strip()
    return "unknown"


def _parse_vscode_chat_file(path: Path, project_key: Optional[str]) -> List[UsageRecord]:
    key = project_key or "vscode"
    project_name = project_pretty(key) if project_key else "(unknown)"

    lines: List[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        _log.warning("⚠️ copilot_usage: cannot read %s: %s", path, exc)
        return []

    records: List[UsageRecord] = []
    for req in _replay_vscode_requests(lines):
        credits = req.get("copilotCredits")
        if credits is None:
            continue  # response never completed / no billing data for this request
        ts_ms = req.get("timestamp")
        try:
            ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            ts = datetime.now(tz=timezone.utc)

        records.append(
            UsageRecord(
                session_id=req.get("requestId") or path.stem,
                project_key=key,
                project_name=project_name,
                model=_resolved_model_from_request(req),
                ts=ts,
                input_tokens=int(req.get("promptTokens") or 0),
                output_tokens=int(req.get("completionTokens") or 0),
                cache_creation_tokens=0,
                cache_read_tokens=0,
                vendor="copilot",
                credits_usd=float(credits) / 100,
            )
        )
    return records


def _load_vscode_chat_file(path: Path, project_key: Optional[str]) -> List[UsageRecord]:
    return load_cached(
        path, _vscode_file_cache, lambda p: _parse_vscode_chat_file(p, project_key)
    )


def _vscode_records() -> List[UsageRecord]:
    """Scan all VS Code Copilot Chat session logs and return usage records."""
    records: List[UsageRecord] = []

    if not _VSCODE_WORKSPACE_STORAGE_DIR.exists():
        return records

    try:
        hash_dirs = [p for p in _VSCODE_WORKSPACE_STORAGE_DIR.iterdir() if p.is_dir()]
    except OSError as exc:
        _log.warning(
            "⚠️ copilot_usage: cannot list %s: %s", _VSCODE_WORKSPACE_STORAGE_DIR, exc
        )
        return records

    for hdir in hash_dirs:
        chat_dir = hdir / "chatSessions"
        if not chat_dir.is_dir():
            continue
        project_key = _read_workspace_project_key(hdir)
        try:
            jsonl_files = list(chat_dir.glob("*.jsonl"))
        except OSError:
            continue
        for jf in jsonl_files:
            records.extend(_load_vscode_chat_file(jf, project_key))

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def all_records() -> List[UsageRecord]:
    """Return every Copilot usage record from both the CLI and VS Code."""
    return _cli_records() + _vscode_records()
