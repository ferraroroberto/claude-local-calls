"""Unit tests for the GitHub Copilot usage parser (issue #231).

These run without a real ``~/.copilot`` / VS Code tree: each test writes
synthetic session files into a temp dir and points the parser at it.

Key guarantees:
- CLI: ``totalNanoAiu / 1e9 / 100`` is the credits->USD conversion, one
  record per model in ``modelMetrics``, sessions without ``events.jsonl``
  are skipped.
- VS Code: the minimal patch replay extracts ``copilotCredits`` /
  ``resolvedModel`` / token counts without needing a general JSON-patch
  engine, and only ``.jsonl`` files (never the empty ``.json`` skeletons)
  are read.
- ``record_costs()`` returns a Copilot record's ``credits_usd`` directly,
  never falling through to the Claude family pricing table.
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from src import code_usage, copilot_usage, usage_common, usage_pricing


def _write_yaml(path: Path, cwd: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"id: {path.parent.name}\ncwd: {cwd}\n", encoding="utf-8")


def _write_jsonl(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(obj) for obj in lines) + "\n", encoding="utf-8"
    )


@pytest.fixture
def copilot_dirs(tmp_path, monkeypatch):
    """Point both parsers at temp dirs and clear their mtime caches."""
    cli_dir = tmp_path / "session-state"
    vscode_dir = tmp_path / "workspaceStorage"
    monkeypatch.setattr(copilot_usage, "_CLI_SESSION_STATE_DIR", cli_dir)
    monkeypatch.setattr(copilot_usage, "_VSCODE_WORKSPACE_STORAGE_DIR", vscode_dir)
    copilot_usage._cli_file_cache.clear()
    copilot_usage._vscode_file_cache.clear()
    copilot_usage._vscode_workspace_cache.clear()
    return cli_dir, vscode_dir


def _shutdown_event(ts: str, model_metrics: dict, session_id: str = "sess-1") -> dict:
    return {
        "type": "session.shutdown",
        "timestamp": ts,
        "data": {"sessionId": session_id, "modelMetrics": model_metrics},
    }


# ---------------------------------------------------------------------------
# Part A — CLI session logs
# ---------------------------------------------------------------------------


def test_cli_nano_aiu_to_credits_conversion(copilot_dirs):
    cli_dir, _ = copilot_dirs
    sdir = cli_dir / "uuid-1"
    _write_yaml(sdir / "workspace.yaml", "E:\\automation\\demo")
    _write_jsonl(sdir / "events.jsonl", [
        _shutdown_event(
            "2026-06-22T13:47:52.492Z",
            {
                "claude-haiku-4.5": {
                    "requests": {"count": 1, "cost": 0.33},
                    "usage": {
                        "inputTokens": 17846, "outputTokens": 103,
                        "cacheReadTokens": 0, "cacheWriteTokens": 17836,
                    },
                    "totalNanoAiu": 2282000000,
                }
            },
        )
    ])

    records = copilot_usage._cli_records()
    assert len(records) == 1
    r = records[0]
    assert r.vendor == "copilot"
    assert r.model == "claude-haiku-4.5"
    assert r.project_key == usage_common.encode_project_key("E:\\automation\\demo")
    assert r.input_tokens == 17846
    assert r.output_tokens == 103
    assert r.cache_creation_tokens == 17836
    assert r.credits_usd == pytest.approx(0.02282)
    assert r.ts.tzinfo is timezone.utc


def test_cli_one_record_per_model(copilot_dirs):
    cli_dir, _ = copilot_dirs
    sdir = cli_dir / "uuid-2"
    _write_yaml(sdir / "workspace.yaml", "E:\\automation\\demo")
    _write_jsonl(sdir / "events.jsonl", [
        _shutdown_event(
            "2026-06-22T14:00:00.000Z",
            {
                "gpt-5-mini": {"usage": {"inputTokens": 100, "outputTokens": 10}, "totalNanoAiu": 1_000_000_000},
                "claude-sonnet-4.5": {"usage": {"inputTokens": 200, "outputTokens": 20}, "totalNanoAiu": 2_000_000_000},
            },
        )
    ])

    records = copilot_usage._cli_records()
    assert len(records) == 2
    models = {r.model for r in records}
    assert models == {"gpt-5-mini", "claude-sonnet-4.5"}


def test_cli_skips_sessions_without_events(copilot_dirs):
    cli_dir, _ = copilot_dirs
    sdir = cli_dir / "uuid-3"
    _write_yaml(sdir / "workspace.yaml", "E:\\automation\\demo")
    # No events.jsonl written — session never cleanly shut down.

    assert copilot_usage._cli_records() == []


# ---------------------------------------------------------------------------
# Part C — VS Code chat session logs
# ---------------------------------------------------------------------------


def _vscode_patch_lines() -> list:
    return [
        {"kind": 0, "v": {"requests": []}},
        {"kind": 2, "k": ["requests"], "i": 0, "v": {"requestId": "req-1", "timestamp": 1700000000000}},
        # Irrelevant field — must be ignored by the minimal replay.
        {"kind": 1, "k": ["requests", 0, "response"], "v": [{"value": "hello"}]},
        {"kind": 1, "k": ["requests", 0, "result"], "v": {
            "metadata": {"resolvedModel": "gpt-5-mini"},
            "details": "GPT-5 mini • 0.8 credits",
        }},
        {"kind": 1, "k": ["requests", 0, "promptTokens"], "v": 30970},
        {"kind": 1, "k": ["requests", 0, "completionTokens"], "v": 1308},
        {"kind": 1, "k": ["requests", 0, "copilotCredits"], "v": 0.77665},
    ]


def test_vscode_replay_extracts_credits_and_model(copilot_dirs, tmp_path):
    _, vscode_dir = copilot_dirs
    hash_dir = vscode_dir / "abc123"
    hash_dir.mkdir(parents=True)
    (hash_dir / "workspace.json").write_text(
        json.dumps({"workspace": "file:///e%3A/automation/oracle-to-gcp.code-workspace"}),
        encoding="utf-8",
    )
    _write_jsonl(hash_dir / "chatSessions" / "sess-a.jsonl", _vscode_patch_lines())

    records = copilot_usage._vscode_records()
    assert len(records) == 1
    r = records[0]
    assert r.vendor == "copilot"
    assert r.model == "gpt-5-mini"
    assert r.input_tokens == 30970
    assert r.output_tokens == 1308
    assert r.credits_usd == pytest.approx(0.0077665)
    assert r.project_key == usage_common.encode_project_key("E:\\automation\\oracle-to-gcp")


def test_vscode_skips_empty_skeleton_json(copilot_dirs):
    _, vscode_dir = copilot_dirs
    hash_dir = vscode_dir / "def456"
    hash_dir.mkdir(parents=True)
    (hash_dir / "workspace.json").write_text(
        json.dumps({"workspace": "file:///e%3A/automation/demo.code-workspace"}), encoding="utf-8"
    )
    skeleton_dir = hash_dir / "chatSessions"
    skeleton_dir.mkdir()
    (skeleton_dir / "sess-b.json").write_text(
        json.dumps({"version": 3, "requests": [], "sessionId": "sess-b"}), encoding="utf-8"
    )

    # Only *.jsonl is scanned — the .json skeleton contributes nothing.
    assert copilot_usage._vscode_records() == []


def test_vscode_request_without_credits_is_skipped(copilot_dirs):
    _, vscode_dir = copilot_dirs
    hash_dir = vscode_dir / "ghi789"
    hash_dir.mkdir(parents=True)
    (hash_dir / "workspace.json").write_text(
        json.dumps({"workspace": "file:///e%3A/automation/demo.code-workspace"}), encoding="utf-8"
    )
    lines = [
        {"kind": 0, "v": {"requests": []}},
        {"kind": 2, "k": ["requests"], "i": 0, "v": {"requestId": "req-2", "timestamp": 1700000000000}},
        # Response never completed — no copilotCredits patch ever arrives.
    ]
    _write_jsonl(hash_dir / "chatSessions" / "sess-c.jsonl", lines)

    assert copilot_usage._vscode_records() == []


# ---------------------------------------------------------------------------
# Cost pricing + summary merge
# ---------------------------------------------------------------------------


def test_record_costs_copilot_returns_credits_directly():
    """A copilot record must not fall through to the Claude pricing table —
    even when the resolved model name matches a Claude family substring."""
    from datetime import datetime

    r = usage_common.UsageRecord(
        session_id="s", project_key="k", project_name="k", model="claude-sonnet-4.5",
        ts=datetime(2026, 7, 11, tzinfo=timezone.utc),
        input_tokens=1_000_000, output_tokens=100_000,
        cache_creation_tokens=0, cache_read_tokens=0,
        vendor="copilot", credits_usd=0.05,
    )
    input_cost, output_cost, cache_cost = usage_pricing.record_costs(r)
    assert input_cost == pytest.approx(0.05)
    assert output_cost == 0.0
    assert cache_cost == 0.0


def test_summary_by_vendor_merges_copilot(copilot_dirs):
    cli_dir, _ = copilot_dirs
    sdir = cli_dir / "uuid-4"
    _write_yaml(sdir / "workspace.yaml", "E:\\automation\\demo")
    _write_jsonl(sdir / "events.jsonl", [
        _shutdown_event(
            "2026-06-22T14:00:00.000Z",
            {"gpt-5-mini": {"usage": {"inputTokens": 100, "outputTokens": 10}, "totalNanoAiu": 1_000_000_000}},
        )
    ])

    summary = code_usage.get_summary("all", "copilot")
    assert summary["vendor"] == "copilot"
    vendors = {row["vendor"] for row in summary["by_vendor"]}
    assert vendors == {"copilot"}
    assert summary["totals"]["requests"] == 1
