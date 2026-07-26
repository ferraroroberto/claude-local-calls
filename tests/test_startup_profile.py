"""Unit tests for src/startup_profile.py (issues #265, #430).

Covers the tolerant load contract (missing/unparseable file -> defaults),
atomic save + validation, cache invalidation on save, cache correctness
across a swapped DEFAULT_PROFILE_PATH (the pattern host_profile._load_config
already relies on for test isolation), and the #430 migration: the retired
``models`` key is ignored on read and never written back.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

import pytest  # noqa: E402

from src import startup_profile as sp  # noqa: E402


def test_missing_file_returns_defaults(tmp_path, monkeypatch):
    # Both the live file and the example template absent → pure defaults.
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", tmp_path / "does-not-exist.json")
    monkeypatch.setattr(sp, "EXAMPLE_PROFILE_PATH", tmp_path / "no-example.json")
    profile = sp.load_startup_profile()
    assert profile == sp.StartupProfile()


def test_falls_back_to_example_when_live_file_absent(tmp_path, monkeypatch):
    """Fresh clone (no live file) reads the committed example template (#304)."""
    example = tmp_path / "startup_profile.example.json"
    example.write_text(json.dumps({
        "docker": False,
        "langfuse": False,
    }), encoding="utf-8")
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", tmp_path / "startup_profile.json")
    monkeypatch.setattr(sp, "EXAMPLE_PROFILE_PATH", example)

    profile = sp.load_startup_profile()
    assert profile.docker is False
    assert profile.langfuse is False
    assert profile.agentsview is True


def test_explicit_path_ignores_example_fallback(tmp_path, monkeypatch):
    """An explicit (test) path is honoured verbatim — never the example."""
    example = tmp_path / "startup_profile.example.json"
    example.write_text(json.dumps({"docker": False}), encoding="utf-8")
    monkeypatch.setattr(sp, "EXAMPLE_PROFILE_PATH", example)
    profile = sp.load_startup_profile(str(tmp_path / "missing.json"))
    assert profile == sp.StartupProfile()


def test_unparseable_file_returns_defaults(tmp_path, monkeypatch):
    target = tmp_path / "startup_profile.json"
    target.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", target)
    profile = sp.load_startup_profile()
    assert profile == sp.StartupProfile()


def test_loads_committed_shape(tmp_path, monkeypatch):
    target = tmp_path / "startup_profile.json"
    target.write_text(json.dumps({
        "docker": False,
        "langfuse": True,
        "agentsview": False,
    }), encoding="utf-8")
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", target)
    profile = sp.load_startup_profile()
    assert profile.docker is False
    assert profile.langfuse is True
    assert profile.agentsview is False


def test_legacy_keys_are_ignored(tmp_path, monkeypatch):
    """Stale keys from older live files load cleanly and are dropped:
    ``mac_mini_sync`` (retired #374 — peer sync is reconcile-driven) and
    ``models`` (retired #430 — the autostart set derives from models.yaml).
    Neither exists on the profile, so neither round-trips on save."""
    target = tmp_path / "startup_profile.json"
    target.write_text(json.dumps({
        "docker": True,
        "langfuse": True,
        "mac_mini_sync": True,           # legacy #374 key
        "models": ["qwen35_4b", "piper"],  # legacy #430 key
    }), encoding="utf-8")
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", target)
    profile = sp.load_startup_profile()
    assert profile.docker is True
    assert not hasattr(profile, "mac_mini_sync")
    assert not hasattr(profile, "models")
    assert set(profile.as_dict()) == {"docker", "langfuse", "agentsview"}

    # normalize (the save path) also ignores lingering legacy keys — a
    # not-yet-updated peer PATCHing the old shape must never 400.
    clean = sp.normalize_profile({"mac_mini_sync": True, "models": ["piper"]})
    assert set(clean.as_dict()) == {"docker", "langfuse", "agentsview"}


def test_cache_busts_when_default_path_swapped(tmp_path, monkeypatch):
    """Two different resolved paths must never share a cache slot."""
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"docker": True}), encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"docker": False}), encoding="utf-8")

    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", a)
    first = sp.load_startup_profile()
    assert first.docker is True

    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", b)
    second = sp.load_startup_profile()
    assert second.docker is False, "stale cache hit from path a's slot"


def test_normalize_rejects_non_dict():
    with pytest.raises(ValueError):
        sp.normalize_profile("nope")


def test_save_writes_atomically_and_busts_cache(tmp_path, monkeypatch):
    target = tmp_path / "startup_profile.json"
    monkeypatch.setattr(sp, "DEFAULT_PROFILE_PATH", target)
    # No example template either, so the prime below is the true default.
    monkeypatch.setattr(sp, "EXAMPLE_PROFILE_PATH", tmp_path / "startup_profile.example.json")

    # Prime the cache with the (missing-file) default.
    assert sp.load_startup_profile().docker is True

    saved = sp.save_startup_profile({"docker": False, "langfuse": True})
    assert saved.docker is False

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["docker"] is False
    assert "models" not in on_disk  # #430: never written back

    # Cache must reflect the new state, not the primed default.
    reread = sp.load_startup_profile()
    assert reread.docker is False
