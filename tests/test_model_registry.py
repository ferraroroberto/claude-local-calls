"""Registry parsing and per-host filtering."""

from __future__ import annotations

import yaml

from src import host_profile, model_registry


def _write_config(tmp_path, content: dict):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump(content), encoding="utf-8")
    return cfg


def _patch_config_path(monkeypatch, cfg_path):
    # _load_config in both modules reads module-level CONFIG_PATH at call time,
    # so monkeypatching the attribute (no reload) is enough.
    monkeypatch.setattr(host_profile, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(model_registry, "CONFIG_PATH", cfg_path, raising=False)


def test_resolves_hostname_match(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, {
        "hub": {"port": 8000},
        "hosts": {
            "pc":  {"platform": "win32", "hostname": "TEST-PC", "enabled": ["qwen"]},
            "mac": {"platform": "darwin", "default": True, "enabled": []},
        },
        "models": {
            "qwen": {"display_name": "qwen3.5-9b", "backend": "openai", "port": 8081},
            "claude": {"display_name": "claude-haiku-4-5", "backend": "claude"},
        },
    })
    _patch_config_path(monkeypatch, cfg)
    monkeypatch.delenv("CLAUDE_LOCAL_CALLS_HOST", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "test-pc")

    prof = host_profile.resolve()
    assert prof.id == "pc"
    assert prof.enabled == ["qwen"]

    ids = {m.id for m in model_registry.enabled_models()}
    assert "claude" in ids and "qwen" in ids


def test_env_override_wins(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, {
        "hub": {"port": 8000},
        "hosts": {
            "pc":  {"platform": "win32", "default": True, "enabled": ["qwen", "glm"]},
            "mac": {"platform": "darwin", "enabled": ["qwen"]},
        },
        "models": {
            "qwen": {"display_name": "qwen3.5-9b", "backend": "openai", "port": 8081},
            "glm":  {"display_name": "glm-4.5-air", "backend": "openai", "port": 8082},
            "claude": {"display_name": "claude-haiku-4-5", "backend": "claude"},
        },
    })
    _patch_config_path(monkeypatch, cfg)
    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "mac")

    prof = host_profile.resolve()
    assert prof.id == "mac"
    ids = {m.id for m in model_registry.enabled_models()}
    assert "glm" not in ids
    assert "qwen" in ids


def test_resolve_by_alias(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, {
        "hub": {"port": 8000},
        "hosts": {
            "pc": {"platform": "win32", "default": True, "enabled": ["qwen"]},
        },
        "models": {
            "qwen": {
                "display_name": "qwen3.5-9b",
                "backend": "openai",
                "port": 8081,
                "aliases": ["qwen", "qwen3.5"],
            },
        },
    })
    _patch_config_path(monkeypatch, cfg)
    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "pc")

    m = model_registry.resolve("qwen3.5")
    assert m is not None
    assert m.id == "qwen"
    assert model_registry.resolve("nonexistent") is None


def test_model_url_from_port(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, {
        "hub": {"port": 8000},
        "hosts": {"pc": {"platform": "win32", "default": True, "enabled": ["qwen"]}},
        "models": {
            "qwen": {"display_name": "qwen3.5-9b", "backend": "openai", "port": 8081},
        },
    })
    _patch_config_path(monkeypatch, cfg)
    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "pc")

    m = model_registry.resolve("qwen3.5-9b")
    assert m.url == "http://127.0.0.1:8081/v1"
