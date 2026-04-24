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


def test_gemma_per_host_filtering(tmp_path, monkeypatch):
    """All three gemma rows must show on pc-cuda and stay hidden on mac-mini-m4."""
    cfg = _write_config(tmp_path, {
        "hub": {"port": 8000},
        "hosts": {
            "pc-cuda":     {"platform": "win32", "default": True, "enabled": ["qwen", "glm", "gemma3_12b", "gemma3_27b", "gemma3n_e4b"]},
            "mac-mini-m4": {"platform": "darwin", "enabled": ["qwen"]},
        },
        "models": {
            "qwen":        {"display_name": "qwen3.5-9b",     "backend": "openai", "port": 8081},
            "glm":         {"display_name": "glm-4.5-air",    "backend": "openai", "port": 8082},
            "gemma3_12b":  {"display_name": "gemma3-12b-it",  "backend": "openai", "port": 8083},
            "gemma3_27b":  {"display_name": "gemma3-27b-it",  "backend": "openai", "port": 8084},
            "gemma3n_e4b": {"display_name": "gemma3n-e4b-it", "backend": "openai", "port": 8085},
        },
    })
    _patch_config_path(monkeypatch, cfg)

    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "pc-cuda")
    names_pc = {m.display_name for m in model_registry.enabled_models()}
    assert {"qwen3.5-9b", "glm-4.5-air", "gemma3-12b-it", "gemma3-27b-it", "gemma3n-e4b-it"} <= names_pc
    assert model_registry.resolve("gemma3-12b-it").port == 8083
    assert model_registry.resolve("gemma3-27b-it").port == 8084
    assert model_registry.resolve("gemma3n-e4b-it").port == 8085

    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "mac-mini-m4")
    names_mac = {m.display_name for m in model_registry.enabled_models()}
    assert "gemma3-12b-it" not in names_mac
    assert "gemma3-27b-it" not in names_mac
    assert "gemma3n-e4b-it" not in names_mac
    assert model_registry.resolve("gemma3-12b-it") is None
    assert model_registry.resolve("gemma3-27b-it") is None
    assert model_registry.resolve("gemma3n-e4b-it") is None


def test_whisper_entry(tmp_path, monkeypatch):
    """Whisper is a distinct backend; runs on 8090, surfaces on pc-cuda only."""
    cfg = _write_config(tmp_path, {
        "hub": {"port": 8000},
        "hosts": {
            "pc-cuda":     {"platform": "win32", "default": True, "enabled": ["qwen", "whisper"]},
            "mac-mini-m4": {"platform": "darwin", "enabled": ["qwen"]},
        },
        "models": {
            "qwen":    {"display_name": "qwen3.5-9b",    "backend": "openai",  "port": 8081},
            "whisper": {
                "display_name": "whisper-large-v3-turbo",
                "backend": "whisper",
                "engine": "whisper-server",
                "port": 8090,
                "hf_repo": "ggerganov/whisper.cpp",
                "hf_pattern": "ggml-large-v3-turbo.bin",
                "model_path": "models/ggml-large-v3-turbo.bin",
                "args": ["--threads", "4", "--gpu", "1"],
            },
        },
    })
    _patch_config_path(monkeypatch, cfg)

    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "pc-cuda")
    m = model_registry.resolve("whisper-large-v3-turbo")
    assert m is not None
    assert m.id == "whisper"
    assert m.backend == "whisper"
    assert m.engine == "whisper-server"
    assert m.port == 8090
    assert m.url == "http://127.0.0.1:8090/v1"
    assert "--gpu" in m.args

    monkeypatch.setenv("CLAUDE_LOCAL_CALLS_HOST", "mac-mini-m4")
    assert model_registry.resolve("whisper-large-v3-turbo") is None


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
