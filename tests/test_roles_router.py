"""Unit tests for app_web/routers/roles.py (issue #373).

GET /admin/api/roles reads config/models.yaml's `roles:` section through the
same host_profile._load_config() cache model_registry.audio_role_chain()
uses, and returns a flat role_key -> {model_id, display_name, notes,
fallback} map — audio sub-roles dotted (audio.transcribe, ...) so a caller
never has to know the YAML's nesting shape.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from fastapi.testclient import TestClient  # noqa: E402

from src import model_registry  # noqa: E402
from src import server as server_mod  # noqa: E402


_BASE_CONFIG = {
    "hub": {"port": 8000},
    "hosts": {
        "pc": {"platform": "win32", "default": True, "enabled": ["qwen", "parakeet", "whisper", "piper"]},
    },
    "models": {
        "qwen": {"display_name": "qwen3.5-9b", "backend": "openai", "port": 8081},
        "gemma": {"display_name": "gemma4-26b-a4b-it", "backend": "openai", "port": 8087},
        "parakeet": {"display_name": "Parakeet", "backend": "openai", "port": 8092},
        "whisper": {"display_name": "whisper-large-v3-turbo", "backend": "whisper", "port": 8090},
        "piper": {"display_name": "piper-tts", "backend": "tts", "port": 8096},
    },
    "roles": {
        "agentic_light": {"model_id": "qwen", "notes": "fast lane"},
        "agentic_heavy": {"model_id": "gemma", "notes": "deep lane"},
        "audio": {
            "transcribe": {"model_id": "parakeet", "fallback": ["whisper"]},
            "translate": {"model_id": "whisper"},
            "speech": {"model_id": "piper"},
        },
    },
}


def test_get_roles_flattens_audio_and_resolves_display_names(write_config):
    write_config(_BASE_CONFIG)

    client = TestClient(server_mod.app)
    r = client.get("/admin/api/roles")
    assert r.status_code == 200, r.text
    body = r.json()["roles"]

    assert set(body) == {
        "agentic_light", "agentic_heavy",
        "audio.transcribe", "audio.translate", "audio.speech",
    }

    assert body["agentic_light"] == {
        "model_id": "qwen", "display_name": "qwen3.5-9b",
        "notes": "fast lane", "fallback": [],
    }
    assert body["agentic_heavy"]["display_name"] == "gemma4-26b-a4b-it"

    # Audio role carries its ordered fallback chain, display names resolved.
    assert body["audio.transcribe"] == {
        "model_id": "parakeet", "display_name": "Parakeet",
        "notes": None, "fallback": ["whisper"],
    }
    # A role with no fallback configured yields an empty list, not null/missing.
    assert body["audio.translate"]["fallback"] == []
    assert body["audio.speech"]["model_id"] == "piper"


def test_get_roles_unknown_model_id_falls_back_to_the_raw_id(write_config):
    """A role pointing at a model_id with no matching row (typo, or the row was
    since removed from `models:`) still returns a usable entry — the id itself
    stands in for display_name rather than erroring the whole endpoint."""
    cfg = dict(_BASE_CONFIG)
    cfg["roles"] = {"agentic_light": {"model_id": "ghost-model"}}
    write_config(cfg)

    client = TestClient(server_mod.app)
    body = client.get("/admin/api/roles").json()["roles"]
    assert body["agentic_light"]["model_id"] == "ghost-model"
    assert body["agentic_light"]["display_name"] == "ghost-model"
    assert body["agentic_light"]["fallback"] == []


def test_get_roles_empty_section_returns_empty_map(write_config):
    cfg = dict(_BASE_CONFIG)
    cfg.pop("roles", None)
    write_config(cfg)

    client = TestClient(server_mod.app)
    r = client.get("/admin/api/roles")
    assert r.status_code == 200, r.text
    assert r.json() == {"roles": {}}
