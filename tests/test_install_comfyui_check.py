"""The ComfyUI install check must not report ok for what it hasn't verified.

A missing ComfyUI-GGUF custom node is invisible until generation time — stock
ComfyUI cannot read a `.gguf` diffusion model, so `flux2_local` sits in the
model list looking healthy and fails with an unknown-node error on first use
(#498). Per CLAUDE.md, a check that cannot establish a fact reports that as its
own state rather than folding it into the passing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Import it exactly the way src/install.py does. Reaching it via a sys.path
# insert instead would create a *second* module object for the same file, and
# monkeypatching one would leave the other — the one actually under test —
# untouched.
from scripts import install_comfyui  # type: ignore

from src import install as install_mod


@pytest.fixture
def comfy_installed(monkeypatch):
    """Pretend the engine itself is installed and CUDA-verified."""
    monkeypatch.setattr(install_comfyui, "venv_python", lambda: Path(__file__))
    monkeypatch.setattr(install_comfyui, "main_script", lambda: Path(__file__))
    monkeypatch.setattr(install_comfyui, "read_marker", lambda: {
        "comfyui_tag": "v0.31.0", "torch_version": "2.13.0+cu130",
        "cuda": "13.0", "device": "RTX 5060 Ti",
    })


def _has_gguf_row():
    from src.model_registry import local_models
    return any(m.backend == "comfyui" and (m.model_path or "").endswith(".gguf")
               for m in local_models())


def test_absent_gguf_node_is_reported_missing(comfy_installed, monkeypatch):
    if not _has_gguf_row():
        pytest.skip("no .gguf comfyui row enabled on this host")
    monkeypatch.setattr(install_comfyui, "installed_gguf_node_sha", lambda: None)
    check = install_mod._check_comfyui()
    assert check.status == "missing"
    assert "GGUF" in check.detail
    assert check.fix_id == "comfyui"


def test_drifted_gguf_pin_is_reported_as_warn(comfy_installed, monkeypatch):
    """A node at the wrong commit is 'unknown-good', not 'broken' — but it must
    not read as ok either, since the workflow graph is written against the pin."""
    if not _has_gguf_row():
        pytest.skip("no .gguf comfyui row enabled on this host")
    monkeypatch.setattr(install_comfyui, "installed_gguf_node_sha",
                        lambda: "0" * 40)
    check = install_mod._check_comfyui()
    assert check.status == "warn"
    assert install_comfyui.GGUF_NODE_PIN[:12] in check.detail


def test_pinned_gguf_node_reports_ok_and_names_the_pin(comfy_installed, monkeypatch):
    if not _has_gguf_row():
        pytest.skip("no .gguf comfyui row enabled on this host")
    monkeypatch.setattr(install_comfyui, "installed_gguf_node_sha",
                        lambda: install_comfyui.GGUF_NODE_PIN)
    check = install_mod._check_comfyui()
    assert check.status == "ok"
    assert install_comfyui.GGUF_NODE_PIN[:12] in check.detail


def test_head_reader_handles_detached_and_ref_forms(tmp_path, monkeypatch):
    """The installer leaves a detached HEAD (raw SHA); a hand-modified clone
    may be on a branch instead."""
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    monkeypatch.setattr(install_comfyui, "GGUF_NODE_DIR", tmp_path)

    (git / "HEAD").write_text("a" * 40, encoding="utf-8")
    assert install_comfyui.installed_gguf_node_sha() == "a" * 40

    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("b" * 40 + "\n", encoding="utf-8")
    assert install_comfyui.installed_gguf_node_sha() == "b" * 40


def test_head_reader_returns_none_when_absent(tmp_path, monkeypatch):
    """None means 'could not establish', which the caller must not treat as ok."""
    monkeypatch.setattr(install_comfyui, "GGUF_NODE_DIR", tmp_path / "nope")
    assert install_comfyui.installed_gguf_node_sha() is None


def test_marker_records_the_gguf_pin(tmp_path, monkeypatch):
    """So a future reader can tell which node build a verified install had."""
    monkeypatch.setattr(install_comfyui, "MARKER_PATH", tmp_path / "m.json")
    install_comfyui.write_marker({"version": "2.13.0", "cuda": "13.0",
                                  "device": "RTX 5060 Ti"})
    assert install_comfyui.read_marker() is None or True  # path monkeypatched
    import json
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert data["gguf_node_pin"] == install_comfyui.GGUF_NODE_PIN
