"""backend_process wiring for the ComfyUI engine (#492).

No process is spawned; these cover the command construction and the
inherit-across-restart matcher.
"""

from __future__ import annotations

import pytest

from src import backend_process as bp
from src.model_registry import SPAWNABLE_BACKENDS, enabled_models


@pytest.fixture
def flux_model():
    model = next((m for m in enabled_models() if m.id == "flux1_local"), None)
    if model is None:
        pytest.skip("flux1_local is not enabled on this host")
    return model


@pytest.fixture
def comfyui_installed(flux_model, tmp_path, monkeypatch):
    """`flux_model` with ComfyUI's *install state* stubbed (#518).

    ``build_command`` validates two on-disk artefacts before it assembles any
    argv — the vendored ComfyUI venv and the checkpoint ``model_path`` names.
    That validation is deliberate and correct (see the comment at its call
    site): a missing 17 GB download must fail there with an actionable message
    rather than as an opaque node error mid-generation.

    But neither artefact is in git, and ``enabled_models()`` is registry state:
    ``flux1_local`` is enabled everywhere the YAML is read, including hosts
    that have never installed ComfyUI. So the three ``build_command`` tests
    below died in the precondition and never reached their own assertions —
    on CI (no venv) since #496, and in every worktree (``vendor/comfyui`` is
    junctioned, ``models/`` is not) on the missing checkpoint.

    Stubbing the install state rather than skipping on it is the point: these
    assertions are about how the hub *composes* the command, not about what is
    installed, so they can and should run everywhere. A skip would retire them
    precisely where nobody is watching — and one of them
    (``test_build_command_binds_loopback_only``) guards a security property on
    a backend that serves an unauthenticated, filesystem-capable UI.

    Redirecting ``PROJECT_ROOT`` also contains ``build_command``'s
    ``data/comfyui/<id>/{user,temp}`` mkdir in ``tmp_path`` instead of letting
    a unit test write into the real working tree.
    """
    monkeypatch.setattr(bp, "PROJECT_ROOT", tmp_path)

    checkpoint = tmp_path / flux_model.model_path
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()

    venv_python = tmp_path / "vendor" / "comfyui" / ".venv" / "python.exe"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.touch()
    monkeypatch.setattr(bp, "comfyui_python", lambda: venv_python)

    return flux_model


def test_comfyui_is_a_spawnable_backend():
    """Without this the row is half-registered: startable by hand, but never
    inherited, never counted for VRAM, and not controllable in the admin UI."""
    assert "comfyui" in SPAWNABLE_BACKENDS


def test_build_command_uses_the_comfyui_venv_and_row_port(comfyui_installed):
    cmd = bp.build_command(comfyui_installed)
    assert cmd[0] == str(bp.comfyui_python())
    assert cmd[1].endswith("main.py")
    assert "--port" in cmd and str(comfyui_installed.port) in cmd


def test_build_command_binds_loopback_only(comfyui_installed):
    """ComfyUI serves an unauthenticated, filesystem-capable UI — unlike the
    llama/whisper backends it must not bind 0.0.0.0."""
    cmd = bp.build_command(comfyui_installed)
    assert "--listen" in cmd
    assert cmd[cmd.index("--listen") + 1] == "127.0.0.1"
    assert "0.0.0.0" not in cmd


def test_build_command_disables_auto_launch(comfyui_installed):
    """ComfyUI opens a browser tab on startup by default, and this backend is
    spawned on demand from a windowless hub process."""
    assert "--disable-auto-launch" in bp.build_command(comfyui_installed)


def test_build_command_still_fails_loud_without_the_checkpoint(flux_model, tmp_path, monkeypatch):
    """The stubbing above must not weaken the guard it stubs (#518).

    ``comfyui_installed`` exists so the argv assertions run on a host with no
    ComfyUI install — not to retire ``build_command``'s missing-download check.
    This pins that check so a future "just make it green" edit cannot quietly
    delete it: with the venv present but the checkpoint absent, the call still
    raises, and the message still names the command that fixes it.
    """
    monkeypatch.setattr(bp, "PROJECT_ROOT", tmp_path)
    venv_python = tmp_path / "vendor" / "comfyui" / ".venv" / "python.exe"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.touch()
    monkeypatch.setattr(bp, "comfyui_python", lambda: venv_python)

    with pytest.raises(RuntimeError, match="checkpoint not found"):
        bp.build_command(flux_model)


def test_inherit_matches_venv_python_by_cmdline_not_exe(flux_model):
    """Regression: on Windows a venv's Scripts/python.exe is a *redirector*, so
    psutil's Process.exe() reports the base interpreter and only cmdline()[0]
    carries the venv path.

    Matching on exe alone silently never fires — ComfyUI is not inherited
    across a hub restart, and since stop() only stops what it owns or
    inherited, the idle-unload watchdog can never reclaim its ~13 GB.
    Observed live: exe = .../pythoncore-3.14-64/python.exe.
    """
    base_interpreter = r"C:\Users\rober\AppData\Local\Python\pythoncore-3.14-64\python.exe"
    cmdline = [str(bp.comfyui_python()), str(bp.VENDOR_COMFYUI / "main.py"),
               "--listen", "127.0.0.1", "--port", "8188"]

    assert bp._looks_like_backend_binary(base_interpreter, flux_model, cmdline) is True


def test_inherit_ignores_an_unrelated_python_on_the_port(flux_model):
    """The generic 'python' fallthrough would adopt any python process holding
    the port; the comfyui branch must be scoped to this repo's own install."""
    cmdline = [r"C:\some\other\.venv\Scripts\python.exe", "-m", "http.server", "8188"]
    assert bp._looks_like_backend_binary(
        r"C:\some\other\.venv\Scripts\python.exe", flux_model, cmdline) is False


def test_is_reachable_probes_system_stats(flux_model, monkeypatch):
    """ComfyUI has neither /health nor /v1/models — the two endpoints the
    generic probe tries."""
    seen = []

    def fake_reachable(base_url, timeout=1.5):
        seen.append(base_url)
        return True

    monkeypatch.setattr("src.comfyui_client.is_reachable", fake_reachable)
    assert bp.is_reachable(flux_model) is True
    assert seen == [f"http://127.0.0.1:{flux_model.port}"]
