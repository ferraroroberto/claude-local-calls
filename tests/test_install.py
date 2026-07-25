"""Sanity tests for src.install — run every check, assert shape + fix wiring."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from src import install as install_mod


def test_run_all_checks_returns_nonempty_report():
    report = install_mod.run_all_checks()
    assert len(report.checks) >= 5
    ids = {c.id for c in report.checks}
    for expected in ("python", "deps", "host", "claude_cli", "gpu", "llama_cpp"):
        assert expected in ids, f"missing check {expected!r}"
    # Every check has a known status glyph.
    for c in report.checks:
        assert c.status in ("ok", "warn", "missing", "error"), c.status


def test_worst_status_ordering():
    from src.install import Check, Report
    r = Report(checks=[
        Check("a", "a", "ok"),
        Check("b", "b", "warn"),
        Check("c", "c", "missing"),
    ])
    assert r.worst_status == "missing"
    assert r.ok is False

    r2 = Report(checks=[Check("a", "a", "ok"), Check("b", "b", "warn")])
    assert r2.worst_status == "warn"
    assert r2.ok is True


def test_fix_fn_for_known_ids():
    from src.install import Check
    assert install_mod.fix_fn_for(Check("x", "x", "missing", fix_id="deps")) is not None
    assert install_mod.fix_fn_for(Check("x", "x", "missing", fix_id="llama_cpp")) is not None
    assert install_mod.fix_fn_for(Check("x", "x", "missing", fix_id="download_qwen")) is not None
    assert install_mod.fix_fn_for(Check("x", "x", "missing", fix_id="systemd_unit")) is not None
    assert install_mod.fix_fn_for(Check("x", "x", "missing", fix_id=None)) is None
    assert install_mod.fix_fn_for(Check("x", "x", "ok")) is None


# --------------------------------------------------------------- systemd unit (#368)


def test_check_systemd_unit_missing_when_unit_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(install_mod, "_systemd_unit_path", lambda: tmp_path / "absent.service")
    c = install_mod._check_systemd_unit()
    assert c.status == "missing"
    assert c.fix_id == "systemd_unit"


def test_check_systemd_unit_warns_when_present_but_inactive(monkeypatch, tmp_path):
    unit = tmp_path / "local-llm-hub.service"
    unit.write_text("unit", encoding="utf-8")
    monkeypatch.setattr(install_mod, "_systemd_unit_path", lambda: unit)

    class _R:
        def __init__(self, out):
            self.stdout = out

    def _fake_run(cmd, **kwargs):
        # cmd = ["systemctl", "is-active"|"is-enabled", name]
        return _R("inactive\n" if cmd[1] == "is-active" else "enabled\n")

    monkeypatch.setattr(install_mod.subprocess, "run", _fake_run)
    c = install_mod._check_systemd_unit()
    assert c.status == "warn"
    assert c.fix_id == "systemd_unit"
    assert "is-active=inactive" in c.detail


def test_check_systemd_unit_ok_when_active(monkeypatch, tmp_path):
    unit = tmp_path / "local-llm-hub.service"
    unit.write_text("unit", encoding="utf-8")
    monkeypatch.setattr(install_mod, "_systemd_unit_path", lambda: unit)

    class _R:
        def __init__(self, out):
            self.stdout = out

    def _fake_run(cmd, **kwargs):
        return _R("active\n" if cmd[1] == "is-active" else "enabled\n")

    monkeypatch.setattr(install_mod.subprocess, "run", _fake_run)
    c = install_mod._check_systemd_unit()
    assert c.status == "ok"
    assert c.fix_id is None


def test_fix_systemd_unit_refuses_off_linux(monkeypatch):
    monkeypatch.setattr(install_mod.sys, "platform", "win32")
    import pytest
    with pytest.raises(RuntimeError, match="only applies on Linux"):
        install_mod._fix_systemd_unit()


def test_gpu_check_uses_nvidia_smi_on_linux(monkeypatch):
    """The Linux branch reuses the same nvidia-smi probe as Windows (#368)."""
    monkeypatch.setattr(install_mod.sys, "platform", "linux")
    sentinel = install_mod.Check("gpu", "GPU / accelerator detected", "ok", "GTX 1070")
    called = {"n": 0}

    def _fake_probe():
        called["n"] += 1
        return sentinel

    monkeypatch.setattr(install_mod, "_nvidia_smi_gpu_check", _fake_probe)
    c = install_mod._check_gpu()
    assert called["n"] == 1
    assert c is sentinel


def _reset_cache(monkeypatch):
    monkeypatch.setattr(install_mod, "_cached_report", None)
    monkeypatch.setattr(install_mod, "_cached_at", 0.0)


def _counting_venv_check(monkeypatch):
    """Replace the cheapest check with a call-counting stub so cache
    hit/miss is observable without shelling out to claude/nvidia-smi."""
    calls = {"n": 0}

    def _stub():
        calls["n"] += 1
        from src.install import Check
        return Check("python", "stub", "ok")

    monkeypatch.setattr(install_mod, "_check_python_venv", _stub)
    return calls


def test_use_cache_true_reuses_a_recent_report(monkeypatch):
    _reset_cache(monkeypatch)
    calls = _counting_venv_check(monkeypatch)

    install_mod.run_all_checks(use_cache=True)
    assert calls["n"] == 1  # cache was empty -> ran fresh

    install_mod.run_all_checks(use_cache=True)
    assert calls["n"] == 1  # cache hit -> did not re-run the battery


def test_use_cache_true_falls_back_to_fresh_when_stale(monkeypatch):
    _reset_cache(monkeypatch)
    calls = _counting_venv_check(monkeypatch)

    install_mod.run_all_checks(use_cache=True)
    assert calls["n"] == 1

    # Simulate the TTL having elapsed.
    monkeypatch.setattr(install_mod, "_cached_at", 0.0)
    install_mod.run_all_checks(use_cache=True)
    assert calls["n"] == 2  # cache expired -> ran fresh again


def test_use_cache_false_always_runs_fresh(monkeypatch):
    _reset_cache(monkeypatch)
    calls = _counting_venv_check(monkeypatch)

    install_mod.run_all_checks()  # default use_cache=False
    install_mod.run_all_checks()
    assert calls["n"] == 2  # no caching applied on the default path


def test_run_all_checks_always_refreshes_the_cache_for_later_use_cache_calls(monkeypatch):
    _reset_cache(monkeypatch)
    calls = _counting_venv_check(monkeypatch)

    install_mod.run_all_checks()  # fresh, non-cached call — still warms the cache
    assert calls["n"] == 1
    install_mod.run_all_checks(use_cache=True)
    assert calls["n"] == 1  # served from the cache the plain call just warmed


def test_linux_llama_build_hint_uses_configured_arch(monkeypatch):
    from scripts import install_llama_cpp

    monkeypatch.setenv("LOCAL_LLM_HUB_CUDA_ARCH", "75")
    hint = install_llama_cpp._linux_cuda_build_hint()
    assert "sm_75" in hint
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in hint
    assert "-DGGML_CUDA=ON" in hint
    assert "llama-server" in hint


def test_linux_whisper_build_hint_uses_configured_arch(monkeypatch):
    from scripts import install_whisper_cpp

    monkeypatch.setenv("LOCAL_LLM_HUB_CUDA_ARCH", "61")
    hint = install_whisper_cpp._linux_cuda_build_hint()
    assert "sm_61" in hint
    assert "-DCMAKE_CUDA_ARCHITECTURES=61" in hint
    assert "whisper-server" in hint
    assert install_whisper_cpp.PINNED_TAG in hint


# --------------------------------------------------------------- macOS from-source build (#413)


def test_find_macos_asset_none_when_no_arm64_asset():
    from scripts import install_whisper_cpp

    release = {"assets": [{"name": "whisper-bin-x64.zip"}, {"name": "whisper-v1.8.6-xcframework.zip"}]}
    assert install_whisper_cpp._find_macos_asset(release) is None


def test_find_macos_asset_matches_when_present():
    from scripts import install_whisper_cpp

    release = {"assets": [{"name": "whisper-bin-x64.zip"}, {"name": "whisper-bin-arm64.zip"}]}
    picked = install_whisper_cpp._find_macos_asset(release)
    assert picked["name"] == "whisper-bin-arm64.zip"


def test_pick_assets_no_longer_handles_darwin(monkeypatch):
    from scripts import install_whisper_cpp
    import pytest

    monkeypatch.setattr(install_whisper_cpp.sys, "platform", "darwin")
    with pytest.raises(install_whisper_cpp.InstallError, match="unsupported platform"):
        install_whisper_cpp._pick_assets({"assets": []})


def test_macos_find_cmake_prefers_path(monkeypatch):
    from scripts import install_whisper_cpp

    monkeypatch.setattr(install_whisper_cpp.shutil, "which", lambda name: "/usr/bin/cmake")
    assert install_whisper_cpp._macos_find_cmake() == "/usr/bin/cmake"


def test_macos_find_cmake_falls_back_to_homebrew_path(monkeypatch):
    from scripts import install_whisper_cpp

    monkeypatch.setattr(install_whisper_cpp.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        install_whisper_cpp.Path, "exists",
        lambda self: self.as_posix() == "/opt/homebrew/bin/cmake",
    )
    assert install_whisper_cpp._macos_find_cmake() == "/opt/homebrew/bin/cmake"


def test_macos_find_cmake_none_when_absent_everywhere(monkeypatch):
    from scripts import install_whisper_cpp

    monkeypatch.setattr(install_whisper_cpp.shutil, "which", lambda name: None)
    monkeypatch.setattr(install_whisper_cpp.Path, "exists", lambda self: False)
    assert install_whisper_cpp._macos_find_cmake() is None


def test_macos_build_prereqs_raises_actionable_message_when_cmake_missing(monkeypatch):
    from scripts import install_whisper_cpp
    import pytest

    monkeypatch.setattr(install_whisper_cpp, "_macos_find_cmake", lambda: None)
    with pytest.raises(install_whisper_cpp.InstallError, match="brew install cmake"):
        install_whisper_cpp._macos_build_prereqs()


def test_macos_build_prereqs_raises_actionable_message_when_clt_missing(monkeypatch):
    from scripts import install_whisper_cpp
    import pytest

    monkeypatch.setattr(install_whisper_cpp, "_macos_find_cmake", lambda: "/opt/homebrew/bin/cmake")

    class _R:
        returncode = 2

    monkeypatch.setattr(install_whisper_cpp.subprocess, "run", lambda *a, **k: _R())
    with pytest.raises(install_whisper_cpp.InstallError, match="xcode-select --install"):
        install_whisper_cpp._macos_build_prereqs()


def test_macos_build_prereqs_ok_returns_cmake_path(monkeypatch):
    from scripts import install_whisper_cpp

    monkeypatch.setattr(install_whisper_cpp, "_macos_find_cmake", lambda: "/opt/homebrew/bin/cmake")

    class _R:
        returncode = 0

    monkeypatch.setattr(install_whisper_cpp.subprocess, "run", lambda *a, **k: _R())
    assert install_whisper_cpp._macos_build_prereqs() == "/opt/homebrew/bin/cmake"


def test_macos_build_from_source_flattens_dylibs_and_fixes_rpath(monkeypatch, tmp_path):
    """Exercises the real flatten + symlink-preservation + rpath-fix control
    flow against a synthetic build tree shaped like the real whisper.cpp
    cmake output (nested dirs, versioned real .dylib + symlink chain) —
    verified against an actual macOS build on mac-mini-m4 during development
    (#413); this test locks in the Python-side file/process orchestration."""
    from scripts import install_whisper_cpp

    vendor_dir = tmp_path / "vendor"
    monkeypatch.setattr(install_whisper_cpp, "VENDOR_DIR", vendor_dir)
    monkeypatch.setattr(install_whisper_cpp.sys, "platform", "darwin")
    monkeypatch.setattr(install_whisper_cpp, "_macos_build_prereqs", lambda: "/opt/homebrew/bin/cmake")

    def _fake_mkdtemp(prefix=None):
        d = tmp_path / "tmproot"
        d.mkdir()
        return str(d)

    monkeypatch.setattr(install_whisper_cpp.tempfile, "mkdtemp", _fake_mkdtemp)

    install_name_tool_calls = []

    def _fake_run(cmd, **kwargs):
        if cmd[0] == "git":
            src_dir = Path(cmd[-1])
            src_dir.mkdir(parents=True)
        elif cmd[0] == "/opt/homebrew/bin/cmake" and cmd[1] == "-B":
            build_dir = Path(cmd[2])
            build_dir.mkdir(parents=True)
        elif cmd[0] == "/opt/homebrew/bin/cmake" and cmd[1] == "--build":
            build_dir = Path(cmd[2])
            bin_dir = build_dir / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "whisper-server").write_bytes(b"MACHO-BINARY")
            base_real = build_dir / "ggml" / "src" / "libggml-base.0.13.1.dylib"
            base_real.parent.mkdir(parents=True)
            base_real.write_bytes(b"MACHO-DYLIB")
            # Real whisper.cpp cmake output symlinks the unversioned/soname
            # names to the fully-versioned real file. Creating actual OS
            # symlinks needs elevated privilege on Windows dev machines, so
            # these are plain files here — is_symlink() is monkeypatched
            # below by name to stand in for that OS-level fact instead.
            (build_dir / "ggml" / "src" / "libggml-base.0.dylib").write_bytes(b"MACHO-DYLIB")
            (build_dir / "ggml" / "src" / "libggml-base.dylib").write_bytes(b"MACHO-DYLIB")
        elif cmd[0] == "install_name_tool":
            install_name_tool_calls.append(cmd[-1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(install_whisper_cpp.subprocess, "run", _fake_run)

    symlink_names = {"libggml-base.0.dylib", "libggml-base.dylib"}
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name in symlink_names)

    install_whisper_cpp._macos_build_from_source()

    assert (vendor_dir / "whisper-server").read_bytes() == b"MACHO-BINARY"
    assert (vendor_dir / "libggml-base.0.13.1.dylib").read_bytes() == b"MACHO-DYLIB"
    assert (vendor_dir / "libggml-base.0.dylib").is_symlink()
    assert (vendor_dir / "libggml-base.dylib").is_symlink()
    # Only the real (non-symlink) files get install_name_tool -add_rpath —
    # symlinks share the target's inode and would error as a duplicate.
    assert sorted(install_name_tool_calls) == sorted([
        str(vendor_dir / "whisper-server"),
        str(vendor_dir / "libggml-base.0.13.1.dylib"),
    ])
    # The temp clone/build tree is cleaned up afterward.
    assert not (tmp_path / "tmproot").exists()


def test_detect_cuda_arch_empty_without_nvidia_smi(monkeypatch):
    from scripts import _lib

    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_lib.subprocess, "run", _boom)
    assert _lib.detect_cuda_arch() == ""


def test_kokoro_installer_warms_spanish_voice_assets(tmp_path, monkeypatch):
    from scripts import install_tts

    model_path = tmp_path / "models" / "kokoro" / "kokoro-v1.0.int8.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"onnx")
    (model_path.parent / "voices-v1.0.bin").write_bytes(b"voices")
    calls: list[dict] = []

    class _FakeKokoro:
        def __init__(self, model, voices):
            assert model == str(model_path)
            assert voices == str(model_path.parent / "voices-v1.0.bin")

        def create(self, text, **kwargs):
            calls.append({"text": text, **kwargs})
            return [0.0], 24000

    monkeypatch.setattr(install_tts, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(install_tts, "enabled_models", lambda: [SimpleNamespace(
        backend="tts",
        tts_engine="kokoro",
        model_path="models/kokoro/kokoro-v1.0.int8.onnx",
    )])
    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=_FakeKokoro))

    install_tts._warm_kokoro()

    assert [(call["voice"], call["lang"]) for call in calls] == [
        ("am_michael", "en-us"),
        ("ef_dora", "es"),
        ("em_alex", "es"),
    ]
