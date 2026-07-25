"""src/config_write.py — the git-backed models.yaml write path (#424).

Three legs, each a hard safety property of the write-through-to-git design:

* **Validation** — schema + the #375 VRAM budget as a config-time hard
  gate, all computed before any file/git side effect.
* **Comment-preserving YAML editing** — ruamel round-trip on a copy of the
  real config/models.yaml: only the intended lines change, every comment
  survives (the file's comments are load-bearing; a lossy rewrite is a
  regression even if semantically equal).
* **Git transaction** — on throwaway temp repos: commits exactly the one
  file with the config-bot message and pushes; refuses a dirty tree or a
  non-main branch; a failed push rolls file *and* history back.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml as pyyaml

os.environ.setdefault("LOCAL_LLM_HUB_HOST", "tower")

from src import config_write as cw  # noqa: E402
from src.host_profile import CONFIG_PATH  # noqa: E402


def _chain(*ids, cpu=()):
    return [{"id": i, "cpu": i in cpu} for i in ids]


# --------------------------------------------------------------- validation

def test_validate_ok_for_status_quo_orpheus():
    """The live config's own orpheus placement must validate clean — the
    gate can never reject the state the fleet already runs."""
    assert cw.validate_placement("orpheus", _chain("tower", "gaming"), "eager", None) == []


def test_validate_rejects_unknown_model():
    errs = cw.validate_placement("nope", _chain("tower"), "eager", None)
    assert errs and "unknown model" in errs[0]


def test_validate_rejects_subscription_and_virtual_rows():
    assert "subscription" in cw.validate_placement(
        "claude_haiku", _chain("tower"), "eager", None
    )[0]
    assert "virtual alias" in cw.validate_placement(
        "qwen35_4b_nothink", _chain("tower"), "eager", None
    )[0]


def test_validate_rejects_unknown_host_and_duplicates():
    errs = cw.validate_placement("orpheus", _chain("tower", "atlantis"), "eager", None)
    assert any("unknown host" in e for e in errs)
    errs = cw.validate_placement(
        "orpheus", [{"id": "tower", "cpu": False}, {"id": "tower", "cpu": True}],
        "eager", None,
    )
    assert any("repeats a host" in e for e in errs)


def test_validate_rejects_empty_chain():
    errs = cw.validate_placement("orpheus", [], "eager", None)
    assert any("at least one host" in e for e in errs)


def test_validate_enforces_cross_enable_contract():
    """openclaw declares no enabled: list — placing orpheus there must fail
    the every-chain-member-cross-enables rule."""
    errs = cw.validate_placement("orpheus", _chain("tower", "openclaw"), "eager", None)
    assert any("openclaw" in e and "enabled" in e for e in errs)


def test_validate_rejects_bad_startup_and_idle_shapes():
    errs = cw.validate_placement("orpheus", _chain("tower", "gaming"), "sometimes", None)
    assert any("startup must be" in e for e in errs)
    errs = cw.validate_placement("orpheus", _chain("tower", "gaming"), "on_demand", 0)
    assert any("positive integer" in e for e in errs)
    errs = cw.validate_placement("orpheus", _chain("tower", "gaming"), "eager", 30)
    assert any("only applies to startup: on_demand" in e for e in errs)


def test_validate_hard_rejects_vram_overcommit():
    """Flipping gemma4_26b (13400 MB) to eager on tower must overcommit the
    16384 MB ceiling against the other eager GPU rows — a hard 400, with the
    arithmetic in the message so the UI rejection is self-explanatory."""
    errs = cw.validate_placement("gemma4_26b", _chain("tower"), "eager", None)
    assert len(errs) == 1
    assert "overcommits" in errs[0]
    assert "16384 MB ceiling" in errs[0]


def test_validate_on_demand_skips_vram_gate():
    """The same gemma4_26b placement is fine as on_demand — transient
    residency is src.on_demand's advisory warning, not a config error."""
    assert cw.validate_placement("gemma4_26b", _chain("tower"), "on_demand", 30) == []


def test_validate_cpu_tier_holds_no_vram():
    """A cpu-flagged chain member contributes nothing — whisper's degraded
    tower tier must not count 2000 MB against tower's eager set."""
    errs = cw.validate_placement(
        "whisper", _chain("gaming", "mac-mini-m4", "tower", cpu=("tower",)),
        "eager", None,
    )
    assert errs == []


def test_normalize_chain_accepts_strings_and_dicts():
    chain, errs = cw.normalize_chain(["tower", {"id": "gaming", "cpu": True}])
    assert errs == []
    assert chain == [{"id": "tower", "cpu": False}, {"id": "gaming", "cpu": True}]
    _, errs = cw.normalize_chain("tower")
    assert errs and "must be a list" in errs[0]
    _, errs = cw.normalize_chain([{"cpu": True}])
    assert errs and "missing its host id" in errs[0]


# ------------------------------------------------------------- yaml editing

@pytest.fixture()
def yaml_copy(tmp_path: Path) -> Path:
    """A byte-exact copy of the real config/models.yaml — the strongest
    possible round-trip fixture: whatever ruamel mangles here it would
    mangle live."""
    dst = tmp_path / "models.yaml"
    shutil.copyfile(CONFIG_PATH, dst)
    return dst


def _changed_lines(before: str, after: str):
    """(removed, added) line lists, ignoring order-stable identical lines."""
    import difflib

    removed, added = [], []
    for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0
    ):
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return removed, added


def test_yaml_edit_scalar_touches_exactly_one_line(yaml_copy: Path):
    before = yaml_copy.read_text(encoding="utf-8")
    changed = cw.edit_models_yaml(
        yaml_copy, "gemma4_26b", _chain("tower"), "on_demand", 31
    )
    assert changed is True
    after = yaml_copy.read_text(encoding="utf-8")
    removed, added = _changed_lines(before, after)
    assert removed == ["    idle_unload_minutes: 30"]
    assert added == ["    idle_unload_minutes: 31"]
    # Semantics round-trip through the ordinary pyyaml reader too.
    data = pyyaml.safe_load(after)
    assert data["models"]["gemma4_26b"]["idle_unload_minutes"] == 31


def test_yaml_edit_noop_leaves_file_byte_identical(yaml_copy: Path):
    before = yaml_copy.read_bytes()
    changed = cw.edit_models_yaml(
        yaml_copy, "gemma4_26b", _chain("tower"), "on_demand", 30
    )
    assert changed is False
    assert yaml_copy.read_bytes() == before


def test_yaml_edit_chain_rewrite_preserves_comments(yaml_copy: Path):
    """Reordering whisper's chain rewrites only its `hosts:` line; the huge
    comment block above it (the #405 pilot rationale) survives untouched."""
    before = yaml_copy.read_text(encoding="utf-8")
    changed = cw.edit_models_yaml(
        yaml_copy, "whisper",
        _chain("mac-mini-m4", "gaming", "tower", cpu=("tower",)),
        "eager", None,
    )
    assert changed is True
    after = yaml_copy.read_text(encoding="utf-8")
    removed, added = _changed_lines(before, after)
    assert len(removed) == 1 and removed[0].strip().startswith("hosts:")
    assert len(added) == 1
    assert "mac-mini-m4" in added[0] and added[0].index("mac-mini-m4") < added[0].index("gaming")
    # Every comment line of the original file is still present.
    before_comments = [l for l in before.splitlines() if l.lstrip().startswith("#")]
    after_lines = set(after.splitlines())
    missing = [c for c in before_comments if c not in after_lines]
    assert missing == []
    data = pyyaml.safe_load(after)
    assert data["models"]["whisper"]["hosts"] == [
        "mac-mini-m4", "gaming", {"id": "tower", "cpu": True},
    ]


def test_yaml_edit_single_host_to_chain_and_new_keys(yaml_copy: Path):
    """A bare `host:` row (chatterbox) edited to a two-host on-demand chain:
    `host:` is replaced by `hosts:` in place, and startup/idle are inserted
    as new keys — all other rows untouched."""
    before = yaml_copy.read_text(encoding="utf-8")
    changed = cw.edit_models_yaml(
        yaml_copy, "chatterbox", _chain("tower", "gaming"), "on_demand", 15
    )
    assert changed is True
    after = yaml_copy.read_text(encoding="utf-8")
    data = pyyaml.safe_load(after)
    row = data["models"]["chatterbox"]
    assert "host" not in row
    assert row["hosts"] == ["tower", "gaming"]
    assert row["startup"] == "on_demand"
    assert row["idle_unload_minutes"] == 15
    # No collateral damage: every other model row parses identically.
    before_models = pyyaml.safe_load(before)["models"]
    after_models = data["models"]
    for mid in before_models:
        if mid != "chatterbox":
            assert after_models[mid] == before_models[mid], mid


def test_yaml_edit_removes_idle_key_when_cleared(yaml_copy: Path):
    changed = cw.edit_models_yaml(
        yaml_copy, "gemma4_26b", _chain("tower"), "on_demand", None
    )
    assert changed is True
    data = pyyaml.safe_load(yaml_copy.read_text(encoding="utf-8"))
    assert "idle_unload_minutes" not in data["models"]["gemma4_26b"]


# ----------------------------------------------------------- git transaction

def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True, creationflags=flags,
    )


@pytest.fixture()
def git_pair(tmp_path: Path):
    """(work, origin): a bare origin and a clone with config/models.yaml
    committed on main — the minimal shape of the live checkout."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _run_git(origin, "init", "--bare", "--initial-branch=main")
    work = tmp_path / "work"
    _run_git(tmp_path, "clone", str(origin), str(work))
    _run_git(work, "config", "user.name", "test")
    _run_git(work, "config", "user.email", "test@example.com")
    # Isolate from the machine's global hooks (this box enforces an
    # author-email allowlist via core.hooksPath) — point the repo at an
    # empty hooks dir so the transaction under test is all that runs.
    hooks = tmp_path / "no-hooks"
    hooks.mkdir()
    _run_git(work, "config", "core.hooksPath", str(hooks))
    cfg = work / "config" / "models.yaml"
    cfg.parent.mkdir()
    cfg.write_text("models: {}\n", encoding="utf-8")
    (work / "other.txt").write_text("hello\n", encoding="utf-8")
    _run_git(work, "add", ".")
    _run_git(work, "commit", "-m", "init")
    _run_git(work, "push", "origin", "main")
    return work, origin


def _log_origin(origin: Path) -> str:
    return _run_git(origin, "log", "--format=%s|%an", "main").stdout


def test_commit_and_push_happy_path(git_pair):
    work, origin = git_pair
    (work / "config" / "models.yaml").write_text("models: {edited: 1}\n", encoding="utf-8")
    sha = cw.commit_and_push(work, cw.CONFIG_RELPATH, "config: edited placement via admin UI")
    assert sha
    log = _log_origin(origin)
    assert log.splitlines()[0] == "config: edited placement via admin UI|local-llm-hub config-bot"


def test_preflight_refuses_dirty_tracked_file(git_pair):
    work, _ = git_pair
    (work / "other.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(cw.ConfigWriteError) as exc:
        cw.git_preflight(work, cw.CONFIG_RELPATH)
    assert exc.value.status == 409
    assert "uncommitted tracked changes" in str(exc.value)


def test_preflight_ignores_untracked_scratch(git_pair):
    work, _ = git_pair
    (work / "scratch.tmp").write_text("junk\n", encoding="utf-8")
    cw.git_preflight(work, cw.CONFIG_RELPATH)  # must not raise


def test_preflight_refuses_off_main_branch(git_pair):
    work, _ = git_pair
    _run_git(work, "checkout", "-b", "feature/x")
    with pytest.raises(cw.ConfigWriteError) as exc:
        cw.git_preflight(work, cw.CONFIG_RELPATH)
    assert exc.value.status == 409
    assert "feature/x" in str(exc.value)


def test_commit_refuses_when_unrelated_file_also_dirty(git_pair):
    """The only-models.yaml-in-the-commit guarantee: a concurrent edit to
    another tracked file aborts before anything is staged."""
    work, origin = git_pair
    (work / "config" / "models.yaml").write_text("models: {edited: 1}\n", encoding="utf-8")
    (work / "other.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(cw.ConfigWriteError) as exc:
        cw.commit_and_push(work, cw.CONFIG_RELPATH, "config: nope")
    assert exc.value.status == 409
    assert "only change" in str(exc.value)
    # Nothing was committed or pushed.
    assert "nope" not in _log_origin(origin)


def test_push_failure_rolls_back_file_and_history(git_pair):
    work, origin = git_pair
    pre = (work / "config" / "models.yaml").read_bytes()
    pre_head = _run_git(work, "rev-parse", "HEAD").stdout.strip()
    (work / "config" / "models.yaml").write_text("models: {edited: 1}\n", encoding="utf-8")
    # Point origin at a path that cannot exist → push must fail.
    _run_git(work, "remote", "set-url", "origin", str(work / "no-such-remote.git"))
    with pytest.raises(cw.ConfigWriteError) as exc:
        cw.commit_and_push(work, cw.CONFIG_RELPATH, "config: doomed")
    assert exc.value.status == 502
    assert "NOT persisted" in str(exc.value)
    assert (work / "config" / "models.yaml").read_bytes() == pre
    assert _run_git(work, "rev-parse", "HEAD").stdout.strip() == pre_head
    assert _run_git(work, "status", "--porcelain").stdout.strip() == ""


# ------------------------------------------------------------- write host

def test_write_host_gate(monkeypatch):
    assert cw.write_host_id() == "tower"
    assert cw.is_write_host() is True
    monkeypatch.setenv("LOCAL_LLM_HUB_HOST", "mac-mini-m4")
    assert cw.is_write_host() is False
