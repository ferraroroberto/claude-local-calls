"""Sanity tests for src.install — run every check, assert shape + fix wiring."""

from __future__ import annotations

import os

os.environ.setdefault("CLAUDE_LOCAL_CALLS_HOST", "pc-cuda")

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
    assert install_mod.fix_fn_for(Check("x", "x", "missing", fix_id=None)) is None
    assert install_mod.fix_fn_for(Check("x", "x", "ok")) is None
