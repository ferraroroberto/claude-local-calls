"""Testing page — run unit tests and the end-to-end smoke test."""

from __future__ import annotations

import os
import subprocess
import sys

import streamlit as st

from src import server_process as sp


def _child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(sp.PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_child_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return 124, f"timeout after {timeout}s\n{e.stdout or ''}\n{e.stderr or ''}"
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, out


def render() -> None:
    st.title("✅ Testing")
    st.caption(
        "Unit tests run in isolation (no real `claude` calls). The smoke "
        "test hits the live server and makes two real calls through your "
        "Claude subscription."
    )

    reachable = sp.is_reachable()
    st.markdown(
        f"Server status: **{'reachable ✅' if reachable else 'unreachable ❌'}** "
        f"at `{sp.BASE_URL}`"
    )

    st.divider()
    st.subheader("Unit tests (pytest)")

    if st.button("Run unit tests", type="primary", key="run_unit"):
        with st.spinner("pytest -q …"):
            code, out = _run(
                [sys.executable, "-m", "pytest", "-q", "tests"],
                timeout=60,
            )
        (st.success if code == 0 else st.error)(f"exit code {code}")
        st.code(out or "(no output)", language="log")

    st.divider()
    st.subheader("End-to-end smoke test")
    st.caption(
        "Requires the server running (start it in the **🛰 Server** tab). "
        "Hits `/v1/messages` via raw HTTP and via the official `anthropic` "
        "SDK with `base_url` overridden."
    )

    if st.button("Run smoke test", type="primary", disabled=not reachable, key="run_smoke"):
        with st.spinner("scripts/smoke_test.py …"):
            code, out = _run(
                [sys.executable, "scripts/smoke_test.py"],
                timeout=180,
            )
        (st.success if code == 0 else st.error)(f"exit code {code}")
        st.code(out or "(no output)", language="log")

    if not reachable:
        st.info("Start the server in the **🛰 Server** tab to enable the smoke test.")
