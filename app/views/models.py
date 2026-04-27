"""Per-backend start/stop/health/log panel for each enabled model."""

from __future__ import annotations

import streamlit as st

from src import backend_process as lp
from src.model_registry import enabled_models


def _render_claude_card(m) -> None:
    st.subheader(f"🤖 {m.display_name}")
    st.caption(
        "Anthropic subscription via `claude -p`. Lives inside the hub process; "
        "controlled from the 🛰 Server tab."
    )
    if m.aliases:
        st.caption("Aliases: " + ", ".join(m.aliases))


def _render_local_card(m) -> None:
    running = lp.is_running(m.id)
    reachable = lp.is_reachable(m) if running else False

    is_whisper = m.backend == "whisper" or m.engine == "whisper-server"
    glyph = "🎙" if is_whisper else "🦙"
    engine_label = "whisper-server" if is_whisper else "llama-server"
    st.subheader(f"{glyph} {m.display_name}")
    st.caption(f"`{engine_label}` on :{m.port} — id `{m.id}`")

    cols = st.columns(4)
    cols[0].metric("Process", "running" if running else "stopped")
    cols[1].metric("PID", str(lp.pid(m.id)) if running else "—")
    cols[2].metric("Health", "ok" if reachable else "—")
    cols[3].metric("Log lines", f"{len(lp.log_lines(m.id))}")

    ctrl = st.columns([1, 1, 1, 4])
    with ctrl[0]:
        if st.button("▶ Start", key=f"start_{m.id}", type="primary",
                     disabled=running, width="stretch"):
            ok, msg = lp.start(m.id)
            (st.success if ok else st.warning)(msg)
            st.rerun()
    with ctrl[1]:
        if st.button("■ Stop", key=f"stop_{m.id}",
                     disabled=not running, width="stretch"):
            ok, msg = lp.stop(m.id)
            (st.success if ok else st.warning)(msg)
            st.rerun()
    with ctrl[2]:
        if st.button("🔄 Refresh", key=f"refresh_{m.id}", width="stretch"):
            st.rerun()

    with st.expander("Log tail", expanded=False):
        lines = lp.log_lines(m.id)
        body = "\n".join(lines[-400:]) if lines else "(no output yet — start the backend)"
        st.code(body, language="log")

    with st.expander("Launch args", expanded=False):
        try:
            cmd = lp.build_command(m)
            st.code(" ".join(cmd), language="bash")
        except Exception as e:
            st.warning(str(e))


def render() -> None:
    st.title("🧠 Models")
    st.caption(
        "One card per model enabled for this host. Local `llama-server` "
        "backends can be started/stopped individually; the hub on :8000 "
        "routes requests by `model` name."
    )

    models = enabled_models()
    if not models:
        st.warning("No models enabled for this host — check `config/models.yaml`.")
        return

    for m in models:
        if m.backend == "claude":
            _render_claude_card(m)
        elif m.backend in ("openai", "whisper"):
            _render_local_card(m)
        else:
            st.write(f"unknown backend {m.backend!r} for {m.display_name}")
        st.divider()
