"""Server control page — start / stop the FastAPI process + tail its log."""

from __future__ import annotations

import streamlit as st

from src import server_process as sp


def render() -> None:
    st.title("🛰 Server")

    lan = sp.lan_url()
    local_md = f"[{sp.BASE_URL}]({sp.BASE_URL})"
    if lan:
        st.markdown(f"**Local:** {local_md}  \n**LAN:** [{lan}]({lan})")
        st.caption(
            "The server binds on 0.0.0.0, so other machines on the LAN can "
            "reach it at the LAN URL. On Windows, allow the Python process "
            "through the firewall on the first run."
        )
    else:
        st.markdown(f"**Local:** {local_md}")
        st.caption("No LAN route detected — reachable from this machine only.")

    running = sp.is_running()
    reachable = sp.is_reachable() if running else False

    cols = st.columns(4)
    cols[0].metric("Process", "running" if running else "stopped")
    cols[1].metric("PID", str(sp.pid()) if running else "—")
    cols[2].metric("Health", "ok" if reachable else "—")
    cols[3].metric("Log lines", f"{len(sp.log_lines())}")

    ctrl = st.columns([1, 1, 1, 4])
    with ctrl[0]:
        if st.button("▶ Start", type="primary", disabled=running, use_container_width=True):
            ok, msg = sp.start()
            (st.success if ok else st.warning)(msg)
            st.rerun()
    with ctrl[1]:
        if st.button("■ Stop", disabled=not running, use_container_width=True):
            ok, msg = sp.stop()
            (st.success if ok else st.warning)(msg)
            st.rerun()
    with ctrl[2]:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    strays = [] if running else sp.stray_pids_on_port()
    if strays:
        pretty = ", ".join(str(p) for p in strays)
        st.warning(
            f"Port {sp.PORT} is held by another process (PID {pretty}) — "
            "probably a stale server from a previous session. Start will "
            "fail with WinError 10048 until it's gone."
        )
        if st.button(f"💀 Kill stray process on port {sp.PORT}", use_container_width=False):
            ok, msg = sp.kill_stray_on_port()
            (st.success if ok else st.error)(msg)
            st.rerun()

    st.divider()

    st.markdown("**Server log** (stdout + stderr)")
    lines = sp.log_lines()
    body = "\n".join(lines[-400:]) if lines else "(no output yet — start the server)"
    st.code(body, language="log")

    st.caption(
        "The process is managed by this Streamlit session. Stopping the app "
        "will also stop the server. For standalone use, run `launchers/run_hub.bat`."
    )
