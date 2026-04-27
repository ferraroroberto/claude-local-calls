"""First-run / health page. One row per check with a Fix button when applicable."""

from __future__ import annotations

import streamlit as st

from src import install as install_mod


_BADGE = {
    "ok":      ("OK",   "#1f7a3d"),
    "warn":    ("WARN", "#b8890a"),
    "missing": ("NEED", "#8a4b10"),
    "error":   ("FAIL", "#a43030"),
}


def _badge_html(status: str) -> str:
    label, color = _BADGE.get(status, ("?", "#555"))
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:6px;"
        f"background:{color};color:white;font-weight:600;font-size:0.8em;"
        f"font-family:monospace;'>{label}</span>"
    )


def render() -> None:
    st.title("🩺 Install")
    st.caption(
        "Per-host setup check. Green means ready; yellow/red means the hub, "
        "a backend, or a model won't work yet. Fix buttons run the same code "
        "`python -m src.install --fix` would run from the shell."
    )

    refresh = st.button("🔄 Re-run all checks", width="content")
    if refresh:
        st.rerun()

    run_all = st.button("🛠  Run all fixes", type="primary")

    report = install_mod.run_all_checks()
    st.caption(f"overall status: **{report.worst_status}**  ·  {len(report.checks)} checks")

    if run_all:
        for c in list(report.checks):
            if c.status not in ("missing", "error"):
                continue
            fn = install_mod.fix_fn_for(c)
            if fn is None:
                continue
            with st.status(f"Fixing {c.label} ...", expanded=True) as s:
                try:
                    fn()
                    s.update(label=f"Fixed {c.label}", state="complete")
                except Exception as e:
                    s.update(label=f"Fix failed: {c.label}", state="error")
                    s.write(str(e))
        st.rerun()

    st.divider()

    for c in report.checks:
        fix_fn = install_mod.fix_fn_for(c)
        cols = st.columns([1, 4, 3, 2])
        cols[0].markdown(_badge_html(c.status), unsafe_allow_html=True)
        cols[1].markdown(f"**{c.label}**")
        cols[2].markdown(f"<span style='color:#888'>{c.detail}</span>", unsafe_allow_html=True)
        if fix_fn is not None and c.status in ("missing", "error"):
            if cols[3].button("Fix", key=f"fix_{c.id}"):
                with st.status(f"Running fix for {c.label} ...", expanded=True) as s:
                    try:
                        fix_fn()
                        s.update(label=f"Done: {c.label}", state="complete")
                    except Exception as e:
                        s.update(label=f"Failed: {c.label}", state="error")
                        s.write(str(e))
                st.rerun()

    st.divider()
    with st.expander("What each check does"):
        st.markdown(
            "- **Python / venv** - interpreter is >=3.10 and running from this repo's `.venv`.\n"
            "- **Deps** - `fastapi`, `uvicorn`, `httpx`, `pyyaml`, `huggingface_hub`, etc. are importable.\n"
            "- **Host profile** - which row in `config/models.yaml` applies to this machine.\n"
            "- **`claude` CLI** - Claude Code CLI on PATH (needed for the Claude backend).\n"
            "- **GPU / accelerator** - NVIDIA on Windows, Apple Silicon on macOS.\n"
            "- **llama.cpp binary** - `vendor/llama.cpp/llama-server[.exe]` present and runnable.\n"
            "- **Model present** - GGUF files for each enabled local model.\n"
            "- **Ports free** - hub (8000) + each enabled backend port are free or held by our process.\n"
        )
