"""Model comparison page — renders `docs/model-comparison.md` verbatim.

Single source of truth: the markdown file is what lands in the repo
and the docs; this page just reads and displays it. Adding a model =
one table row in one file.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_PATH = PROJECT_ROOT / "docs" / "model-comparison.md"


def render() -> None:
    if not DOC_PATH.exists():
        st.title("📊 Model comparison")
        st.error(f"`{DOC_PATH.relative_to(PROJECT_ROOT)}` not found.")
        return

    text = DOC_PATH.read_text(encoding="utf-8")
    # Drop the leading H1 — Streamlit wraps pages in its own chrome;
    # the nav label already carries the page title.
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        # Trim a blank line that would otherwise sit at the top.
        while lines and not lines[0].strip():
            lines.pop(0)
    body = "\n".join(lines)

    st.title("📊 Model comparison")
    st.caption(
        f"Rendered from `{DOC_PATH.relative_to(PROJECT_ROOT)}` — edit that "
        "file to update this page."
    )
    st.markdown(body, unsafe_allow_html=False)
