"""Streamlit entry point for claude-local-calls.

Run with:

    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from views import comparison, install, models, playground, server, testing, welcome

st.set_page_config(
    page_title="claude-local-calls",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded",
)

_STYLES_DIR = Path(__file__).resolve().parent / "styles"


def _inject_css(filename: str) -> None:
    css = (_STYLES_DIR / filename).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


with st.sidebar:
    if st.toggle("☀  Light mode", value=False, key="light_mode"):
        _inject_css("light.css")

nav_pages = [
    st.Page(welcome.render,    title="Welcome",    icon="👋", url_path="welcome",    default=True),
    st.Page(install.render,    title="Install",    icon="🩺", url_path="install"),
    st.Page(server.render,     title="Server",     icon="🛰",  url_path="server"),
    st.Page(comparison.render, title="Comparison", icon="📊", url_path="comparison"),
    st.Page(models.render,     title="Models",     icon="🧠", url_path="models"),
    st.Page(testing.render,    title="Testing",    icon="✅", url_path="testing"),
    st.Page(playground.render, title="Playground", icon="💬", url_path="playground"),
]

pg = st.navigation(nav_pages, position="sidebar")
pg.run()
