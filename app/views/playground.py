"""Playground — send a prompt to the local API, show reply + token counts."""

from __future__ import annotations

import time

import httpx
import streamlit as st

from src import server_process as sp
from src.model_registry import enabled_models


def _model_options() -> list[str]:
    names: list[str] = []
    for m in enabled_models():
        names.extend(m.all_names)
    # de-dupe while preserving order
    return list(dict.fromkeys(names))


def render() -> None:
    st.title("💬 Playground")
    st.caption(f"POSTs to `{sp.BASE_URL}/v1/messages`")

    reachable = sp.is_reachable()
    if not reachable:
        st.warning(
            "Server unreachable. Start it in the **🛰 Server** tab, then "
            "come back here."
        )

    cols = st.columns([1, 2])
    with cols[0]:
        options = _model_options()
        model = st.selectbox("Model", options, index=0) if options else None
        apply_max = st.checkbox(
            "Apply max_tokens",
            value=False,
            help="When off, no cap is sent — `claude -p` decides when to stop.",
        )
        max_tokens = st.number_input(
            "max_tokens",
            min_value=16,
            max_value=8192,
            value=512,
            step=16,
            disabled=not apply_max,
        )
    with cols[1]:
        system = st.text_area(
            "System prompt (optional)",
            value="",
            height=100,
            placeholder="e.g. Answer in one sentence.",
        )

    prompt = st.text_area(
        "Your message",
        height=160,
        placeholder="Type a prompt and press Send…",
        key="pg_prompt",
    )

    send = st.button(
        "Send",
        type="primary",
        disabled=not reachable or not prompt.strip() or not model,
    )

    if send:
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if apply_max:
            payload["max_tokens"] = int(max_tokens)
        if system.strip():
            payload["system"] = system

        t0 = time.time()
        try:
            with st.spinner(f"calling {model} via hub …"):
                r = httpx.post(
                    f"{sp.BASE_URL}/v1/messages",
                    json=payload,
                    timeout=300.0,
                )
        except Exception as e:
            st.error(f"request failed: {e}")
            return

        elapsed = time.time() - t0

        if r.status_code != 200:
            st.error(f"HTTP {r.status_code}")
            st.code(r.text, language="json")
            return

        body = r.json()
        text = body["content"][0]["text"] if body.get("content") else ""
        usage = body.get("usage", {})

        in_new = int(usage.get("input_tokens", 0) or 0)
        cache_r = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_w = int(usage.get("cache_creation_input_tokens", 0) or 0)
        in_total = in_new + cache_r + cache_w

        metrics = st.columns(6)
        metrics[0].metric("input (total)", in_total)
        metrics[1].metric("input (new)", in_new)
        metrics[2].metric("cache read", cache_r)
        metrics[3].metric("cache write", cache_w)
        metrics[4].metric("output tokens", usage.get("output_tokens", 0))
        metrics[5].metric("elapsed", f"{elapsed:.1f}s")
        st.caption(
            "`input (total) = input (new) + cache read + cache write`. "
            "Claude Code caches the system preamble, tools, and often your "
            "message itself, so `input (new)` is usually tiny — the real "
            "prompt size is the total."
        )

        st.markdown("**Response**")
        st.markdown(text or "_(empty response)_")

        with st.expander("Raw JSON response"):
            st.json(body)
