"""Welcome page — introduction + usage instructions."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("👋 claude-local-calls")
    st.caption(
        "A local Anthropic-compatible API backed by `claude -p` and your "
        "Claude Code subscription."
    )

    st.markdown(
        """
        ### What this is

        A tiny FastAPI server that exposes `POST /v1/messages` with the
        same shape as the official **Anthropic Messages API**, but
        routes each call through the local **`claude -p`** CLI. Point
        any client — including the official `anthropic` SDK — at
        `http://127.0.0.1:8000` and your existing code keeps working,
        charged to your Claude subscription instead of API credits.

        ### Using this app

        - **🛰 Server** — start / stop the FastAPI process and watch its log live.
        - **📊 Comparison** — per-model specs table (params, quant, size, context, docs links).
        - **🧠 Models** — start / stop each local llama-server backend and tail its log.
        - **✅ Testing** — run unit tests and the end-to-end smoke test.
        - **💬 Playground** — send a prompt, pick a model, see the reply and token counts.

        ### Use it from your own code
        """
    )

    st.markdown("**Python (official Anthropic SDK):**")
    st.code(
        '''from anthropic import Anthropic

client = Anthropic(api_key="local-dummy", base_url="http://127.0.0.1:8000")
msg = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)
print("in:", msg.usage.input_tokens, "out:", msg.usage.output_tokens)
''',
        language="python",
    )

    st.markdown("**Raw HTTP (curl):**")
    st.code(
        'curl -s http://127.0.0.1:8000/v1/messages \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"model":"claude-haiku-4-5","max_tokens":64,'
        '"messages":[{"role":"user","content":"hi"}]}\'',
        language="bash",
    )

    st.markdown(
        """
        ### LAN access

        The server binds on `0.0.0.0`, so other machines on your
        network (another laptop, a VM, an agent) can call it directly.
        Open the **🛰 Server** tab to see the clickable **LAN** URL for
        this machine — point the remote client's `base_url` at that
        URL instead of `127.0.0.1`:

        ```python
        client = Anthropic(
            api_key="local-dummy",
            base_url="http://192.168.1.42:8000",   # your LAN IP here
        )
        ```

        On Windows the first run will prompt to allow Python through
        the firewall — accept on **Private** networks only. There is
        no authentication, so only run on trusted networks (home /
        office LAN you own) and never port-forward to the internet.

        ### Layout

        - **`app/`** — this Streamlit UI (entry point + views).
        - **`src/`** — non-UI Python: CLI wrapper, FastAPI server, process manager.
        - **`tests/`** — pytest unit tests.
        - **`scripts/`** — end-to-end smoke test.

        ### Requirements

        - The **`claude`** CLI on `PATH` (Claude Code must be installed and logged in).
        - Python deps already installed in `.venv` from `requirements.txt`.

        ### Caveats (intentional — lightweight)

        No streaming, no images, no tool use, multi-turn is flattened.
        See the **Backlog for improvement** in `README.md` for the full
        list of what a faithful Anthropic-API parity would add.
        """
    )
