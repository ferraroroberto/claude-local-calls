"""End-to-end smoke test - requires a running server and real `claude` auth.

Usage (in two terminals):

    # Terminal 1
    .venv\\Scripts\\python -m src.server

    # Terminal 2
    .venv\\Scripts\\python scripts/smoke_test.py

Exercises both the raw HTTP endpoint and the official `anthropic` SDK
pointed at the local server, proving it's a drop-in replacement.
"""

from __future__ import annotations

import sys

import httpx
from anthropic import Anthropic

BASE_URL = "http://127.0.0.1:8000"
MODEL = "claude-haiku-4-5"


def via_raw_http() -> str:
    print("-> raw HTTP to /v1/messages")
    r = httpx.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        },
        timeout=120.0,
    )
    r.raise_for_status()
    body = r.json()
    text = body["content"][0]["text"]
    print(f"  model={body['model']}  usage={body['usage']}")
    print(f"  text={text!r}")
    return text


def via_sdk() -> str:
    print("-> anthropic SDK with base_url override")
    client = Anthropic(api_key="local-dummy", base_url=BASE_URL)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=64,
        system="Answer in one word.",
        messages=[{"role": "user", "content": "Capital of France?"}],
    )
    text = msg.content[0].text
    print(f"  model={msg.model}  usage=in:{msg.usage.input_tokens} out:{msg.usage.output_tokens}")
    print(f"  text={text!r}")
    return text


def main() -> int:
    try:
        h = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        h.raise_for_status()
    except Exception as e:
        print(f"server not reachable at {BASE_URL}: {e}")
        print("start it with: .venv\\Scripts\\python -m src.server")
        return 1

    try:
        t1 = via_raw_http()
        t2 = via_sdk()
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    if not t1 or not t2:
        print("FAIL: empty response")
        return 1
    print("OK -- both paths returned text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
