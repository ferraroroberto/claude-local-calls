"""Standalone worker for ``tests/test_event_loop.py``'s aborted-connection
bombardment (issue #441).

The two Windows-only tests that drive this deliberately corrupt OS-level
asyncio/proactor state -- that's the whole point of
``test_proactor_loop_dies_on_aborted_connections``, which documents a real
``ProactorEventLoop`` bug (issue #222). #416's original investigation into
the full-suite's order-dependent ``asyncio.run()`` failures named this file
as the prime suspect for leaking that corruption into later tests sharing
the same pytest process; #417 fixed a different, confirmed leak
(``wire_observatory_loop``'s orphaned background tasks) but never actually
ruled this one in or out. Rather than trust that leaving 400+ raw aborted
socket connections and a deliberately-killed proactor loop behind is safe,
this worker runs the bombardment in its own process -- whatever it perturbs
dies with it.

Usage: ``python -m tests._event_loop_worker <selector|proactor>``
Exit 0 = expected outcome (listener survived for selector, died as
documented for proactor). Exit 1 = unexpected outcome.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading


async def _noop_handler(reader, writer):
    writer.close()


def _abort_connect_sync(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    try:
        s.settimeout(0.5)
        s.connect(("127.0.0.1", port))
    except OSError:
        pass
    finally:
        s.close()  # SO_LINGER(1, 0) forces an RST instead of a FIN


async def _still_accepting(port: int) -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=1.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def _bombard_with_aborts(rounds: int, burst: int) -> None:
    server = await asyncio.start_server(_noop_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        for _ in range(rounds):
            threads = [
                threading.Thread(target=_abort_connect_sync, args=(port,))
                for _ in range(burst)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            await asyncio.sleep(0.02)
            assert await _still_accepting(
                port
            ), "listener died on an aborted client connection (issue #222)"
    finally:
        server.close()
        await server.wait_closed()


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "selector":
        asyncio.run(
            _bombard_with_aborts(rounds=10, burst=20), loop_factory=asyncio.SelectorEventLoop
        )
        return 0
    if mode == "proactor":
        try:
            asyncio.run(
                _bombard_with_aborts(rounds=10, burst=20), loop_factory=asyncio.ProactorEventLoop
            )
        except AssertionError:
            return 0  # documents the bug this issue fixes -- see module docstring
        print("proactor loop unexpectedly survived the abort bombardment", file=sys.stderr)
        return 1
    print(f"unknown mode: {mode!r} (expected 'selector' or 'proactor')", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
