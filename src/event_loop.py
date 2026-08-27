"""Windows event-loop shim for uvicorn (issue #222).

On Windows, asyncio's default proactor event loop closes its listening
socket the moment ``accept()`` raises any ``OSError`` (see CPython's
``proactor_events.py:_start_serving``'s accept loop) -- and a client
aborting a connection mid-request (a dropped LAN client, a killed process,
a network hiccup) surfaces as exactly such an ``OSError`` (WinError 64,
"The specified network name is no longer available"). One aborted client
and the listener is gone; the process stays alive but every subsequent
connection fails until a manual restart.

This hub binds ``0.0.0.0`` (LAN-wide -- ``tower`` and a Mac Mini dial
each other's hub over the LAN) and proxies long-running audio/LLM
streaming traffic, so the exposure here is broader than a single-host,
phone-facing webapp: any LAN client dropping a connection mid-request can
trigger the wedge on any of this repo's three uvicorn-served processes
(the main hub plus the tts/parakeet shim servers).

The selector event loop's accept path has no such failure mode -- verified
empirically (in the sister app-launcher repo, issue #388): 800 concurrent
aborted connections against a bare ``SelectorEventLoop`` server left it
accepting fine, while the same abuse killed a ``ProactorEventLoop`` server
after ~20. None of this repo's servers spawn asyncio subprocesses
in-process (backends are launched via plain ``subprocess.Popen`` in
``src/backend_process.py``, never ``asyncio.create_subprocess_*``), so the
selector loop's lack of subprocess support is a non-issue here.

Wired into every ``uvicorn.run(...)`` call in this repo -- ``src/server.py``
(the main hub), ``src/tts_server.py``, and ``src/parakeet_server.py`` -- via
``loop=LOOP_FACTORY``. Each of those is a programmatic ``uvicorn.run()``
call (not a CLI ``-m uvicorn`` invocation), so this is the only wiring
point needed; ``src/backend_process.py::build_command()`` just launches
``python -m src.<module>``, and each module's own ``main()`` already sets
the loop.

For a *custom* ``loop=`` value (anything outside uvicorn's built-in
``none``/``auto``/``asyncio``/``uvloop`` names), uvicorn imports the
target and uses it directly as the final zero-arg
``Callable[[], asyncio.AbstractEventLoop]`` passed to ``asyncio.run`` --
unlike the built-in names, it is *not* called with a ``use_subprocess=``
kwarg first (that indirection only applies to the built-in factories in
``uvicorn.config.LOOP_FACTORIES``). So ``selector_loop_factory`` below
must itself return an *instantiated* loop, not a loop class.
"""

from __future__ import annotations

import asyncio
import sys


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


LOOP_FACTORY = "src.event_loop:selector_loop_factory"
