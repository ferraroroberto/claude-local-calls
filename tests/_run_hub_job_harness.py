"""Test-only harness for ``run_backend._self_contain`` (#507).

Not a pytest test module (leading underscore keeps it out of collection).
Invoked by ``tests/test_run_backend_containment.py`` as
``python -m tests._run_hub_job_harness <marker-path>`` to prove the actual
``_self_contain`` wiring — not a re-implementation of it: this harness calls
the real function, then spawns a grandchild the same way
``backend_process.start`` spawns an on-demand backend
(``CREATE_NEW_PROCESS_GROUP`` — deliberately detached so a restart's
CTRL_BREAK doesn't reach it), reports both PIDs to the marker file, then
blocks — so the test can force-kill *this* process externally and observe
whether the grandchild dies with it, with no cooperating code running in the
victim process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_backend import _self_contain  # noqa: E402
from src.server_process import WIN_NEW_GROUP  # noqa: E402


def main(argv: list) -> int:
    marker = Path(argv[0])
    _self_contain()

    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        creationflags=WIN_NEW_GROUP,
    )
    marker.write_text(json.dumps({
        "self_pid": os.getpid(),
        "grandchild_pid": grandchild.pid,
    }))
    time.sleep(120)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
