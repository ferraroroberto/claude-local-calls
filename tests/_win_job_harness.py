"""Test-only harness process for ``src.win_job`` containment (#468).

Not a pytest test module (leading underscore keeps it out of collection).
Invoked by ``tests/test_win_job.py`` as ``python -m tests._win_job_harness
<marker-path>`` to exercise the "spawner killed mid-run" guarantee from a
real second process: this harness creates a kill-on-close job, joins it
itself, spawns a grandchild that just sleeps, reports both PIDs to the
marker file, then blocks — so the test can force-kill *this* process
externally and observe whether ``KILL_ON_JOB_CLOSE`` takes the grandchild
down with it, with no cooperating code running in the victim process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import win_job  # noqa: E402 — path insert above must run first
from src.no_window import NO_WINDOW  # noqa: E402


def main(argv: list) -> int:
    marker = Path(argv[0])
    job = win_job.create_kill_on_close_job()
    if job is None:
        marker.write_text(json.dumps({"error": "no job object (not Windows?)"}))
        return 1
    win_job.assign_pid(job, os.getpid())

    grandchild = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        creationflags=NO_WINDOW,
    )
    marker.write_text(json.dumps({
        "self_pid": os.getpid(),
        "grandchild_pid": grandchild.pid,
    }))
    time.sleep(120)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
