"""Idempotently enable SearXNG's `json` output format on a generated
settings.yml (home-automation#321, local-llm-hub#438/#439).

JSON is off by default (SearXNG's abuse-prevention default) but the HA voice
search function and the home-automation webapp's status card both need
`format=json`. The container generates `settings.yml` itself on first run
(baking in a fresh random `secret_key`), so the fix can't be a committed
file — this script patches only `search.formats` on the already-generated
file, leaving every other key (especially `secret_key`) untouched. Run after
every `docker compose up`; a no-op once the format is already present.

Exit codes: 0 = already enabled, no change made; 1 = format added, caller
should restart the container for SearXNG to pick it up; 2 = error (bad
usage, or the file never appeared in time).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

WAIT_TIMEOUT_S = 30
POLL_INTERVAL_S = 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ensure_json_format.py <settings.yml path>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])

    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while not path.exists():
        if time.monotonic() >= deadline:
            print(f"timed out waiting for {path} to be generated", file=sys.stderr)
            return 2
        time.sleep(POLL_INTERVAL_S)

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 2 ** 16

    data = yaml.load(path.read_text(encoding="utf-8"))
    if data is None:
        data = CommentedMap()

    search = data.get("search")
    formats = search.get("formats") if search is not None else None
    if formats is not None and "json" in formats:
        print("json format already enabled — no change")
        return 0

    if search is None:
        search = CommentedMap()
        data["search"] = search
    if formats is None:
        search["formats"] = CommentedSeq(["html", "json"])
    else:
        formats.append("json")

    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)

    print("added json to search.formats — restart required to take effect")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
