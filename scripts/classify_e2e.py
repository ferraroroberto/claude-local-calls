"""Diff-proportionate e2e routing for the pre-ship gate (project-scaffolding#180).

`scripts/verify-before-ship.ps1` used to run the whole `tests/e2e` suite
unconditionally, regardless of what the diff touched. Proven first in
`app-launcher` (issue #568, PR #574): map the changed-file set of the current
branch to a coverage tier so the gate's browser phase runs a slice
proportionate to the diff -- never weaker for real UI/behaviour changes, and
fail-safe to the full suite whenever the diff is ambiguous, mixed, or touches
anything not confidently narrow.

Unlike app-launcher's classifier, the path -> tier rules here are NOT
hardcoded in this module -- every consumer's layout differs. They are
declared per-project in that repo's own `.fleet.toml` under an `[e2e]` table
(see the `[e2e]` block in this repo's own `.fleet.toml` for a worked example,
and `docs/e2e-routing.md` for the schema reference). This module only
supplies the mechanism: load the declared rules, classify the diff against
them in order, first match wins.

Tiers:

  * ``skip``    every changed path matched a declared "none" rule (no
                browser impact) -> no browser suite runs.
  * ``static``  the worst-matching path is a declared "static" rule (e.g. an
                image asset or an inert vendored HTML fragment) -> a narrow
                smoke target, declared as ``static_pytest_target``.
  * ``full``    any path matched a declared "full" rule, OR any path matched
                *no* rule at all, OR the project has no `[e2e]` table
                declared, OR the diff is empty -- runs
                ``full_pytest_target`` (the whole e2e suite by default). This
                is the fail-safe default: uncertainty always escalates to
                full coverage, never narrows it.

CLI: prints ``E2E_*=`` key/value lines (parsed by the PowerShell gate) plus a
human summary on stderr. Run standalone to see how the current branch would
route:

    python scripts/classify_e2e.py            # classify the live diff
    python scripts/classify_e2e.py a.js b.svg  # classify an explicit file list
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEET_TOML = REPO_ROOT / ".fleet.toml"


class Category(IntEnum):
    """Per-path coverage requirement. Higher wins across the diff."""

    NONE = 0    # declared "none" rule -- no browser impact
    STATIC = 1  # declared "static" rule -- a narrow smoke target is enough
    FULL = 2    # declared "full" rule, or no rule matched at all


_TIER_NAMES = {"none": Category.NONE, "static": Category.STATIC, "full": Category.FULL}


@dataclass
class Rule:
    """One declared `[[e2e.rule]]` entry. First matching rule wins."""

    tier: Category
    prefix: str | None = None
    path: str | None = None
    extensions: tuple[str, ...] | None = None
    label: str = "rule"

    def matches(self, path: str, ext: str) -> bool:
        if self.path is not None:
            return path == self.path
        if self.prefix is not None and not path.startswith(self.prefix):
            return False
        if self.extensions is not None and ext not in self.extensions:
            return False
        # A rule needs at least one real matcher -- a bare {tier=...} row
        # with no prefix/path/extensions would match everything by accident.
        return self.prefix is not None or self.extensions is not None


@dataclass
class E2EConfig:
    rules: list[Rule]
    static_pytest_target: str = "tests/e2e"
    static_browsers: tuple[str, ...] = ("chromium",)
    full_pytest_target: str = "tests/e2e"
    # "declared" = usable [e2e] table found; anything else is a fail-safe
    # reason string surfaced in the routing output.
    source: str = "missing"


def load_config(fleet_toml: Path = FLEET_TOML) -> E2EConfig:
    """Read the `[e2e]` table from *fleet_toml*.

    Missing file, unparsable TOML, absent `[e2e]` table, or an `[e2e]` table
    with zero usable rules all return a config with ``source != "declared"``
    -- `classify()` then fail-safes every path to FULL rather than guessing.
    """
    if not fleet_toml.is_file():
        return E2EConfig(rules=[], source="missing")
    try:
        data = tomllib.loads(fleet_toml.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return E2EConfig(rules=[], source="invalid")

    e2e = data.get("e2e")
    if not isinstance(e2e, dict):
        return E2EConfig(rules=[], source="missing")

    rules: list[Rule] = []
    for raw in e2e.get("rule", []):
        if not isinstance(raw, dict) or "tier" not in raw:
            continue
        tier = _TIER_NAMES.get(str(raw["tier"]).lower())
        if tier is None:
            continue
        exts = raw.get("extensions")
        rules.append(
            Rule(
                tier=tier,
                prefix=raw.get("prefix"),
                path=raw.get("path"),
                extensions=tuple(str(e).lower() for e in exts) if exts else None,
                label=str(raw.get("label") or raw.get("prefix") or raw.get("path") or "rule"),
            )
        )

    if not rules:
        return E2EConfig(rules=[], source="empty")

    return E2EConfig(
        rules=rules,
        static_pytest_target=str(e2e.get("static_pytest_target", "tests/e2e")),
        static_browsers=tuple(e2e.get("static_browsers", ["chromium"])),
        full_pytest_target=str(e2e.get("full_pytest_target", "tests/e2e")),
        source="declared",
    )


def _classify_one(path: str, rules: list[Rule]) -> tuple[Category, str]:
    """Map one repo-relative (posix) path to its coverage category + label.

    Rules are tried in declaration order, first match wins. A path matching
    no rule at all is the fail-safe: treated as FULL, labelled "unclassified".
    """
    name = path.rsplit("/", 1)[-1]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    for rule in rules:
        if rule.matches(path, ext):
            return rule.tier, rule.label
    return Category.FULL, "unclassified"


@dataclass
class Routing:
    tier: str                       # "skip" | "static" | "full"
    browsers: list[str]
    pytest_target: str              # "" when tier == skip
    reasons: list[str] = field(default_factory=list)


def classify(paths: list[str], config: E2EConfig) -> Routing:
    """Route a set of changed paths to an e2e tier per *config*.

    Fail-safe to FULL whenever: the project has no usable `[e2e]`
    declaration, the diff is empty, or any changed path matches no declared
    rule. Uncertainty always escalates to full coverage, never narrows it.
    """
    if config.source != "declared":
        reason = {
            "missing": "no [e2e] table declared in .fleet.toml",
            "invalid": ".fleet.toml could not be parsed",
            "empty": "[e2e] table declared but has no usable rule entries",
        }.get(config.source, config.source)
        return Routing("full", [], config.full_pytest_target or "tests/e2e",
                        [f"{reason} -- fail-safe full suite"])

    examples: dict[str, str] = {}
    top = Category.NONE
    for raw in paths:
        path = raw.strip().replace("\\", "/")
        if not path:
            continue
        cat, label = _classify_one(path, config.rules)
        top = max(top, cat)
        examples.setdefault(f"{cat.name}:{label}", path)

    def reasons_for(cat: Category) -> list[str]:
        return [
            f"{key.split(':', 1)[1]}: {ex}"
            for key, ex in sorted(examples.items())
            if key.startswith(f"{cat.name}:")
        ]

    if not examples:
        # Empty diff (e.g. run on a clean tree): can't prove narrow -> full.
        return Routing("full", [], config.full_pytest_target, ["empty-diff: no changed files"])

    if top == Category.FULL:
        return Routing("full", [], config.full_pytest_target, reasons_for(Category.FULL))
    if top == Category.STATIC:
        return Routing(
            "static", list(config.static_browsers), config.static_pytest_target,
            reasons_for(Category.STATIC),
        )
    return Routing("skip", [], "", reasons_for(Category.NONE))


# --------------------------------------------------------------------- git diff
# This module is copied byte-identical into adopter repos (fleet-config's
# `e2e_route.py bootstrap`, then hash-verified against this file), so it must
# stay self-contained -- it deliberately does NOT import this repo's shared
# `src/no_window.py`, which would not resolve in a consumer's tree. Deriving
# the flag locally here is the documented vendor-verbatim exception, not drift.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _run_git(args: list[str]) -> list[str]:
    # The gate runs this 4-5 times per invocation (symbolic-ref, rev-parse, two
    # diffs, ls-files). A console-less parent -- a tray, a scheduled task, an
    # agent shell -- would get a console window flashed for each one without
    # CREATE_NO_WINDOW; the closed stdin stops git ever waiting on a prompt.
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
    except OSError:
        return []
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _main_ref() -> str:
    """origin/main (or origin/<default>) when present, else main."""
    head = _run_git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
    if head:
        return head[0].replace("refs/remotes/", "", 1)
    if _run_git(["rev-parse", "--verify", "--quiet", "origin/main"]):
        return "origin/main"
    return "main"


def changed_files() -> list[str]:
    """Changed files on the current branch vs main, incl. the working tree.

    Union of: committed since the merge-base with main (``main...HEAD``),
    tracked working-tree edits (staged + unstaged, ``git diff HEAD``), and
    untracked new files. So a *pre-commit* gate run classifies correctly.
    """
    ref = _main_ref()
    files: set[str] = set()
    files.update(_run_git(["diff", "--name-only", f"{ref}...HEAD"]))
    files.update(_run_git(["diff", "--name-only", "HEAD"]))
    files.update(_run_git(["ls-files", "--others", "--exclude-standard"]))
    return sorted(files)


def main(argv: list[str]) -> int:
    paths = argv[1:] if len(argv) > 1 else changed_files()
    config = load_config()
    routing = classify(paths, config)

    # Machine-readable block the PowerShell gate parses (^E2E_ lines only).
    print(f"E2E_TIER={routing.tier}")
    print(f"E2E_BROWSERS={','.join(routing.browsers)}")
    print(f"E2E_PYTEST_TARGET={routing.pytest_target}")
    print(f"E2E_REASON={' | '.join(routing.reasons) if routing.reasons else '(none)'}")

    # Human summary (ignored by the gate parser).
    print("", file=sys.stderr)
    print(f"e2e routing: tier={routing.tier} "
          f"browsers={routing.browsers or 'suite-default'}", file=sys.stderr)
    for reason in routing.reasons:
        print(f"  - {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
