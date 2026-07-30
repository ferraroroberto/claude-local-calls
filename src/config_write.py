"""Git-backed write path for config/models.yaml placement edits (#424).

The admin UI's editable placement cards write *through to git*: an accepted
edit mutates ``config/models.yaml`` comment-preservingly (ruamel.yaml — the
file's comments are load-bearing), commits with the config-bot message
convention (``config: <model> placement via admin UI``) and pushes to
``origin main``. The YAML stays the single source of truth; satellites
converge by pulling (the #181 sync endpoints + the drift loop below).

Safety contract, in order:

* **Single writer** — only the host named by ``hub.config_write_host`` in
  models.yaml (tower) may write; everything else 403s at the router.
* **Validate before any file change** — schema checks plus the #375
  VRAM-budget math: a placement whose *eager, GPU-tier* chain members
  overcommit a host's declared ``vram_mb`` ceiling is hard-rejected
  (config-time overcommit is a standing error, unlike the transient
  runtime overcommit ``src.on_demand`` merely warns about).
* **Never commit unrelated work** — the write refuses when the repo has any
  dirty *tracked* file (untracked scratch is ignored — never staged, the
  ``git add`` is path-scoped), refuses off ``main``, and re-checks after the
  edit that models.yaml is the *only* change in the commit.
* **No silent local-only divergence** — a failed push rolls the commit and
  the file back to the pre-edit state and surfaces the error to the caller.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .host_profile import (
    CONFIG_PATH,
    PROJECT_ROOT,
    _CONFIG_CACHE,
    _load_config,
    all_hosts,
    resolve as resolve_host,
)
from .model_registry import (
    STARTUP_EAGER,
    STARTUP_ON_DEMAND,
    _parse_host_chain,
    _parse_startup,
    all_models,
)
from .no_window import NO_WINDOW

logger = logging.getLogger(__name__)

CONFIG_RELPATH = "config/models.yaml"
COMMIT_MESSAGE_TEMPLATE = "config: {model} placement via admin UI"
# Distinct bot identity so `git log config/models.yaml` reads honestly:
# UI-driven commits are visibly not the human's hand-edits. Only the *name*
# is overridden — the author email stays whatever the repo/global git config
# provides, because this machine's commit hook enforces an email allowlist
# (an invented bot email would be blocked at commit time).
_BOT_NAME = "local-llm-hub config-bot"
_GIT_TIMEOUT_S = 60

# The one branch config writes may land on — the fleet's canonical branch.
WRITE_BRANCH = "main"


class ConfigWriteError(RuntimeError):
    """A refused or failed config write, carrying the HTTP status the admin
    router should surface (400 validation, 403 wrong host, 409 repo not in a
    writable state, 502 push failed-and-rolled-back)."""

    def __init__(self, detail: str, status: int = 500) -> None:
        super().__init__(detail)
        self.status = status


# --------------------------------------------------------------------- host
def write_host_id() -> Optional[str]:
    """The single host id allowed to write (``hub.config_write_host``), or
    ``None`` when the config declares no writer (writes disabled fleet-wide)."""
    cfg = _load_config()
    value = (cfg.get("hub") or {}).get("config_write_host")
    return str(value) if value else None


def is_write_host() -> bool:
    """True only on the declared single-writer host (#424's tower)."""
    writer = write_host_id()
    if not writer:
        return False
    try:
        return resolve_host().id == writer
    except Exception:  # noqa: BLE001 — hostless tooling contexts
        return False


# ---------------------------------------------------------------------- git
def _git(*args: str, repo: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Run one git command against ``repo`` — captured, windowless, bounded."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
        check=False,
    )


def _dirty_tracked(repo: Path) -> List[str]:
    """``git status --porcelain`` entries for *tracked* files only.

    Untracked (``??``) entries are ignored throughout the write path: they
    can never end up in the commit (the ``git add`` is path-scoped) and a
    stray scratch file must not block config edits forever.
    """
    result = _git("status", "--porcelain", repo=repo)
    if result.returncode != 0:
        raise ConfigWriteError(
            f"git status failed: {(result.stderr or result.stdout).strip()}", 500
        )
    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
    return [ln for ln in lines if not ln.startswith("??")]


def _current_branch(repo: Path) -> str:
    result = _git("symbolic-ref", "--short", "HEAD", repo=repo)
    if result.returncode != 0:
        raise ConfigWriteError(
            f"cannot determine current branch: {(result.stderr or '').strip()}", 409
        )
    return (result.stdout or "").strip()


def _head_sha(repo: Path, short: bool = False) -> str:
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    result = _git(*args, repo=repo)
    if result.returncode != 0:
        raise ConfigWriteError(
            f"git rev-parse failed: {(result.stderr or '').strip()}", 500
        )
    return (result.stdout or "").strip()


def git_preflight(repo: Path = PROJECT_ROOT, rel_path: str = CONFIG_RELPATH) -> None:
    """Refuse the write before touching any file unless the repo is in the
    one state a config-bot commit is safe in: on ``main`` with no dirty
    tracked files at all (a hand-edit in progress — even to models.yaml
    itself — must never be silently bundled into a bot commit)."""
    branch = _current_branch(repo)
    if branch != WRITE_BRANCH:
        raise ConfigWriteError(
            f"repo is on branch {branch!r} — config writes require {WRITE_BRANCH!r}",
            409,
        )
    dirty = _dirty_tracked(repo)
    if dirty:
        raise ConfigWriteError(
            "repo has uncommitted tracked changes — refusing to mix them into a "
            f"config commit: {', '.join(d.strip() for d in dirty)}",
            409,
        )


def commit_and_push(
    repo: Path, rel_path: str, message: str
) -> str:
    """Commit exactly ``rel_path`` and push to ``origin main``.

    Re-verifies that ``rel_path`` is the *only* dirty tracked file (a
    concurrent edit elsewhere aborts the write), path-scopes the ``git add``,
    commits under the config-bot identity, and pushes. A failed push resets
    hard to the pre-commit HEAD — file and history roll back together, so
    the checkout never diverges silently from the remote. Returns the short
    sha of the pushed commit.
    """
    dirty = _dirty_tracked(repo)
    normalized = {d[3:].strip().replace("\\", "/") for d in dirty}
    if normalized != {rel_path}:
        raise ConfigWriteError(
            f"expected {rel_path!r} to be the only change, found: "
            f"{', '.join(d.strip() for d in dirty) or '(nothing)'}",
            409,
        )
    pre_head = _head_sha(repo)
    add = _git("add", "--", rel_path, repo=repo)
    if add.returncode != 0:
        raise ConfigWriteError(f"git add failed: {(add.stderr or '').strip()}", 500)
    commit = _git(
        "-c", f"user.name={_BOT_NAME}",
        "commit", "-m", message,
        repo=repo,
    )
    if commit.returncode != 0:
        _git("reset", "--hard", pre_head, repo=repo)
        raise ConfigWriteError(
            f"git commit failed: {(commit.stderr or commit.stdout).strip()}", 500
        )
    push = _git("push", "origin", WRITE_BRANCH, repo=repo)
    if push.returncode != 0:
        reset = _git("reset", "--hard", pre_head, repo=repo)
        rollback_note = (
            "rolled back to pre-edit state"
            if reset.returncode == 0
            else f"ROLLBACK FAILED ({(reset.stderr or '').strip()}) — repo needs manual attention"
        )
        raise ConfigWriteError(
            "git push failed — the edit was NOT persisted "
            f"({rollback_note}). Push error: {(push.stderr or push.stdout).strip()}",
            502,
        )
    return _head_sha(repo, short=True)


# ------------------------------------------------------------- config sha
# HEAD sha of models.yaml — the admin UI's config-version indicator. A git
# subprocess per read would ride the 5 s Models-tab poll, so cache briefly;
# a successful write invalidates explicitly.
_SHA_TTL_S = 15.0
_sha_cache: Tuple[float, str] = (0.0, "")
_sha_lock = threading.Lock()


def config_sha(fresh: bool = False) -> str:
    """Short HEAD sha of ``config/models.yaml`` — same value on every hub
    that has converged, so comparing chips across /admin pages shows drift."""
    global _sha_cache
    now = time.monotonic()
    with _sha_lock:
        ts, sha = _sha_cache
        if not fresh and sha and now - ts < _SHA_TTL_S:
            return sha
    result = _git("log", "-n", "1", "--format=%h", "--", CONFIG_RELPATH)
    value = (result.stdout or "").strip() if result.returncode == 0 else ""
    value = value or "unknown"
    with _sha_lock:
        _sha_cache = (now, value)
    return value


def _invalidate_caches() -> None:
    """After a successful write: re-read the YAML (registry/host profile) and
    re-derive the config sha, so this hub converges without a restart —
    every consumer loop re-reads the registry each pass."""
    _CONFIG_CACHE.clear()
    config_sha(fresh=True)


# --------------------------------------------------------------- validation
def normalize_chain(raw: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Payload ``hosts`` → ``[{"id": str, "cpu": bool}, ...]`` + shape errors.

    Accepts bare host-id strings or ``{id, cpu}`` dicts, mirroring the YAML
    shapes ``_parse_host_chain`` accepts.
    """
    if not isinstance(raw, list):
        return [], ["hosts must be a list of host ids or {id, cpu} entries"]
    chain: List[Dict[str, Any]] = []
    errors: List[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            host_id = str(entry.get("id") or "").strip()
            if not host_id:
                errors.append("chain entry is missing its host id")
                continue
            chain.append({"id": host_id, "cpu": bool(entry.get("cpu"))})
        elif isinstance(entry, str) and entry.strip():
            chain.append({"id": entry.strip(), "cpu": False})
        else:
            errors.append(f"unrecognized chain entry: {entry!r}")
    return chain, errors


def _eager_gpu_hosts(
    hosts: List[str], cpu_hosts: List[str], startup: str
) -> List[str]:
    """The hosts where a model counts against the VRAM ceiling: eager policy
    only (an on-demand row's residency is transient by design — its load-time
    overcommit is ``src.on_demand``'s advisory warning, not a config error),
    excluding ``cpu``-flagged degraded tiers (CPU offload holds no VRAM)."""
    if startup != STARTUP_EAGER:
        return []
    cpu = set(cpu_hosts)
    return [h for h in hosts if h not in cpu]


def _overcommit_errors(
    model_id: str, chain: List[Dict[str, Any]], startup: str
) -> List[str]:
    """The #375 budget math as a hard config-time gate: with the edit
    applied, no host with a declared ``vram_mb`` ceiling that this model
    would sit on (eager, non-cpu) may have its eager GPU-tier set exceed
    the ceiling. Matches the worst-case chain-membership arithmetic the
    YAML's own host comments use (gaming's 6200-of-8192 note)."""
    candidate_hosts = _eager_gpu_hosts(
        [e["id"] for e in chain], [e["id"] for e in chain if e["cpu"]], startup
    )
    if not candidate_hosts:
        return []
    ceilings = {h.id: h.vram_mb for h in all_hosts()}
    errors: List[str] = []
    others = [m for m in all_models(apply_cpu_offload=False) if m.id != model_id]
    candidate_est = next(
        (m.est_vram_mb or 0 for m in all_models(apply_cpu_offload=False) if m.id == model_id),
        0,
    )
    for host_id in candidate_hosts:
        ceiling = ceilings.get(host_id)
        if not ceiling:
            continue  # no declared ceiling (Apple silicon) — never gates
        resident = [
            (m.id, m.est_vram_mb or 0)
            for m in others
            if host_id in _eager_gpu_hosts(m.hosts, m.cpu_hosts, m.startup)
            and (m.est_vram_mb or 0) > 0
        ]
        projected = sum(mb for _, mb in resident) + candidate_est
        if projected > ceiling:
            listing = ", ".join(f"{mid} ~{mb} MB" for mid, mb in resident) or "none"
            errors.append(
                f"placing {model_id} (~{candidate_est} MB) on {host_id} overcommits its "
                f"VRAM budget: ~{projected} MB estimated vs the {ceiling} MB ceiling "
                f"(other eager GPU models there: {listing})"
            )
    return errors


def validate_placement(
    model_id: str,
    chain: List[Dict[str, Any]],
    startup: str,
    idle_unload_minutes: Optional[int],
) -> List[str]:
    """All validation errors for a proposed placement edit (empty = valid).

    Runs entirely against the in-memory registry — nothing here touches the
    file or git, so a rejection is guaranteed side-effect-free.
    """
    models = {m.id: m for m in all_models(apply_cpu_offload=False)}
    m = models.get(model_id)
    if m is None:
        return [f"unknown model {model_id!r}"]
    if m.backend in ("claude", "gemini"):
        return ["subscription-backed rows have no placement to edit"]
    if m.virtual:
        return ["virtual alias shares another row's process — edit the parent row instead"]

    errors: List[str] = []
    host_profiles = {h.id: h for h in all_hosts()}
    ids = [e["id"] for e in chain]
    if not ids:
        errors.append("hosts chain must contain at least one host")
    unknown = [h for h in ids if h not in host_profiles]
    if unknown:
        errors.append(f"unknown host(s): {', '.join(unknown)}")
    if len(set(ids)) != len(ids):
        errors.append("hosts chain repeats a host")
    for host_id in ids:
        profile = host_profiles.get(host_id)
        if profile is not None and model_id not in profile.enabled:
            errors.append(
                f"host {host_id!r} does not list {model_id!r} in its enabled: list — "
                "every chain member must cross-enable the model (and pre-stage its weights)"
            )
    if startup not in (STARTUP_EAGER, STARTUP_ON_DEMAND):
        errors.append(
            f"startup must be {STARTUP_EAGER!r} or {STARTUP_ON_DEMAND!r}, got {startup!r}"
        )
    if idle_unload_minutes is not None:
        if not isinstance(idle_unload_minutes, int) or isinstance(idle_unload_minutes, bool) \
                or idle_unload_minutes < 1:
            errors.append("idle_unload_minutes must be a positive integer (or null)")
        elif startup != STARTUP_ON_DEMAND:
            errors.append(
                "idle_unload_minutes only applies to startup: on_demand "
                "(eager models are never idle-unloaded)"
            )
    if errors:
        return errors
    return _overcommit_errors(model_id, chain, startup)


# ---------------------------------------------------------------- yaml edit
def edit_models_yaml(
    path: Path,
    model_id: str,
    chain: List[Dict[str, Any]],
    startup: str,
    idle_unload_minutes: Optional[int],
) -> bool:
    """Apply the placement edit to the YAML file comment-preservingly.

    ruamel.yaml round-trip mode: every comment, blank line, and quote style
    outside the touched keys survives byte-for-byte (the file's comments are
    load-bearing — a lossy rewrite is a regression even if semantically
    equal; pinned by tests). Only keys whose values actually change are
    touched. Returns True when the file changed.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    yaml = YAML()
    yaml.preserve_quotes = True
    # Never re-wrap: the file's long flow sequences / strings must round-trip
    # on one line exactly as authored.
    yaml.width = 2 ** 16
    # The file's block-sequence style (`    - item` under a 2-space key) —
    # without this ruamel re-indents every list in the file on dump.
    yaml.indent(mapping=2, sequence=4, offset=2)

    text = path.read_text(encoding="utf-8")
    data = yaml.load(text)
    row = (data.get("models") or {}).get(model_id)
    if row is None:
        raise ConfigWriteError(f"model {model_id!r} not found in {path.name}", 400)

    changed = False
    current_hosts, current_cpu = _parse_host_chain(row)
    new_ids = [e["id"] for e in chain]
    new_cpu = [e["id"] for e in chain if e["cpu"]]
    if current_hosts != new_ids or set(current_cpu) != set(new_cpu):
        seq = CommentedSeq()
        for entry in chain:
            if entry["cpu"]:
                item = CommentedMap([("id", entry["id"]), ("cpu", True)])
                item.fa.set_flow_style()
                seq.append(item)
            else:
                seq.append(entry["id"])
        seq.fa.set_flow_style()
        # `hosts:` supersedes a bare `host:` — never leave two sources of
        # truth for entry one (the registry would ignore `host:` anyway).
        anchor_key = "hosts" if "hosts" in row else ("host" if "host" in row else None)
        if anchor_key == "host":
            keys = list(row.keys())
            pos = keys.index("host")
            del row["host"]
            row.insert(pos, "hosts", seq)
        elif anchor_key == "hosts":
            row["hosts"] = seq
        else:
            row.insert(len(row), "hosts", seq)
        changed = True

    def _insert_after(existing_keys: List[str], key: str, value: Any) -> None:
        keys = list(row.keys())
        pos = len(keys)
        for anchor in existing_keys:
            if anchor in keys:
                pos = keys.index(anchor) + 1
                break
        row.insert(pos, key, value)

    if _parse_startup(row) != startup:
        if "startup" in row:
            row["startup"] = startup
        else:
            _insert_after(["hosts", "host", "port"], "startup", startup)
        changed = True

    current_idle = (
        int(row["idle_unload_minutes"]) if row.get("idle_unload_minutes") is not None else None
    )
    if current_idle != idle_unload_minutes:
        if idle_unload_minutes is None:
            if "idle_unload_minutes" in row:
                del row["idle_unload_minutes"]
        elif "idle_unload_minutes" in row:
            row["idle_unload_minutes"] = idle_unload_minutes
        else:
            _insert_after(["startup"], "idle_unload_minutes", idle_unload_minutes)
        changed = True

    if not changed:
        return False

    buf = StringIO()
    yaml.dump(data, buf)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return True


# ------------------------------------------------------------- orchestration
def apply_placement(
    model_id: str,
    chain: List[Dict[str, Any]],
    startup: str,
    idle_unload_minutes: Optional[int],
) -> Dict[str, Any]:
    """The full write transaction (#424): validate → git preflight →
    comment-preserving edit → verify-only-our-file → commit → push.

    Raises :class:`ConfigWriteError` with the right HTTP status at every
    refusal point; any failure after the file was touched restores the
    original bytes, so the checkout is never left half-written.
    """
    errors = validate_placement(model_id, chain, startup, idle_unload_minutes)
    if errors:
        raise ConfigWriteError("; ".join(errors), 400)

    git_preflight(PROJECT_ROOT, CONFIG_RELPATH)

    original = CONFIG_PATH.read_bytes()
    try:
        changed = edit_models_yaml(
            CONFIG_PATH, model_id, chain, startup, idle_unload_minutes
        )
        if not changed:
            return {
                "ok": True,
                "changed": False,
                "commit": None,
                "config_sha": config_sha(fresh=True),
                "detail": "no changes — placement already matches",
            }
        message = COMMIT_MESSAGE_TEMPLATE.format(model=model_id)
        sha = commit_and_push(PROJECT_ROOT, CONFIG_RELPATH, message)
    except ConfigWriteError:
        CONFIG_PATH.write_bytes(original)
        raise
    except Exception as exc:  # noqa: BLE001 — never leave a half-written file
        CONFIG_PATH.write_bytes(original)
        raise ConfigWriteError(f"config write failed: {exc}", 500) from exc

    _invalidate_caches()
    logger.info("✅ config write: %s → commit %s (pushed to origin/%s)",
                model_id, sha, WRITE_BRANCH)
    return {
        "ok": True,
        "changed": True,
        "commit": sha,
        "config_sha": config_sha(fresh=True),
        "detail": message,
    }


# --------------------------------------------------------------- drift loop
# Periodic convergence net under the push-triggered sync: a satellite that
# was down when the tower pushed (and so missed its sync) is re-synced when
# it reappears with a stale models.yaml sha. One attempt per (peer, local
# sha) pair — a peer whose pull keeps failing is not restart-hammered.
CONFIG_DRIFT_POLL_S = 600
_DRIFT_BOOT_DELAY_S = 120


async def config_drift_sync_loop() -> None:
    """Write-host only: poll hub peers' ``/admin/api/version`` for their
    ``config_sha`` and fire the #181 sync (git pull + hub restart) at any
    peer whose models.yaml lags this host's."""
    if not is_write_host():
        return
    from . import remote_bootstrap, remote_stats
    from .host_profile import get_host, hub_port
    from .http_client import get_async_client
    from .model_registry import hub_peer_ids

    attempted: Dict[str, str] = {}
    await asyncio.sleep(_DRIFT_BOOT_DELAY_S)
    while True:
        try:
            local = await asyncio.to_thread(config_sha, True)
            if local != "unknown":
                for peer in hub_peer_ids():
                    profile = get_host(peer)
                    if profile is None or profile.dormant:
                        continue
                    address = await remote_stats.dial_address_async(profile)
                    if not address:
                        continue
                    try:
                        r = await get_async_client().get(
                            f"http://{address}:{hub_port()}/admin/api/version",
                            timeout=5.0,
                        )
                        peer_sha = (r.json() or {}).get("config_sha")
                    except Exception:  # noqa: BLE001 — peer down/mid-restart
                        continue
                    # An older build reports no config_sha — skip rather than
                    # guess (the regular reconcile/sync paths update builds).
                    if not peer_sha or peer_sha == local:
                        continue
                    if attempted.get(peer) == local:
                        continue
                    attempted[peer] = local
                    logger.info(
                        "🔃 config drift: %s has models.yaml %s, local is %s — syncing",
                        peer, peer_sha, local,
                    )
                    result = await remote_bootstrap.sync_host(peer)
                    logger.info("config drift sync of %s: %s", peer, result)
        except Exception as exc:  # noqa: BLE001 — the loop must survive anything
            logger.warning("config drift pass raised: %s", exc)
        await asyncio.sleep(CONFIG_DRIFT_POLL_S)


__all__ = [
    "COMMIT_MESSAGE_TEMPLATE",
    "ConfigWriteError",
    "apply_placement",
    "commit_and_push",
    "config_drift_sync_loop",
    "config_sha",
    "edit_models_yaml",
    "git_preflight",
    "is_write_host",
    "normalize_chain",
    "validate_placement",
    "write_host_id",
]
