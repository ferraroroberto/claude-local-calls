"""SQLite persistence for diagnostics captures (issue #315).

The repo's first SQLite store. The database lives in the fleet runtime-data
root (``<root>/local-llm-hub/diagnostics.db`` — see
``src/runtime_data.py``), off whichever drive this repo happens to be cloned
onto; it is per-machine runtime state, never committed. The schema is versioned
through ``PRAGMA user_version`` with a forward-only migration ladder so it can
evolve without hand-editing shipped tables.

Connection policy: one short-lived connection per operation, WAL journal. The
sampler writes once per tick (~15 s) and readers are user-driven, so pooling
would buy nothing and a long-lived handle across an asyncio task boundary is
a footgun. WAL keeps a reader from blocking the sampler's writes.

Every row carries the ``machine_id`` of the run that produced it, so DBs
copied off different machines merge cleanly for offline comparison.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from src.runtime_data import runtime_db_path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# The hub is on around the clock and writes a diagnostics row per request, so
# its default belongs on a fast local drive rather than on whichever drive this
# repo was cloned onto — a spinning HDD here (project-scaffolding#243).
# ``LLM_HUB_DIAGNOSTICS_DB_PATH`` (env) overrides it for a single process;
# :func:`set_db_path` still wins over both for tests.
DEFAULT_DB_PATH = runtime_db_path(
    "local-llm-hub", "diagnostics.db", env_var="LLM_HUB_DIAGNOSTICS_DB_PATH"
)

SCHEMA_VERSION = 2

# Overridable for tests (same mutable-module-global pattern as
# startup_profile.DEFAULT_PROFILE_PATH).
_db_path: Optional[Path] = None


def db_path() -> Path:
    """The active database path (test-overridable via :func:`set_db_path`)."""
    return _db_path or DEFAULT_DB_PATH


def set_db_path(path: Optional[Path]) -> None:
    """Point the store at another file (``None`` restores the default)."""
    global _db_path
    _db_path = Path(path) if path else None


# --------------------------------------------------------------- connection


# Concurrent first-open retry (#326). Several connections opening a brand-new
# DB at once all try to change the journal mode to WAL, which needs a brief
# exclusive lock — and SQLite does *not* run the busy handler for a journal-mode
# change, so the losers get "database is locked" returned immediately, not after
# `busy_timeout`. The winner sets WAL + migrates in microseconds, so a short
# bounded retry converges (probe: 12/20 rounds locked → 0/20). An uncontended
# open succeeds on the first attempt and pays nothing.
_OPEN_RETRIES = 12
_OPEN_BACKOFF_S = 0.02


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a migrated connection, committing on clean exit.

    ``check_same_thread=False`` because the sampler runs its writes through
    ``asyncio.to_thread`` — the connection never outlives the call, so no two
    threads ever hold it at once."""
    conn = _open_migrated(db_path())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _open_migrated(target: Path) -> sqlite3.Connection:
    """Open a WAL-mode, migrated connection, retrying the transient "database is
    locked" a concurrent first-open can raise on the journal-mode change or the
    migration (#326 — SQLite skips the busy handler for a journal-mode change,
    so ``busy_timeout`` alone does not cover it). The migration is committed in
    its own transaction so concurrent openers see the new schema and stop
    racing it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[sqlite3.OperationalError] = None
    for attempt in range(_OPEN_RETRIES):
        conn = sqlite3.connect(str(target), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _migrate(conn)
            conn.commit()
            return conn
        except sqlite3.OperationalError as exc:
            conn.close()
            if "locked" not in str(exc).lower():
                raise
            last = exc
            time.sleep(_OPEN_BACKOFF_S * (attempt + 1))
    raise last if last is not None else sqlite3.OperationalError(
        "could not open diagnostics DB after concurrent-open retries")


def _migrate(conn: sqlite3.Connection) -> None:
    """Forward-only schema ladder keyed on ``PRAGMA user_version``.

    Never edit a shipped step — add the next one. A DB from a newer build is
    left alone rather than downgraded (the reader may simply see extra
    columns it ignores)."""
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current >= SCHEMA_VERSION:
        return
    if current < 1:
        conn.executescript(_SCHEMA_V1)   # CREATE TABLE IF NOT EXISTS — idempotent
    if current < 2:
        # Per-collector coverage: which signals could actually be read this run,
        # so "we weren't allowed to look" stops rendering as "nothing there"
        # (#322). Unlike v1's CREATE-IF-NOT-EXISTS, `ALTER TABLE ADD COLUMN` has
        # no idempotent form, so two connections opening a fresh DB at once (a
        # write racing the hub's background `_init_diagnostics`) would both pass
        # the version gate and both ALTER — the loser raising "duplicate column".
        # Guard on the actual column, and still swallow a lost race.
        _add_column(conn, "runs", "coverage_json", "TEXT NOT NULL DEFAULT '{}'")
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    logger.info("🗃️ diagnostics schema migrated %d → %d", current, SCHEMA_VERSION)


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Idempotent, concurrency-safe ``ALTER TABLE ADD COLUMN``."""
    have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in have:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as exc:
        # A concurrent migrator added it between our check and our ALTER.
        if "duplicate column" not in str(exc).lower():
            raise


_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    machine_id   TEXT NOT NULL,
    os           TEXT NOT NULL,
    hostname     TEXT NOT NULL DEFAULT '',
    started_at   REAL NOT NULL,
    ended_at     REAL,
    interval_s   REAL NOT NULL,
    duration_s   REAL,
    trigger      TEXT NOT NULL DEFAULT 'manual',
    status       TEXT NOT NULL DEFAULT 'running',
    is_baseline  INTEGER NOT NULL DEFAULT 0,
    note         TEXT NOT NULL DEFAULT '',
    params_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs (started_at DESC);

CREATE TABLE IF NOT EXISTS samples (
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts             REAL NOT NULL,
    cpu_percent    REAL,
    per_core_json  TEXT,
    load_avg_json  TEXT,
    ram_used_gb    REAL,
    ram_total_gb   REAL,
    ram_percent    REAL,
    swap_used_gb   REAL,
    swap_total_gb  REAL,
    swap_percent   REAL,
    disk_used_gb   REAL,
    disk_total_gb  REAL,
    disk_percent   REAL,
    disk_io_json   TEXT,
    net_io_json    TEXT,
    gpu_json       TEXT,
    process_count  INTEGER,
    PRIMARY KEY (run_id, ts)
);

CREATE TABLE IF NOT EXISTS process_samples (
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts           REAL NOT NULL,
    pid          INTEGER NOT NULL,
    ppid         INTEGER,
    name         TEXT NOT NULL DEFAULT '',
    cmdline      TEXT NOT NULL DEFAULT '',
    app_id       TEXT NOT NULL DEFAULT 'unattributed',
    cpu_percent  REAL,
    rss_bytes    INTEGER,
    num_threads  INTEGER,
    status       TEXT,
    create_time  REAL
);
CREATE INDEX IF NOT EXISTS idx_proc_run_ts  ON process_samples (run_id, ts);
CREATE INDEX IF NOT EXISTS idx_proc_run_app ON process_samples (run_id, app_id);

CREATE TABLE IF NOT EXISTS ports (
    run_id  TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts      REAL NOT NULL,
    port    INTEGER NOT NULL,
    proto   TEXT NOT NULL DEFAULT 'tcp',
    address TEXT NOT NULL DEFAULT '',
    pid     INTEGER,
    name    TEXT NOT NULL DEFAULT '',
    app_id  TEXT NOT NULL DEFAULT 'unattributed'
);
CREATE INDEX IF NOT EXISTS idx_ports_run ON ports (run_id, ts);

CREATE TABLE IF NOT EXISTS verdicts (
    run_id        TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    computed_at   REAL NOT NULL,
    level         TEXT NOT NULL,
    findings_json TEXT NOT NULL DEFAULT '[]'
);
"""


# ------------------------------------------------------------------- writes


def new_run_id() -> str:
    """A sortable, collision-free run id: ``<epoch-ms>-<6 hex>``."""
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


def create_run(
    *,
    machine_id: str,
    os_name: str,
    hostname: str = "",
    interval_s: float,
    duration_s: Optional[float],
    trigger: str = "manual",
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Insert a ``running`` run row and return its id."""
    run_id = new_run_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, machine_id, os, hostname, started_at, interval_s,"
            " duration_s, trigger, status, params_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)",
            (
                run_id, machine_id, os_name, hostname, time.time(), float(interval_s),
                float(duration_s) if duration_s is not None else None, trigger,
                json.dumps(params or {}),
            ),
        )
    return run_id


def finish_run(run_id: str, *, status: str = "complete") -> None:
    """Mark a run finished. Idempotent — a second call is a no-op update."""
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET ended_at = ?, status = ? WHERE run_id = ?",
            (time.time(), status, run_id),
        )


def close_orphan_runs() -> int:
    """Close runs still marked ``running`` from a previous hub process.

    Called at hub startup: the sampler lives in-process, so any run still
    open in the DB belongs to a hub that died mid-capture. Marking them
    ``interrupted`` (rather than deleting) keeps whatever they did record
    minable, and stops a stale row from looking like a live capture."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE runs SET ended_at = COALESCE(ended_at, ?), status = 'interrupted'"
            " WHERE status = 'running'",
            (time.time(),),
        )
        count = cur.rowcount or 0
    if count:
        logger.info("🧹 diagnostics: closed %d orphaned run(s) from a previous hub", count)
    return count


@dataclass(frozen=True)
class SystemSample:
    """One system-level tick. JSON-shaped fields are pre-serialized by the
    sampler so this stays a dumb carrier."""

    ts: float
    cpu_percent: Optional[float]
    per_core: List[float]
    load_avg: Optional[List[float]]
    ram: Dict[str, float]
    swap: Dict[str, float]
    disk: Dict[str, float]
    disk_io: Dict[str, Any]
    net_io: Dict[str, Any]
    gpus: List[Dict[str, Any]]
    process_count: int


def write_sample(
    run_id: str,
    sample: SystemSample,
    processes: Iterable[Dict[str, Any]],
    listening_ports: Iterable[Dict[str, Any]],
) -> None:
    """Persist one tick: the system row, its process inventory, its ports.

    All three go in a single transaction so a reader never sees a system row
    whose process rows haven't landed yet."""
    proc_rows = [
        (
            run_id, sample.ts, p.get("pid"), p.get("ppid"), p.get("name") or "",
            p.get("cmdline") or "", p.get("app_id") or "unattributed",
            p.get("cpu_percent"), p.get("rss_bytes"), p.get("num_threads"),
            p.get("status"), p.get("create_time"),
        )
        for p in processes
    ]
    port_rows = [
        (
            run_id, sample.ts, q.get("port"), q.get("proto") or "tcp",
            q.get("address") or "", q.get("pid"), q.get("name") or "",
            q.get("app_id") or "unattributed",
        )
        for q in listening_ports
    ]
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO samples (run_id, ts, cpu_percent, per_core_json,"
            " load_avg_json, ram_used_gb, ram_total_gb, ram_percent, swap_used_gb,"
            " swap_total_gb, swap_percent, disk_used_gb, disk_total_gb, disk_percent,"
            " disk_io_json, net_io_json, gpu_json, process_count)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, sample.ts, sample.cpu_percent, json.dumps(sample.per_core),
                json.dumps(sample.load_avg) if sample.load_avg else None,
                sample.ram.get("used_gb"), sample.ram.get("total_gb"), sample.ram.get("percent"),
                sample.swap.get("used_gb"), sample.swap.get("total_gb"), sample.swap.get("percent"),
                sample.disk.get("used_gb"), sample.disk.get("total_gb"), sample.disk.get("percent"),
                json.dumps(sample.disk_io), json.dumps(sample.net_io),
                json.dumps(sample.gpus), sample.process_count,
            ),
        )
        if proc_rows:
            conn.executemany(
                "INSERT INTO process_samples (run_id, ts, pid, ppid, name, cmdline,"
                " app_id, cpu_percent, rss_bytes, num_threads, status, create_time)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                proc_rows,
            )
        if port_rows:
            conn.executemany(
                "INSERT INTO ports (run_id, ts, port, proto, address, pid, name, app_id)"
                " VALUES (?,?,?,?,?,?,?,?)",
                port_rows,
            )


def save_verdict(run_id: str, level: str, findings: List[Dict[str, Any]]) -> None:
    """Persist (or replace) a run's health verdict."""
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO verdicts (run_id, computed_at, level, findings_json)"
            " VALUES (?, ?, ?, ?)",
            (run_id, time.time(), level, json.dumps(findings)),
        )


def save_coverage(run_id: str, coverage: Dict[str, Any]) -> None:
    """Persist the per-collector coverage map for a run (#322)."""
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET coverage_json = ? WHERE run_id = ?",
            (json.dumps(coverage or {}), run_id),
        )


def proc_readability(run_id: str) -> Dict[str, int]:
    """How many of a run's distinct processes had readable memory / CPU.

    Counts by DISTINCT pid, not by row: privilege is stable across a capture,
    so a process is readable or not for the whole run, and counting rows would
    just weight it by how many ticks it survived. ``rss_bytes``/``cpu_percent``
    are stored NULL (never 0) when psutil is denied, which is exactly what lets
    this distinguish 'unreadable' from a genuine zero (#322)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT pid) AS total,"
            " COUNT(DISTINCT CASE WHEN rss_bytes IS NOT NULL THEN pid END) AS mem_ok,"
            " COUNT(DISTINCT CASE WHEN cpu_percent IS NOT NULL THEN pid END) AS cpu_ok"
            " FROM process_samples WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return {
        "total": int(row["total"] or 0),
        "mem_ok": int(row["mem_ok"] or 0),
        "cpu_ok": int(row["cpu_ok"] or 0),
    }


def set_baseline(run_id: str) -> None:
    """Mark one run as *the* baseline for its machine (exactly one wins)."""
    with connect() as conn:
        row = conn.execute("SELECT machine_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        conn.execute("UPDATE runs SET is_baseline = 0 WHERE machine_id = ?", (row["machine_id"],))
        conn.execute("UPDATE runs SET is_baseline = 1 WHERE run_id = ?", (run_id,))


def set_note(run_id: str, note: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE runs SET note = ? WHERE run_id = ?", (note[:500], run_id))


def delete_run(run_id: str) -> None:
    """Delete a run and (via ON DELETE CASCADE) every row it produced."""
    with connect() as conn:
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


# -------------------------------------------------------------------- reads


def _run_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    if "params_json" in d:
        try:
            d["params"] = json.loads(d.pop("params_json") or "{}")
        except json.JSONDecodeError:
            d["params"] = {}
    if "coverage_json" in d:
        try:
            d["coverage"] = json.loads(d.pop("coverage_json") or "{}")
        except json.JSONDecodeError:
            d["coverage"] = {}
    d["is_baseline"] = bool(d.get("is_baseline"))
    return d


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Recent runs, newest first, each with its verdict level + sample count."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT r.*, v.level AS verdict_level,"
            " (SELECT COUNT(*) FROM samples s WHERE s.run_id = r.run_id) AS sample_count"
            " FROM runs r LEFT JOIN verdicts v ON v.run_id = r.run_id"
            " ORDER BY r.started_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [_run_dict(r) for r in rows]


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT r.*, v.level AS verdict_level, v.findings_json,"
            " (SELECT COUNT(*) FROM samples s WHERE s.run_id = r.run_id) AS sample_count"
            " FROM runs r LEFT JOIN verdicts v ON v.run_id = r.run_id WHERE r.run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    run = _run_dict(row)
    try:
        run["findings"] = json.loads(run.pop("findings_json") or "[]")
    except json.JSONDecodeError:
        run["findings"] = []
    return run


def baseline_run(machine_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE machine_id = ? AND is_baseline = 1"
            " ORDER BY started_at DESC LIMIT 1",
            (machine_id,),
        ).fetchone()
    return _run_dict(row) if row else None


def samples(run_id: str) -> List[Dict[str, Any]]:
    """Every system tick of a run, oldest first, JSON columns decoded."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM samples WHERE run_id = ? ORDER BY ts", (run_id,)
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        for src, dst in (
            ("per_core_json", "per_core"), ("load_avg_json", "load_avg"),
            ("disk_io_json", "disk_io"), ("net_io_json", "net_io"), ("gpu_json", "gpus"),
        ):
            raw = d.pop(src, None)
            try:
                d[dst] = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                d[dst] = None
        out.append(d)
    return out


def process_aggregates(run_id: str) -> List[Dict[str, Any]]:
    """Per-process rollup across a run, keyed by the identity that survives a
    restart (app_id + name + cmdline) rather than PID.

    Grouping on cmdline is deliberate: the venv ``pythonw`` redirector spawns
    a stub *and* a real process per launch, so a PID-keyed rollup double-counts
    a single app."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT app_id, name, cmdline,"
            "       COUNT(DISTINCT pid)  AS pid_count,"
            "       COUNT(*)             AS observations,"
            "       AVG(cpu_percent)     AS avg_cpu,"
            "       MAX(cpu_percent)     AS peak_cpu,"
            "       AVG(rss_bytes)       AS avg_rss,"
            "       MAX(rss_bytes)       AS peak_rss,"
            "       MAX(num_threads)     AS peak_threads"
            " FROM process_samples WHERE run_id = ?"
            " GROUP BY app_id, name, cmdline"
            " ORDER BY avg_cpu DESC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def app_aggregates(run_id: str) -> List[Dict[str, Any]]:
    """Per-*app* rollup — the headline "app-launcher: 3 procs / 800 MB" view.

    ``peak_procs`` is the largest per-tick distinct-PID count, computed from
    the per-tick counts rather than over the whole run, so a process that
    restarted mid-run doesn't inflate the app's apparent concurrency."""
    with connect() as conn:
        rows = conn.execute(
            "WITH per_tick AS ("
            "  SELECT app_id, ts, COUNT(DISTINCT pid) AS procs,"
            "         SUM(rss_bytes) AS rss, SUM(cpu_percent) AS cpu"
            "  FROM process_samples WHERE run_id = ? GROUP BY app_id, ts"
            ")"
            " SELECT app_id, AVG(procs) AS avg_procs, MAX(procs) AS peak_procs,"
            "        AVG(rss) AS avg_rss, MAX(rss) AS peak_rss,"
            "        AVG(cpu) AS avg_cpu, MAX(cpu) AS peak_cpu"
            " FROM per_tick GROUP BY app_id ORDER BY avg_rss DESC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def process_count_timeline(run_id: str) -> List[Dict[str, Any]]:
    """Distinct-PID count per app per tick — the "how many python processes
    over time" series the whole feature was asked for."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, app_id, COUNT(DISTINCT pid) AS procs"
            " FROM process_samples WHERE run_id = ? GROUP BY ts, app_id ORDER BY ts",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def listening_ports(run_id: str) -> List[Dict[str, Any]]:
    """The distinct listening sockets seen during a run, with their owner."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT port, proto, app_id, name, MIN(ts) AS first_seen, MAX(ts) AS last_seen,"
            "       COUNT(DISTINCT ts) AS ticks"
            " FROM ports WHERE run_id = ? GROUP BY port, proto, app_id, name ORDER BY port",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- retention


def prune(retention_days: int = 90) -> Dict[str, int]:
    """Drop raw rows from runs older than ``retention_days``.

    Run metadata, verdicts, and baselines are kept forever — they are tiny and
    they are what long-horizon drift comparison reads. Only the bulky raw
    tables age out, which is what actually bounds the file size. Called
    opportunistically at capture start, so there is no timer to keep alive."""
    if retention_days <= 0:
        return {"runs_pruned": 0}
    cutoff = time.time() - retention_days * 86400
    with connect() as conn:
        rows = conn.execute(
            "SELECT run_id FROM runs WHERE started_at < ? AND is_baseline = 0"
            " AND run_id IN (SELECT DISTINCT run_id FROM samples)",
            (cutoff,),
        ).fetchall()
        stale = [r["run_id"] for r in rows]
        for run_id in stale:
            conn.execute("DELETE FROM process_samples WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM ports WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM samples WHERE run_id = ?", (run_id,))
            conn.execute("UPDATE runs SET status = 'pruned' WHERE run_id = ?", (run_id,))
    if stale:
        logger.info("🧹 diagnostics: pruned raw samples from %d run(s)", len(stale))
    return {"runs_pruned": len(stale)}


def db_size_bytes() -> int:
    try:
        return db_path().stat().st_size
    except OSError:
        return 0
