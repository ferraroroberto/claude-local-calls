"""Shared test configuration.

Disables OpenTelemetry SDK in unit tests so we don't:
  * try to open a gRPC connection to a non-existent OTLP endpoint
  * log "OTel initialised" / "OTLP export failed" lines that pollute test output
  * leak background BatchSpanProcessor threads between test sessions

The trace_id middleware + GenAI helpers are exercised independently via
their own unit tests against the disabled-mode no-ops.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# Disable the optional AgentsView integration (issue #280): empty base URL
# means no probe and no background refresh threads — hermetic even on a dev
# box that has a real AgentsView serving on :8080.
os.environ.setdefault("AGENTSVIEW_BASE_URL", "")

import pytest  # noqa: E402
import yaml  # noqa: E402


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    """Write a throwaway ``models.yaml`` and point the config readers at it.

    Four test modules each carried their own ``_write_config`` /
    ``_patch_config_path`` pair — two of them with a comment saying they
    mirror ``test_model_registry.py``'s (#470). This is that pair, once:
    dump ``content`` to a temp ``models.yaml``, repoint both modules'
    ``CONFIG_PATH`` (each ``_load_config()`` reads the module attribute at
    call time, so no reload is needed) and drop ``host_profile``'s parsed
    cache so the next read actually re-parses the new file.

    Pass ``dirpath`` to write a *second* config in the same test — the file
    name is fixed, so a distinct directory is what makes it a distinct file.
    """
    from src import host_profile, model_registry

    def _write(content: dict, *, dirpath=None) -> Path:
        cfg = Path(dirpath or tmp_path) / "models.yaml"
        cfg.write_text(yaml.safe_dump(content), encoding="utf-8")
        monkeypatch.setattr(host_profile, "CONFIG_PATH", cfg)
        monkeypatch.setattr(model_registry, "CONFIG_PATH", cfg, raising=False)
        host_profile._CONFIG_CACHE.clear()
        return cfg

    return _write


@pytest.fixture(autouse=True)
def _isolate_code_usage_history(tmp_path):
    """Point the Code-tab history snapshot at a per-test temp file (#280).

    ``get_summary()`` folds records into and reads synthetic records from
    ``data/code_usage_history.json`` — without this, unit tests would write
    into the repo's real snapshot and read the dev machine's history back
    into their assertions.
    """
    from src import code_usage_history

    code_usage_history._reset_for_tests(tmp_path / "code_usage_history.json")
    yield
    code_usage_history._reset_for_tests(None)


@pytest.fixture(autouse=True)
def _isolate_claude_code_otel_store(tmp_path, monkeypatch):
    """Point the OTel usage store at a per-test temp file (#280 follow-up).

    ``get_summary()`` tops Claude up with OTel deltas — without this, tests
    would read the dev machine's real ``data/telemetry`` history into their
    assertions.  Tests that want OTel data monkeypatch/write it themselves.
    """
    from src import claude_code_otel as cco

    monkeypatch.setattr(cco, "_DATA_DIR", tmp_path / "telemetry")
    monkeypatch.setattr(cco, "_DATA_FILE", tmp_path / "telemetry" / "usage.jsonl")
    cco._reset_for_tests()
    yield
    cco._reset_for_tests()


@pytest.fixture(autouse=True)
def _hermetic_remote_probes(monkeypatch):
    """Keep unit tests off the real network and remote-stats caches clean (#396).

    ``remote_stats.dial_address`` now sits under every peer-connect path
    (model proxy, SSH ops, peer health), and for a host with a ``tailscale:``
    fallback a cache-miss resolve TCP-probes real addresses. Stubbing the
    lowest-level ``_probe_port`` to "nothing answers" makes every unstubbed
    resolve deterministically pick the LAN primary with zero sockets — tests
    that exercise the fallback itself monkeypatch ``_probe_liveness_ports``
    (or ``_probe_port``) on top of this. The per-host caches are cleared on
    both sides so a cached liveness/dial route never leaks between tests.
    """
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_probe_port", lambda address, port: False)
    caches = (
        remote_stats._cache,
        remote_stats._liveness_cache,
        remote_stats._dial_cache,
        remote_stats._active_route,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


@pytest.fixture(autouse=True)
def _reset_shared_http_clients():
    """Reset the hub's shared httpx client singletons around every test.

    The hub reuses one pooled ``httpx.AsyncClient`` / ``httpx.Client`` across
    requests (issue #165) and caches it module-side. Tests that monkeypatch
    ``httpx.AsyncClient`` / ``httpx.Client`` to a fake need the cache cleared so
    ``get_async_client()`` / ``get_sync_client()`` reconstruct the patched class
    fresh, and so a real client built in one test never leaks into the next.
    """
    from src import http_client

    http_client._async_client = None
    http_client._sync_client = None
    yield
    http_client._async_client = None
    http_client._sync_client = None
