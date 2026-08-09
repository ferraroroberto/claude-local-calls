"""Anti-drift guard for this repo's `.fleet.toml` `[e2e]` table (#180-style routing).

Loads the real `.fleet.toml` and asserts representative paths land in the
tier their rule intends — an edit that silently under-routes a real e2e
surface fails here. Mechanism + fail-safe behavior are covered by
project-scaffolding's own `tests/test_classify_e2e.py`; this file only
guards this repo's declared rules.
"""

from __future__ import annotations

from pathlib import Path

from scripts.classify_e2e import load_config, classify

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_FLEET_TOML = REPO_ROOT / ".fleet.toml"


def test_real_fleet_toml_declares_usable_e2e_table() -> None:
    cfg = load_config(REAL_FLEET_TOML)
    assert cfg.source == "declared", (
        "this repo's .fleet.toml must declare a usable [e2e] table"
    )
    assert cfg.rules, "at least one [[e2e.rule]] must be declared"


def test_real_rules_route_representative_paths() -> None:
    cfg = load_config(REAL_FLEET_TOML)

    def tier(*paths: str) -> str:
        return classify(list(paths), cfg).tier

    # The admin SPA (routers + static CSS/JS/HTML) -> full.
    assert tier("app_web/routers/hub.py") == "full"
    assert tier("app_web/static/js/models.js") == "full"
    assert tier("app_web/static/_vendored/card/card.css") == "full"

    # The e2e suite itself + its boot conftest -> full.
    assert tier("tests/e2e/test_smoke.py") == "full"
    assert tier("tests/e2e/conftest.py") == "full"

    # Inert static assets -> static.
    assert tier("app_web/static/icons/foo.svg") == "static"
    assert tier("app_web/static/_vendored/nav/nav-tabs.html") == "static"

    # Backend / tray / tests / docs / markdown -> skip.
    assert tier("src/server_process.py") == "skip"
    assert tier("tray/tray.py") == "skip"
    assert tier("tests/test_server_process.py") == "skip"
    assert tier("docs/architecture.mmd") == "skip"
    assert tier("README.md") == "skip"

    # Mixed real diff (backend + admin SPA) -> full.
    assert tier("src/server_process.py", "app_web/routers/hub.py") == "full"
