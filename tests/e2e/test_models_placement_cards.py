"""End-to-end tests for the Models tab's read-only placement cards (#423).

UI over the extended ``/admin/api/models`` payload (placement intent per row
+ per-host VRAM budget context). Same discipline as the fleet-placement grid
suite: the GET contract is pinned with route interception so the render is
deterministic and no real backend is started — the SPA is the unit under
test (the API itself is covered in ``tests/test_models_router.py``).

What they lock in:
  * chain pills render in priority order with the effective owner
    highlighted and the degraded ``cpu: true`` tier marked;
  * the startup badge reads the policy + live state truthfully — ``eager``,
    ``on-demand · loaded``, ``on-demand · idle-unloaded``;
  * the owner host's budget bar renders used-vs-ceiling, tints ``over`` on
    overcommit, and is absent for a host with no declared ceiling;
  * subscription rows (claude) carry no placement section at all.
"""

from __future__ import annotations

import json

FAKE_MODELS = {
    "models": [
        {
            # Eager multi-host row served by its preferred host: tower owner,
            # gaming fallback (the #422 orpheus shape).
            "id": "orpheus", "display_name": "orpheus-tts", "backend": "tts",
            "engine": "tts-server", "port": 8093, "url": None, "aliases": [],
            "controllable": True, "ownership": "ours", "pid": 4242,
            "reachable": True, "model_path": None, "host": "tower",
            "preferred_host": "tower", "failover": False,
            "placement": {
                "chain": [
                    {"id": "tower", "cpu": False},
                    {"id": "gaming", "cpu": False},
                ],
                "startup": "eager", "idle_unload_minutes": None,
                "est_vram_mb": 2200,
            },
        },
        {
            # On-demand row currently idle-unloaded (stopped, waits for a
            # request) — the gemma4_26b shape.
            "id": "gemma4_26b", "display_name": "gemma4-26b-a4b-it",
            "backend": "openai", "engine": "llama-server", "port": 8087,
            "url": None, "aliases": ["agentic_heavy"], "controllable": True,
            "ownership": "none", "pid": None, "reachable": False,
            "model_path": None, "host": "tower",
            "placement": {
                "chain": [{"id": "tower", "cpu": False}],
                "startup": "on_demand", "idle_unload_minutes": 30,
                "est_vram_mb": 13400,
            },
        },
        {
            # Remote-owned chain with a degraded cpu last resort; its owner
            # host (gaming) overcommits its ceiling in host_budgets below.
            "id": "whisper", "display_name": "whisper-large-v3-turbo",
            "backend": "whisper", "engine": "whisper-server", "port": 8090,
            "url": None, "aliases": [], "controllable": True,
            "ownership": "ours", "pid": 777, "reachable": True,
            "model_path": None, "host": "gaming",
            "preferred_host": "gaming", "failover": False,
            "placement": {
                "chain": [
                    {"id": "gaming", "cpu": False},
                    {"id": "mac-mini-m4", "cpu": False},
                    {"id": "tower", "cpu": True},
                ],
                "startup": "eager", "idle_unload_minutes": None,
                "est_vram_mb": 2000,
            },
        },
        {
            # Row owned by a host with no declared VRAM ceiling — no bar.
            "id": "parakeet", "display_name": "parakeet-tdt-0.6b-v3",
            "backend": "whisper", "engine": "parakeet-server", "port": 8098,
            "url": None, "aliases": ["parakeet"], "controllable": True,
            "ownership": "ours", "pid": 555, "reachable": True,
            "model_path": None, "host": "mac-mini-m4",
            "placement": {
                "chain": [{"id": "mac-mini-m4", "cpu": False}],
                "startup": "eager", "idle_unload_minutes": None,
                "est_vram_mb": 0,
            },
        },
        {
            # Subscription row — no placement key, no placement section.
            "id": "claude_haiku", "display_name": "claude-haiku-4-5",
            "backend": "claude", "engine": None, "port": None, "url": None,
            "aliases": ["claude_haiku"], "controllable": False,
            "ownership": "none", "pid": None, "reachable": True,
            "model_path": None, "host": "tower",
        },
    ],
    "host_budgets": {
        "tower": {"vram_mb": 16384, "resident_est_vram_mb": 4300},
        "gaming": {"vram_mb": 8192, "resident_est_vram_mb": 9000},
        "mac-mini-m4": {"vram_mb": None, "resident_est_vram_mb": 0},
    },
}


def _install_routes(page):
    def handler(route):
        route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps(FAKE_MODELS),
        )

    page.route("**/admin/api/models", handler)


def _open_models_tab(page, admin_url):
    # Show every row, not just active ones — the idle-unloaded on-demand row
    # is exactly what the cards must describe truthfully.
    page.add_init_script(
        "localStorage.setItem('llmhub.models.activeOnly', 'false');"
    )
    _install_routes(page)
    page.goto(admin_url, wait_until="load")
    page.click("#tabModels")
    page.wait_for_selector("#paneModels", state="visible", timeout=5000)
    page.wait_for_selector("#modelsList .app-item .placement", state="visible", timeout=10000)


def _item(page, model_id):
    return page.locator(f'#modelsList .app-item[data-id="{model_id}"]')


def test_chain_pills_render_with_owner_highlight(page, admin_url):
    _open_models_tab(page, admin_url)

    orpheus = _item(page, "orpheus")
    pills = orpheus.locator(".place-pill")
    assert pills.count() == 2
    assert pills.nth(0).inner_text() == "tower"
    assert "owner" in (pills.nth(0).get_attribute("class") or "")
    assert pills.nth(1).inner_text() == "gaming"
    assert "owner" not in (pills.nth(1).get_attribute("class") or "")
    # The static VRAM estimate rides the same line (2200 MB → ~2.1 GB).
    assert "2.1 GB VRAM" in orpheus.locator(".placement-vram").inner_text()


def test_degraded_cpu_tier_is_marked(page, admin_url):
    _open_models_tab(page, admin_url)

    whisper = _item(page, "whisper")
    pills = whisper.locator(".place-pill")
    assert pills.count() == 3
    # Owner highlight follows the *effective* owner (gaming), not position.
    assert "owner" in (pills.nth(0).get_attribute("class") or "")
    last = pills.nth(2)
    assert "tower" in last.inner_text() and "cpu" in last.inner_text()
    assert "owner" not in (last.get_attribute("class") or "")


def test_startup_badge_reads_policy_and_live_state(page, admin_url):
    _open_models_tab(page, admin_url)

    orpheus_badges = _item(page, "orpheus").locator(".placement .badge")
    assert orpheus_badges.count() == 1
    assert orpheus_badges.first.inner_text() == "eager"

    gemma_badge = _item(page, "gemma4_26b").locator(".placement .badge").first
    assert gemma_badge.inner_text() == "on-demand · idle-unloaded"


def test_budget_bar_used_vs_ceiling_and_overcommit_tint(page, admin_url):
    _open_models_tab(page, admin_url)

    # Tower (orpheus's owner): 4.2 / 16.0 GB, within budget → no .over tint.
    orpheus_budget = _item(page, "orpheus").locator(".placement-budget")
    assert orpheus_budget.count() == 1
    label = orpheus_budget.locator(".placement-budget-label").inner_text()
    assert "tower" in label and "4.2" in label and "16.0" in label
    bar_cls = orpheus_budget.locator(".placement-budget-bar").get_attribute("class") or ""
    assert "over" not in bar_cls

    # Gaming (whisper's owner) overcommits 9000/8192 → the .over tint.
    whisper_bar = _item(page, "whisper").locator(".placement-budget-bar")
    assert "over" in (whisper_bar.get_attribute("class") or "")

    # Mac Mini declares no ceiling — no bar at all, never a fake denominator.
    assert _item(page, "parakeet").locator(".placement-budget").count() == 0


def test_subscription_row_has_no_placement_section(page, admin_url):
    _open_models_tab(page, admin_url)

    claude = _item(page, "claude_haiku")
    assert claude.count() == 1, "subscription row should still render"
    assert claude.locator(".placement").count() == 0
