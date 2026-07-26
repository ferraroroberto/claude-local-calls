"""End-to-end tests for the Hub tab's "Model decisions" card (issue #373).

Route-intercepts both backing endpoints so the render is deterministic and no
real peer probe (fleet-placement's per-host TCP liveness check) ever runs —
same discipline as ``test_fleet_placement_tab.py``. The API contract itself
has its own unit tests in ``tests/test_roles_router.py``.

What this locks in:
  * the card is folded by default and fetches nothing until expanded (no
    background poll — mirrors the System Map lazy-load pattern);
  * expanding it renders one row per configured role and one row per fleet
    host;
  * "View fleet summary" jumps to the Models tab and opens the existing
    Fleet summary card instead of duplicating it here.
"""

from __future__ import annotations

import json

FAKE_ROLES = {
    "roles": {
        "agentic_light": {
            "model_id": "qwen35_4b", "display_name": "Qwen3.5 4B",
            "notes": "fast lane", "fallback": [],
        },
        "audio.transcribe": {
            "model_id": "parakeet", "display_name": "Parakeet",
            "notes": None, "fallback": ["whisper"],
        },
    },
}

FAKE_PLACEMENT = {
    "placement": {"tower": ["whisper"]},
    "hosts": [
        {
            "id": "tower", "display_name": "Tower", "icon": "monitor",
            "local": True, "reachable": True, "can_ssh": False, "runs_hub": True,
            "eligible": [{"id": "whisper", "display_name": "Whisper Turbo"}],
            "placed": ["whisper"], "running": ["whisper"],
        },
    ],
}


def _install_routes(page):
    def roles_handler(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps(FAKE_ROLES))

    def placement_handler(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps(FAKE_PLACEMENT))

    page.route("**/admin/api/roles", roles_handler)
    page.route("**/admin/api/fleet-placement", placement_handler)


def test_roles_card_folded_by_default_no_fetch(page, admin_url):
    """Collapsed on load; neither backing endpoint is called until expanded."""
    calls = []

    def roles_handler(route):
        calls.append(1)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(FAKE_ROLES))

    page.route("**/admin/api/roles", roles_handler)

    page.goto(admin_url, wait_until="load")
    page.wait_for_selector("#rolesCard", state="attached", timeout=5000)
    assert page.eval_on_selector("#rolesCard", "el => el.open") is False
    page.wait_for_timeout(300)  # give any accidental eager fetch a chance to fire
    assert calls == [], "the roles card must not fetch while collapsed"


def test_roles_card_renders_on_expand(page, admin_url):
    _install_routes(page)
    page.goto(admin_url, wait_until="load")
    page.eval_on_selector("#rolesCard", "el => { el.open = true; }")

    page.wait_for_selector("#rolesList .startup-row", state="visible", timeout=10000)
    roles_rows = page.locator("#rolesList .startup-row")
    assert roles_rows.count() == 2
    assert "Qwen3.5 4B" in roles_rows.nth(0).inner_text()
    # dotted audio.transcribe role renders with its fallback chain visible
    assert "fallback: whisper" in roles_rows.nth(1).inner_text().lower()

    placement_rows = page.locator("#rolesPlacementList .startup-row")
    assert placement_rows.count() == 1
    assert "Tower" in placement_rows.first.inner_text()
    assert "whisper" in placement_rows.first.inner_text().lower()


def test_view_placement_button_jumps_to_models_tab(page, admin_url):
    _install_routes(page)
    page.goto(admin_url, wait_until="load")
    page.eval_on_selector("#rolesCard", "el => { el.open = true; }")
    page.wait_for_selector("#rolesList .startup-row", state="visible", timeout=10000)

    page.click("#rolesViewPlacementBtn")
    page.wait_for_selector("#paneModels", state="visible", timeout=5000)
    assert page.eval_on_selector("#fleetPlacementCard", "el => el.open") is True
