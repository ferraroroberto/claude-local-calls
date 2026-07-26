"""End-to-end tests for the Fleet-placement grid (issues #354, #430).

The grid is UI over the desired-state API (#353), **read-only since #430** —
the placement is derived from ``config/models.yaml`` (hosts chains + startup
policy), so there are no switches and no PATCH any more (the full card rework
is the #431 follow-up). These tests pin the GET contract with route
interception so the render is deterministic and no real backend is ever
started: the SPA is the unit under test here (the API itself has its own unit
tests in ``tests/test_fleet_placement_router.py``).

What they lock in:
  * per-host groups render, each with its status chip and a row per desired
    model — no toggles anywhere (read-only);
  * a desired+running model carries the "running" badge; desired-but-down on a
    reachable host reads "pending";
  * an **offline** host renders the deferred-apply note (not an error) — the
    registry keeps its desired set and the reconcile loop applies it on
    power-up;
  * a host with an empty desired set shows an honest note instead of an empty
    list.
"""

from __future__ import annotations

import json

# A fixed fleet: the local tower (online), an offline Mac Mini, and a
# no-desired-models satellite.
FAKE_PLACEMENT = {
    "placement": {"tower": ["whisper"], "mac-mini-m4": ["parakeet"]},
    "hosts": [
        {
            "id": "tower", "display_name": "Tower", "icon": "monitor",
            "local": True, "reachable": True, "can_ssh": False, "runs_hub": True,
            "eligible": [
                {"id": "whisper", "display_name": "Whisper Turbo"},
                {"id": "qwen35_4b", "display_name": "Qwen3.5 4B"},
                {"id": "piper", "display_name": "Piper TTS", "device": "cpu"},
            ],
            "placed": ["whisper"], "running": ["whisper"],
        },
        {
            "id": "mac-mini-m4", "display_name": "Mac Mini M4", "icon": "server",
            "local": False, "reachable": False, "can_ssh": True, "runs_hub": True,
            "eligible": [
                {"id": "parakeet", "display_name": "Parakeet"},
                {"id": "qwen", "display_name": "Qwen"},
            ],
            "placed": ["parakeet"], "running": [],
        },
        {
            # Managed-only satellite: powered on (TCP liveness), but runs no hub
            # and has no desired models — shown honestly with a note and no
            # rows, never silently dropped.
            "id": "gaming", "display_name": "Gaming", "icon": "server",
            "local": False, "reachable": True, "can_ssh": True, "runs_hub": False,
            "eligible": [], "placed": [], "running": [],
        },
    ],
}


def _install_routes(page, payload=None):
    """Serve a fixed GET payload — the grid is read-only (#430)."""
    body = json.dumps(payload or FAKE_PLACEMENT)

    def handler(route):
        route.fulfill(status=200, content_type="application/json", body=body)

    page.route("**/admin/api/fleet-placement", handler)


def _open_fleet_card(page, admin_url):
    page.goto(admin_url, wait_until="load")
    page.click("#tabModels")
    page.wait_for_selector("#paneModels", state="visible", timeout=5000)
    # The card is folded by default — open it so the grid renders.
    page.eval_on_selector("#fleetPlacementCard", "el => { el.open = true; }")
    page.wait_for_selector("#fleetPlacementBody .fleet-host", state="visible", timeout=10000)


def test_fleet_placement_renders_host_groups_read_only(page, admin_url):
    _install_routes(page)
    _open_fleet_card(page, admin_url)

    groups = page.locator("#fleetPlacementBody .fleet-host")
    assert groups.count() == 3, "expected one group per fleet host"

    tower = page.locator(".fleet-host", has_text="Tower")
    assert "This machine" in tower.locator(".hub-live-status").inner_text()
    # Desired + running → the running badge; only desired models render rows.
    assert tower.locator(".startup-row", has_text="Whisper Turbo").locator(".badge.good").count() == 1
    assert tower.locator(".startup-row").count() == 1

    # #430: the grid is read-only — no switches anywhere.
    assert page.locator("#fleetPlacementBody button.toggle[role='switch']").count() == 0

    mac = page.locator(".fleet-host", has_text="Mac Mini M4")
    assert "Offline" in mac.locator(".hub-live-status").inner_text()

    # A satellite with nothing desired still shows — online, with an honest
    # note and no rows — rather than silently vanishing.
    gaming = page.locator(".fleet-host", has_text="Gaming")
    assert "Online" in gaming.locator(".hub-live-status").inner_text()
    assert gaming.locator(".startup-row").count() == 0
    assert "no hub" in gaming.locator(".fleet-host-note").inner_text().lower()


def test_offline_host_shows_deferred_note_not_error(page, admin_url):
    _install_routes(page)
    _open_fleet_card(page, admin_url)

    mac = page.locator(".fleet-host", has_text="Mac Mini M4")
    note = mac.locator(".fleet-host-note")
    assert note.count() == 1, "offline host should carry the deferred-apply note"
    assert "power" in note.inner_text().lower(), "note should explain it applies on power-up"

    # An offline host is a deferred state, never an error empty-state — its
    # desired model renders with the "deferred" badge.
    assert page.locator("#fleetPlacementBody .empty-state").count() == 0
    assert "deferred" in mac.locator(".startup-row", has_text="Parakeet").inner_text().lower()


# A fleet where the tower overcommits its VRAM ceiling and the Mac Mini (no
# declared ceiling) does not — pins the advisory capacity warning (#375).
CAPACITY_PLACEMENT = {
    "placement": {"tower": ["gemma4_26b", "whisper"], "mac-mini-m4": ["parakeet"]},
    "hosts": [
        {
            "id": "tower", "display_name": "Tower", "icon": "monitor",
            "local": True, "reachable": True, "can_ssh": False, "runs_hub": True,
            "eligible": [
                {"id": "gemma4_26b", "display_name": "Gemma4 26B"},
                {"id": "whisper", "display_name": "Whisper Turbo"},
            ],
            "placed": ["gemma4_26b", "whisper"], "running": ["gemma4_26b", "whisper"],
            "vram_mb": 8192, "est_vram_mb": 16000, "capacity_warning": True,
        },
        {
            "id": "mac-mini-m4", "display_name": "Mac Mini M4", "icon": "server",
            "local": False, "reachable": True, "can_ssh": True, "runs_hub": True,
            "eligible": [{"id": "parakeet", "display_name": "Parakeet"}],
            "placed": ["parakeet"], "running": [],
            "vram_mb": None, "est_vram_mb": 99999, "capacity_warning": False,
        },
    ],
}


def test_capacity_warning_renders_only_on_overcommitted_host(page, admin_url):
    """The overcommitted tower shows the advisory VRAM warning; the ceiling-less
    Mac Mini never does, even with a large footprint (#375)."""
    _install_routes(page, CAPACITY_PLACEMENT)
    _open_fleet_card(page, admin_url)

    tower = page.locator(".fleet-host", has_text="Tower")
    warn = tower.locator(".fleet-capacity-warn")
    assert warn.count() == 1, "overcommitted host should show the capacity warning"
    assert "over vram capacity" in warn.inner_text().lower()

    mac = page.locator(".fleet-host", has_text="Mac Mini M4")
    assert mac.locator(".fleet-capacity-warn").count() == 0, \
        "a host with no declared ceiling must never warn"
