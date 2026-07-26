"""End-to-end tests for the Fleet summary card (issues #354, #430, #431).

The card is a **read-only** per-machine summary over the desired-state API
(#353): per host — models running now (live), models loadable but not
running (chain members incl. on-demand), and estimated capacity (GPU est vs
ceiling, RAM where known). Zero controls beyond the collapse (#431) — no
switches, no refresh button, no sync icons; placement is edited in the model
cards / config/models.yaml. These tests pin the GET contract with route
interception so the render is deterministic and no real backend is ever
started: the SPA is the unit under test here (the API itself has its own
unit tests in ``tests/test_fleet_placement_router.py``).

What they lock in:
  * per-host groups render with a status chip, a Running list (live state,
    honest ``none``), a Loadable list (eligible-but-not-running, incl. the
    cpu device hint), and a Capacity line;
  * #434: Running/Loadable render one model per line (``.fleet-model-line``,
    muted caption size, right-aligned by the value column) — the old inline
    ``·``-joined flow is gone;
  * #434: the capacity line is the same shape on every machine — ``GPU ~est
    / ceiling · RAM used/total`` with live RAM where the API carries a
    ``ram`` block, the declared total otherwise, and the system-RAM total as
    the GPU denominator on a unified-memory host (no bare "est" outlier);
  * **zero interactive controls** inside the card body — no toggles and no
    refresh button anywhere in the card;
  * an **offline** host renders the deferred-apply note (not an error) — the
    registry keeps its desired set and the reconcile loop applies it on
    power-up;
  * a host with nothing placeable shows an honest note instead of empty rows;
  * a backend the hub merely adopted from an external sibling on a
    mutex-shared port is labelled ``· external`` (#431), never claimed as
    hub-run without qualification;
  * the advisory capacity warning (#375) renders only on the overcommitted
    host.
"""

from __future__ import annotations

import json

# A fixed fleet: the local tower (online), an offline Mac Mini, and a
# nothing-placeable machine.
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
            "placed": ["whisper"], "running": ["whisper"], "external": [],
            "vram_mb": 16384, "est_vram_mb": 2048, "capacity_warning": False,
            "ram_mb": 131072,
            "ram": {"used_gb": 45.5, "total_gb": 128.0, "percent": 35.5},
        },
        {
            "id": "mac-mini-m4", "display_name": "Mac Mini M4", "icon": "server",
            "local": False, "reachable": False, "can_ssh": True, "runs_hub": True,
            "eligible": [
                {"id": "parakeet", "display_name": "Parakeet"},
                {"id": "qwen", "display_name": "Qwen"},
            ],
            "placed": ["parakeet"], "running": [], "external": [],
            "vram_mb": None, "est_vram_mb": 4096, "capacity_warning": False,
            "ram_mb": 16384, "ram": None,  # offline — no live figure (#434)
        },
        {
            # Enrolled as a managed machine, serves no models — shown honestly
            # with a note and no rows, never silently dropped.
            "id": "openclaw", "display_name": "OpenClaw", "icon": "laptop",
            "local": False, "reachable": True, "can_ssh": True, "runs_hub": False,
            "eligible": [], "placed": [], "running": [], "external": [],
            "vram_mb": None, "est_vram_mb": 0, "capacity_warning": False,
            "ram_mb": None, "ram": None,
        },
    ],
}


def _install_routes(page, payload=None):
    """Serve a fixed GET payload — the card is read-only (#430/#431)."""
    body = json.dumps(payload or FAKE_PLACEMENT)

    def handler(route):
        route.fulfill(status=200, content_type="application/json", body=body)

    page.route("**/admin/api/fleet-placement", handler)


def _open_fleet_card(page, admin_url):
    page.goto(admin_url, wait_until="load")
    page.click("#tabModels")
    page.wait_for_selector("#paneModels", state="visible", timeout=5000)
    # The card is folded by default — open it so the summary renders.
    page.eval_on_selector("#fleetPlacementCard", "el => { el.open = true; }")
    page.wait_for_selector("#fleetPlacementBody .fleet-host", state="visible", timeout=10000)


def _row_value(host_locator, label):
    """The right-hand value text of one summary line ("Running", …)."""
    return host_locator.locator(
        ".startup-row", has_text=label
    ).locator(".roles-row-value").inner_text()


def test_fleet_summary_renders_read_only_host_groups(page, admin_url):
    _install_routes(page)
    _open_fleet_card(page, admin_url)

    groups = page.locator("#fleetPlacementBody .fleet-host")
    assert groups.count() == 3, "expected one group per fleet host"

    tower = page.locator(".fleet-host", has_text="Tower")
    assert "This machine" in tower.locator(".hub-live-status").inner_text()
    # Running = live state; Loadable = eligible-but-not-running, with the
    # low-emphasis cpu device hint riding the CPU-resident row. One model
    # per line (#434) — .fleet-model-line, muted caption size, no inline
    # `·` joiner between models.
    running_row = tower.locator(".startup-row", has_text="Running")
    running_lines = running_row.locator(".fleet-model-line")
    assert running_lines.count() == 1
    assert "Whisper Turbo" in running_lines.first.inner_text()
    assert "muted" in (running_lines.first.get_attribute("class") or "")
    assert "small" in (running_lines.first.get_attribute("class") or "")

    loadable_row = tower.locator(".startup-row", has_text="Loadable")
    loadable_lines = loadable_row.locator(".fleet-model-line")
    assert loadable_lines.count() == 2, "one line per loadable model"
    texts = [loadable_lines.nth(i).inner_text() for i in range(2)]
    assert any("Qwen3.5 4B" in t for t in texts)
    assert any("Piper TTS" in t and "cpu" in t for t in texts)
    for i in range(2):
        cls = loadable_lines.nth(i).get_attribute("class") or ""
        assert "muted" in cls and "small" in cls

    # Capacity (#434): homogeneous shape — GPU est/ceiling · live RAM used/total.
    capacity = _row_value(tower, "Capacity")
    assert "GPU ~2.0 GB / 16.0 GB" in capacity
    assert "RAM 45.5 / 128 GB" in capacity

    # #431: a summary, not a control plane — zero interactive controls in the
    # whole card (no switches, no refresh button, no sync icons).
    assert page.locator("#fleetPlacementCard button").count() == 0
    assert page.locator("#fleetPlacementRefreshBtn").count() == 0

    # A machine with nothing placeable still shows — online, with an honest
    # note and no rows — rather than silently vanishing.
    claw = page.locator(".fleet-host", has_text="OpenClaw")
    assert "Online" in claw.locator(".hub-live-status").inner_text()
    assert claw.locator(".startup-row").count() == 0
    assert "no models placeable" in claw.locator(".fleet-host-note").inner_text().lower()


def test_offline_host_shows_deferred_note_not_error(page, admin_url):
    _install_routes(page)
    _open_fleet_card(page, admin_url)

    mac = page.locator(".fleet-host", has_text="Mac Mini M4")
    assert "Offline" in mac.locator(".hub-live-status").inner_text()
    note = mac.locator(".fleet-host-note")
    assert note.count() == 1, "offline host should carry the deferred-apply note"
    assert "power" in note.inner_text().lower(), "note should explain it applies on power-up"

    # An offline host is a deferred state, never an error empty-state — its
    # summary still renders, with an honest empty Running line.
    assert page.locator("#fleetPlacementBody .empty-state").count() == 0
    assert "none" in _row_value(mac, "Running").lower()
    assert "Parakeet" in _row_value(mac, "Loadable")

    # #434: identical capacity shape on the ceiling-less unified-memory host
    # — system RAM is the GPU denominator, declared total when offline (no
    # live figure), never a bare "GPU ~X GB est" outlier.
    capacity = _row_value(mac, "Capacity")
    assert "GPU ~4.0 GB / 16.0 GB" in capacity
    assert "RAM 16 GB" in capacity
    assert "est" not in capacity


def test_external_adopted_backend_labelled(page, admin_url):
    """A live backend served by a foreign adopted process (voice-transcriber's
    whisper-server on the mutex-shared :8090) carries the distinct
    ``· external`` marker — the summary must not claim the hub runs it (#431)."""
    payload = json.loads(json.dumps(FAKE_PLACEMENT))
    payload["hosts"][0]["external"] = ["whisper"]
    _install_routes(page, payload)
    _open_fleet_card(page, admin_url)

    tower = page.locator(".fleet-host", has_text="Tower")
    running = _row_value(tower, "Running")
    assert "Whisper Turbo" in running and "external" in running


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
            "external": [],
            "vram_mb": 8192, "est_vram_mb": 16000, "capacity_warning": True,
            "ram_mb": 131072,
        },
        {
            "id": "mac-mini-m4", "display_name": "Mac Mini M4", "icon": "server",
            "local": False, "reachable": True, "can_ssh": True, "runs_hub": True,
            "eligible": [{"id": "parakeet", "display_name": "Parakeet"}],
            "placed": ["parakeet"], "running": [], "external": [],
            "vram_mb": None, "est_vram_mb": 99999, "capacity_warning": False,
            "ram_mb": None,
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
