"""End-to-end tests for the Models tab's placement editor (#424).

Same discipline as ``test_models_placement_cards.py``: the GET contract is
pinned with route interception so the render is deterministic; the PUT is
intercepted too, so no real config write ever happens — the SPA is the unit
under test (validation + the git transaction are covered in
``tests/test_config_write.py`` / ``tests/test_models_router.py``).

What they lock in:
  * the editor only offers itself where the server says this hub may write
    (``config.write_enabled`` — satellites render read-only cards), and
    never on a virtual-alias row (``placement.editable: false``);
  * the full edit surface drives the PUT payload faithfully — reorder /
    remove / add chain hosts, cpu-tier toggle, startup policy, idle window;
  * a validation rejection (400) surfaces its detail *inline* in the open
    editor — the VRAM arithmetic is the message — and nothing closes;
  * a successful save closes the editor and refreshes;
  * the config-version chip renders the models.yaml sha in the card header.
"""

from __future__ import annotations

import copy
import json

BASE_CONFIG = {
    "sha": "abc1234",
    "write_enabled": True,
    "write_host": "tower",
    "fleet_hosts": [
        {"id": "tower", "vram_mb": 16384},
        {"id": "mac-mini-m4", "vram_mb": None},
        {"id": "openclaw", "vram_mb": None},
        {"id": "gaming", "vram_mb": 8192},
    ],
}

FAKE_MODELS = {
    "models": [
        {
            # Multi-host chain with a degraded cpu last resort — the whisper
            # shape; the editor's reorder/remove/cpu controls act on this.
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
                "est_vram_mb": 2000, "editable": True,
            },
        },
        {
            # On-demand row — the idle-window edit target (gemma shape).
            "id": "gemma4_26b", "display_name": "gemma4-26b-a4b-it",
            "backend": "openai", "engine": "llama-server", "port": 8087,
            "url": None, "aliases": ["agentic_heavy"], "controllable": True,
            "ownership": "none", "pid": None, "reachable": False,
            "model_path": None, "host": "tower",
            "placement": {
                "chain": [{"id": "tower", "cpu": False}],
                "startup": "on_demand", "idle_unload_minutes": 30,
                "est_vram_mb": 13400, "editable": True,
            },
        },
        {
            # Virtual alias — placement rendered but never editable.
            "id": "qwen35_4b_nothink", "display_name": "qwen3.5-4b-nothink",
            "backend": "openai", "engine": None, "port": 8088, "url": None,
            "aliases": [], "controllable": False, "ownership": "none",
            "pid": None, "reachable": True, "model_path": None, "host": "tower",
            "placement": {
                "chain": [{"id": "tower", "cpu": False}],
                "startup": "eager", "idle_unload_minutes": None,
                "est_vram_mb": 0, "editable": False,
            },
        },
    ],
    "host_budgets": {
        "tower": {"vram_mb": 16384, "resident_est_vram_mb": 4300},
        "gaming": {"vram_mb": 8192, "resident_est_vram_mb": 4000},
    },
    "config": BASE_CONFIG,
}


def _payload(write_enabled=True):
    body = copy.deepcopy(FAKE_MODELS)
    body["config"]["write_enabled"] = write_enabled
    return body


def _open_models_tab(page, admin_url, body, put_handler=None):
    page.add_init_script(
        "localStorage.setItem('llmhub.models.activeOnly', 'false');"
    )

    def handler(route):
        if route.request.method == "PUT" and put_handler is not None:
            put_handler(route)
            return
        route.fulfill(
            status=200, content_type="application/json", body=json.dumps(body)
        )

    # ``**`` (not ``*``): Playwright's single star never crosses a ``/``,
    # and the PUT lands on /admin/api/models/<id>/placement — both the GET
    # and the PUT must be intercepted for the suite to stay hermetic.
    page.route("**/admin/api/models**", handler)
    page.goto(admin_url, wait_until="load")
    page.click("#tabModels")
    page.wait_for_selector("#paneModels", state="visible", timeout=5000)
    page.wait_for_selector("#modelsList .app-item .placement", state="visible", timeout=10000)


def _item(page, model_id):
    return page.locator(f'#modelsList .app-item[data-id="{model_id}"]')


def _open_editor(page, model_id):
    _item(page, model_id).locator(".placement-edit-btn").click()
    page.wait_for_selector(
        f'#modelsList .app-item[data-id="{model_id}"] .placement-editor',
        state="visible", timeout=5000,
    )
    return _item(page, model_id).locator(".placement-editor")


def test_config_sha_chip_renders(page, admin_url):
    _open_models_tab(page, admin_url, _payload())
    assert page.locator("#modelsConfigSha").inner_text() == "cfg abc1234"


def test_no_edit_affordance_when_not_write_host(page, admin_url):
    """A satellite hub (write_enabled false) renders the cards read-only —
    zero edit buttons anywhere, exactly the pre-#424 surface."""
    _open_models_tab(page, admin_url, _payload(write_enabled=False))
    assert page.locator("#modelsList .placement").count() >= 3
    assert page.locator(".placement-edit-btn").count() == 0


def test_virtual_alias_row_is_never_editable(page, admin_url):
    _open_models_tab(page, admin_url, _payload())
    assert _item(page, "whisper").locator(".placement-edit-btn").count() == 1
    assert _item(page, "qwen35_4b_nothink").locator(".placement-edit-btn").count() == 0


def test_full_edit_drives_put_payload(page, admin_url):
    """Reorder + remove + cpu-toggle + add + policy edits land in the PUT
    body exactly as drafted, and a 200 closes the editor."""
    captured = {}

    def put_handler(route):
        captured["url"] = route.request.url
        captured["body"] = route.request.post_data_json
        route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps({"ok": True, "changed": True, "commit": "beefcaf",
                             "config_sha": "beefcaf"}),
        )

    _open_models_tab(page, admin_url, _payload(), put_handler=put_handler)
    editor = _open_editor(page, "whisper")

    # Chain: [gaming, mac-mini-m4, tower·cpu] → remove mac-mini-m4, move
    # tower to the front, clear its cpu flag, then add mac-mini-m4 back.
    editor.locator('.pe-host-row[data-host="mac-mini-m4"] .pe-remove').click()
    editor.locator('.pe-host-row[data-host="tower"] .pe-up').click()
    editor.locator('.pe-host-row[data-host="tower"] .pe-cpu').uncheck()
    editor.locator(".pe-add-select").select_option("mac-mini-m4")
    editor.locator(".pe-add-btn").click()

    editor.locator(".pe-startup").select_option("on_demand")
    editor.locator(".pe-idle").fill("45")
    editor.locator(".pe-save").click()

    page.wait_for_selector(".placement-editor", state="detached", timeout=5000)
    assert "/admin/api/models/whisper/placement" in captured["url"]
    assert captured["body"] == {
        "hosts": [
            {"id": "tower", "cpu": False},
            {"id": "gaming", "cpu": False},
            {"id": "mac-mini-m4", "cpu": False},
        ],
        "startup": "on_demand",
        "idle_unload_minutes": 45,
    }


def test_validation_rejection_shows_inline_and_keeps_editor_open(page, admin_url):
    """A 400 (the #375 VRAM hard gate) renders its detail inline in the open
    editor — the arithmetic is the message — and nothing is lost."""
    detail = (
        "placing gemma4_26b (~13400 MB) on tower overcommits its VRAM budget: "
        "~25700 MB estimated vs the 16384 MB ceiling"
    )

    def put_handler(route):
        route.fulfill(
            status=400, content_type="application/json",
            body=json.dumps({"detail": detail}),
        )

    _open_models_tab(page, admin_url, _payload(), put_handler=put_handler)
    editor = _open_editor(page, "gemma4_26b")
    editor.locator(".pe-startup").select_option("eager")
    editor.locator(".pe-save").click()

    err = editor.locator(".pe-error")
    page.wait_for_selector(".placement-editor .pe-error", state="visible", timeout=5000)
    assert "overcommits" in err.inner_text()
    # Editor stays open with the draft intact for correction.
    assert editor.locator(".pe-startup").input_value() == "eager"


def test_idle_input_disabled_for_eager_policy(page, admin_url):
    _open_models_tab(page, admin_url, _payload())
    editor = _open_editor(page, "whisper")
    assert editor.locator(".pe-idle").is_disabled()
    editor.locator(".pe-startup").select_option("on_demand")
    assert editor.locator(".pe-idle").is_enabled()
