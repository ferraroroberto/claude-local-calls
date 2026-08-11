"""End-to-end tests for the SPA Claude Code usage tab (issue #20).

Boots the hub with OTel disabled (set in tests/e2e/conftest.py).

The JSONL parser reads from ~/.claude/projects/ — on a CI runner or a
dev machine those files exist (Claude Code writes them automatically).
The /admin/api/code/usage/summary endpoint is expected to return a valid
JSON dict even when no JSONL files are present (empty-state path).

Checks:
  - Tab button is visible and clickable.
  - Switching to the tab hides the other panes.
  - The four counter elements are rendered (may show "—" when empty).
  - The summary API returns a well-formed JSON response.
  - Phone-size screenshot is saved.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.usefixtures("admin_url")

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
PHONE_VIEWPORT = {"width": 390, "height": 844}
# Pane-switch DOM/CSS transition can overrun a tight budget under runner
# contention (issue #177) — give it the same headroom as the button wait.
PANE_TIMEOUT = 10000
# The summary endpoint cold-scans every vendor's full session history on the
# hub's first call (no mtime cache yet — each e2e session boots a fresh hub
# subprocess, so every run pays this once). Measured ~19.7s for period=all,
# vendor=all on this box's ~300k-record Claude history (#491) — right at the
# old 20s budget, so it tipped over under host contention. The
# ``_warm_code_usage_cache`` fixture below now pays that cost once, up front,
# before any assertion races it; every call below runs against a warm mtime
# cache, so 20s stays a comfortable margin rather than a bet against the
# cold-scan clock.
API_TIMEOUT = 20.0
# Generous ceiling for the one-off cold scan itself — not a race, just a
# backstop against a genuinely hung hub.
_WARMUP_TIMEOUT = 90.0


@pytest.fixture(scope="session", autouse=True)
def _warm_code_usage_cache(admin_url):
    """Pay the cold-scan cost once, before any test assertion depends on it.

    ``get_summary`` builds every vendor's mtime cache from scratch on the
    hub's first call (#491) — cheap once warm, but ~20s cold on this box's
    transcript volume. Firing the most expensive combination
    (``period=all&vendor=all``) here means every test below — including the
    SPA's own polling in ``test_code_usage_tab_loads`` /
    ``test_code_usage_tab_phone_screenshot`` — runs against an already-warm
    cache instead of racing the scan under its own fixed timeout. Duration is
    logged (not just swallowed) so a growing cold-scan cost stays visible
    over time rather than silently hidden behind ever-larger budgets.
    """
    base = admin_url.rstrip("/") + "/api/code/usage/summary"
    t0 = time.monotonic()
    r = httpx.get(base, params={"period": "all", "vendor": "all"}, timeout=_WARMUP_TIMEOUT)
    elapsed = time.monotonic() - t0
    print(f"\n[code-usage] cold-scan warm-up: {elapsed:.2f}s (status {r.status_code})")
    assert r.status_code == 200, f"warm-up call failed: {r.text}"


@pytest.fixture(autouse=True)
def _no_console_errors(page):
    errs = []

    def _on_console(msg):
        if msg.type != "error":
            return
        text = msg.text
        # PWA icons are optional placeholders; ignore their 404s.
        if "icon-180" in text or "icon-512" in text:
            return
        errs.append(text)

    page.on("console", _on_console)
    yield
    if errs:
        raise AssertionError("console errors: " + " | ".join(errs))


def test_code_usage_tab_loads(page, admin_url):
    page.goto(admin_url, wait_until="domcontentloaded")
    # Tab button must be present and visible.
    page.wait_for_selector("#tabCodeUsage", state="visible", timeout=5000)
    page.click("#tabCodeUsage")
    page.wait_for_selector("#paneCodeUsage", state="visible", timeout=PANE_TIMEOUT)
    # Other panes must be hidden — wait for state, don't race the DOM.
    page.wait_for_selector("#paneHub", state="hidden", timeout=PANE_TIMEOUT)
    page.wait_for_selector("#paneModels", state="hidden", timeout=PANE_TIMEOUT)
    page.wait_for_selector("#panePlayground", state="hidden", timeout=PANE_TIMEOUT)
    page.wait_for_selector("#paneTelemetry", state="hidden", timeout=PANE_TIMEOUT)
    # All four counter elements must be present (content may be "—" or a value).
    assert page.locator("#cldRequests").count() == 1
    assert page.locator("#cldInputTok").count() == 1
    assert page.locator("#cldOutputTok").count() == 1
    assert page.locator("#cldCacheRead").count() == 1


def test_code_usage_api_returns_valid_json(admin_url):
    """The /admin/api/code/usage/summary endpoint must return a 200 with
    the expected keys, for every valid period value."""
    base = admin_url.rstrip("/") + "/api/code/usage/summary"
    for period in ("today", "week", "month", "all"):
        r = httpx.get(base, params={"period": period}, timeout=API_TIMEOUT)
        assert r.status_code == 200, f"period={period}: {r.text}"
        body = r.json()
        for key in ("period", "vendor", "totals", "daily", "by_model", "by_project", "by_vendor", "recent_sessions"):
            assert key in body, f"period={period}: missing key {key!r}"
        assert body["period"] == period
        assert isinstance(body["totals"], dict)
        # Equivalent-API-cost fields (issue #52) — present and numeric.
        for cost_key in ("input_cost", "output_cost", "cache_read_cost"):
            assert cost_key in body["totals"], f"period={period}: missing {cost_key!r}"
            assert isinstance(body["totals"][cost_key], (int, float))
        assert isinstance(body["daily"], list)
        assert isinstance(body["by_model"], list)
        assert isinstance(body["by_project"], list)
        assert isinstance(body["by_vendor"], list)
        assert isinstance(body["recent_sessions"], list)


def test_code_usage_api_vendor_param(admin_url):
    """The vendor query param (claude | codex | copilot | all) is accepted and
    echoed, and by_vendor rows only ever carry the requested vendor(s)
    (issues #71, #231)."""
    base = admin_url.rstrip("/") + "/api/code/usage/summary"
    for vendor in ("all", "claude", "codex", "copilot"):
        r = httpx.get(base, params={"period": "all", "vendor": vendor}, timeout=API_TIMEOUT)
        assert r.status_code == 200, f"vendor={vendor}: {r.text}"
        body = r.json()
        assert body["vendor"] == vendor
        seen = {row["vendor"] for row in body["by_vendor"]}
        if vendor == "all":
            # agy is a first-class curated vendor (issue #280); any other
            # AgentsView-discovered vendor is a legitimate extra when a live
            # AgentsView is serving on this machine.
            extra = set((body.get("agentsview") or {}).get("vendors") or [])
            assert seen <= {"claude", "codex", "copilot", "agy"} | extra
        else:
            assert seen <= {vendor}
    # Unknown vendor falls back to "all".
    r = httpx.get(base, params={"period": "all", "vendor": "bogus"}, timeout=API_TIMEOUT)
    assert r.status_code == 200
    assert r.json()["vendor"] == "all"


def test_period_toggle_changes_counters(page, admin_url):
    """Clicking 'Week' toggles the active button; counters update.

    The period toggle sits inside the first card — we fire the click via JS
    to avoid any viewport-clipping issues in the headless runner.
    """
    page.set_viewport_size({"width": 800, "height": 900})
    page.goto(admin_url, wait_until="domcontentloaded")
    page.click("#tabCodeUsage")
    page.wait_for_selector("#paneCodeUsage", state="visible", timeout=PANE_TIMEOUT)
    # Give the first poll a moment to land.
    page.wait_for_timeout(2000)
    # Fire the click via JS so viewport clipping doesn't block us.
    page.evaluate(
        "document.querySelector('#cldPeriodSeg button[data-period=\"week\"]').click()"
    )
    active_period = page.evaluate(
        "document.querySelector('#cldPeriodSeg button.active')?.dataset.period"
    )
    assert active_period == "week", f"expected 'week', got {active_period!r}"


def test_vendor_toggle_changes_selector(page, admin_url):
    """Clicking 'Codex' toggles the active vendor button and reveals the
    per-vendor card only in 'All' mode (issue #71)."""
    page.set_viewport_size({"width": 800, "height": 900})
    page.goto(admin_url, wait_until="domcontentloaded")
    page.click("#tabCodeUsage")
    page.wait_for_selector("#paneCodeUsage", state="visible", timeout=PANE_TIMEOUT)
    page.wait_for_timeout(2000)
    # Default is "all" → per-vendor card visible.
    assert page.evaluate(
        "document.querySelector('#cldVendorSeg button.active')?.dataset.vendor"
    ) == "all"
    # Switch to Codex via JS to avoid viewport clipping.
    page.evaluate(
        "document.querySelector('#cldVendorSeg button[data-vendor=\"codex\"]').click()"
    )
    active_vendor = page.evaluate(
        "document.querySelector('#cldVendorSeg button.active')?.dataset.vendor"
    )
    assert active_vendor == "codex", f"expected 'codex', got {active_vendor!r}"
    # Per-vendor card is hidden when a single vendor is selected. Condition
    # wait, not a fixed sleep: the vendor-switch re-render sits behind the
    # summary fetch, which cold-scans full multi-vendor history and overruns
    # any fixed budget under host contention (issue #361 — same flake class
    # as the pane switches, #177).
    page.wait_for_selector("#cldVendorCard", state="hidden", timeout=PANE_TIMEOUT)


def test_code_usage_tab_phone_screenshot(page, admin_url, browser_name):
    page.set_viewport_size(PHONE_VIEWPORT)
    page.goto(admin_url, wait_until="domcontentloaded")
    page.click("#tabCodeUsage")
    page.wait_for_selector("#paneCodeUsage", state="visible", timeout=PANE_TIMEOUT)
    # Wait for the first poll to complete (counters fill in).
    page.wait_for_function(
        "document.getElementById('cldRequests') && "
        "document.getElementById('cldRequests').textContent !== ''",
        timeout=8000,
    )
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT_DIR / f"code-usage-390x844-{browser_name}.png"
    page.screenshot(path=str(out), full_page=True)
    assert out.exists() and out.stat().st_size > 0
