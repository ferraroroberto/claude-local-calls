"""Regression test for the copilot_billing UTC-midnight cache gap (#529).

`get_daily_credits` caches each day's usage; "today" (still accruing) is
refreshed on a short TTL while past days are meant to be cached forever
*once immutable*. The bug: the cache-forever check was gated purely on
`d == today` at call time, so a day cached while it was still "today" (in
its final `_TODAY_REFRESH_SECS` window before UTC midnight) was never
re-fetched once the date rolled over -- any late spend in that window was
permanently dropped. The fix stamps each entry `settled` and forces exactly
one re-fetch the first time a not-yet-settled day is seen as no longer
"today".
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from src import copilot_billing as cb


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Hermetic per-test state -- these are module-level caches (#529)."""
    monkeypatch.setattr(cb, "_day_cache", {})
    monkeypatch.setattr(cb, "_username_cache", "octocat")
    monkeypatch.setattr(cb, "_unavailable", None)
    monkeypatch.setenv(cb._PAT_ENV, "dummy-pat")
    monkeypatch.setattr(cb, "get_async_client", lambda: None)
    yield


class _FakeNow(datetime):
    """Patchable ``datetime.now()`` so the test controls the UTC clock."""

    _current: datetime = datetime(2026, 8, 27, 23, 59, 30, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):  # noqa: D102 - trivial override
        return cls._current


def _set_now(monkeypatch, when: datetime) -> None:
    _FakeNow._current = when
    monkeypatch.setattr(cb, "datetime", _FakeNow)


def test_day_cached_pre_midnight_is_refetched_once_after_rollover(monkeypatch):
    aug27 = date(2026, 8, 27)
    aug28 = date(2026, 8, 28)

    calls_for_aug27: list[float] = []

    async def _fake_fetch_day(client, pat, username, d):
        if d == aug27:
            calls_for_aug27.append(1)
            if len(calls_for_aug27) == 1:
                # Fetched while Aug 27 was still "today" -- misses the
                # last 30s of spend before midnight.
                return [{"model": "gpt-4", "netAmount": 10.0}]
            # Fetched again after the date rolled over -- the day is now
            # over and this is the true, final total.
            return [{"model": "gpt-4", "netAmount": 15.0}]
        return []

    monkeypatch.setattr(cb, "_fetch_day", _fake_fetch_day)

    # First call: Aug 27, 23:59:30 UTC -- Aug 27 is "today", gets cached
    # with settled=False.
    _set_now(monkeypatch, datetime(2026, 8, 27, 23, 59, 30, tzinfo=timezone.utc))
    result1 = _run(cb.get_daily_credits(days=1))
    assert result1["available"] is True
    assert calls_for_aug27 == [1]
    assert cb._day_cache[aug27]["settled"] is False
    aug27_rows_1 = [r for r in result1["daily"] if r["date"] == "2026-08-27"]
    assert aug27_rows_1[0]["credits"] == 10.0

    # Second call: Aug 28, 00:00:05 UTC -- Aug 27 has rolled out of the
    # window's "today" slot. Pre-fix, `days=1`'s window no longer even
    # contains Aug 27, so widen to days=2 to keep it in range and prove
    # the settle re-fetch happens instead of serving the stale cached 10.0.
    _set_now(monkeypatch, datetime(2026, 8, 28, 0, 0, 5, tzinfo=timezone.utc))
    result2 = _run(cb.get_daily_credits(days=2))
    assert calls_for_aug27 == [1, 1]  # one more fetch happened for aug27
    assert cb._day_cache[aug27]["settled"] is True
    aug27_rows_2 = [r for r in result2["daily"] if r["date"] == "2026-08-27"]
    assert aug27_rows_2[0]["credits"] == 15.0, (
        "late pre-midnight spend must be picked up by the one-time settle "
        "re-fetch, not lost to the stale is_today-only cache check"
    )

    # Third call, same day: no further re-fetch for the now-settled Aug 27.
    result3 = _run(cb.get_daily_credits(days=2))
    assert calls_for_aug27 == [1, 1]
    aug27_rows_3 = [r for r in result3["daily"] if r["date"] == "2026-08-27"]
    assert aug27_rows_3[0]["credits"] == 15.0


def test_day_first_seen_already_in_the_past_settles_immediately(monkeypatch):
    """A day that was never observed as "today" this process needs only
    the one fetch -- it's immutable from the moment it's first cached."""
    past_day = date(2026, 8, 20)

    calls: list[date] = []

    async def _fake_fetch_day(client, pat, username, d):
        calls.append(d)
        return [{"model": "gpt-4", "netAmount": 5.0}]

    monkeypatch.setattr(cb, "_fetch_day", _fake_fetch_day)
    _set_now(monkeypatch, datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc))

    _run(cb.get_daily_credits(days=14))
    assert calls.count(past_day) == 1
    assert cb._day_cache[past_day]["settled"] is True

    _run(cb.get_daily_credits(days=14))
    assert calls.count(past_day) == 1  # still cached forever, no re-fetch
