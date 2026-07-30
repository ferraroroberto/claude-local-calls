"""Per-vendor $/Mtok pricing tables and per-record cost estimation.

Extracted out of ``code_usage.py`` (#451) — three independent pricing-table
loaders (Claude/OpenAI/Gemini) plus the cross-vendor cost dispatcher
(:func:`record_costs`) were one of five unrelated concerns that module mixed
together. ``code_usage.py`` imports :func:`record_costs` for its
``get_summary`` aggregation; nothing else here depends back on it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from src.usage_common import UsageRecord, model_display

_log = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Anthropic API list prices (USD per million tokens), keyed by model family.
# Loaded once from config/claude_pricing.json; this dict is the fallback used
# when that file is missing or unreadable, so cost display degrades gracefully.
_PRICING_PATH: Path = _PROJECT_ROOT / "config" / "claude_pricing.json"
_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "Fable":  {"input": 10.0, "output": 50.0, "cache_write": 12.50, "cache_read": 1.00},
    "Opus":   {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},
    "Sonnet": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
    "Haiku":  {"input": 1.0, "output": 5.0,  "cache_write": 1.25, "cache_read": 0.10},
}

# OpenAI API list prices (USD per million tokens), keyed by display model id
# (what model_display returns for a Codex model, e.g. "GPT-5.5").  Used to
# show the equivalent metered-API cost of host-side Codex usage.  Loaded once
# from config/openai_pricing.json; this dict is the fallback when that file is
# missing or unreadable.  Codex's cached_input tokens are a *subset* of input
# (not additive), so the cost path prices the non-cached remainder at "input"
# and the cached portion at "cached_input".
_OPENAI_PRICING_PATH: Path = _PROJECT_ROOT / "config" / "openai_pricing.json"
_OPENAI_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "GPT-5.6-SOL": {"input": 5.0,  "cached_input": 0.50, "output": 30.0},
    "GPT-5.5":     {"input": 5.0,  "cached_input": 0.50, "output": 30.0},
    "GPT-5.5 Pro": {"input": 30.0, "cached_input": 0.0,  "output": 180.0},
    "GPT-5.4":     {"input": 2.5,  "cached_input": 0.25, "output": 15.0},
}

# Gemini API list prices in USD per million tokens, keyed by family (what
# gemini_family collapses an AGY model name to).  Used to show the
# equivalent metered-API cost of AGY/Antigravity usage (#280) — an estimate
# against Google list prices, same idea as Codex vs OpenAI.  AGY cache reads
# are reported separately/additively (Claude-style), priced at "cache_read".
_GEMINI_PRICING_PATH: Path = _PROJECT_ROOT / "config" / "gemini_pricing.json"
_GEMINI_PRICING_FALLBACK: Dict[str, Dict[str, float]] = {
    "pro":        {"input": 2.0,  "output": 12.0, "cache_read": 0.20},
    "flash":      {"input": 0.30, "output": 2.50, "cache_read": 0.03},
    "flash-lite": {"input": 0.10, "output": 0.40, "cache_read": 0.01},
}

# Memoized price tables, keyed by cache slot ("claude" / "openai" / "gemini");
# populated lazily by ``_load_priced_table``.
_priced_table_cache: Dict[str, Dict[str, Dict[str, float]]] = {}


def _load_priced_table(
    path: Path,
    top_key: str,
    fallback: Dict[str, Dict[str, float]],
    cache_slot: str,
) -> Dict[str, Dict[str, float]]:
    """Return a per-family/per-model price table, loaded once from config and cached.

    Shared loader for the Claude/OpenAI/Gemini pricing tables (each used to
    carry its own near-identical ~35-line copy differing only in the config
    path, the JSON top-level key, and which cache slot to memoize into).
    Falls back to ``fallback`` when ``path`` is missing or malformed, so the
    cost display never hard-fails on a fresh checkout.
    """
    cached = _priced_table_cache.get(cache_slot)
    if cached is not None:
        return cached

    pricing: Dict[str, Dict[str, float]] = dict(fallback)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        table = raw.get(top_key) or {}
        if isinstance(table, dict) and table:
            pricing = {
                name: {k: float(v) for k, v in rates.items()}
                for name, rates in table.items()
                if isinstance(rates, dict)
            }
    except (OSError, ValueError, TypeError) as exc:
        _log.warning(
            "⚠️ usage_pricing: using fallback pricing (%s unreadable): %s",
            path, exc,
        )
    _priced_table_cache[cache_slot] = pricing
    return pricing


def load_pricing() -> Dict[str, Dict[str, float]]:
    """Return the per-family Claude price table, loaded once from config and cached."""
    return _load_priced_table(_PRICING_PATH, "families", _PRICING_FALLBACK, "claude")


def load_openai_pricing() -> Dict[str, Dict[str, float]]:
    """Return the per-model OpenAI price table, loaded once and cached."""
    return _load_priced_table(_OPENAI_PRICING_PATH, "models", _OPENAI_PRICING_FALLBACK, "openai")


def gemini_family(model: str) -> str:
    """Collapse an AGY model name to a Gemini pricing family.

    AgentsView surfaces both raw ids (``gemini-3.1-pro-preview``) and
    display names (``Gemini 3.1 Pro (High)``) — family matching by
    substring covers both.  Unknown names return "" (prices at zero, no
    fabricated cost).
    """
    m = model.lower().replace(" ", "-")
    if "gemini" not in m:
        return ""
    if "flash-lite" in m:
        return "flash-lite"
    if "flash" in m:
        return "flash"
    if "pro" in m:
        return "pro"
    return ""


def load_gemini_pricing() -> Dict[str, Dict[str, float]]:
    """Return the per-family Gemini price table, loaded once and cached."""
    return _load_priced_table(_GEMINI_PRICING_PATH, "families", _GEMINI_PRICING_FALLBACK, "gemini")


def record_costs(r: "UsageRecord") -> Tuple[float, float, float]:
    """Return ``(input_cost, output_cost, cache_read_cost)`` in USD for one record.

    Priced against the record's own model, so a mixed-model / mixed-vendor
    period is summed correctly.  Unknown models price at zero (no fabricated
    cost).  The cost maps into the same three tiles the SPA shows.

    Claude: input-tile cost folds in cache-creation tokens (5-min cache-write
    rate) to mirror the tile (``input + cache_creation``).

    Codex: ``cached_input`` tokens are a *subset* of ``input_tokens``, so the
    non-cached remainder is priced at the input rate and the cached portion at
    the (cheaper) cached_input rate — no double counting.  Reasoning tokens are
    already inside ``output_tokens`` and bill at the output rate.  The >272K
    long-context surcharge (2x input / 1.5x output) is not modelled — this is an
    estimate, and per-request context size isn't tracked.

    AGY (#280): priced per tile against Gemini API list prices
    (``config/gemini_pricing.json``), same estimate-vs-list-prices idea as
    Codex — AgentsView's own ``cost_usd`` is unreliable here (it can't price
    the display-name model ids ``antigravity-cli`` sessions carry).  Cache
    reads are reported separately/additively (Claude-style), priced at the
    implicit-caching rate.

    Copilot and any other AgentsView-sourced vendor: ``credits_usd`` carries
    the record's own dollar figure — Copilot's *exact* billed AI Credits
    (#231) — returned directly as the input-cost slot so it still nets into
    the same three-tile total the SPA sums, never re-priced against a hub
    rate table.  This must be checked before the Claude family-substring
    fallback below, or a Copilot call that resolved to e.g.
    ``claude-sonnet-4.5`` would get silently re-priced at the Anthropic
    subscription rate instead of its own credit charge.
    """
    if r.vendor == "agy":
        rates = load_gemini_pricing().get(gemini_family(r.model))
        if not rates:
            return 0.0, 0.0, 0.0
        input_cost = r.input_tokens * rates.get("input", 0.0) / 1_000_000
        output_cost = r.output_tokens * rates.get("output", 0.0) / 1_000_000
        cache_read_cost = (
            r.cache_read_tokens * rates.get("cache_read", 0.0) / 1_000_000
        )
        return input_cost, output_cost, cache_read_cost

    if r.vendor not in ("claude", "codex"):
        return r.credits_usd, 0.0, 0.0

    if r.vendor == "codex":
        rates = load_openai_pricing().get(model_display(r.model))
        if not rates:
            return 0.0, 0.0, 0.0
        non_cached_input = max(r.input_tokens - r.cache_read_tokens, 0)
        input_cost = non_cached_input * rates.get("input", 0.0) / 1_000_000
        output_cost = r.output_tokens * rates.get("output", 0.0) / 1_000_000
        cache_read_cost = (
            r.cache_read_tokens * rates.get("cached_input", 0.0) / 1_000_000
        )
        return input_cost, output_cost, cache_read_cost

    rates = load_pricing().get(model_display(r.model))
    if not rates:
        return 0.0, 0.0, 0.0
    input_cost = (
        r.input_tokens * rates.get("input", 0.0)
        + r.cache_creation_tokens * rates.get("cache_write", 0.0)
    ) / 1_000_000
    output_cost = r.output_tokens * rates.get("output", 0.0) / 1_000_000
    cache_read_cost = r.cache_read_tokens * rates.get("cache_read", 0.0) / 1_000_000
    return input_cost, output_cost, cache_read_cost
