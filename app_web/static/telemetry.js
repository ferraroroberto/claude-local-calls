/* Telemetry tab — stack health, per-model leaderboard, live trace feed.
 *
 * Reuses the existing patterns:
 *   - jsonApi / postJson / eventStream from api.js
 *   - els + state from state.js
 *   - .card, .counters, .card-list dense, .hub-live-status from styles.css
 *
 * SSE is only opened while the tab is visible — onTabChange in main.js
 * calls startTelemetryStream / stopTelemetryStream.
 */

import { els, state } from './state.js';
import { jsonApi, postJson, eventStream, toast, escapeHtml, fmtClock, fmtTok, fmtCost, tokPair, renderCounterTable, modelLabel, modelLabelText } from './api.js';
import { icon } from './_vendored/icons/icons.js';

const HEALTH_POLL_MS = 8000;
const METRICS_POLL_MS = 5000;
const CC_POLL_MS = 10000;
const TRACE_RING_MAX = 50;

let healthPollHandle = null;
let metricsPollHandle = null;
let ccPollHandle = null;

// --------------------------------------------------------- health
export async function fetchTelemetryHealth() {
  try {
    const body = await jsonApi('/admin/api/telemetry/health');
    state.telHealth = body;
    renderHealth(body);
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    setHealthDot('danger', 'unreachable');
  }
}

function renderHealth(body) {
  if (!body) return;
  if (!body.otel_enabled) {
    setHealthDot('warn', 'OTel disabled');
  } else if (body.langfuse_reachable) {
    setHealthDot('good', 'live');
  } else {
    setHealthDot('danger', 'Langfuse offline');
  }
  if (els.telOtelState) els.telOtelState.textContent = body.otel_enabled ? 'on' : 'off';
  if (els.telHashMode) els.telHashMode.textContent = body.hash_prompts ? 'hashed' : 'raw';
  if (els.telEndpoint) els.telEndpoint.textContent = body.otel_endpoint || '—';
  // Point the header 🔗 button at the client-reachable Langfuse base, not
  // the hub's internal langfuse_host (which is always localhost-ish and
  // would fail from mobile/Tailscale). Same derivation as the per-row
  // trace deep-links.
  if (els.telOpenLangfuse) {
    els.telOpenLangfuse.href = langfuseUiBase();
  }
  if (els.telOfflineHint) {
    els.telOfflineHint.hidden = !!body.langfuse_reachable || !body.otel_enabled;
  }
}

function setHealthDot(kind, text) {
  if (!els.telHealth) return;
  els.telHealth.classList.remove('good', 'warn', 'danger');
  if (kind) els.telHealth.classList.add(kind);
  if (els.telHealthText) els.telHealthText.textContent = text;
}

// --------------------------------------------------------- per-model leaderboard
export async function fetchTelemetryMetrics() {
  try {
    const body = await jsonApi('/admin/api/telemetry/metrics');
    state.telCounters = body.counters || [];
    renderTelCounters();
    renderSummary(body.summary || {});
  } catch (_) { /* ignore */ }
}

function renderTelCounters() {
  renderCounterTable(els.telCountersTable, state.telCounters);
}

function renderSummary(s) {
  if (!els.telSummary) return;
  const reqs = s.requests || 0;
  const errs = s.errors || 0;
  const upS = Math.round(s.since_uptime_s || 0);
  const errRate = reqs ? ((errs / reqs) * 100).toFixed(1) : '0.0';
  let uptime = upS + 's';
  if (upS >= 60) uptime = Math.floor(upS / 60) + 'm ' + (upS % 60) + 's';
  if (upS >= 3600) uptime = Math.floor(upS / 3600) + 'h ' + Math.floor((upS % 3600) / 60) + 'm';
  els.telSummary.textContent = reqs + ' req · ' + errRate + '% err · since ' + uptime;
}

// --------------------------------------------------------- Claude Code (host CLI) panel — issue #68
export async function fetchClaudeCodeUsage() {
  try {
    const body = await jsonApi('/admin/api/telemetry/claude-code/usage?period=' + state.telCcPeriod);
    state.telCcSummary = body;
    renderClaudeCodeUsage(body);
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
  }
}

function renderClaudeCodeUsage(body) {
  const tbl = els.telCcTable;
  const empty = els.telCcEmpty;
  if (!tbl) return;
  const tbody = tbl.querySelector('tbody');
  if (!tbody) return;
  const rows = (body && body.rows) || [];
  tbody.innerHTML = '';
  if (empty) empty.hidden = rows.length > 0;
  rows.forEach(function (r) {
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td>' + escapeHtml(r.date || '—') + '</td>' +
      '<td>' + escapeHtml(r.project || '—') + '</td>' +
      '<td>' + escapeHtml(r.model) + '</td>' +
      '<td>' + escapeHtml(r.query_source) + '</td>' +
      '<td>' + fmtTok(r.input) + '</td>' +
      '<td>' + fmtTok(r.output) + '</td>' +
      '<td>' + fmtTok(r.cache_read) + '</td>' +
      '<td>' + fmtTok(r.cache_creation) + '</td>' +
      '<td>' + (fmtCost(r.cost_usd) || '—') + '</td>';
    tbody.appendChild(tr);
  });
  if (els.telCcSummary) {
    const totals = (body && body.totals) || {};
    const reqTok = (totals.input || 0) + (totals.output || 0);
    els.telCcSummary.textContent = reqTok
      ? tokPair(totals.input, totals.output) + ' tok · ' + (fmtCost(totals.cost_usd) || '≈ $0.00')
      : 'no data yet';
  }
}

// --------------------------------------------------------- live trace feed
export function startTelemetryStream() {
  if (state.telStreamCtl) return;
  try {
    const es = eventStream('/admin/api/telemetry/stream', {
      message: function (data) { ingestTrace(data); },
      error: function () { /* EventSource auto-reconnects */ },
    });
    state.telStreamCtl = es;
    // Exposed on window so the e2e suite can probe the stream's
    // readyState (== 1 once OPEN) before firing test requests —
    // otherwise the request can race ahead of the subscription on
    // slow CI runners.
    try { window.__telStream = es; } catch (_) {}
  } catch (exc) {
    console.warn('telemetry stream failed:', exc);
  }
}

export function stopTelemetryStream() {
  if (state.telStreamCtl) {
    try { state.telStreamCtl.close(); } catch (_) {}
    state.telStreamCtl = null;
    try { window.__telStream = null; } catch (_) {}
  }
}

function ingestTrace(rec) {
  if (!rec || typeof rec !== 'object') return;
  const traces = state.telTraces;
  // Dedup by ts — the SSE seed re-sends the same 20 records on reconnect.
  const exists = traces.some(function (t) { return t.ts === rec.ts && t.model === rec.model; });
  if (exists) return;
  traces.unshift(rec);
  if (traces.length > TRACE_RING_MAX) traces.length = TRACE_RING_MAX;
  renderTraces();
}

function renderTraces() {
  const list = els.telTracesList;
  const empty = els.telTracesEmpty;
  const badge = els.telTracesBadge;
  if (!list) return;
  const traces = state.telTraces;
  if (badge) badge.textContent = String(traces.length);
  if (empty) empty.hidden = traces.length > 0;
  list.innerHTML = '';
  traces.forEach(function (rec) {
    const li = document.createElement('li');
    li.className = 'tel-trace-row';
    if (rec.trace_id) li.classList.add('clickable');
    li.dataset.traceId = rec.trace_id || '';
    const statusCls = rec.status >= 500 ? 'err' : (rec.status >= 400 ? 'warn' : 'ok');
    const tsStr = fmtClock(rec.ts) || '—';
    const latency = (rec.latency_ms || 0).toFixed(0) + 'ms';
    // "requested → served @host" when the hub resolved one to the other; the
    // tooltip carries the same string unclipped for a narrow viewport (#412).
    const modelHtml = modelLabel(rec, '—');
    const modelTitle = modelLabelText(rec, '—');
    const tid = (rec.trace_id || '').slice(0, 8) || '—';
    const expanded = state.telExpandedTraceId && state.telExpandedTraceId === rec.trace_id;
    if (expanded) li.classList.add('expanded');

    li.innerHTML =
      '<span class="req-time">' + tsStr + '</span>' +
      '<span class="req-model" title="' + escapeAttr(modelTitle) + '">' + modelHtml + '</span>' +
      '<span class="req-latency">' + latency + '</span>' +
      '<span class="req-status ' + statusCls + '">' + rec.status + '</span>';

    const meta = document.createElement('div');
    meta.className = 'tel-trace-meta';
    meta.innerHTML =
      '<span class="muted small">trace</span> <code>' + escapeHtml(tid) + '</code>' +
      (rec.backend ? ' <span class="muted small">·</span> ' + escapeHtml(rec.backend) : '') +
      (rec.in_tok || rec.out_tok ? ' <span class="muted small">·</span> in ' + rec.in_tok + ' / out ' + rec.out_tok : '') +
      (rec.trace_id ? ' <span class="muted small">·</span> <span class="tel-expand-hint">' + (expanded ? icon('chevron-down') + ' tap to collapse' : icon('chevron-right') + ' tap for detail') + '</span>' : '');
    li.appendChild(meta);

    if (rec.trace_id) {
      const actions = document.createElement('div');
      actions.className = 'tel-trace-actions';
      actions.innerHTML =
        '<button type="button" class="ghost-btn tel-thumb" data-thumbs="1" aria-label="Thumbs up">' + icon('thumbs-up') + '</button>' +
        '<button type="button" class="ghost-btn tel-thumb" data-thumbs="-1" aria-label="Thumbs down">' + icon('thumbs-down') + '</button>' +
        '<a class="ghost-btn tel-deeplink" target="_blank" rel="noopener" href="' + langfuseTraceUrl(rec.trace_id) + '">' + icon('external-link') + 'Langfuse</a>';
      li.appendChild(actions);
    }

    if (rec.error_detail) {
      const err = document.createElement('div');
      err.className = 'tel-trace-err muted small';
      err.textContent = rec.error_detail.slice(0, 280);
      li.appendChild(err);
    }

    if (expanded) {
      const detail = document.createElement('div');
      detail.className = 'tel-trace-detail';
      detail.innerHTML = '<span class="muted small">loading…</span>';
      li.appendChild(detail);
      // Fetch the detail payload asynchronously and patch the panel.
      hydrateDetailPanel(detail, rec.trace_id);
    }

    list.appendChild(li);
  });
}

async function hydrateDetailPanel(target, traceId) {
  let body;
  try {
    body = await jsonApi('/admin/api/telemetry/trace/' + encodeURIComponent(traceId));
  } catch (exc) {
    target.innerHTML = '<span class="muted small">detail fetch failed: '
      + escapeHtml(exc.message || 'unknown') + '</span>';
    return;
  }
  const lf = body.langfuse || {};
  const parts = [];
  parts.push('<div class="tel-detail-row"><span class="muted small">trace_id</span><code>'
    + escapeHtml(body.trace_id) + '</code></div>');

  if (lf.available) {
    parts.push('<div class="tel-detail-section"><div class="muted small">Prompt</div>'
      + '<pre class="tel-detail-pre">' + escapeHtml(stringifyPayload(lf.input)) + '</pre></div>');
    parts.push('<div class="tel-detail-section"><div class="muted small">Completion</div>'
      + '<pre class="tel-detail-pre">' + escapeHtml(stringifyPayload(lf.output)) + '</pre></div>');
  } else {
    const msg = lf.fetch_error || 'Langfuse stack offline — start it for full prompt/completion bodies.';
    parts.push('<div class="tel-detail-section muted small">' + escapeHtml(msg) + '</div>');
  }
  target.innerHTML = parts.join('');
}

function stringifyPayload(value) {
  if (value == null) return '(empty)';
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value, null, 2); }
  catch (_) { return String(value); }
}

function langfuseUiBase() {
  // Pick the host the user's browser can actually reach Langfuse at.
  //
  //   1. If an explicit LANGFUSE_PUBLIC_URL is configured, use it
  //      verbatim. Right for Cloudflare-tunnel setups where the
  //      Langfuse UI lives on its own tunneled hostname.
  //   2. Otherwise reuse the hostname the SPA itself was loaded from
  //      and swap the port for Langfuse's. Works automatically across
  //      localhost / LAN / Tailscale because the hub and Langfuse run
  //      on the same machine — the hostname the client used to reach
  //      :8000 also reaches :3000.
  const h = state.telHealth || {};
  if (h.langfuse_public_url) return h.langfuse_public_url;
  const port = h.langfuse_port || 3000;
  const loc = window.location;
  return loc.protocol + '//' + loc.hostname + ':' + port;
}

export function langfuseTraceUrl(traceId) {
  const h = state.telHealth || {};
  const base = langfuseUiBase();
  const pid = h.langfuse_project_id || '';
  // Langfuse v3 path includes the project_id. When we haven't resolved
  // it yet (offline, missing keys, first probe pending) fall back to
  // the host root — Langfuse routes to the right project after login.
  if (pid) {
    return base + '/project/' + encodeURIComponent(pid) +
      '/traces/' + encodeURIComponent(traceId);
  }
  return base + '/trace/' + encodeURIComponent(traceId);
}

// --------------------------------------------------------- feedback
async function postFeedback(traceId, value) {
  try {
    await postJson('/admin/api/trace/' + encodeURIComponent(traceId) + '/feedback', {
      thumbs: value,
    });
    toast(value > 0 ? 'Thumbs up sent' : (value < 0 ? 'Thumbs down sent' : 'noted'), 'good');
  } catch (exc) {
    toast('feedback failed: ' + (exc.message || 'unknown'), 'error');
  }
}

// --------------------------------------------------------- helpers
/* escapeHtml / fmtClock live in api.js (sibling dedup, #211). */
function escapeAttr(s) { return escapeHtml(s); }

// --------------------------------------------------------- lifecycle
export function wireTelemetry() {
  if (els.telCcPeriodSeg) {
    els.telCcPeriodSeg.addEventListener('click', function (e) {
      const btn = e.target.closest('button[data-period]');
      if (!btn) return;
      const next = btn.dataset.period;
      if (next === state.telCcPeriod) return;
      state.telCcPeriod = next;
      els.telCcPeriodSeg.querySelectorAll('button').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      fetchClaudeCodeUsage().catch(function () {});
    });
  }
  if (!els.telTracesList) return;
  // Delegated click handler — thumbs buttons + Langfuse deep-link short
  // circuit; everything else on a row toggles the detail panel.
  els.telTracesList.addEventListener('click', function (ev) {
    const btn = ev.target.closest('.tel-thumb');
    if (btn) {
      const row = btn.closest('.tel-trace-row');
      const traceId = row && row.dataset ? row.dataset.traceId : '';
      if (!traceId) return;
      const value = parseInt(btn.dataset.thumbs || '0', 10);
      postFeedback(traceId, value);
      ev.stopPropagation();
      return;
    }
    // Let the Langfuse anchor handle its own navigation.
    if (ev.target.closest('a.tel-deeplink')) return;

    const row = ev.target.closest('.tel-trace-row');
    if (!row || !row.dataset || !row.dataset.traceId) return;
    const tid = row.dataset.traceId;
    state.telExpandedTraceId = (state.telExpandedTraceId === tid) ? '' : tid;
    renderTraces();
  });
}

export function startTelemetryPolls() {
  if (healthPollHandle) return;
  fetchTelemetryHealth();
  fetchTelemetryMetrics();
  fetchClaudeCodeUsage();
  startTelemetryStream();
  healthPollHandle = setInterval(function () {
    fetchTelemetryHealth().catch(function () {});
  }, HEALTH_POLL_MS);
  metricsPollHandle = setInterval(function () {
    fetchTelemetryMetrics().catch(function () {});
  }, METRICS_POLL_MS);
  ccPollHandle = setInterval(function () {
    fetchClaudeCodeUsage().catch(function () {});
  }, CC_POLL_MS);
}

export function stopTelemetryPolls() {
  if (healthPollHandle) { clearInterval(healthPollHandle); healthPollHandle = null; }
  if (metricsPollHandle) { clearInterval(metricsPollHandle); metricsPollHandle = null; }
  if (ccPollHandle) { clearInterval(ccPollHandle); ccPollHandle = null; }
  stopTelemetryStream();
}
