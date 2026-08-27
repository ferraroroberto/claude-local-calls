/* Claude Code usage tab (issues #20, #50).
 *
 * Polls /admin/api/code/usage/summary?period=<period> every 30 s while
 * the tab is visible.  Data comes from ~/.claude/projects/**\/*.jsonl —
 * no subprocess, no proxy.
 *
 * Reuses: jsonApi from api.js, els + state from state.js,
 *         .card, .counters, .card-list.dense, .badge, .empty from styles.css
 *
 * Charts (issue #50): requires Chart.js 4.x loaded globally via CDN before
 * this module executes (see the <script> tag just before main.js in index.html).
 */

import { els, state } from './state.js';
import { jsonApi, fmtTok, fmtCost, tokPair, escapeHtml, renderTable } from './api.js';

// ---------------------------------------------------------------------------
// Chart constants (issue #50)
// ---------------------------------------------------------------------------

// Fixed colours for the Claude model families. Hardcoded (not from CSS vars)
// because the Chart.js canvas cannot read CSS custom properties. Fills are the
// energy-tab's translucent 0.18 alpha over a solid 2px line (#215).
const CLAUDE_COLORS = {
  Haiku:  { bg: 'rgba(74,138,243,0.18)',  border: 'rgba(74,138,243,0.85)'  },
  Sonnet: { bg: 'rgba(76,175,80,0.18)',   border: 'rgba(76,175,80,0.85)'   },
  Opus:   { bg: 'rgba(240,161,0,0.18)',   border: 'rgba(240,161,0,0.85)'   },
  Fable:  { bg: 'rgba(63,81,181,0.18)',   border: 'rgba(63,81,181,0.85)'   },
};
const CLAUDE_ORDER = ['Haiku', 'Sonnet', 'Opus', 'Fable'];
// Extra colours assigned in first-seen order to non-Claude families (Codex
// GPT models, etc.) so each gets its own series instead of collapsing to grey.
const EXTRA_COLORS = [
  { bg: 'rgba(186,104,200,0.18)', border: 'rgba(186,104,200,0.85)' }, // purple
  { bg: 'rgba(0,188,212,0.18)',   border: 'rgba(0,188,212,0.85)'   }, // cyan
  { bg: 'rgba(233,30,99,0.18)',   border: 'rgba(233,30,99,0.85)'   }, // pink
  { bg: 'rgba(121,85,72,0.18)',   border: 'rgba(121,85,72,0.85)'   }, // brown
  { bg: 'rgba(255,112,67,0.18)',  border: 'rgba(255,112,67,0.85)'  }, // deep orange
];
// Final fallback for an absent/unattributable model ("unknown").
const OTHER_COLOR = { bg: 'rgba(154,154,154,0.14)', border: 'rgba(154,154,154,0.65)' };

// Retained Chart.js instances — destroyed before each recreation to prevent leaks.
let _chartInput  = null;
let _chartOutput = null;
let _chartReqs   = null;
let _chartCache  = null;

const POLL_MS = 30_000;
let _pollHandle = null;

// ---------------------------------------------------------------------------
// Public lifecycle — called from main.js
// ---------------------------------------------------------------------------

export function wireCodeUsage() {
  // Period toggle (Day / Week / Month / All)
  if (els.cldPeriodSeg) {
    els.cldPeriodSeg.addEventListener('click', function (e) {
      const btn = e.target.closest('button[data-period]');
      if (!btn) return;
      const next = btn.dataset.period;
      if (next === state.cldPeriod) return;     // nothing to do
      state.cldPeriod = next;
      els.cldPeriodSeg.querySelectorAll('button').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      // Immediate re-fetch so the switch feels instant.
      fetchSummary().catch(function () {});
    });
  }

  // Vendor selector (All / Claude / Codex) — issue #71.
  if (els.cldVendorSeg) {
    els.cldVendorSeg.addEventListener('click', function (e) {
      const btn = e.target.closest('button[data-vendor]');
      if (!btn) return;
      const next = btn.dataset.vendor;
      if (next === state.cldVendor) return;
      state.cldVendor = next;
      els.cldVendorSeg.querySelectorAll('button').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      fetchSummary().catch(function () {});
    });
  }
}

export function startCodeUsagePolls() {
  fetchSummary().catch(function () {});
  _pollHandle = setInterval(function () {
    fetchSummary().catch(function () {});
  }, POLL_MS);
}

export function stopCodeUsagePolls() {
  if (_pollHandle !== null) {
    clearInterval(_pollHandle);
    _pollHandle = null;
  }
}

// ---------------------------------------------------------------------------
// Data fetch
// ---------------------------------------------------------------------------

async function fetchSummary() {
  try {
    const body = await jsonApi(
      '/admin/api/code/usage/summary?period=' + state.cldPeriod +
      '&vendor=' + state.cldVendor
    );
    state.cldSummary = body;
    syncAgentsviewVendors(body.agentsview);
    // If the server coerced an unknown vendor back to "all" (e.g. the hub
    // restarted and hasn't re-discovered an AgentsView vendor yet), resync
    // the selector so the UI and the data agree.
    if (body.vendor && body.vendor !== state.cldVendor) {
      state.cldVendor = body.vendor;
      if (els.cldVendorSeg) {
        els.cldVendorSeg.querySelectorAll('button').forEach(function (b) {
          b.classList.toggle('active', b.dataset.vendor === body.vendor);
        });
      }
    }
    render(body);
    // Copilot billing card is a separate authoritative source (issue #231,
    // part B) — only fetched while that vendor tab is actually selected, so
    // the GitHub API isn't hit on every 30s poll regardless of tab.
    if (state.cldVendor === 'copilot') {
      fetchCopilotBilling().catch(function () {});
    } else if (els.cldCopilotBillingCard) {
      els.cldCopilotBillingCard.hidden = true;
    }
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    setFreshness('error fetching data');
  }
}

// ---------------------------------------------------------------------------
// Copilot official billing card (issue #231, part B)
// ---------------------------------------------------------------------------

async function fetchCopilotBilling() {
  if (!els.cldCopilotBillingCard) return;
  try {
    const body = await jsonApi('/admin/api/code/copilot/billing');
    renderCopilotBilling(body);
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    renderCopilotBilling({ available: false, reason: 'error fetching data', daily: [] });
  }
}

function renderCopilotBilling(body) {
  const card = els.cldCopilotBillingCard;
  if (!card) return;
  card.hidden = state.cldVendor !== 'copilot';
  if (state.cldVendor !== 'copilot' || !body) return;

  if (els.cldCopilotBillingAsOf) {
    els.cldCopilotBillingAsOf.textContent = body.as_of
      ? 'as of ' + new Date(body.as_of).toLocaleTimeString() : '';
  }

  // `available: false` forces the empty treatment regardless of `daily` —
  // there's no billing data worth showing, just a reason.
  const rows = (body.available && body.daily) || [];
  if (els.cldCopilotBillingEmptyMsg) {
    els.cldCopilotBillingEmptyMsg.textContent = body.available === false
      ? (body.reason || 'Not available.')
      : 'No billing data for this window.';
  }
  const totalCredits = rows.reduce(function (sum, r) { return sum + (Number(r.credits) || 0); }, 0);
  set(els.cldCopilotBillingTotal, rows.length ? totalCredits.toFixed(2) : '—');

  renderTable(els.cldCopilotBillingTable, els.cldCopilotBillingEmpty, rows, function (r) {
    return '<tr>' +
      '<td>' + escapeHtml(r.date) + '</td>' +
      '<td>' + escapeHtml(r.model) + '</td>' +
      '<td>' + (Number(r.credits) || 0).toFixed(2) + '</td>' +
      '<td>' + (fmtCost(r.usd) || '—') + '</td>' +
      '</tr>';
  });
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function render(body) {
  if (!body) return;
  renderCldCounters(body);
  renderDeltas(body);
  renderCharts(body);
  renderVendorTable(body.by_vendor || []);
  renderModelTable(body.by_model || []);
  renderProjectTable(body.by_project || []);
  renderSessions(body.recent_sessions || []);
  setFreshness('updated ' + new Date().toLocaleTimeString());
}

function renderCldCounters(body) {
  if (!body) return;
  const bucket = body.totals || {};
  set(els.cldRequests, fmtNum(bucket.requests));
  // Grand-total equivalent metered cost under the requests tile (issue #71).
  const totalCost = (bucket.input_cost || 0) + (bucket.output_cost || 0) + (bucket.cache_read_cost || 0);
  set(els.cldTotalCost, fmtCost(totalCost));
  set(els.cldInputTok, fmtTok(
    (bucket.input_tokens || 0) + (bucket.cache_creation_tokens || 0)
  ));
  set(els.cldOutputTok, fmtTok(bucket.output_tokens));
  set(els.cldCacheRead, fmtTok(bucket.cache_read_tokens));
  // Equivalent metered-API cost under the three token tiles (issue #52, #71).
  set(els.cldInputCost, fmtCost(bucket.input_cost));
  set(els.cldOutputCost, fmtCost(bucket.output_cost));
  set(els.cldCacheCost, fmtCost(bucket.cache_read_cost));
  // Codex reasoning tokens — a subset of output, shown for transparency (#71).
  const reasoning = bucket.reasoning_output_tokens || 0;
  set(els.cldOutputReasoning, reasoning ? 'incl. ' + fmtTok(reasoning) + ' reasoning' : '');
}

function renderVendorTable(rows) {
  // Per-vendor card only makes sense when viewing every vendor at once.
  if (els.cldVendorCard) els.cldVendorCard.hidden = state.cldVendor !== 'all';
  renderTable(els.cldVendorTable, els.cldVendorEmpty, rows, function (r) {
    const totalIn = (r.input_tokens || 0) + (r.cache_creation_tokens || 0);
    const cost = (r.input_cost || 0) + (r.output_cost || 0) + (r.cache_read_cost || 0);
    return '<tr>' +
      '<td>' + escapeHtml(vendorLabel(r.vendor)) + '</td>' +
      '<td>' + fmtNum(r.requests) + '</td>' +
      '<td>' + tokPair(totalIn, r.output_tokens) + '</td>' +
      '<td class="muted">' + fmtTok(r.cache_read_tokens) + '</td>' +
      '<td>' + (fmtCost(cost) || '—') + '</td>' +
      '</tr>';
  });
}

function vendorLabel(v) {
  if (v === 'claude') return 'Claude';
  if (v === 'codex') return 'Codex';
  if (v === 'copilot') return 'Copilot';
  if (v === 'agy') return 'AGY';
  // Any other AgentsView-sourced vendor (issue #280): auto-label from the slug.
  return v ? v.charAt(0).toUpperCase() + v.slice(1) : '—';
}

// ---------------------------------------------------------------------------
// AgentsView gap-fill vendors (issue #280)
// ---------------------------------------------------------------------------

/* Inject a selector button per AgentsView-discovered vendor (additive only —
 * the server keeps vendors sticky across outages, so an active button never
 * vanishes mid-session). Click handling is delegated on the container in
 * wireCodeUsage(), so injected buttons need no listeners. The offline hint
 * only shows when vendors are known AND the service is down — a machine that
 * never ran AgentsView stays silent. */
function syncAgentsviewVendors(av) {
  if (!av || !els.cldVendorSeg) return;
  (av.vendors || []).forEach(function (v) {
    if (els.cldVendorSeg.querySelector('button[data-vendor="' + v + '"]')) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.dataset.vendor = v;
    btn.textContent = vendorLabel(v);
    els.cldVendorSeg.appendChild(btn);
  });
  if (els.cldAgentsviewHint) {
    els.cldAgentsviewHint.hidden =
      !(av.vendors && av.vendors.length && !av.reachable);
  }
}

function renderModelTable(rows) {
  renderTable(els.cldModelTable, els.cldModelEmpty, rows, function (r) {
    const totalIn = (r.input_tokens || 0) + (r.cache_creation_tokens || 0);
    return '<tr>' +
      '<td class="td-trunc" title="' + escapeHtml(r.model) + '">' + escapeHtml(r.model) + '</td>' +
      '<td>' + fmtNum(r.requests) + '</td>' +
      '<td>' + tokPair(totalIn, r.output_tokens) + '</td>' +
      '<td class="muted">' + fmtTok(r.cache_read_tokens) + '</td>' +
      '</tr>';
  });
}

function renderProjectTable(rows) {
  renderTable(els.cldProjectTable, els.cldProjectEmpty, rows, function (r) {
    const totalIn = (r.input_tokens || 0) + (r.cache_creation_tokens || 0);
    const name = r.project || r.project_key;
    return '<tr>' +
      '<td class="td-trunc" title="' + escapeHtml(name) + '">' + escapeHtml(name) + '</td>' +
      '<td>' + fmtNum(r.requests) + '</td>' +
      '<td>' + tokPair(totalIn, r.output_tokens) + '</td>' +
      '</tr>';
  });
}

function renderSessions(sessions) {
  if (!els.cldSessionsList) return;
  const badge = els.cldSessionsBadge;
  const empty = els.cldSessionsEmpty;

  if (badge) badge.textContent = sessions.length;
  if (!sessions.length) {
    els.cldSessionsList.innerHTML = '';
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;

  els.cldSessionsList.innerHTML = sessions.map(function (s) {
    const totalIn = (s.input_tokens || 0) + (s.cache_creation_tokens || 0);
    const firstTs = fmtTs(s.first_ts);
    const lastTs = fmtTs(s.last_ts);
    const proj = escapeHtml(s.project || s.project_key || '');
    const model = escapeHtml(modelShort(s.model || ''));
    const sid = escapeHtml((s.session_id || '').slice(0, 8));
    return '<li>' +
      '<span class="cld-sess-time">' + lastTs + '</span>' +
      '<span class="cld-sess-project">' + proj + '</span>' +
      '<span class="cld-sess-model">' + model + '</span>' +
      '<div class="cld-sess-meta">' +
        '<code>' + sid + '</code>' +
        ' · ' + firstTs + ' → ' + lastTs +
        ' · ' + fmtNum(s.requests) + ' req' +
        ' · ' + fmtTok(totalIn) + '↑ ' + fmtTok(s.output_tokens) + '↓' +
      '</div>' +
      '</li>';
  }).join('');
}

// ---------------------------------------------------------------------------
// Delta badges (issue #50)
// ---------------------------------------------------------------------------

function renderDeltas(body) {
  const prev = body.prev_totals;        // absent when period === 'all'
  const curr = body.totals || {};
  const hide = state.cldPeriod === 'all' || !prev;

  function apply(el, c, p) {
    if (!el) return;
    if (hide) { el.hidden = true; return; }
    // Zero base: percentage is undefined. Show "new" when this metric appeared
    // this period (e.g. Codex with no prior-week data) instead of hiding the
    // badge — the comparison window itself is valid (issue #71).
    if (p === 0) {
      if (c > 0) { el.hidden = false; el.className = 'cld-delta up'; el.textContent = 'new'; }
      else { el.hidden = true; }
      return;
    }
    el.hidden = false;
    const pct = Math.round((c - p) / p * 100);
    el.className = pct > 0 ? 'cld-delta up' : pct < 0 ? 'cld-delta down' : 'cld-delta';
    el.textContent = pct > 0 ? '+' + pct + '% ↑' : pct < 0 ? pct + '% ↓' : '±0%';
  }

  const prevIn  = prev ? (prev.input_tokens || 0) + (prev.cache_creation_tokens || 0) : 0;
  const currIn  = (curr.input_tokens || 0) + (curr.cache_creation_tokens || 0);
  const prevOut = prev ? (prev.output_tokens || 0) : 0;
  const prevReq = prev ? (prev.requests || 0) : 0;

  const prevCache = prev ? (prev.cache_read_tokens || 0) : 0;

  apply(els.cldDeltaRequests,  curr.requests || 0,           prevReq);
  apply(els.cldDeltaInputTok,  currIn,                        prevIn);
  apply(els.cldDeltaOutputTok, curr.output_tokens || 0,       prevOut);
  apply(els.cldDeltaCacheRead, curr.cache_read_tokens || 0,   prevCache);
}

// ---------------------------------------------------------------------------
// Trend charts (issue #50)
// ---------------------------------------------------------------------------

function renderCharts(body) {
  const card = els.cldChartsCard;
  if (!card) return;

  const ts = body.time_series;
  if (!ts || !ts.length || state.cldPeriod === 'all') {
    card.hidden = true;
    _destroyCharts();
    return;
  }

  card.hidden = false;
  const labels = ts.map(function (b) { return b.label; });
  // Stable family list + colours, shared across all four charts so a model
  // keeps the same colour everywhere (issue #71: GPT models get own series).
  const families = _orderedFamilies(ts);

  _chartInput  = _makeChart(els.cldChartInput,  _chartInput,  labels, ts, families, 'input_tokens',      true);
  _chartOutput = _makeChart(els.cldChartOutput, _chartOutput, labels, ts, families, 'output_tokens',     true);
  _chartReqs   = _makeChart(els.cldChartReqs,   _chartReqs,   labels, ts, families, 'requests',          false);
  _chartCache  = _makeChart(els.cldChartCache,  _chartCache,  labels, ts, families, 'cache_read_tokens', true);
}

// Build the ordered list of chart series from the time-series buckets: Claude
// families first (fixed colours), then every other model (e.g. Codex GPT-5.5)
// each with its own colour, and finally a grey "Other" for unattributable ids.
function _orderedFamilies(ts) {
  const seen = new Set();
  ts.forEach(function (b) {
    Object.keys(b.models || {}).forEach(function (k) { seen.add(k); });
  });

  const series = [];
  CLAUDE_ORDER.forEach(function (k) {
    if (seen.has(k)) series.push({ key: k, label: k, bg: CLAUDE_COLORS[k].bg, border: CLAUDE_COLORS[k].border });
  });

  const rest = Array.from(seen).filter(function (k) {
    return CLAUDE_ORDER.indexOf(k) === -1 && k !== 'unknown';
  }).sort();
  rest.forEach(function (k, i) {
    const c = EXTRA_COLORS[i % EXTRA_COLORS.length];
    series.push({ key: k, label: k, bg: c.bg, border: c.border });
  });

  if (seen.has('unknown')) {
    series.push({ key: 'unknown', label: 'Other', bg: OTHER_COLOR.bg, border: OTHER_COLOR.border });
  }
  return series;
}

function _destroyCharts() {
  if (_chartInput)  { _chartInput.destroy();  _chartInput  = null; }
  if (_chartOutput) { _chartOutput.destroy(); _chartOutput = null; }
  if (_chartReqs)   { _chartReqs.destroy();   _chartReqs   = null; }
  if (_chartCache)  { _chartCache.destroy();  _chartCache  = null; }
}

function _makeChart(canvas, existing, labels, ts, families, field, isTok) {
  if (!canvas) return existing;

  var datasets = [];
  families.forEach(function (fam) {
    var values = ts.map(function (b) { return ((b.models || {})[fam.key] || {})[field] || 0; });
    if (values.every(function (v) { return v === 0; })) return;
    datasets.push({
      label: fam.label,
      data: values,
      fill: true,
      backgroundColor: fam.bg,
      borderColor: fam.border,
      borderWidth: 2,
      tension: 0.3,
      pointRadius: ts.length > 10 ? 0 : 3,
      pointHoverRadius: 4,
    });
  });

  if (existing) { existing.destroy(); }

  /* global Chart */
  return new Chart(canvas, {
    type: 'line',
    data: { labels: labels, datasets: datasets },
    options: _chartOptions(isTok),
  });
}

/* Chart.js cannot read CSS custom properties from the canvas itself, so
 * resolve the theme tokens at (re)creation time — and again on theme flip
 * via restyleCodeUsageCharts(). */
function _cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function _chartOptions(isTok) {
  var gridColor  = _cssVar('--line');
  var tickColor  = _cssVar('--muted');
  var fgColor    = _cssVar('--ink');
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        stacked: true,
        grid: { display: false },
        ticks: { color: tickColor, font: { size: 11 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
      },
      y: {
        stacked: true,
        beginAtZero: true,
        grid: { color: gridColor },
        ticks: {
          color: tickColor,
          font: { size: 11 },
          callback: isTok ? function (v) { return fmtTok(v); } : undefined,
        },
      },
    },
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: fgColor, boxWidth: 12, padding: 12, font: { size: 11 }, usePointStyle: true },
      },
      tooltip: {
        callbacks: isTok ? {
          label: function (ctx) {
            return ' ' + ctx.dataset.label + ': ' + fmtTok(ctx.parsed.y);
          },
        } : {},
      },
    },
  };
}

/* Re-resolve the axis/legend token colors on the live charts after a theme
 * flip (called from main.js's applyTheme). Dataset fills are saturated and
 * theme-stable, so only grid/tick/legend colors need the update. */
export function restyleCodeUsageCharts() {
  var gridColor = _cssVar('--line');
  var tickColor = _cssVar('--muted');
  var fgColor   = _cssVar('--ink');
  [_chartInput, _chartOutput, _chartReqs, _chartCache].forEach(function (chart) {
    if (!chart) return;
    chart.options.scales.x.ticks.color = tickColor;
    chart.options.scales.y.grid.color = gridColor;
    chart.options.scales.y.ticks.color = tickColor;
    chart.options.plugins.legend.labels.color = fgColor;
    chart.update('none');
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function set(el, text) {
  if (el) el.textContent = text;
}

function setFreshness(text) {
  if (els.cldFreshness) els.cldFreshness.textContent = text;
}

function fmtNum(n) {
  if (n === undefined || n === null) return '—';
  return Number(n).toLocaleString();
}

/* fmtTok / fmtCost live in api.js (shared with the Hub/OTel tables, #215). */

function fmtTs(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch (_) { return iso.slice(11, 16); }
}

function modelShort(m) {
  const lm = m.toLowerCase();
  if (lm.includes('fable')) return 'Fable';
  if (lm.includes('opus')) return 'Opus';
  if (lm.includes('sonnet')) return 'Sonnet';
  if (lm.includes('haiku')) return 'Haiku';
  return m.length > 20 ? m.slice(0, 20) + '…' : m;
}
