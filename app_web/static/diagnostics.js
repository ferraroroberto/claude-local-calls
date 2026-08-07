/* Machine diagnostics (issue #315) — the drill-in behind the this-machine
 * card's 🔬 Diagnostics row.
 *
 * Shape mirrors machines.js deliberately: a poll started only while the
 * surface is visible (here: while the dialog is open, not while the tab is),
 * template-string rendering, delegated clicks, escapeHtml on every value, and
 * the five design.md async states.
 *
 * The dialog is the vendored native <dialog> shell (_vendored/modal) driven
 * with showModal()/close(), so the fleet nav auto-hides itself while it is
 * open (body:has(dialog[open])) and Esc closes for free.
 */

import { els, state } from './state.js';
import { jsonApi, postJson, putJson, api, toast, escapeHtml } from './api.js';
import { icon } from './_vendored/icons/icons.js';
import { setSwitch } from './_vendored/switch/switch.js';

const POLL_MS = 5000;
let pollHandle = null;

// ------------------------------------------------------------------ format
function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.floor(seconds);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm';
  const h = Math.floor(m / 60);
  const remM = m % 60;
  return h + 'h' + (remM ? ' ' + remM + 'm' : '');
}

function fmtWhen(epochSeconds) {
  if (!epochSeconds) return '—';
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

function fmtMb(mb) {
  if (!Number.isFinite(mb)) return '—';
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
}

/* Fixed-decimal formatting, `digits` decimal places (default 0) — distinct
 * contract from code_usage.js's fmtNum, which takes one arg and returns
 * toLocaleString() comma-grouping instead (design-drift audit #384). */
function fmtFixed(n, digits) {
  return Number.isFinite(n) ? n.toFixed(digits === undefined ? 0 : digits) : '—';
}

/* Verdict → the status vocabulary the rest of the SPA already uses
 * (design.md: status colors signal state, never decoration). */
function verdictMeta(level) {
  if (level === 'critical') return { cls: 'danger', label: 'Critical' };
  if (level === 'warning') return { cls: 'warn', label: 'Warning' };
  if (level === 'healthy') return { cls: 'good', label: 'Healthy' };
  return { cls: '', label: 'No capture yet' };
}

// ------------------------------------------------------------------- fetch
async function fetchDiagnosticsStatus() {
  try {
    const body = await jsonApi('/admin/api/diagnostics/status');
    state.diagStatus = body;
    state.diagDataState = 'ready';
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    state.diagDataState = state.diagStatus ? 'stale' : 'error';
  }
  renderDiagnostics();
}

async function fetchRuns() {
  try {
    const body = await jsonApi('/admin/api/diagnostics/runs?limit=50');
    state.diagRuns = body.runs || [];
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    state.diagRuns = state.diagRuns || [];
  }
  renderDiagnostics();
}

async function fetchSummary(runId) {
  state.diagSummary = null;
  state.diagDrift = null;
  state.diagSummaryState = 'loading';
  renderDiagnostics();
  try {
    const [summary, drift] = await Promise.all([
      jsonApi('/admin/api/diagnostics/runs/' + encodeURIComponent(runId)),
      jsonApi('/admin/api/diagnostics/runs/' + encodeURIComponent(runId) + '/drift'),
    ]);
    state.diagSummary = summary;
    state.diagDrift = drift;
    state.diagSummaryState = 'ready';
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    state.diagSummaryState = 'error';
  }
  renderDiagnostics();
}

// ------------------------------------------------------------------ render
function renderCapture(status) {
  const active = status && status.active;
  if (active) {
    const pct = active.duration_s
      ? Math.min(100, (active.elapsed_s / active.duration_s) * 100)
      : 0;
    const remaining = active.duration_s
      ? fmtDuration(active.remaining_s) + ' remaining'
      : 'open-ended';
    return '<div class="diag-capture is-running">'
      + '<p class="diag-capture-line">' + icon('activity')
      + '<span>Capturing · ' + escapeHtml(String(active.samples_written)) + ' samples · '
      + escapeHtml(remaining) + '</span></p>'
      + '<div class="gauge-track"><div class="gauge-fill" style="width:' + pct.toFixed(0) + '%"></div></div>'
      + '<div class="diag-capture-actions">'
      + '<button type="button" class="ghost-btn danger" data-diag-action="stop">'
      + icon('square') + '<span>Stop capture</span></button>'
      + '</div></div>';
  }
  return '<div class="diag-capture">'
    + '<label class="diag-field"><span>Capture window</span>'
    + '<select id="diagDuration" class="select-native">'
    + '<option value="900">15 minutes</option>'
    + '<option value="3600" selected>1 hour</option>'
    + '<option value="7200">2 hours</option>'
    + '<option value="28800">8 hours</option>'
    + '</select></label>'
    + '<label class="diag-field"><span>Sample every</span>'
    + '<select id="diagInterval" class="select-native">'
    + '<option value="5">5 seconds</option>'
    + '<option value="15" selected>15 seconds</option>'
    + '<option value="60">1 minute</option>'
    + '</select></label>'
    + '<div class="diag-capture-actions">'
    + '<button type="button" class="button-primary" data-diag-action="start">'
    + icon('play') + '<span>Start capture</span></button>'
    + '<button type="button" class="ghost-btn" data-diag-action="snapshot">'
    + icon('flask-conical') + '<span>One-shot snapshot</span></button>'
    + '</div></div>';
}

function renderRuns(runs, selectedId) {
  if (!runs || !runs.length) {
    return '<div class="empty-state">'
      + '<svg class="icon empty-state-icon" aria-hidden="true"><use href="#i-clock"></use></svg>'
      + '<p class="empty-state-message">No captures recorded yet.</p></div>';
  }
  return '<div class="diag-runs">' + runs.map(function (r) {
    const v = verdictMeta(r.verdict_level);
    const selected = r.run_id === selectedId;
    const badges = [];
    if (r.is_baseline) badges.push('<span class="diag-badge accent">Baseline</span>');
    if (r.trigger && r.trigger !== 'manual') {
      badges.push('<span class="diag-badge">' + escapeHtml(r.trigger) + '</span>');
    }
    if (r.status && r.status !== 'complete') {
      badges.push('<span class="diag-badge">' + escapeHtml(r.status) + '</span>');
    }
    return '<button type="button" class="diag-run-row' + (selected ? ' is-selected' : '') + '"'
      + ' data-diag-run="' + escapeHtml(r.run_id) + '">'
      + '<span class="diag-run-main">'
      + '<span class="hub-live-status ' + v.cls + '"><span class="dot"></span>'
      + '<span>' + escapeHtml(v.label) + '</span></span>'
      + badges.join('')
      + '</span>'
      + '<span class="diag-run-meta muted">' + escapeHtml(fmtWhen(r.started_at))
      + ' · ' + escapeHtml(String(r.sample_count || 0)) + ' samples</span>'
      + '</button>';
  }).join('') + '</div>';
}

function renderFindings(verdict) {
  const findings = (verdict && verdict.findings) || [];
  if (!findings.length) {
    return '<p class="muted small">No threshold was crossed — this window looks healthy.</p>';
  }
  return '<ul class="diag-findings">' + findings.map(function (f) {
    const v = verdictMeta(f.level);
    return '<li class="diag-finding">'
      + '<span class="diag-badge ' + v.cls + '">' + escapeHtml(v.label) + '</span>'
      + '<span>' + escapeHtml(f.summary) + '</span>'
      + '<code class="diag-rule muted">' + escapeHtml(f.rule) + '</code>'
      + '</li>';
  }).join('') + '</ul>';
}

function statRow(label, entry, unit) {
  const avg = entry && Number.isFinite(entry.avg) ? fmtFixed(entry.avg, 0) + (unit || '') : '—';
  const peak = entry && Number.isFinite(entry.peak) ? fmtFixed(entry.peak, 0) + (unit || '') : '—';
  return '<tr><th scope="row">' + escapeHtml(label) + '</th><td>' + avg + '</td><td>' + peak + '</td></tr>';
}

function renderSummary(summary, drift) {
  if (!summary) return '';
  const res = summary.resources || {};
  const run = summary.run || {};
  const parts = [];

  parts.push('<h3 class="diag-h3">' + icon('stethoscope') + 'Verdict</h3>');
  parts.push(renderFindings(summary.verdict));

  parts.push('<h3 class="diag-h3">' + icon('chart-column') + 'Resource envelope</h3>');
  parts.push('<div class="diag-table-wrap"><table class="diag-table">'
    + '<thead><tr><th scope="col">Metric</th><th scope="col">Avg</th><th scope="col">Peak</th></tr></thead><tbody>'
    + statRow('CPU', res.cpu, '%')
    + statRow('RAM', res.ram, '%')
    + statRow('Swap', res.swap, '%')
    + statRow('Disk', res.disk, '%')
    + statRow('Processes', res.process_count, '')
    + ((res.gpu_peak_vram || []).map(function (g) {
      return '<tr><th scope="row">VRAM · ' + escapeHtml(g.name) + '</th><td>—</td><td>'
        + fmtFixed(g.peak_percent, 0) + '%</td></tr>';
    }).join(''))
    + '</tbody></table></div>');

  const apps = summary.apps || [];
  parts.push('<h3 class="diag-h3">' + icon('boxes') + 'Load by app</h3>');
  if (!apps.length) {
    parts.push('<p class="muted small">No process data in this run.</p>');
  } else {
    parts.push('<div class="diag-table-wrap"><table class="diag-table"><thead><tr>'
      + '<th scope="col">App</th><th scope="col">Procs</th><th scope="col">Memory</th><th scope="col" title="Percent of the whole machine">CPU</th>'
      + '</tr></thead><tbody>'
      + apps.map(function (a) {
        return '<tr><th scope="row">' + escapeHtml(a.app_id) + '</th>'
          + '<td>' + escapeHtml(String(a.peak_procs)) + '</td>'
          + '<td>' + escapeHtml(fmtMb(a.peak_rss_mb)) + '</td>'
          + '<td>' + fmtFixed(a.peak_cpu, 0) + '%</td></tr>';
      }).join('')
      + '</tbody></table></div>');
  }

  const procs = summary.top_processes_by_rss || [];
  if (procs.length) {
    parts.push('<h3 class="diag-h3">' + icon('cpu') + 'Heaviest processes</h3>');
    parts.push('<div class="diag-table-wrap"><table class="diag-table"><thead><tr>'
      + '<th scope="col">Process</th><th scope="col">App</th><th scope="col">PIDs</th><th scope="col">Memory</th>'
      + '</tr></thead><tbody>'
      + procs.map(function (p) {
        return '<tr><th scope="row" title="' + escapeHtml(p.cmdline || '') + '">'
          + escapeHtml(p.name || '—') + '</th>'
          + '<td>' + escapeHtml(p.app_id || '') + '</td>'
          + '<td>' + escapeHtml(String(p.pid_count || 0)) + '</td>'
          + '<td>' + escapeHtml(fmtMb(p.peak_rss_mb)) + '</td></tr>';
      }).join('')
      + '</tbody></table></div>');
  }

  const ports = summary.ports || [];
  if (ports.length) {
    parts.push('<h3 class="diag-h3">' + icon('radio') + 'Listening ports</h3>');
    parts.push('<div class="diag-table-wrap"><table class="diag-table"><thead><tr>'
      + '<th scope="col">Port</th><th scope="col">Owner</th><th scope="col">Process</th>'
      + '</tr></thead><tbody>'
      + ports.map(function (q) {
        return '<tr><th scope="row">' + escapeHtml(String(q.port)) + '</th>'
          + '<td>' + escapeHtml(q.app_id || '') + '</td>'
          + '<td>' + escapeHtml(q.name || '—') + '</td></tr>';
      }).join('')
      + '</tbody></table></div>');
  }

  if (drift && drift.baseline) {
    parts.push('<h3 class="diag-h3">' + icon('chart-line') + 'Drift vs baseline</h3>');
    parts.push('<p class="muted small">Baseline captured ' + escapeHtml(fmtWhen(drift.baseline.started_at)) + '.</p>');
    const changes = drift.changes || [];
    if (changes.length) {
      parts.push('<ul class="diag-drift">' + changes.map(function (c) {
        const sign = c.delta > 0 ? '+' : '';
        const cls = c.delta > 0 ? 'warn' : (c.delta < 0 ? 'good' : '');
        return '<li><span>' + escapeHtml(c.label) + '</span>'
          + '<span class="diag-delta ' + cls + '">' + escapeHtml(String(c.before)) + ' → '
          + escapeHtml(String(c.now)) + ' (' + sign + escapeHtml(String(c.delta)) + ')</span></li>';
      }).join('') + '</ul>');
    }
    if ((drift.new_apps || []).length) {
      parts.push('<p class="small">New since baseline: <strong>'
        + escapeHtml(drift.new_apps.join(', ')) + '</strong></p>');
    }
    (drift.ports || []).forEach(function (p) {
      parts.push('<p class="small muted">Port ' + escapeHtml(String(p.port)) + ' ('
        + escapeHtml(p.app_id || '') + ') — ' + escapeHtml(p.status) + '</p>');
    });
  }

  parts.push('<div class="diag-run-actions">'
    + '<button type="button" class="ghost-btn" data-diag-action="baseline">'
    + icon('tag') + '<span>' + (run.is_baseline ? 'Baseline' : 'Mark as baseline') + '</span></button>'
    + '<button type="button" class="ghost-btn" data-diag-action="report">'
    + icon('scroll-text') + '<span>Health report</span></button>'
    + '<button type="button" class="ghost-btn" data-diag-action="export">'
    + icon('download') + '<span>Export JSON</span></button>'
    + '<button type="button" class="ghost-btn danger" data-diag-action="delete">'
    + icon('circle-x') + '<span>Delete</span></button>'
    + '</div>');

  return '<div class="diag-summary">' + parts.join('') + '</div>';
}

function renderSettings(status) {
  const s = (status && status.settings) || {};
  const hours = s.scheduled_interval_hours || 24;
  return '<div class="diag-settings">'
    + '<label class="diag-field diag-switch-row">'
    + '<span>Daily snapshot<br><span class="muted small">One automatic sample so trends exist without pressing anything. Adds no process.</span></span>'
    // Rendered by the vendored setSwitch() after innerHTML lands (see
    // renderDiagnostics) — that helper is the single write path keeping the
    // class and aria-checked in lockstep, so we never hand-author the markup.
    + '<button type="button" data-diag-action="toggle-schedule" data-diag-switch="'
    + (s.scheduled_enabled ? '1' : '0') + '" aria-label="Daily diagnostics snapshot"></button>'
    + '</label>'
    + '<label class="diag-field"><span>Snapshot every</span>'
    + '<select id="diagScheduleHours" class="select-native">'
    + [6, 12, 24, 48].map(function (h) {
      return '<option value="' + h + '"' + (Number(hours) === h ? ' selected' : '') + '>'
        + (h === 24 ? 'day' : h + ' hours') + '</option>';
    }).join('')
    + '</select></label>'
    + '<label class="diag-field"><span>Keep raw samples</span>'
    + '<select id="diagRetention" class="select-native">'
    + [30, 90, 180, 365].map(function (d) {
      return '<option value="' + d + '"' + (Number(s.retention_days) === d ? ' selected' : '') + '>'
        + d + ' days</option>';
    }).join('')
    + '</select></label>'
    + '<p class="muted small">Database ' + escapeHtml(fmtMb((s.db_size_bytes || 0) / (1024 * 1024)))
    + ' · run metadata and verdicts are kept indefinitely.</p>'
    + '<div class="diag-capture-actions">'
    + '<button type="button" class="ghost-btn" data-diag-action="save-settings">'
    + icon('save') + '<span>Save settings</span></button></div>'
    + '</div>';
}

function renderDiagnostics() {
  // The card-level chip (rendered by machines.js) tracks the same state.
  updateCardChip();
  if (!els.diagBody) return;
  if (!els.diagDialog || !els.diagDialog.open) return;

  const status = state.diagStatus;
  const ds = state.diagDataState;
  if (ds === 'loading' && !status) {
    els.diagBody.innerHTML = '<div class="empty-state">'
      + '<svg class="icon empty-state-icon" aria-hidden="true"><use href="#i-stethoscope"></use></svg>'
      + '<p class="empty-state-message">Reading diagnostics status…</p></div>';
    return;
  }
  if (ds === 'error' && !status) {
    els.diagBody.innerHTML = '<div class="empty-state">'
      + '<svg class="icon empty-state-icon" aria-hidden="true"><use href="#i-triangle-alert"></use></svg>'
      + '<p class="empty-state-message">Could not read diagnostics status.</p>'
      + '<button type="button" class="empty-state-action" data-diag-action="retry">Retry</button></div>';
    return;
  }

  const staleNote = ds === 'stale'
    ? '<p class="muted small" role="status">Live data unavailable — showing the last known state.</p>'
    : '';

  els.diagBody.innerHTML = staleNote
    + '<section class="diag-section"><h3 class="diag-h3">' + icon('play') + 'Capture</h3>'
    + renderCapture(status) + '</section>'
    + '<section class="diag-section"><h3 class="diag-h3">' + icon('clock') + 'Runs</h3>'
    + renderRuns(state.diagRuns, state.diagSelectedRun) + '</section>'
    + (state.diagSelectedRun
      ? '<section class="diag-section">'
        + (state.diagSummaryState === 'loading'
          ? '<p class="muted small">Reading run…</p>'
          : (state.diagSummaryState === 'error'
            ? '<p class="muted small">Could not read that run.</p>'
            : renderSummary(state.diagSummary, state.diagDrift)))
        + '</section>'
      : '')
    + '<details class="diag-section diag-settings-block"><summary class="diag-h3">'
    + icon('wrench') + 'Settings</summary>' + renderSettings(status) + '</details>';

  // Paint the schedule switch through the vendored write path.
  const sw = els.diagBody.querySelector('[data-diag-switch]');
  if (sw) setSwitch(sw, sw.dataset.diagSwitch === '1');
}

/* The compact entry point on the this-machine card: last verdict, or live
 * capture progress. machines.js re-renders the card list on its own poll, so
 * this only refreshes the chip's text/class in place. */
function updateCardChip() {
  const chip = document.querySelector('[data-diag-chip]');
  if (!chip) return;
  const status = state.diagStatus;
  const active = status && status.active;
  if (active) {
    chip.className = 'hub-live-status accent';
    chip.innerHTML = '<span class="dot"></span><span>Capturing · '
      + escapeHtml(fmtDuration(active.elapsed_s)) + '</span>';
    return;
  }
  const latest = (state.diagRuns || [])[0];
  const v = verdictMeta(latest && latest.verdict_level);
  chip.className = 'hub-live-status ' + v.cls;
  chip.innerHTML = '<span class="dot"></span><span>' + escapeHtml(v.label) + '</span>';
}

// ----------------------------------------------------------------- actions
async function startCapture() {
  const durationEl = document.getElementById('diagDuration');
  const intervalEl = document.getElementById('diagInterval');
  const payload = {
    duration_s: Number(durationEl && durationEl.value) || 3600,
    interval_s: Number(intervalEl && intervalEl.value) || 15,
  };
  try {
    await postJson('/admin/api/diagnostics/start', payload);
    toast('Capture started.', 'good');
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Could not start the capture.', 'error');
  }
  await fetchDiagnosticsStatus();
  await fetchRuns();
}

async function snapshot() {
  toast('Taking a snapshot…');
  try {
    const body = await postJson('/admin/api/diagnostics/snapshot', {});
    toast('Snapshot captured.', 'good');
    state.diagSelectedRun = body.run_id;
    await fetchRuns();
    await fetchSummary(body.run_id);
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Snapshot failed.', 'error');
  }
  await fetchDiagnosticsStatus();
}

async function stopCapture() {
  try {
    await postJson('/admin/api/diagnostics/stop', {});
    toast('Capture stopped.', 'good');
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Could not stop the capture.', 'error');
  }
  await fetchDiagnosticsStatus();
  await fetchRuns();
}

async function markBaseline() {
  const runId = state.diagSelectedRun;
  if (!runId) return;
  try {
    await postJson('/admin/api/diagnostics/runs/' + encodeURIComponent(runId) + '/baseline', {});
    toast('Baseline set.', 'good');
    await fetchRuns();
    await fetchSummary(runId);
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Could not set the baseline.', 'error');
  }
}

async function deleteRun() {
  const runId = state.diagSelectedRun;
  if (!runId) return;
  const run = (state.diagRuns || []).find(function (r) { return r.run_id === runId; }) || {};
  const when = run.started_at ? fmtWhen(run.started_at) : runId;
  const baselineWarning = run.is_baseline
    ? ' It is the current baseline — later runs will have nothing to report drift against.'
    : '';
  if (!window.confirm('Delete the capture from ' + when + '?' + baselineWarning
    + ' This cannot be undone.')) return;
  try {
    const res = await api('/admin/api/diagnostics/runs/' + encodeURIComponent(runId), { method: 'DELETE' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    toast('Run deleted.', 'good');
    state.diagSelectedRun = null;
    state.diagSummary = null;
    await fetchRuns();
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Could not delete that run.', 'error');
  }
}

async function download(path, filename) {
  try {
    const res = await api(path);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
    toast('Downloading ' + filename, 'good');
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Could not download that file.', 'error');
  }
}

async function saveSettings(overrides) {
  const hoursEl = document.getElementById('diagScheduleHours');
  const retentionEl = document.getElementById('diagRetention');
  const current = (state.diagStatus && state.diagStatus.settings) || {};
  const payload = Object.assign({
    scheduled_enabled: !!current.scheduled_enabled,
    scheduled_interval_hours: Number(hoursEl && hoursEl.value) || 24,
    retention_days: Number(retentionEl && retentionEl.value) || 90,
  }, overrides || {});
  try {
    const body = await putJson('/admin/api/diagnostics/settings', payload);
    state.diagStatus = Object.assign({}, state.diagStatus, { settings: body.settings });
    toast('Settings saved.', 'good');
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    toast('Could not save settings.', 'error');
  }
  renderDiagnostics();
}

// ------------------------------------------------------------------ dialog
export function openDiagnostics() {
  if (!els.diagDialog || !els.diagDialog.showModal) return;
  els.diagDialog.showModal();
  state.diagDataState = state.diagStatus ? 'ready' : 'loading';
  renderDiagnostics();
  fetchDiagnosticsStatus();
  fetchRuns();
  startPoll();
}

function closeDiagnostics() {
  stopPoll();
  if (els.diagDialog && els.diagDialog.open) els.diagDialog.close();
}

/* Poll only while the dialog is open — the same "don't poll an invisible
 * surface" rule machines.js applies per tab. */
function startPoll() {
  if (pollHandle) return;
  pollHandle = setInterval(function () {
    fetchDiagnosticsStatus();
    // While a capture runs, the run list gains samples worth reflecting.
    if (state.diagStatus && state.diagStatus.capturing) fetchRuns();
  }, POLL_MS);
}

function stopPoll() {
  if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
}

/* Kept exported so machines.js can refresh the card chip without opening
 * the dialog (one cheap read per Machines-tab poll). */
export async function refreshDiagnosticsChip() {
  try {
    const [status, runs] = await Promise.all([
      jsonApi('/admin/api/diagnostics/status'),
      jsonApi('/admin/api/diagnostics/runs?limit=1'),
    ]);
    state.diagStatus = status;
    if (!state.diagRuns || !state.diagRuns.length) state.diagRuns = runs.runs || [];
    else if ((runs.runs || []).length) state.diagRuns[0] = runs.runs[0];
    state.diagDataState = 'ready';
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
  }
  updateCardChip();
}

function onBodyClick(ev) {
  const runBtn = ev.target.closest('[data-diag-run]');
  if (runBtn) {
    state.diagSelectedRun = runBtn.dataset.diagRun;
    renderDiagnostics();
    fetchSummary(state.diagSelectedRun);
    return;
  }
  const btn = ev.target.closest('[data-diag-action]');
  if (!btn || btn.disabled) return;
  const action = btn.dataset.diagAction;
  const runId = state.diagSelectedRun;
  if (action === 'start') startCapture();
  else if (action === 'snapshot') snapshot();
  else if (action === 'stop') stopCapture();
  else if (action === 'baseline') markBaseline();
  else if (action === 'delete') deleteRun();
  else if (action === 'retry') { state.diagDataState = 'loading'; renderDiagnostics(); fetchDiagnosticsStatus(); }
  else if (action === 'save-settings') saveSettings();
  else if (action === 'toggle-schedule') {
    const on = btn.getAttribute('aria-checked') === 'true';
    saveSettings({ scheduled_enabled: !on });
  } else if (action === 'report' && runId) {
    download('/admin/api/diagnostics/runs/' + encodeURIComponent(runId) + '/report',
      'diagnostics-' + runId + '.md');
  } else if (action === 'export' && runId) {
    download('/admin/api/diagnostics/runs/' + encodeURIComponent(runId) + '/export',
      'diagnostics-' + runId + '.json');
  }
}

export function wireDiagnostics() {
  if (els.diagCloseBtn) els.diagCloseBtn.addEventListener('click', closeDiagnostics);
  if (els.diagBody) els.diagBody.addEventListener('click', onBodyClick);
  if (els.diagDialog) {
    // Esc fires 'close' on a native <dialog> — stop the poll either way so a
    // dismissed dialog never leaves a timer running.
    els.diagDialog.addEventListener('close', stopPoll);
  }
}
