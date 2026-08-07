/* Hub tab — status card, services, install panel, live request stream,
 * counters, errors, log tail. The four diagnostic surfaces are vendored
 * disclosure cards, folded by default (#215).
 */

import { els, state } from './state.js';
import { jsonApi, postJson, eventStream, toast, escapeHtml, fmtClock, renderCounterTable, shortGpu, modelLabel } from './api.js';
import { langfuseTraceUrl, fetchTelemetryHealth } from './telemetry.js';
import { icon } from './_vendored/icons/icons.js';

// --------------------------------------------------------- status / urls
export async function fetchHubStatus() {
  try {
    const body = await jsonApi('/admin/api/hub/status');
    state.status = body;
    if (els.hubPid) els.hubPid.textContent = body.pid || '—';
    if (els.hubUptime) els.hubUptime.textContent = fmtUptime(body.uptime_s);
    setHubLive('good', 'up');
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    setHubLive('danger', 'unreachable');
  }
}

function setHubLive(kind, text) {
  if (!els.hubLiveStatus) return;
  els.hubLiveStatus.classList.remove('good', 'warn', 'danger');
  if (kind) els.hubLiveStatus.classList.add(kind);
  if (els.hubLiveStatusText) els.hubLiveStatusText.textContent = text;
}

function fmtUptime(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 60) return Math.floor(seconds) + 's';
  if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + Math.floor(seconds % 60) + 's';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h + 'h ' + m + 'm';
}

// --------------------------------------------------------- counters
export async function fetchCounters() {
  try {
    const body = await jsonApi('/admin/api/hub/counters');
    state.counters = body.counters || [];
    renderCounters();
  } catch (_) { /* ignore */ }
}

function renderCounters() {
  renderCounterTable(els.countersTable, state.counters);
}

// --------------------------------------------------------- live requests
function prependRequest(rec) {
  // Dedup by ts. The SSE seed (20 most-recent records) is re-sent every
  // time the EventSource (re)connects — and the browser auto-reconnects
  // after network blips, a tab-switch round-trip, or any uvicorn keepalive
  // drop. Without dedup each reconnect duplicates whatever's still in the
  // server-side ring. `ts` is the middleware's wall-clock float seconds,
  // unique per request because the recording `finally` only fires once.
  const isDup = function (arr) {
    return rec && rec.ts != null && arr.some(function (r) { return r.ts === rec.ts; });
  };
  if (!isDup(state.liveRequests)) {
    state.liveRequests = [rec].concat(state.liveRequests).slice(0, 50);
    renderRequests();
  }
  if (rec.status >= 400 && !isDup(state.recentErrors)) {
    state.recentErrors = [rec].concat(state.recentErrors).slice(0, 50);
    renderErrors();
  }
}

// A trace deep-link should only render when clicking it would actually
// reach a Langfuse trace: the hub itself up, Docker up, and Langfuse
// reachable. Otherwise the link would land on a connection error — so we
// hide it (the row still shows tokens, just no `trace` affordance). Same
// signals the Services card uses (issue #27).
function traceLinkReady() {
  const svc = state.services || {};
  const docker = svc.docker || {};
  const lf = svc.langfuse || {};
  return !!state.status && docker.running === true && lf.reachable === true;
}

function renderRequests() {
  const items = state.liveRequests || [];
  const traceUp = traceLinkReady();
  const list = els.liveRequestsList;
  if (!list) return;
  if (els.liveRequestsBadge) els.liveRequestsBadge.textContent = items.length;
  if (els.liveRequestsEmpty) els.liveRequestsEmpty.hidden = items.length > 0;
  list.innerHTML = '';
  items.forEach(function (r) {
    const li = document.createElement('li');
    const cls = r.status >= 500 ? 'err' : r.status >= 400 ? 'warn' : 'ok';
    // Identical deep-link to the Telemetry tab: shared langfuseTraceUrl()
    // derives the client-reachable Langfuse host (Tailscale/LAN/localhost
    // transparent) + project_id, opened in a new tab. Only shown when the
    // stack is up (see traceLinkReady).
    const traceCol = (r.trace_id && traceUp)
      ? ('<a href="' + langfuseTraceUrl(r.trace_id) + '" target="_blank" rel="noopener" title="' + escapeHtml(r.trace_id) + '">trace</a>')
      : '';
    li.innerHTML =
      '<span class="muted">' + fmtClock(r.ts) + '</span>' +
      '<span>' + modelLabel(r, '(no model)') + ' <span class="muted">' + escapeHtml(r.backend || '') + '</span></span>' +
      '<span class="req-status ' + cls + '">' + r.status + ' · ' + r.latency_ms + ' ms</span>' +
      '<span class="muted">' + (r.in_tok || 0) + ' / ' + (r.out_tok || 0) + ' tok ' + traceCol + '</span>';
    list.appendChild(li);
  });
}

function renderErrors() {
  const items = state.recentErrors || [];
  const list = els.recentErrorsList;
  if (!list) return;
  if (els.recentErrorsBadge) els.recentErrorsBadge.textContent = items.length;
  if (els.recentErrorsEmpty) els.recentErrorsEmpty.hidden = items.length > 0;
  list.innerHTML = '';
  items.forEach(function (r) {
    const li = document.createElement('li');
    li.innerHTML =
      '<span class="muted">' + fmtClock(r.ts) + '</span>' +
      '<span>' + modelLabel(r, '(no model)') + ' <span class="muted">' + escapeHtml(r.backend || '') + '</span></span>' +
      '<span class="req-status err">' + r.status + '</span>' +
      '<span class="muted">' + escapeHtml((r.error_detail || '').slice(0, 80)) + '</span>';
    list.appendChild(li);
  });
}

// --------------------------------------------------------- log tail
let logBuf = [];

function appendLogLine(line) {
  if (state.logPaused) return;
  logBuf.push(line);
  if (logBuf.length > 800) logBuf = logBuf.slice(-800);
  const pre = els.hubLog;
  if (!pre) return;
  pre.textContent = logBuf.join('\n');
  pre.scrollTop = pre.scrollHeight;
}

// --------------------------------------------------------- streams
export function startHubStreams() {
  stopHubStreams();
  state.hubStreamCtl = eventStream('/admin/api/hub/requests/stream', {
    message: function (data) {
      if (!data || typeof data !== 'object') return;
      prependRequest(data);
    },
  });
  state.hubLogStreamCtl = eventStream('/admin/api/hub/log/tail', {
    message: function (data) {
      if (typeof data === 'string') appendLogLine(data);
    },
  });
}

export function stopHubStreams() {
  if (state.hubStreamCtl) { try { state.hubStreamCtl.close(); } catch (_) {} state.hubStreamCtl = null; }
  if (state.hubLogStreamCtl) { try { state.hubLogStreamCtl.close(); } catch (_) {} state.hubLogStreamCtl = null; }
}

// --------------------------------------------------------- install panel
export async function fetchInstallStatus() {
  try {
    const body = await jsonApi('/admin/api/install/status');
    state.installRows = body.checks || [];
    renderInstall(body);
  } catch (_) { /* ignore */ }
}

function renderInstall(body) {
  if (!els.installRows || !els.installSummary) return;
  const checks = body.checks || [];
  const overall = body.worst_status || 'ok';
  els.installSummary.textContent = checks.length + ' checks · overall ' + overall;
  els.installSummary.className = 'collapse-count overall-' + overall;
  els.installRows.innerHTML = '';
  checks.forEach(function (c) {
    const row = document.createElement('div');
    row.className = 'install-row install-' + c.status;
    const glyph = c.status === 'ok' ? icon('circle-check')
      : c.status === 'warn' ? icon('triangle-alert')
      : c.status === 'missing' ? icon('circle-help')
      : icon('circle-x');
    row.innerHTML =
      '<span class="install-glyph">' + glyph + '</span>' +
      '<span class="install-label">' + escapeHtml(c.label) + '</span>' +
      '<span class="install-detail muted small">' + escapeHtml(c.detail || '') + '</span>';
    if (c.fix_id && (c.status === 'missing' || c.status === 'error')) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'ghost-btn';
      btn.innerHTML = icon('wrench') + escapeHtml(c.fix_label || 'Fix');
      btn.addEventListener('click', async function () {
        btn.disabled = true;
        btn.textContent = 'Running…';
        try {
          await postJson('/admin/api/install/fix', { fix_id: c.fix_id });
          toast('Fixed ' + c.id, 'good');
          await fetchInstallStatus();
        } catch (exc) {
          toast(String(exc.message || exc), 'error');
          btn.disabled = false;
          btn.innerHTML = icon('wrench') + 'Retry';
        }
      });
      row.appendChild(btn);
    }
    els.installRows.appendChild(row);
  });
}

// --------------------------------------------------------- services card (issue #27)
export async function fetchServicesStatus() {
  try {
    const body = await jsonApi('/admin/api/services/status');
    state.services = body;
    renderServices();
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    // Probe error itself — render an "unreachable" state. `probeFailed`
    // tells renderServices() this isn't a real launchable=false verdict
    // (we never got far enough to check the install path), so it must
    // not show the "Docker Desktop install not found" hint — that would
    // be a fabricated claim about install state from a fetch failure.
    state.services = { docker: { running: false, error: String(exc.message || exc) },
                       langfuse: { reachable: false, error: '' },
                       launchable: false, platform: '', probeFailed: true };
    renderServices();
  }
}

function setStatusPill(rootEl, textEl, kind, text) {
  if (!rootEl) return;
  rootEl.classList.remove('good', 'warn', 'danger');
  if (kind) rootEl.classList.add(kind);
  if (textEl) textEl.textContent = text;
}

function renderServices() {
  const body = state.services;
  if (!body) return;
  const docker = body.docker || {};
  const lf = body.langfuse || {};

  const dockerKind = docker.running ? 'good' : 'danger';
  const dockerLabel = docker.running ? 'up' : 'down';
  setStatusPill(els.dockerStatus, els.dockerStatusText, dockerKind, dockerLabel);
  if (els.dockerDetail) {
    els.dockerDetail.textContent = docker.running
      ? (docker.server_version ? 'engine ' + docker.server_version : '')
      : (docker.error || '');
  }
  // Start/Stop (#284) — only offered where launch_docker_desktop() actually
  // knows how (same `launchable` gate the combined Launch button already
  // uses: win32 + a found Docker Desktop install).
  if (els.dockerStartBtn) {
    els.dockerStartBtn.hidden = !(body.launchable && !docker.running);
    els.dockerStartBtn.disabled = state.dockerBusy;
  }
  if (els.dockerStopBtn) {
    els.dockerStopBtn.hidden = !(body.launchable && docker.running);
    els.dockerStopBtn.disabled = state.dockerBusy;
  }

  // Langfuse: "up" when reachable, "partial" when Docker is up but Langfuse
  // isn't (containers starting / never launched), "down" otherwise.
  let lfKind = 'danger';
  let lfLabel = 'down';
  if (lf.reachable) { lfKind = 'good'; lfLabel = 'up'; }
  else if (docker.running) { lfKind = 'warn'; lfLabel = 'down'; }
  setStatusPill(els.langfuseStatus, els.langfuseStatusText, lfKind, lfLabel);
  if (els.langfuseDetail) {
    els.langfuseDetail.textContent = lf.reachable ? '' : (lf.error || '');
  }
  if (els.langfuseStartBtn) {
    els.langfuseStartBtn.hidden = lf.reachable;
    els.langfuseStartBtn.disabled = state.langfuseBusy;
  }
  if (els.langfuseStopBtn) {
    els.langfuseStopBtn.hidden = !lf.reachable;
    els.langfuseStopBtn.disabled = state.langfuseBusy;
  }

  // AgentsView (#280): optional external indexer feeding the Code tab's
  // AGY vendor. "not installed" is a warn, not a danger — the hub is fully
  // functional without it.
  const av = body.agentsview || {};
  const avEnabled = !!av.host;
  let avKind = 'danger';
  let avLabel = 'down';
  if (av.reachable) { avKind = 'good'; avLabel = 'up'; }
  else if (!avEnabled) { avKind = 'warn'; avLabel = 'disabled'; }
  else if (av.installed === false) { avKind = 'warn'; avLabel = 'not installed'; }
  setStatusPill(els.agentsviewStatus, els.agentsviewStatusText, avKind, avLabel);
  if (els.agentsviewDetail) {
    els.agentsviewDetail.textContent = av.reachable
      ? (av.version ? av.version : '')
      : (av.installed === false ? 'see docs/code-usage-agentsview.md' : (av.error || ''));
  }
  if (els.agentsviewStartBtn) {
    els.agentsviewStartBtn.hidden = !(avEnabled && !av.reachable && av.installed);
    els.agentsviewStartBtn.disabled = state.agentsviewBusy;
  }
  if (els.agentsviewStopBtn) {
    els.agentsviewStopBtn.hidden = !(avEnabled && av.reachable);
    els.agentsviewStopBtn.disabled = state.agentsviewBusy;
  }

  // Peer rows (#179, generalized #372): the pills don't factor into the
  // overall status/launch-button logic above — each tells its own peer's
  // story. See renderPeerRows() below.
  renderPeerRows();

  // Overall pill summarises both.
  let overallKind = 'good';
  let overallText = 'all up';
  if (!docker.running && !lf.reachable) { overallKind = 'danger'; overallText = 'both down'; }
  else if (!docker.running) { overallKind = 'danger'; overallText = 'docker down'; }
  else if (!lf.reachable) { overallKind = 'warn'; overallText = 'langfuse down'; }
  else if (avEnabled && !av.reachable) { overallKind = 'warn'; overallText = 'agentsview down'; }
  setStatusPill(els.servicesOverall, els.servicesOverallText, overallKind, overallText);

  // Launch button + hint visibility.
  const anyDown = !docker.running || !lf.reachable;
  const showActions = anyDown && body.launchable && !state.servicesLaunching;
  if (els.servicesActions) els.servicesActions.hidden = !(anyDown && body.launchable);
  if (els.servicesHint) {
    if (body.probeFailed) {
      // Status probe itself failed — we don't know the install state,
      // so surface the real error instead of guessing.
      els.servicesHint.textContent = 'Status check failed: ' + (docker.error || 'unknown error') + '.';
      els.servicesHint.hidden = false;
    } else if (anyDown && !body.launchable) {
      const hint = body.platform === 'darwin'
        ? 'Start Docker manually: `open -a Docker`, then `./start_langfuse.sh`.'
        : body.platform === 'linux'
          ? 'Start Docker manually: `sudo systemctl start docker`, then `./start_langfuse.sh`.'
          : 'Docker Desktop install not found — install from docker.com.';
      els.servicesHint.textContent = hint;
      els.servicesHint.hidden = false;
    } else {
      els.servicesHint.hidden = true;
    }
  }
  if (els.servicesLaunchBtn) {
    els.servicesLaunchBtn.disabled = !showActions;
    els.servicesLaunchBtn.innerHTML = state.servicesLaunching
      ? 'Launching… (up to ~90s)'
      : icon('rocket') + 'Launch Docker + Langfuse';
  }
}

async function onServicesLaunchClick() {
  if (state.servicesLaunching) return;
  state.servicesLaunching = true;
  renderServices();
  try {
    const result = await postJson('/admin/api/services/launch', {});
    const steps = (result && result.steps) || [];
    const summary = steps.map(function (s) { return s.name + ':' + s.status; }).join(' · ');
    if (result.ok) {
      toast('Services launched · ' + summary, 'good');
    } else {
      const first = steps.find(function (s) { return s.status === 'error'; });
      const detail = first ? (first.name + ': ' + first.detail) : summary;
      toast('Launch failed — ' + detail, 'error');
    }
  } catch (exc) {
    toast('Launch failed: ' + (exc.message || exc), 'error');
  } finally {
    state.servicesLaunching = false;
    await fetchServicesStatus();
  }
}

// Peer rows (#179, generalized #372): one row per hub-running peer, driven
// by the `peers` list on /admin/api/services/status — never hardcoded (was
// issue #245's original point for the single Mac Mini row; still holds for
// N peers). Renders like machines.js's renderMachineCard()/onMachinesListClick
// — an innerHTML template + one delegated click listener on the container —
// rather than binding a listener per row, since the row count varies.
function renderPeerRows() {
  const container = els.peerRows;
  if (!container) return;
  const peers = (state.services && state.services.peers) || [];
  container.innerHTML = peers.map(renderPeerRow).join('');
}

function renderPeerRow(peer) {
  const busy = !!(state.peerBusyIds && state.peerBusyIds[peer.host_id]);
  const kind = peer.reachable ? 'good' : 'danger';
  const label = peer.reachable ? 'up' : 'down';
  // git_sha_match is null until both sides answer; only warn on an explicit
  // false, never on "haven't compared yet" (#181).
  const outOfSync = peer.reachable && peer.git_sha_match === false;
  const detail = !peer.reachable
    ? escapeHtml(peer.error || '')
    : outOfSync
      ? '<span class="badge warn">out of sync</span> ' +
        escapeHtml(peer.remote_git_sha || '?') + ' vs ' + escapeHtml(peer.local_git_sha || '?')
      : '';
  // Wake/Sync: mirrors the Docker/Langfuse launch-button pattern — one
  // action visible at a time depending on reachability.
  const wakeHtml = busy ? 'Waking…' : icon('play') + 'Wake';
  const syncHtml = busy ? 'Syncing…' : icon('refresh-cw') + 'Sync';
  return '<div class="services-row" data-host-id="' + escapeHtml(peer.host_id) + '">'
    + '<span class="services-label">' + icon('signal') + escapeHtml(peer.display_name || peer.host_id) + '</span>'
    + '<span class="hub-live-status ' + kind + '"><span class="dot"></span><span>' + label + '</span></span>'
    + '<span class="muted small services-detail">' + detail + '</span>'
    + '<button type="button" class="ghost-btn" data-action="bootstrap"' + (peer.reachable ? ' hidden' : '') + (busy ? ' disabled' : '') + '>' + wakeHtml + '</button>'
    + '<button type="button" class="ghost-btn" data-action="sync"' + (peer.reachable ? '' : ' hidden') + (busy ? ' disabled' : '') + '>' + syncHtml + '</button>'
    + '</div>';
}

function onPeerRowsClick(ev) {
  const btn = ev.target.closest('button[data-action]');
  if (!btn || btn.disabled) return;
  const row = btn.closest('.services-row');
  const hostId = row && row.dataset ? row.dataset.hostId : '';
  if (!hostId) return;
  const action = btn.dataset.action; // 'bootstrap' | 'sync'
  onPeerAction(hostId, action, action === 'bootstrap' ? 'woken up' : 'synced');
}

async function onPeerAction(hostId, action, pastTense) {
  if (state.peerBusyIds && state.peerBusyIds[hostId]) return;
  state.peerBusyIds = Object.assign({}, state.peerBusyIds);
  state.peerBusyIds[hostId] = true;
  renderPeerRows();
  const peers = (state.services && state.services.peers) || [];
  const found = peers.find(function (p) { return p.host_id === hostId; });
  const label = (found && found.display_name) || hostId;
  try {
    await postJson('/admin/api/hosts/' + hostId + '/' + action, {});
    toast(label + ' ' + pastTense, 'good');
  } catch (exc) {
    toast(label + ' ' + action + ' failed: ' + (exc.message || exc), 'error');
  } finally {
    const nextBusy = Object.assign({}, state.peerBusyIds);
    delete nextBusy[hostId];
    state.peerBusyIds = nextBusy;
    await fetchServicesStatus();
  }
}

// One start/stop action on a service row: busy flag → re-render → POST →
// {ok, steps}-driven toast → clear the flag and refresh (#284). AgentsView,
// Docker Desktop and Langfuse all ride this one helper (#470 — AgentsView
// kept its own hand-rolled copy when #284 generalized the shape). `busyKey`
// is the `state` flag the row's buttons read to disable themselves.
async function onServiceAction(label, busyKey, path, verb, doneLabel) {
  if (state[busyKey]) return;
  state[busyKey] = true;
  renderServices();
  try {
    const result = await postJson(path, {});
    if (result.ok) {
      toast(label + ' ' + doneLabel, 'good');
    } else {
      const first = (result.steps || []).find(function (s) { return s.status === 'error'; });
      toast(label + ' ' + verb + ' failed — ' + (first ? first.detail : 'unknown'), 'error');
    }
  } catch (exc) {
    toast(label + ' ' + verb + ' failed: ' + (exc.message || exc), 'error');
  } finally {
    state[busyKey] = false;
    await fetchServicesStatus();
  }
}

// AgentsView's start endpoint is `/launch`, not `/start` — hence the full
// path per action rather than a shared prefix + verb.
function onAgentsviewStartClick() {
  return onServiceAction('AgentsView', 'agentsviewBusy',
    '/admin/api/services/agentsview/launch', 'start', 'started');
}
function onAgentsviewStopClick() {
  return onServiceAction('AgentsView', 'agentsviewBusy',
    '/admin/api/services/agentsview/stop', 'stop', 'stopped');
}

// Docker Desktop + Langfuse individual start/stop (#284) — complementing
// (not replacing) the combined "Launch Docker + Langfuse" recovery button.
function onDockerStartClick() {
  return onServiceAction('Docker', 'dockerBusy', '/admin/api/services/docker/start', 'start', 'started');
}
function onDockerStopClick() {
  return onServiceAction('Docker', 'dockerBusy', '/admin/api/services/docker/stop', 'stop', 'stopped');
}
function onLangfuseStartClick() {
  return onServiceAction('Langfuse', 'langfuseBusy', '/admin/api/services/langfuse/start', 'start', 'started');
}
function onLangfuseStopClick() {
  return onServiceAction('Langfuse', 'langfuseBusy', '/admin/api/services/langfuse/stop', 'stop', 'stopped');
}

// --------------------------------------------------------- wire buttons
export function wireHub() {
  function togglePause() {
    state.logPaused = !state.logPaused;
    if (els.hubLogPauseBtn) {
      els.hubLogPauseBtn.innerHTML = state.logPaused
        ? icon('play') + 'Resume'
        : icon('pause') + 'Pause';
    }
  }
  if (els.hubLogPauseBtn) els.hubLogPauseBtn.addEventListener('click', togglePause);

  if (els.installFixAllBtn) {
    els.installFixAllBtn.addEventListener('click', async function () {
      els.installFixAllBtn.disabled = true;
      const original = els.installFixAllBtn.textContent;
      els.installFixAllBtn.textContent = 'Running…';
      try {
        await postJson('/admin/api/install/fix-all', {});
        toast('Fix-all complete.', 'good');
        await fetchInstallStatus();
      } catch (exc) {
        toast(String(exc.message || exc), 'error');
      } finally {
        els.installFixAllBtn.disabled = false;
        els.installFixAllBtn.textContent = original;
      }
    });
  }
  if (els.installRefreshBtn) {
    els.installRefreshBtn.addEventListener('click', function () { fetchInstallStatus(); });
  }

  if (els.servicesLaunchBtn) {
    els.servicesLaunchBtn.addEventListener('click', onServicesLaunchClick);
  }
  if (els.peerRows) {
    els.peerRows.addEventListener('click', onPeerRowsClick);
  }
  if (els.agentsviewStartBtn) {
    els.agentsviewStartBtn.addEventListener('click', onAgentsviewStartClick);
  }
  if (els.agentsviewStopBtn) {
    els.agentsviewStopBtn.addEventListener('click', onAgentsviewStopClick);
  }
  if (els.dockerStartBtn) {
    els.dockerStartBtn.addEventListener('click', onDockerStartClick);
  }
  if (els.dockerStopBtn) {
    els.dockerStopBtn.addEventListener('click', onDockerStopClick);
  }
  if (els.langfuseStartBtn) {
    els.langfuseStartBtn.addEventListener('click', onLangfuseStartClick);
  }
  if (els.langfuseStopBtn) {
    els.langfuseStopBtn.addEventListener('click', onLangfuseStopClick);
  }

  // Sparklines: lightweight inline-SVG renderer driven by /admin/api/hub/stats.
  setInterval(function () {
    if (state.tab !== 'hub') return;
    renderSparklines();
  }, 2500);

  // Services card — Docker + Langfuse status. Cheaper than the sparkline
  // sweep (two small probes, each capped at 2 s) so 5 s is plenty.
  // Also refresh telemetry health here so the live-request trace links can
  // resolve Langfuse's project_id without the user first visiting the
  // Telemetry tab — the deep-link URL is then byte-identical across tabs.
  setInterval(function () {
    if (state.tab !== 'hub') return;
    if (state.servicesLaunching) return;
    fetchServicesStatus().catch(function () {});
    fetchTelemetryHealth().catch(function () {});
  }, 5000);
}

async function renderSparklines() {
  let stats;
  try {
    stats = await jsonApi('/admin/api/hub/stats');
  } catch (_) { return; }
  const container = els.hubSparklines;
  if (!container) return;
  container.innerHTML = '';
  const history = stats.history || [];
  const g0 = stats.gpus && stats.gpus.length ? stats.gpus[0] : null;
  // Utilization tiles on top, memory tiles on bottom (#288).
  const groups = [
    { label: 'CPU', value: stats.cpu && stats.cpu.percent, series: history.map(function (h) { return h.cpu_percent; }) },
  ];
  if (g0) {
    groups.push({ label: 'GPU util', value: g0.util_percent, series: history.map(function (h) { return h.gpu0_util_percent; }) });
  }
  groups.push({ label: 'RAM', value: stats.ram && stats.ram.percent, series: history.map(function (h) { return h.ram_percent; }) });
  if (g0) {
    groups.push({ label: 'VRAM ' + shortGpu(g0.name), value: g0.vram_percent, series: history.map(function (h) { return h.gpu0_vram_percent; }) });
  }
  groups.forEach(function (g) {
    container.appendChild(buildSparkline(g));
  });
}

function buildSparkline(g) {
  const root = document.createElement('div');
  root.className = 'sparkline';
  const series = (g.series || []).filter(function (v) { return v !== null && v !== undefined && !isNaN(v); });
  const max = 100;
  const w = 140, h = 64;
  let path = '';
  if (series.length >= 2) {
    const step = w / (series.length - 1);
    series.forEach(function (v, i) {
      const x = i * step;
      const y = h - (v / max) * h;
      path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
    });
  }
  // Energy-tab chart look (#215): a 2px accent line over a translucent
  // accent area (fill closed down to the baseline). Both pull from the live
  // --accent token so the tiles re-theme for free.
  const area = path ? path + 'L' + w + ',' + h + ' L0,' + h + ' Z' : '';
  root.innerHTML =
    '<div class="sparkline-label"><span>' + escapeHtml(g.label) + '</span>' +
    '<span>' + (Number.isFinite(g.value) ? Math.round(g.value) + '%' : '—') + '</span></div>' +
    '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none">' +
    (area ? '<path d="' + area + '" fill="color-mix(in srgb, var(--accent) 18%, transparent)" stroke="none"/>' : '') +
    (path ? '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>' : '') +
    '</svg>';
  return root;
}


/* escapeHtml / fmtClock live in api.js (sibling dedup, #211). */
