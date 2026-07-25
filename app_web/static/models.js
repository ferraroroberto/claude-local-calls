/* Models tab — per-backend .app-item row (canonical pattern).
 *
 * Process control only: start / stop / ping / force-stop. Per-model log
 * tailing was tried in #10 but pulled back — adopted backends have no
 * captured stdout, and the central Hub log tab already shows every
 * request that flows through the hub. Detailed per-backend telemetry
 * belongs in a future dedicated tab (#4: OpenTelemetry + Langfuse).
 */

import { els, state, MODELS_ACTIVE_ONLY_KEY } from './state.js';
import { jsonApi, postJson, toast, escapeHtml } from './api.js';
import { mountGlossaryEditor } from './glossary.js';
import { icon } from './_vendored/icons/icons.js';

export async function fetchModels() {
  try {
    const body = await jsonApi('/admin/api/models');
    state.models = body.models || [];
    state.hostBudgets = body.host_budgets || {};
    renderModels();
  } catch (_) { /* ignore */ }
}

// A row counts as "active" only if it's a controllable backend that's
// currently running/adopted. Claude/Gemini are excluded outright — they're
// subscription-backed with no on/off state, always "on", so listing them
// here is just noise rather than a signal (#266).
function isActive(m) {
  return m.controllable && (m.ownership === 'ours' || m.ownership === 'external');
}

function renderModels() {
  const root = els.modelsList;
  if (!root) return;
  const models = state.models || [];
  const visible = state.modelsActiveOnly ? models.filter(isActive) : models;

  if (els.modelsEmpty) {
    els.modelsEmpty.hidden = visible.length > 0;
    const msg = els.modelsEmpty.querySelector('.empty-state-message');
    if (msg) {
      msg.innerHTML = models.length === 0
        ? 'No models enabled for this host — check <code>config/models.yaml</code>.'
        : 'No active models right now — turn off “Active only” to see the full list.';
    }
  }

  // Diff-update so the row identity survives the 5 s poll. Reusing the
  // existing <li> per model id avoids the DOM churn (and any focus /
  // selection loss) of a full innerHTML rebuild.
  const existing = {};
  Array.prototype.forEach.call(root.children, function (node) {
    if (node.classList && node.classList.contains('app-item') && node.dataset.id) {
      existing[node.dataset.id] = node;
    }
  });

  const frag = document.createDocumentFragment();
  visible.forEach(function (m) {
    const prev = existing[m.id];
    if (prev) {
      fillItem(prev, m);
      frag.appendChild(prev);
    } else {
      frag.appendChild(buildItem(m));
    }
  });
  root.replaceChildren(frag);
}

function buildItem(m) {
  const li = document.createElement('li');
  li.className = 'app-item';
  li.dataset.id = m.id;
  fillItem(li, m);
  return li;
}

function fillItem(li, m) {
  const ownership = m.ownership || 'none';
  const reachable = !!m.reachable;
  const adopted = ownership === 'external';
  const localHost = state.status && state.status.host;
  const remote = !!(m.host && localHost && m.host !== localHost);

  // Rebuild the .app-main block in place rather than wiping the whole <li>,
  // so an open dictionary panel (a sibling, below) survives the 5 s poll
  // with its unsaved edits intact.
  let main = li.querySelector(':scope > .app-main');
  if (main) {
    main.replaceChildren();
  } else {
    main = document.createElement('div');
    main.className = 'app-main';
  }

  const titleRow = document.createElement('div');
  titleRow.className = 'app-title-row';
  // The badge lives OUTSIDE .app-title: that span ellipsises long display
  // names (whisper-large-v3-turbo, gemma4-26b-a4b-it), and a pill nested
  // inside it gets pushed past the clip boundary and vanishes. As a sibling
  // in the title row it is never clipped. PID/host details moved to the
  // meta line (#215) — as title-row extras they crushed the name on phones.
  titleRow.innerHTML =
    '<span class="app-title"><span class="app-name">' + escapeHtml(m.display_name) + '</span></span>' + badge(m);

  const icons = document.createElement('div');
  icons.className = 'app-icons';
  const buttons = [];
  if (m.controllable) {
    buttons.push({ act: 'start', glyph: icon('play'), label: 'Start', disabled: ownership !== 'none' });
    buttons.push({ act: 'stop',  glyph: icon('square'), label: 'Stop',  disabled: ownership !== 'ours', danger: true });
  }
  buttons.push({
    act: 'ping', glyph: icon('signal'), label: 'Ping',
    disabled: !reachable && m.backend !== 'claude' && m.backend !== 'gemini',
  });
  if (adopted) {
    buttons.push({ act: 'force-stop', glyph: icon('skull'), label: 'Force stop', danger: true });
  }
  if (m.backend === 'whisper') {
    // The transcription dictionary is shared by every whisper backend, so
    // the same editor opens from any whisper row.
    buttons.push({ act: 'dictionary', glyph: icon('book-open'), label: 'Transcription dictionary' });
  }
  const panelOpen = !!li.querySelector(':scope > .glossary-panel:not([hidden])');
  buttons.forEach(function (b) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'icon-btn' + (b.danger ? ' danger' : '');
    if (b.act === 'dictionary' && panelOpen) btn.classList.add('active');
    btn.dataset.act = b.act;
    btn.disabled = !!b.disabled;
    btn.title = b.label;
    btn.setAttribute('aria-label', b.label);
    btn.innerHTML = b.glyph;
    btn.addEventListener('click', function () { handleAction(m, b.act); });
    icons.appendChild(btn);
  });
  titleRow.appendChild(icons);
  main.appendChild(titleRow);

  const meta = document.createElement('div');
  meta.className = 'app-meta';
  // Host note (#181) + adopted PID live here with the port (#215): in the
  // title row they wrapped the tile to two title lines and squeezed the
  // name to nothing on a phone ("parakeet … on mac-mini-m4").
  meta.textContent =
    m.backend +
    (m.port ? ' · :' + m.port : '') +
    (remote ? ' on ' + m.host : '') +
    // Dynamic host-chain fallback (#342): flag a model currently served
    // off its preferred host, naming where it normally lives.
    (m.failover ? ' · failover (prefers ' + m.preferred_host + ')' : '') +
    (m.pid && adopted ? ' · PID ' + m.pid : '') +
    // Resolved TTS device (cpu/cuda/mps), reported once the backend has
    // finished loading and answered its own /health — see
    // app_web/routers/models.py's _probe_device (#371). Omitted (not
    // guessed) for stopped/loading rows and backends with no device concept.
    (m.device ? ' · ' + m.device : '') +
    (m.aliases && m.aliases.length ? ' · ' + m.aliases.join(', ') : '');
  main.appendChild(meta);

  // Read-only placement card (#423) — declared intent under the runtime meta.
  const placement = buildPlacement(m);
  if (placement) main.appendChild(placement);

  // Keep .app-main as the first child so any dictionary panel stays below it.
  const panel = li.querySelector(':scope > .glossary-panel');
  if (panel) {
    li.insertBefore(main, panel);
  } else if (main.parentNode !== li) {
    li.appendChild(main);
  }
}

/* ---------- read-only placement card (#423) ----------
 * Renders the declared placement *intent* from config/models.yaml (the Phase 1
 * #422 registry fields the API now carries per row): the host chain in
 * priority order with the effective owner highlighted and degraded cpu tiers
 * marked, a startup-policy badge that reads the live state for on-demand
 * rows, the static VRAM estimate, and the owner host's budget bar (resident
 * estimate vs declared ceiling — the #375 grid math, surfaced per row).
 * Subscription rows (claude/gemini) carry no `placement` key and get nothing.
 * Everything here is read-only — mutation stays with start/stop (Phase 3). */
function buildPlacement(m) {
  const p = m.placement;
  if (!p) return null;
  const wrap = document.createElement('div');
  wrap.className = 'placement';

  const line = document.createElement('div');
  line.className = 'placement-line';

  const chain = p.chain || [];
  if (chain.length) {
    const chainEl = document.createElement('span');
    chainEl.className = 'placement-chain';
    chain.forEach(function (entry, i) {
      if (i) {
        const sep = document.createElement('span');
        sep.className = 'placement-arrow';
        sep.textContent = '›';
        sep.setAttribute('aria-hidden', 'true');
        chainEl.appendChild(sep);
      }
      const owner = entry.id === m.host;
      const pill = document.createElement('span');
      pill.className = 'place-pill' + (owner ? ' owner' : '');
      pill.textContent = entry.id + (entry.cpu ? ' · cpu' : '');
      pill.title = (owner ? 'Active owner'
        : (i === 0 ? 'Preferred host' : 'Fallback host'))
        + (entry.cpu ? ' — degraded CPU tier' : '');
      chainEl.appendChild(pill);
    });
    line.appendChild(chainEl);
  }

  line.appendChild(startupBadge(m, p));

  if (p.est_vram_mb) {
    const vr = document.createElement('span');
    vr.className = 'placement-vram muted';
    vr.textContent = '~' + fmtGb(p.est_vram_mb) + ' VRAM';
    vr.title = 'Static worst-case GPU-VRAM estimate (est_vram_mb) — not live telemetry';
    line.appendChild(vr);
  }
  wrap.appendChild(line);

  const budget = budgetBar(m);
  if (budget) wrap.appendChild(budget);
  return wrap;
}

/* Startup policy badge: `eager` is the quiet default; an on-demand row reads
 * its live state honestly — "loaded" while the backend answers, else
 * "idle-unloaded" (stopped by the idle watchdog or never yet demanded). */
function startupBadge(m, p) {
  const el = document.createElement('span');
  if (p.startup === 'on_demand') {
    const idleNote = p.idle_unload_minutes
      ? 'Loads on first request; unloads after ' + p.idle_unload_minutes + ' min idle'
      : 'Loads on first request; stays up until stopped';
    if (m.reachable) {
      el.className = 'badge good';
      el.textContent = 'on-demand · loaded';
    } else {
      el.className = 'badge';
      el.textContent = 'on-demand · idle-unloaded';
    }
    el.title = idleNote;
  } else {
    el.className = 'badge';
    el.textContent = 'eager';
    el.title = 'Always-on: autostarted and kept resident';
  }
  return el;
}

/* The owner host's VRAM budget bar: resident estimate vs declared ceiling.
 * Hosts with no `vram_mb` ceiling (Apple-silicon unified memory) render no
 * bar — there is no honest denominator (#375). Overcommit tints the fill
 * `attention`, matching the fleet grid's advisory warning. */
function budgetBar(m) {
  const b = m.host && state.hostBudgets ? state.hostBudgets[m.host] : null;
  if (!b || !b.vram_mb) return null;
  const used = b.resident_est_vram_mb || 0;
  const pct = Math.min(100, Math.round((used / b.vram_mb) * 100));
  const row = document.createElement('div');
  row.className = 'placement-budget';
  row.title = 'Estimated VRAM of models currently resident on ' + m.host
    + ' vs its ' + fmtGb(b.vram_mb) + ' ceiling — static estimates, not live telemetry';
  const bar = document.createElement('span');
  bar.className = 'placement-budget-bar' + (used > b.vram_mb ? ' over' : '');
  const fill = document.createElement('span');
  fill.style.width = pct + '%';
  bar.appendChild(fill);
  const label = document.createElement('span');
  label.className = 'placement-budget-label muted';
  label.textContent = m.host + ' · ' + fmtGb(used) + ' / ' + fmtGb(b.vram_mb);
  row.appendChild(bar);
  row.appendChild(label);
  return row;
}

// MB → compact "X.X GB" (same formatting as the fleet grid's capacity note).
function fmtGb(mb) {
  return (Number(mb || 0) / 1024).toFixed(1) + ' GB';
}

function badge(m) {
  if (!m.controllable) return ' <span class="badge">' + escapeHtml(m.backend) + '</span>';
  if (m.ownership === 'ours') return ' <span class="badge good">running</span>';
  if (m.ownership === 'external') return ' <span class="badge warn">adopted</span>';
  return ' <span class="badge">stopped</span>';
}

async function handleAction(m, act) {
  if (act === 'start') {
    try {
      await postJson('/admin/api/models/' + encodeURIComponent(m.id) + '/start', {});
      toast('Starting ' + m.display_name + '…', 'good');
      await sleep(800);
      fetchModels();
    } catch (exc) { toast(String(exc.message || exc), 'error'); }
  } else if (act === 'stop') {
    try {
      await postJson('/admin/api/models/' + encodeURIComponent(m.id) + '/stop', {});
      toast('Stopped ' + m.display_name, 'good');
      await sleep(400);
      fetchModels();
    } catch (exc) { toast(String(exc.message || exc), 'error'); }
  } else if (act === 'force-stop') {
    if (!window.confirm('Force-kill the process on :' + m.port + '? This taskkills whoever holds the port (PID ' + (m.pid || '?') + ').')) return;
    try {
      await postJson('/admin/api/models/' + encodeURIComponent(m.id) + '/force-stop', {});
      toast('Force-stopped ' + m.display_name, 'good');
      await sleep(400);
      fetchModels();
    } catch (exc) { toast(String(exc.message || exc), 'error'); }
  } else if (act === 'ping') {
    try {
      const body = await postJson('/admin/api/models/' + encodeURIComponent(m.id) + '/ping', {});
      toast(m.display_name + ' · ' + body.status + ' · ' + body.latency_ms + ' ms', body.ok ? 'good' : 'error');
    } catch (exc) { toast(String(exc.message || exc), 'error'); }
  } else if (act === 'dictionary') {
    toggleDictionaryPanel(m);
  }
}

// Open/close the shared transcription-dictionary editor under a whisper row.
//
// There is exactly one dictionary, so only one editor is ever open at a
// time: opening it on any whisper row first closes any other open panel.
// That makes it visually unambiguous that Turbo and Translate edit the
// same list — you can't have two side-by-side that look independent.
function toggleDictionaryPanel(m) {
  const root = els.modelsList;
  if (!root) return;
  const li = root.querySelector('.app-item[data-id="' + cssEscape(m.id) + '"]');
  if (!li) return;
  const alreadyOpen = !!li.querySelector(':scope > .glossary-panel');
  closeAllDictionaryPanels(root);
  // Clicking the row whose panel was open just closes it (toggle off).
  if (alreadyOpen) return;
  const panel = document.createElement('div');
  panel.className = 'glossary-panel';
  li.appendChild(panel);
  mountGlossaryEditor(panel);
  const btn = li.querySelector('.icon-btn[data-act="dictionary"]');
  if (btn) btn.classList.add('active');
}

function closeAllDictionaryPanels(root) {
  root.querySelectorAll('.glossary-panel').forEach(function (p) { p.remove(); });
  root.querySelectorAll('.icon-btn[data-act="dictionary"].active')
    .forEach(function (b) { b.classList.remove('active'); });
}

function cssEscape(s) {
  return String(s).replace(/["\\]/g, '\\$&');
}

/* escapeHtml lives in api.js (sibling dedup, #211). */

function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

// "Active only" toggle — lives in the card's own collapse-summary header
// (#266) as an .icon-header-btn, same recipe as the Hub card's theme/restart
// buttons, with the toggled-on state borrowed from .app-item .icon-btn.active.
// Persisted like home-automation Plugs' show-hidden localStorage flag.
function renderActiveToggle() {
  const btn = els.modelsActiveToggle;
  if (!btn) return;
  btn.classList.toggle('active', state.modelsActiveOnly);
  btn.setAttribute('aria-pressed', state.modelsActiveOnly ? 'true' : 'false');
  btn.title = state.modelsActiveOnly
    ? 'Showing active only — click to show all'
    : 'Showing all — click to show active only';
  btn.setAttribute('aria-label', btn.title);
}

export function wireModels() {
  try {
    const stored = localStorage.getItem(MODELS_ACTIVE_ONLY_KEY);
    if (stored !== null) state.modelsActiveOnly = stored === 'true';
  } catch (_) { /* private mode */ }
  renderActiveToggle();

  if (els.modelsActiveToggle) {
    els.modelsActiveToggle.addEventListener('click', function (ev) {
      // The button lives inside <summary> — without this, clicking it
      // also fires the <details> element's native open/close toggle.
      ev.preventDefault();
      ev.stopPropagation();
      state.modelsActiveOnly = !state.modelsActiveOnly;
      try {
        localStorage.setItem(MODELS_ACTIVE_ONLY_KEY, String(state.modelsActiveOnly));
      } catch (_) { /* private mode */ }
      renderActiveToggle();
      renderModels();
    });
  }
}
