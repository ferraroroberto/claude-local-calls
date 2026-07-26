/* Models tab — "Fleet summary" card (issue #354, reworked read-only in #431).
 *
 * A per-machine summary over GET /admin/api/fleet-placement: for each fleet
 * host — the models running there right now (live state), the models it
 * could load but isn't running (chain members, including on-demand rows),
 * and its estimated capacity (GPU-VRAM estimate vs declared ceiling, total
 * RAM where known). Zero controls beyond the collapse: placement is edited
 * per model in the Models card (#424) or config/models.yaml — the old
 * toggle grid and its PATCH surface were retired in #430/#431.
 *
 * A powered-off machine still renders its summary — the registry keeps its
 * placement and the reconcile loop applies it when the box powers up, so an
 * unreachable host is a deferred-apply state, never an error. Same five
 * async lifecycle states as the Machines tab (design.md): loading / ready /
 * empty / stale / error.
 */

import { els, state } from './state.js';
import { jsonApi, escapeHtml } from './api.js';
import { icon } from './_vendored/icons/icons.js';
import { emptyStateEl } from './_vendored/empty-state/empty-state.js';

export async function fetchFleetPlacement() {
  try {
    const body = await jsonApi('/admin/api/fleet-placement');
    state.fleetPlacement = body;
    state.fleetPlacementState = (body.hosts || []).length ? 'ready' : 'empty';
    state.fleetPlacementUpdated = Date.now();
    renderFleetPlacement();
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    // Good data from an earlier fetch → stale (keep + label it), else error.
    state.fleetPlacementState = state.fleetPlacement ? 'stale' : 'error';
    renderFleetPlacement();
  }
}

// -------------------------------------------------------------------- render
/* A small, low-emphasis device hint for CPU-resident rows (piper, -ng
 * whisper rows, a chain's degraded cpu tier — #387/#431): a model that
 * contributes 0 to the VRAM sum reads as intentionally exempt rather than
 * an omission. Omitted (not guessed) for every other row, including other
 * 0-VRAM rows that aren't actually CPU (e.g. parakeet on the Mac's ANE). */
function deviceHint(model) {
  if (!model || !model.device) return '';
  return ' <span class="muted small">· ' + escapeHtml(model.device) + '</span>';
}

function hostChip(host) {
  if (host.local) return { cls: 'good', label: 'This machine' };
  if (host.reachable) return { cls: 'good', label: 'Online' };
  return { cls: '', label: 'Offline' };
}

function hostGlyph(host) {
  return host.icon || (host.local ? 'monitor' : 'server');
}

// MB → a compact "X.X GB" label for VRAM figures.
function fmtGb(mb) {
  return (Number(mb || 0) / 1024).toFixed(1) + ' GB';
}

// MB → a whole-GB label for RAM totals (128 GB, not 128.0 GB).
function fmtGbWhole(mb) {
  return Math.round(Number(mb || 0) / 1024) + ' GB';
}

/* An advisory VRAM-overcommit warning (issue #375). Shown only when the host
 * declares a `vram_mb` ceiling and the estimated footprint of its desired +
 * running models exceeds it — a heads-up, never a hard block. CPU-resident
 * rows are already excluded from the sum server-side (#431). Hosts with no
 * declared ceiling (Apple-silicon unified memory, managed-only boxes) never
 * carry it. */
function capacityWarnEl(host) {
  if (!host.capacity_warning) return null;
  const p = document.createElement('p');
  p.className = 'fleet-host-note fleet-capacity-warn small';
  p.title = 'Estimated GPU-VRAM footprint of this host’s desired + running models '
    + '(CPU-resident rows excluded) exceeds its ' + fmtGb(host.vram_mb)
    + ' ceiling. Advisory only.';
  p.innerHTML = icon('triangle-alert')
    + '<span>Over VRAM capacity — ~' + fmtGb(host.est_vram_mb)
    + ' est / ' + fmtGb(host.vram_mb) + ' GPU</span>';
  return p;
}

/* One model's inline name for a summary list: display name + a low-emphasis
 * cpu hint + a distinct "external" marker for a live backend served by a
 * process the hub adopted but never spawned (voice-transcriber's
 * whisper-server on the mutex-shared :8090 — #431: the summary must not
 * claim the hub runs it). */
function modelNameHtml(model, id, external) {
  const name = model ? model.display_name : id;
  let html = '<span class="fleet-model-name">' + escapeHtml(name) + '</span>' + deviceHint(model);
  if (external) {
    html += ' <span class="muted small" title="Served by an external process the hub'
      + ' adopted on a mutex-shared port — not started by this hub">· external</span>';
  }
  return html;
}

/* One summary line: a quiet left label ("Running", "Loadable", "Capacity")
 * and a right-aligned value — the roles-card row idiom, no new CSS. */
function summaryRow(label, valueHtml, title) {
  const li = document.createElement('li');
  li.className = 'startup-row';
  if (title) li.title = title;
  li.innerHTML =
    '<span class="startup-row-label muted small">' + escapeHtml(label) + '</span>'
    + '<span class="roles-row-value">' + valueHtml + '</span>';
  return li;
}

function capacityHtml(host) {
  const parts = [];
  if (host.vram_mb) {
    parts.push('GPU ~' + fmtGb(host.est_vram_mb) + ' / ' + fmtGb(host.vram_mb));
  } else if (host.est_vram_mb) {
    parts.push('GPU ~' + fmtGb(host.est_vram_mb) + ' est');
  }
  if (host.ram_mb) parts.push('RAM ' + fmtGbWhole(host.ram_mb));
  return parts.length
    ? '<span class="muted small">' + parts.join(' · ') + '</span>'
    : '';
}

function buildHostGroup(host) {
  const group = document.createElement('div');
  group.className = 'fleet-host';

  const head = document.createElement('div');
  head.className = 'fleet-host-head';
  const chip = hostChip(host);
  head.innerHTML =
    '<span class="fleet-host-name">' + icon(hostGlyph(host))
    + '<span>' + escapeHtml(host.display_name || host.id) + '</span></span>'
    + '<span class="hub-live-status ' + chip.cls + '"><span class="dot"></span><span>'
    + escapeHtml(chip.label) + '</span></span>';
  group.appendChild(head);

  const warn = capacityWarnEl(host);
  if (warn) group.appendChild(warn);

  const eligible = host.eligible || [];

  // A host with nothing placeable — an honest note instead of empty rows.
  // (openclaw today: enrolled as a managed machine, serves no models.)
  if (!eligible.length) {
    const note = document.createElement('p');
    note.className = 'fleet-host-note muted small';
    note.textContent = 'No models placeable on this machine.';
    group.appendChild(note);
    return group;
  }

  // A manageable host that's powered off: its desired set is kept in the
  // registry and applies on power-up — a deferred state, never a failure.
  if (!host.local && !host.reachable) {
    const note = document.createElement('p');
    note.className = 'fleet-host-note muted small';
    note.textContent = 'Offline — desired models apply when this machine powers up.';
    group.appendChild(note);
  }

  const byId = {};
  eligible.forEach(function (m) { byId[m.id] = m; });
  const running = host.running || [];
  const runningSet = {};
  running.forEach(function (id) { runningSet[id] = true; });
  const externalSet = {};
  (host.external || []).forEach(function (id) { externalSet[id] = true; });
  const loadable = eligible.filter(function (m) { return !runningSet[m.id]; });

  const list = document.createElement('ul');
  list.className = 'startup-list';

  const runningHtml = running.length
    ? running.map(function (id) {
        return modelNameHtml(byId[id], id, !!externalSet[id]);
      }).join('<span class="muted"> · </span>')
    : '<span class="muted small">none</span>';
  list.appendChild(summaryRow('Running', runningHtml,
    'Models serving on this machine right now (live state)'));

  if (loadable.length) {
    const loadableHtml = '<span class="muted small">'
      + loadable.map(function (m) {
          return escapeHtml(m.display_name) + (m.device ? ' · ' + escapeHtml(m.device) : '');
        }).join(' · ')
      + '</span>';
    list.appendChild(summaryRow('Loadable', loadableHtml,
      'Enabled chain members this machine could serve but is not running — '
      + 'includes on-demand rows that load on first request'));
  }

  const cap = capacityHtml(host);
  if (cap) {
    list.appendChild(summaryRow('Capacity', cap,
      'Static estimates (not live telemetry): GPU-VRAM footprint of desired + running '
      + 'models (CPU-resident rows excluded) vs the declared ceiling; total system RAM '
      + 'where documented'));
  }

  group.appendChild(list);
  return group;
}

function renderFleetPlacement() {
  const root = els.fleetPlacementBody;
  if (!root) return;
  const ds = state.fleetPlacementState;
  const note = els.fleetPlacementStaleNote;

  if (ds === 'loading') {
    root.replaceChildren(emptyStateEl('server', 'Reading fleet summary…'));
    if (note) note.hidden = true;
    return;
  }
  if (ds === 'error') {
    root.replaceChildren(emptyStateEl('triangle-alert', 'Could not read the fleet summary.', {
      actionLabel: 'Retry',
      onAction: function () {
        state.fleetPlacementState = 'loading';
        renderFleetPlacement();
        fetchFleetPlacement();
      },
    }));
    if (note) note.hidden = true;
    return;
  }

  const hosts = (state.fleetPlacement && state.fleetPlacement.hosts) || [];
  if (!hosts.length) {
    root.replaceChildren(emptyStateEl('server', 'No machines in the fleet registry.'));
    if (note) note.hidden = true;
    return;
  }

  const frag = document.createDocumentFragment();
  hosts.forEach(function (h) { frag.appendChild(buildHostGroup(h)); });
  root.replaceChildren(frag);

  // Stale: keep the last-known summary, label it, per the design.md async contract.
  if (note) {
    if (ds === 'stale' && state.fleetPlacementUpdated) {
      const t = new Date(state.fleetPlacementUpdated);
      const hh = String(t.getHours()).padStart(2, '0');
      const mm = String(t.getMinutes()).padStart(2, '0');
      note.textContent = 'Last updated ' + hh + ':' + mm + ' · live data unavailable';
      note.hidden = false;
    } else {
      note.hidden = true;
    }
  }
}
