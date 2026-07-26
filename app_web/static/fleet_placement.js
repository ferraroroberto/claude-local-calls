/* Models tab — "Fleet placement" card (issue #354, read-only since #430).
 *
 * A per-machine view over GET /admin/api/fleet-placement — the desired state
 * is now *derived* from config/models.yaml (`hosts:` chains + `startup:
 * eager|on_demand`, #430), so the grid renders it read-only: each host is a
 * group; each model in that host's derived desired set is a row with a live
 * status badge. Changing what runs where is a models.yaml edit, not a toggle
 * here — the old PATCH surface (and config/fleet_placement.json behind it)
 * was retired because it duplicated the registry. The card's full rework
 * (summary layout, on-demand rows) is the #431 follow-up.
 *
 * A powered-off machine still renders its desired models — the registry keeps
 * their placement and the reconcile loop applies it when the box powers up,
 * so an unreachable host is a deferred-apply state, never an error. Same five
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
function placedList(hostId) {
  const p = (state.fleetPlacement && state.fleetPlacement.placement) || {};
  return (p[hostId] || []).slice();
}

/* The live status badge for one desired model on one host.
 * Reachable-but-not-running is "pending" (reconcile will start it); desired
 * on an offline host is "deferred" until it powers up. */
function modelBadge(host, modelId) {
  if ((host.running || []).includes(modelId)) return ' <span class="badge good">running</span>';
  if (host.reachable) return ' <span class="badge warn">pending</span>';
  return ' <span class="badge">deferred</span>';
}

/* A small, low-emphasis device hint for statically CPU-resident models
 * (piper, whisper_translate — #387), so a row that contributes 0 to the
 * VRAM sum reads as intentionally exempt rather than an omission. Reuses
 * models.js's "· cpu" meta-text idiom (#371) — same visual language, no
 * new CSS. Omitted (not guessed) for every other row, including other
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

// MB → a compact "X.X GB" label for the capacity warning.
function fmtGb(mb) {
  return (Number(mb || 0) / 1024).toFixed(1) + ' GB';
}

/* An advisory VRAM-overcommit warning (issue #375). Shown only when the host
 * declares a `vram_mb` ceiling and its desired/running models' estimated
 * footprint exceeds it — a heads-up, never a hard block. Hosts with no
 * declared ceiling (Apple-silicon unified memory, managed-only boxes) never
 * carry it. Mirrors the muted `.fleet-host-note` offline note, tinted as a
 * warning. */
function capacityWarnEl(host) {
  if (!host.capacity_warning) return null;
  const p = document.createElement('p');
  p.className = 'fleet-host-note fleet-capacity-warn small';
  p.title = 'Estimated GPU-VRAM footprint of this host’s desired models '
    + 'exceeds its ' + fmtGb(host.vram_mb) + ' ceiling. Advisory only.';
  p.innerHTML = icon('triangle-alert')
    + '<span>Over VRAM capacity — ~' + fmtGb(host.est_vram_mb)
    + ' desired / ' + fmtGb(host.vram_mb) + ' GPU</span>';
  return p;
}

function buildModelRow(host, model, modelId) {
  const li = document.createElement('li');
  li.className = 'startup-row';

  const label = document.createElement('span');
  label.className = 'startup-row-label';
  const name = model ? model.display_name : modelId;
  label.innerHTML = '<span class="fleet-model-name">' + escapeHtml(name) + '</span>'
    + deviceHint(model)
    + modelBadge(host, modelId);
  li.appendChild(label);
  return li;
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

  const placed = placedList(host.id);

  // A host with nothing in its derived desired set — an honest note instead
  // of an empty list. A managed-only satellite runs no hub (driven directly
  // over SSH); a hub host may simply own only on-demand / no models.
  if (!placed.length) {
    const note = document.createElement('p');
    note.className = 'fleet-host-note muted small';
    note.textContent = host.runs_hub
      ? 'No always-on models for this machine — on-demand rows load on first request.'
      : 'Runs model servers directly (no hub on this machine).';
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
  (host.eligible || []).forEach(function (m) { byId[m.id] = m; });
  const list = document.createElement('ul');
  list.className = 'startup-list';
  placed.forEach(function (id) { list.appendChild(buildModelRow(host, byId[id], id)); });
  group.appendChild(list);
  return group;
}

function renderFleetPlacement() {
  const root = els.fleetPlacementBody;
  if (!root) return;
  const ds = state.fleetPlacementState;
  const note = els.fleetPlacementStaleNote;

  if (ds === 'loading') {
    root.replaceChildren(emptyStateEl('server', 'Reading fleet placement…'));
    if (note) note.hidden = true;
    return;
  }
  if (ds === 'error') {
    root.replaceChildren(emptyStateEl('triangle-alert', 'Could not read fleet placement.', {
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

  // Stale: keep the last-known grid, label it, per the design.md async contract.
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

export function wireFleetPlacement() {
  if (els.fleetPlacementRefreshBtn) {
    els.fleetPlacementRefreshBtn.addEventListener('click', function (ev) {
      // The button lives inside <summary>; stop the native details toggle.
      ev.preventDefault();
      ev.stopPropagation();
      fetchFleetPlacement();
    });
  }
}
