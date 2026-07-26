/* Hub tab — "Model decisions" card (issue #373).
 *
 * Folded by default, styled with the same vendored disclosure idiom as the
 * Hub tab's other collapsibles (_vendored/disclosure). Mirrors app-launcher's
 * System Map lazy-load discipline: no polling while collapsed, both small
 * JSON payloads (GET /admin/api/roles + GET /admin/api/fleet-placement) are
 * fetched only when the <details> is expanded, and re-fetched on every
 * subsequent open so a role/placement change made elsewhere shows up next
 * time you look, without a background timer.
 *
 * Two sections inside: a compact role → model list, and a compact per-host
 * placement summary — the fuller read-only Fleet summary lives on the
 * Models tab (#431); this card links to it rather than duplicating it.
 */

import { els } from './state.js';
import { jsonApi, escapeHtml } from './api.js';
import { setTab } from './tabs.js';

// -------------------------------------------------------------- formatting
// role_key -> a human label without hardcoding every role name: split on the
// audio.* dotted nesting and underscores, title-case each word. Works for
// today's five roles and whatever #342 (or a future role) adds later.
function roleLabel(key) {
  return String(key).split('.').map(function (part) {
    return part.split('_').map(function (w) {
      return w ? w.charAt(0).toUpperCase() + w.slice(1) : w;
    }).join(' ');
  }).join(' · ');
}

function setStatus(msg) {
  if (!els.rolesStatus) return;
  els.rolesStatus.textContent = msg || '';
  els.rolesStatus.hidden = !msg;
}

// ------------------------------------------------------------------ render
function renderRoles(rolesBody) {
  const list = els.rolesList;
  if (!list) return;
  const roles = (rolesBody && rolesBody.roles) || {};
  const keys = Object.keys(roles);
  list.innerHTML = '';
  if (!keys.length) {
    list.innerHTML = '<li class="muted small">No roles configured in config/models.yaml.</li>';
    return;
  }
  keys.forEach(function (key) {
    const entry = roles[key] || {};
    const li = document.createElement('li');
    li.className = 'startup-row';
    const value = escapeHtml(entry.display_name || entry.model_id || '—')
      + ((entry.fallback || []).length
        ? ' <span class="muted small">(+ fallback: ' + escapeHtml(entry.fallback.join(', ')) + ')</span>'
        : '');
    li.innerHTML =
      '<span class="startup-row-label"><span class="fleet-model-name">' + escapeHtml(roleLabel(key)) + '</span></span>'
      + '<span class="roles-row-value">' + value + '</span>';
    list.appendChild(li);
  });
}

function renderPlacement(placementBody) {
  const list = els.rolesPlacementList;
  if (!list) return;
  const hosts = (placementBody && placementBody.hosts) || [];
  list.innerHTML = '';
  if (!hosts.length) {
    list.innerHTML = '<li class="muted small">No fleet hosts configured.</li>';
    return;
  }
  hosts.forEach(function (h) {
    const placed = h.placed || [];
    const li = document.createElement('li');
    li.className = 'startup-row';
    const chipCls = h.local ? 'good' : (h.reachable ? 'good' : '');
    const chipLabel = h.local ? 'this machine' : (h.reachable ? 'online' : 'offline');
    li.innerHTML =
      '<span class="startup-row-label"><span class="fleet-model-name">' + escapeHtml(h.display_name || h.id) + '</span>'
      + '<span class="hub-live-status ' + chipCls + '"><span class="dot"></span><span>' + escapeHtml(chipLabel) + '</span></span></span>'
      + '<span class="roles-row-value muted small">' + (placed.length ? escapeHtml(placed.join(', ')) : 'none placed') + '</span>';
    list.appendChild(li);
  });
}

// ------------------------------------------------------------------- fetch
async function loadRolesCard() {
  setStatus('Loading…');
  try {
    const [rolesBody, placementBody] = await Promise.all([
      jsonApi('/admin/api/roles'),
      jsonApi('/admin/api/fleet-placement'),
    ]);
    renderRoles(rolesBody);
    renderPlacement(placementBody);
    setStatus('');
  } catch (exc) {
    if (String(exc.message) === 'auth required') return;
    setStatus('Could not load role/placement data.');
  }
}

// ------------------------------------------------------------------ wiring
function goToFleetPlacement() {
  setTab('models');
  // The Models pane is `hidden` until the nav flips it visible — give the
  // browser a frame before scrolling so scrollIntoView measures a laid-out
  // target rather than a still-hidden one.
  requestAnimationFrame(function () {
    const card = document.getElementById('fleetPlacementCard');
    if (!card) return;
    card.open = true;
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
}

export function wireRolesCard() {
  if (els.rolesCard) {
    els.rolesCard.addEventListener('toggle', function () {
      if (els.rolesCard.open) loadRolesCard();
    });
  }
  if (els.rolesViewPlacementBtn) {
    els.rolesViewPlacementBtn.addEventListener('click', function (ev) {
      // Lives inside the collapse body, not <summary>, but stop propagation
      // defensively so a future markup move can't accidentally re-toggle it.
      ev.stopPropagation();
      goToFleetPlacement();
    });
  }
}
