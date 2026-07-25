/* HTTP + SSE helpers, bearer-token plumbing, login overlay, toast. */

import { els, state, TOKEN_KEY } from './state.js';

// --------------------------------------------------------------- tokens
export function tokenFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const t = (params.get('token') || '').trim();
  if (!t) return null;
  params.delete('token');
  const newQuery = params.toString();
  const newUrl =
    window.location.pathname +
    (newQuery ? '?' + newQuery : '') +
    window.location.hash;
  window.history.replaceState({}, '', newUrl);
  return t;
}

export function readToken() { return localStorage.getItem(TOKEN_KEY) || ''; }
export function writeToken(t) { if (t) localStorage.setItem(TOKEN_KEY, t); }
export function clearToken() { localStorage.removeItem(TOKEN_KEY); }

export function urlWithToken(path) {
  const token = readToken();
  if (!token) return path;
  const url = new URL(path, window.location.origin);
  url.searchParams.set('token', token);
  return url.pathname + url.search + url.hash;
}

// --------------------------------------------------------------- fetch
export async function api(path, opts) {
  opts = opts || {};
  const headers = new Headers(opts.headers || {});
  const token = readToken();
  if (token) headers.set('Authorization', 'Bearer ' + token);
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    showLogin();
    throw new Error('auth required');
  }
  return res;
}

export async function jsonApi(path, opts) {
  const res = await api(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { body = null; }
  if (!res.ok) {
    const detail = (body && body.detail) || ('HTTP ' + res.status);
    const err = new Error(detail);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

// JSON POST helper.
export function postJson(path, payload) {
  return jsonApi(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
}

// JSON PUT helper.
export function putJson(path, payload) {
  return jsonApi(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
}

// JSON PATCH helper.
export function patchJson(path, payload) {
  return jsonApi(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
}

// --------------------------------------------------------------- SSE
// EventSource doesn't support custom headers, so we attach the bearer
// token via ?token=… (the BearerTokenMiddleware accepts that form).
// Returns the EventSource — caller is responsible for .close().
export function eventStream(path, handlers) {
  const url = urlWithToken(path);
  const es = new EventSource(url);
  if (handlers && handlers.message) {
    es.onmessage = function (ev) {
      let data = ev.data;
      try { data = JSON.parse(ev.data); } catch (_) { /* keep raw */ }
      handlers.message(data, ev);
    };
  }
  if (handlers && handlers.error) {
    es.onerror = handlers.error;
  }
  return es;
}

// --------------------------------------------------------------- login
export function showLogin() {
  if (!els.loginOverlay) return;
  els.loginOverlay.hidden = false;
  // Non-<dialog> overlay: hide the floating nav bar while it's open
  // (the vendored nav's body.nav-hidden hook — _vendored/nav/README.md).
  document.body.classList.add('nav-hidden');
  if (els.loginPassword) { els.loginPassword.value = ''; els.loginPassword.focus(); }
}

export function hideLogin() {
  if (els.loginOverlay) els.loginOverlay.hidden = true;
  document.body.classList.remove('nav-hidden');
}

export function wireLoginForm(onLoginSuccess) {
  if (!els.loginForm) return;
  els.loginForm.addEventListener('submit', async function (ev) {
    ev.preventDefault();
    els.loginError.hidden = true;
    const password = els.loginPassword.value;
    try {
      const res = await fetch('/admin/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });
      const body = await res.json().catch(function () { return null; });
      if (!res.ok || !body || !body.token) {
        const msg = (body && body.detail) || 'Login failed';
        els.loginError.textContent = msg;
        els.loginError.hidden = false;
        return;
      }
      writeToken(body.token);
      hideLogin();
      if (onLoginSuccess) onLoginSuccess();
    } catch (exc) {
      els.loginError.textContent = String(exc.message || exc);
      els.loginError.hidden = false;
    }
  });
}

// --------------------------------------------------------------- toast
let toastTimer = null;
export function toast(msg, kind) {
  if (!els.toast) return;
  els.toast.textContent = msg;
  els.toast.className = 'toast ' + (kind || '');
  els.toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(function () {
    els.toast.hidden = true;
  }, kind === 'error' ? 4500 : 2200);
}

// --------------------------------------------------------------- fmt
/* Shared across hub.js / models.js / telemetry.js — one definition (the
 * sibling-dedup pass of local-llm-hub#211; the copies had drifted). */
export function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]);
  });
}

/* Plain-text label for a request record's model: the requested name, the model
 * that actually served it when the hub resolved one to the other, and the host
 * that served it ("audio_transcribe → parakeet @mac-mini-m4"). The served host
 * is the dimension the #405 drill was missing — a `model=whisper` answered by
 * a machine with no whisper bound was a legitimate remote hop, and no field
 * said so. Only the audio paths populate the served pair, so ordinary chat
 * rows render exactly as before (#412). */
export function modelLabelText(rec, fallback) {
  const requested = (rec && rec.model) || '';
  const served = (rec && rec.served_model) || '';
  const host = (rec && rec.served_host) || '';
  let out = requested || fallback || '';
  if (served && served !== requested) out += ' → ' + served;
  if (host) out += ' @' + host;
  return out;
}

/* Escaped-HTML twin of modelLabelText, with the resolved parts muted. Shared
 * by the Hub tab's live-request / error lists and the Telemetry tab's trace
 * feed so a substitution can never be visible in one pane and invisible in
 * the other (#412). */
export function modelLabel(rec, fallback) {
  const requested = (rec && rec.model) || '';
  const served = (rec && rec.served_model) || '';
  const host = (rec && rec.served_host) || '';
  let html = escapeHtml(requested || fallback || '');
  if (served && served !== requested) {
    html += ' <span class="model-served">&rarr; ' + escapeHtml(served) + '</span>';
  }
  if (host) html += ' <span class="model-host">@' + escapeHtml(host) + '</span>';
  return html;
}

/* Trim the vendor prefix off an nvidia-smi GPU name ("NVIDIA GeForce RTX
 * 4080" -> "RTX 4080") — shared by the Hub sparkline labels and the
 * Machines-tab stats card (#309 sibling-dedup, same pattern as escapeHtml). */
export function shortGpu(name) {
  if (!name) return '';
  return name.replace('NVIDIA ', '').replace('GeForce ', '').trim();
}

/* HH:MM:SS from a unix-seconds timestamp; '' when absent (append a
 * placeholder at the call site where one is wanted). */
export function fmtClock(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}

/* Compact token count — 1.2k / 3.4M (shared by the Hub counters, OTel
 * leaderboard, and Code-usage tables; #215 dedup). */
export function fmtTok(n) {
  if (!n) return '—';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
  return String(n);
}

/* Latency in seconds — two decimals under 1s, one above ("0.35s", "12.4s"),
 * so the p50/p95 columns stay narrow on a phone (#215). */
export function fmtSecs(ms) {
  const n = Number(ms) || 0;
  return (n / 1000).toFixed(n < 995 ? 2 : 1) + 's';
}

/* One "in / out" token cell — merges the former In tok / Out tok columns. */
export function tokPair(inTok, outTok) {
  if (!inTok && !outTok) return '—';
  const one = function (n) { return n ? fmtTok(n) : '0'; };
  return one(inTok) + ' / ' + one(outTok);
}

/* Equivalent metered-API dollar cost — "≈ $1.23" / "≈ <$0.01" / "" when
 * zero/absent (shared by Code-usage tables and the OTel tab's Claude Code
 * panel, #215-style dedup). */
export function fmtCost(n) {
  if (!n) return '';
  if (n < 0.01) return '≈ <$0.01';
  if (n >= 1000) return '≈ $' + Math.round(n).toLocaleString();
  return '≈ $' + n.toFixed(2);
}

export function fmtBytes(n) {
  if (!Number.isFinite(n)) return '—';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
  return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

export function fmtAge(ts) {
  if (!ts) return '';
  const ms = Date.now() - new Date(ts).getTime();
  if (ms < 1000) return 'just now';
  if (ms < 60_000) return Math.floor(ms / 1000) + 's ago';
  if (ms < 3_600_000) return Math.floor(ms / 60_000) + 'm ago';
  return Math.floor(ms / 3_600_000) + 'h ago';
}
