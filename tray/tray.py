"""System-tray launcher — owns the hub process + (optional) Cloudflare tunnel.

Mobile-first design means there's no real desktop UI to surface; the
tray exists so launching ``tray.bat`` brings the hub up alongside
Windows login without keeping a console window open.

Menu (mirrors app-launcher's tray):

    🛰 Hub  http://127.0.0.1:8000          — status header, disabled
    🚀 Open admin                          — opens /admin in the default browser
    📋 Copy local URL                      — clipboard the local URL
    📋 Copy LAN URL                        — clipboard http://<lan-ip>:8000
    📋 Copy Cloudflare URL                 — clipboard https://<tunnel>?token=…
    🧠 Models ▸                            — toggle per-enabled-local-model
    🔄 Restart hub                         — stop + start so a new pull is picked up
    ℹ Status                               — popup with hub state
    Quit

Replaces ``tray/app.py``, ``tray/log_window.py``, and ``tray/config.py``
in one file. The tk log window is gone — logs live in the admin's Hub
tab now that the hub IS the process serving its own admin.
"""

from __future__ import annotations

import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import psutil
import pystray
import yaml

from src.host_profile import CONFIG_PATH, hub_port
from src.model_registry import Model, local_models
from src.no_window import NO_WINDOW
from src.server_process import WIN_NEW_GROUP
from src.webapp_config import append_auth_token, ensure_auth_token, load_webapp_config

from .icon import COLOR_RUNNING, COLOR_STARTING, COLOR_STOPPED, make_icon_image
from .single_instance import cross_process_lock

try:
    from winotify import Notification as _WinToast  # type: ignore
except ImportError:
    _WinToast = None

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Webapp/cloudflared.yml is the canonical place a cloudflared
# config lives — see app-launcher for the same convention. We don't
# write it for the user; we only read the first ingress hostname.
TUNNEL_CONFIG_PATH = PROJECT_ROOT / "webapp" / "cloudflared.yml"


# --------------------------------------------------------------- TrayConfig

@dataclass(frozen=True)
class TrayConfig:
    autostart_hub: bool = True
    hub_ready_timeout_s: float = 30.0


def load_tray_config() -> TrayConfig:
    # Model autostart is not a tray concern: hub startup derives it from the
    # registry (model_registry.desired_model_ids, #430) — the legacy
    # `tray.autostart_models` list is gone.
    raw_path = Path(CONFIG_PATH)
    try:
        cfg = yaml.safe_load(raw_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        logger.warning("⚠️ could not read %s: %s — using tray defaults", raw_path, exc)
        return TrayConfig()

    section = cfg.get("tray") or {}
    return TrayConfig(
        autostart_hub=bool(section.get("autostart_hub", True)),
        hub_ready_timeout_s=float(section.get("hub_ready_timeout_s", 30.0)),
    )


# --------------------------------------------------------------- HubProcess

class HubProcess:
    """Owns the hub subprocess. Adopt-or-spawn semantics — same as the
    legacy :mod:`src.server_process` but here in the tray so the hub
    *is* a child of the tray."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @staticmethod
    def base_url() -> str:
        return f"http://127.0.0.1:{hub_port()}"

    def is_running(self) -> bool:
        with self._lock:
            return self.proc is not None and self.proc.poll() is None

    def is_reachable(self, timeout: float = 0.5) -> bool:
        return _tcp_reachable("127.0.0.1", hub_port(), timeout)

    def adopted(self) -> bool:
        return (not self.is_running()) and self.is_reachable(0.3)

    def start(self) -> Tuple[bool, str]:
        # Race-safe adopt-or-spawn (project-scaffolding#39): serialize the
        # is_running/is_reachable check-then-Popen across processes so two trays
        # can't both spawn the hub. The loser re-checks inside the lock and
        # adopts the now-listening hub. self._lock is in-process only;
        # cross_process_lock adds the cross-process guarantee and fails open so
        # it never blocks startup. Primitive vendored byte-identical from scaffold.
        with cross_process_lock(rf"Global\local-llm-hub-hub-start-{hub_port()}"):
            if self.is_running():
                return True, "already running"
            if self.is_reachable(0.3):
                return True, "adopted external hub"

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            creationflags = WIN_NEW_GROUP
            try:
                with self._lock:
                    self.proc = subprocess.Popen(
                        [sys.executable, "-m", "src.server"],
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        creationflags=creationflags,
                    )
            except Exception as exc:  # noqa: BLE001
                return False, f"failed to launch: {exc}"
            return True, f"started (pid={self.proc.pid})"

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            p = self.proc
        if p is None or p.poll() is not None:
            self.proc = None
            if self.is_reachable(0.3):
                # Something is listening on the port but it isn't a subprocess
                # we spawned (adopted external hub) — there is nothing here we
                # can tear down. Surfacing this distinctly from "not running"
                # lets callers (e.g. _restart_worker) avoid reporting a false
                # "restarted" once start() re-adopts the same stale process.
                return False, "adopted (not ours to stop)"
            return False, "not running"
        try:
            if sys.platform == "win32":
                try:
                    import signal
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                    # Give uvicorn's own shutdown handler (src/server.py's
                    # "shutdown" event) a chance to run before escalating —
                    # terminate() is an immediate hard TerminateProcess on
                    # Windows, so calling it right away never lets the
                    # graceful signal land.
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            return False, f"error stopping: {exc}"
        self.proc = None
        return True, "stopped"

    def wait_ready(self, timeout_s: float = 30.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_reachable(0.5):
                return True
            time.sleep(0.5)
        return False


# --------------------------------------------------------------- TrayApp

EVT_OPEN_ADMIN = "open_admin"
EVT_COPY_LOCAL = "copy_local"
EVT_COPY_LAN = "copy_lan"
EVT_COPY_TUNNEL = "copy_tunnel"
EVT_RESTART_HUB = "restart_hub"
EVT_TOGGLE_MODEL = "toggle_model"
EVT_REFRESH = "refresh"
EVT_STATUS = "status"
EVT_QUIT = "quit"


class TrayApp:
    """The tray's loop runs in pystray's own thread; we feed it via a
    queue to keep menu callbacks fast."""

    def __init__(self, cfg: TrayConfig) -> None:
        self.cfg = cfg
        self.events: "queue.Queue[object]" = queue.Queue()
        self.hub = HubProcess()
        self._icon: Optional[pystray.Icon] = None
        self._stop_event = threading.Event()
        # Remote-owned rows (m.host set to a *different* host, e.g. a
        # Mac-hosted model cross-enabled here for hub routing) are excluded
        # from the tray menu — the tray only launches/stops processes on
        # this machine; use the admin webapp's Models tab to control a
        # remote host's backend (it proxies there automatically).
        self._models: List[Model] = [
            m for m in local_models() if m.backend in ("openai", "whisper", "tts")
        ]
        # The webapp config holds the bearer token we append to copied URLs.
        # Generate one on first boot so we never have an unprotected non-
        # loopback hub by accident.
        self.webapp_cfg = ensure_auth_token()

    # --------------------------------------------------------------- run

    def run(self) -> int:
        self._icon = pystray.Icon(
            "local_llm_hub_tray",
            make_icon_image(COLOR_STARTING),
            "Local LLM Hub",
            menu=self._build_menu(),
        )

        # Pump events on a worker thread so the pystray icon thread isn't
        # blocked. The pystray.Icon.run() call enters a Win32 message loop.
        threading.Thread(target=self._event_pump, daemon=True).start()
        if self.cfg.autostart_hub:
            threading.Thread(target=self._autostart_worker, daemon=True).start()
        threading.Thread(target=self._color_loop, daemon=True).start()

        try:
            self._icon.run()
        finally:
            self._shutdown()
        return 0

    def _shutdown(self) -> None:
        # Hub owns the model children — stopping the hub cascades to
        # them via its shutdown handler (src/server.py).
        if self.hub.is_running():
            try:
                self.hub.stop()
            except Exception as exc:
                logger.debug("hub stop failed: %s", exc)
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
        os._exit(0)

    # ----------------------------------------------------------------- menu

    def _build_menu(self) -> pystray.Menu:
        items = [self._build_model_item(m) for m in self._models]
        if not items:
            items.append(pystray.MenuItem(
                "(no models enabled on this host)", None, enabled=False,
            ))
        models_submenu = pystray.Menu(*items)

        def hub_label(_item: pystray.MenuItem) -> str:
            if self.hub.is_running():
                return f"🛰 Hub  {self.hub.base_url()}"
            if self.hub.adopted():
                return f"🛰 Hub  {self.hub.base_url()}  (adopted)"
            return "🛰 Hub  (stopped)"

        def tunnel_label(_item: pystray.MenuItem) -> str:
            host = _read_tunnel_hostname(TUNNEL_CONFIG_PATH)
            if host:
                return f"📋 Copy Cloudflare URL  ({host})"
            return "📋 Copy Cloudflare URL  (not configured)"

        return pystray.Menu(
            pystray.MenuItem(hub_label, None, enabled=False),
            pystray.MenuItem("🚀 Open admin", lambda: self._enqueue(EVT_OPEN_ADMIN), default=True),
            pystray.MenuItem("📋 Copy local URL", lambda: self._enqueue(EVT_COPY_LOCAL)),
            pystray.MenuItem("📋 Copy LAN URL", lambda: self._enqueue(EVT_COPY_LAN)),
            pystray.MenuItem(
                tunnel_label,
                lambda: self._enqueue(EVT_COPY_TUNNEL),
                enabled=lambda _i: _read_tunnel_hostname(TUNNEL_CONFIG_PATH) is not None,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("🧠 Models", models_submenu),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("🔄 Restart hub", lambda: self._enqueue(EVT_RESTART_HUB)),
            pystray.MenuItem("ℹ Status", lambda: self._enqueue(EVT_STATUS)),
            pystray.MenuItem("🔄 Refresh menu", lambda: self._enqueue(EVT_REFRESH)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda: self._enqueue(EVT_QUIT)),
        )

    def _build_model_item(self, model: Model) -> pystray.MenuItem:
        model_id = model.id
        model_port = model.port

        def on_toggle() -> None:
            self._enqueue((EVT_TOGGLE_MODEL, model_id))

        def is_checked(_item: pystray.MenuItem) -> bool:
            # The hub now owns model subprocesses (we drive it via the
            # admin API). All we can do from here is probe the port —
            # which is enough for the checkbox state.
            return _tcp_reachable("127.0.0.1", model_port, 0.3) if model_port else False

        return pystray.MenuItem(
            f"{model.display_name}  (:{model.port})",
            on_toggle,
            checked=is_checked,
        )

    # ----------------------------------------------------------- event loop

    def _enqueue(self, event: object) -> None:
        self.events.put(event)

    def _event_pump(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self.events.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._handle_event(event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tray event %r raised %s", event, exc)

    def _handle_event(self, event: object) -> None:
        if isinstance(event, tuple):
            kind = event[0]
        else:
            kind = event

        if kind == EVT_OPEN_ADMIN:
            self._open_admin()
        elif kind == EVT_COPY_LOCAL:
            self._copy_url(self.hub.base_url() + "/admin/")
        elif kind == EVT_COPY_LAN:
            lan = _lan_ip()
            if lan:
                self._copy_url(f"http://{lan}:{hub_port()}/admin/")
            else:
                self._notify("LAN", "⚠️ no LAN route detected")
        elif kind == EVT_COPY_TUNNEL:
            host = _read_tunnel_hostname(TUNNEL_CONFIG_PATH)
            if not host:
                self._notify("Cloudflare", "⚠️ cloudflared.yml not configured")
                return
            url = f"https://{host}/admin/"
            self._copy_url(append_auth_token(url, self.webapp_cfg.auth_token))
        elif kind == EVT_RESTART_HUB:
            threading.Thread(target=self._restart_worker, daemon=True).start()
        elif kind == EVT_TOGGLE_MODEL:
            _, model_id = event  # type: ignore[misc]
            threading.Thread(
                target=self._toggle_model_worker, args=(model_id,), daemon=True
            ).start()
        elif kind == EVT_REFRESH:
            self._update_menu()
        elif kind == EVT_STATUS:
            self._notify(
                "Hub",
                f"{'running' if self.hub.is_running() else ('adopted' if self.hub.adopted() else 'stopped')} · "
                f"{self.hub.base_url()}",
            )
        elif kind == EVT_QUIT:
            self._stop_event.set()
            if self._icon is not None:
                self._icon.stop()

    # ------------------------------------------------------------- workers

    def _autostart_worker(self) -> None:
        ok, msg = self.hub.start()
        if not ok:
            self._notify("Hub", f"⚠️ {msg}")
            return
        self._notify("Hub", f"▶ {msg}")
        if not self.hub.wait_ready(self.cfg.hub_ready_timeout_s):
            self._notify("Hub", f"⚠️ not reachable after {self.cfg.hub_ready_timeout_s:.0f}s")
            return
        self._update_menu()
        # Model autostart runs inside the hub startup path so tray and
        # direct hub launches share one behavior.

    def _restart_worker(self) -> None:
        self._notify("Hub", "🔄 restarting…")
        stopped, stop_msg = self.hub.stop()
        if not stopped and stop_msg.startswith("adopted"):
            # The tray didn't spawn this hub, so stop() had nothing to tear
            # down — start() would just re-adopt the same stale process and
            # report success, masking a build that never actually restarted.
            self._notify(
                "Hub",
                "⚠️ hub is running outside the tray (adopted) — can't restart "
                "it from here; use tray.bat --restart or stop it manually",
            )
            return
        time.sleep(0.6)
        ok, msg = self.hub.start()
        if not ok:
            self._notify("Hub", f"⚠️ {msg}")
            return
        if not self.hub.wait_ready(self.cfg.hub_ready_timeout_s):
            self._notify("Hub", f"⚠️ not reachable after {self.cfg.hub_ready_timeout_s:.0f}s")
            return
        self._notify("Hub", "✅ restarted")
        self._update_menu()
        # Hub startup owns configured model autostart; no tray-side duplicate.

    def _toggle_model_worker(self, model_id: str) -> None:
        """Toggle a model's lifecycle via the hub's admin API.

        The hub owns the subprocess. Tray is a thin HTTP client — so
        models started from here show up as "running (managed)" in the
        admin webapp, not "adopted" (which used to happen when the tray
        was the subprocess parent).
        """
        model = next((m for m in self._models if m.id == model_id), None)
        if model is None or model.port is None:
            return
        if _tcp_reachable("127.0.0.1", model.port, 0.3):
            # Try a graceful Stop first; fall back to Force stop if the
            # hub doesn't own it (e.g. a leftover from a previous run).
            ok, msg = self._admin_request("POST", f"/admin/api/models/{model_id}/stop")
            if ok:
                self._notify(model_id, "■ Stopped")
            elif "not running" in (msg or "").lower():
                ok2, msg2 = self._admin_request(
                    "POST", f"/admin/api/models/{model_id}/force-stop"
                )
                self._notify(model_id, "💀 Force-stopped" if ok2 else f"⚠️ {msg2}")
            else:
                self._notify(model_id, f"⚠️ {msg}")
        else:
            self._start_model_worker(model_id, autostart=False)
        self._update_menu()

    def _start_model_worker(self, model_id: str, *, autostart: bool) -> None:
        prefix = "(autostart) " if autostart else ""
        ok, msg = self._admin_request("POST", f"/admin/api/models/{model_id}/start")
        if not ok:
            # "already running" is a 409 — surface as info, not error.
            if "already running" in (msg or "").lower():
                self._notify(f"{prefix}{model_id}", f"✅ {msg}")
            else:
                self._notify(f"{prefix}{model_id}", f"⚠️ {msg}")
            return
        self._notify(f"{prefix}{model_id}", "▶ Starting…")
        threading.Thread(
            target=self._wait_model_ready_worker,
            args=(model_id,),
            daemon=True,
        ).start()

    def _wait_model_ready_worker(self, model_id: str) -> None:
        model = next((m for m in self._models if m.id == model_id), None)
        if model is None or model.port is None:
            return
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if _tcp_reachable("127.0.0.1", model.port, 0.8):
                self._notify(model_id, "✅ Ready")
                self._update_menu()
                return
            time.sleep(1.0)
        self._notify(model_id, "⚠️ not reachable after 120s")
        self._update_menu()

    def _admin_request(self, method: str, path: str) -> Tuple[bool, str]:
        """Talk to the hub's /admin API on loopback. Bearer-token middleware
        bypasses on loopback so we don't need to attach the token here.
        Returns ``(ok, detail)``; ``ok`` is True on 2xx, False on any error.
        """
        url = self.hub.base_url() + path
        try:
            r = httpx.request(method, url, timeout=15.0)
        except httpx.HTTPError as exc:
            return False, f"hub unreachable: {exc}"
        try:
            body = r.json() if r.content else {}
        except Exception:  # noqa: BLE001
            body = {}
        if r.is_success:
            return True, str(body.get("detail") or "ok")
        return False, str(body.get("detail") or f"HTTP {r.status_code}")

    # ---------------------------------------------------- copy / open helpers

    def _open_admin(self) -> None:
        url = self.hub.base_url() + "/admin/"
        token = (self.webapp_cfg.auth_token or "").strip()
        # On the PC itself we go via loopback so the bearer-token middleware
        # exempts us — no need to include the token in the URL.
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001
            self._notify("Admin", f"⚠️ webbrowser.open failed: {exc}")
        # Always also push the LAN+token URL to the clipboard so the user
        # can paste it on their phone right after clicking.
        lan = _lan_ip()
        if lan and token:
            lan_url = append_auth_token(f"http://{lan}:{hub_port()}/admin/", token)
            try:
                _set_clipboard(lan_url)
            except Exception:
                pass

    def _copy_url(self, url: str) -> None:
        try:
            _set_clipboard(url)
            self._notify("Clipboard", f"📋 {url}")
        except Exception as exc:  # noqa: BLE001
            self._notify("Clipboard", f"⚠️ copy failed: {exc}")

    # --------------------------------------------------- icon color loop

    def _color_loop(self) -> None:
        last = None
        while not self._stop_event.is_set():
            try:
                if self.hub.is_reachable(0.4):
                    color = COLOR_RUNNING
                elif self.hub.is_running():
                    color = COLOR_STARTING
                else:
                    color = COLOR_STOPPED
                if color != last and self._icon is not None:
                    try:
                        self._icon.icon = make_icon_image(color)
                    except Exception:
                        pass
                    last = color
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0)

    def _update_menu(self) -> None:
        if self._icon is not None:
            try:
                self._icon.update_menu()
            except Exception as exc:
                logger.debug("update_menu failed: %s", exc)

    def _notify(self, title: str, message: str) -> None:
        if _WinToast is not None:
            try:
                _WinToast(app_id="local-llm-hub", title=title, msg=message).show()
                return
            except Exception as exc:
                logger.debug("winotify failed (%s) — falling back to pystray", exc)
        if self._icon is not None:
            try:
                self._icon.notify(message, title)
                return
            except Exception:
                pass
        logger.info("🔔 %s: %s", title, message)


# --------------------------------------------------------------- helpers


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def _tcp_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _read_tunnel_hostname(config_path: Path) -> Optional[str]:
    """Pull the first ``ingress[].hostname`` out of a cloudflared config.

    Returns ``None`` when the file is missing or unparseable — the tray
    treats either as "no tunnel" and disables the menu item.
    """
    if not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.debug("tunnel config unreadable: %s", exc)
        return None
    ingress = data.get("ingress")
    if not isinstance(ingress, list):
        return None
    for entry in ingress:
        if isinstance(entry, dict):
            host = entry.get("hostname")
            if host:
                return str(host)
    return None


def _set_clipboard(text: str) -> None:
    """Copy *text* to the OS clipboard. Best-effort cross-platform."""
    if sys.platform == "win32":
        # Avoid pyperclip dep — invoke clip.exe directly.
        p = subprocess.Popen(
            ["clip"], stdin=subprocess.PIPE, creationflags=NO_WINDOW,
        )
        p.communicate(input=text.encode("utf-16le"))
        return
    if sys.platform == "darwin":
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        p.communicate(input=text.encode("utf-8"))
        return
    p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
    p.communicate(input=text.encode("utf-8"))


def main() -> int:
    return TrayApp(load_tray_config()).run()

