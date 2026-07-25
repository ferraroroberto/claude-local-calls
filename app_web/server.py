"""FastAPI sub-app for /admin — three-tab control panel for the hub.

Mounted into the main hub's FastAPI app at /admin. Built as a sub-app
rather than a sibling uvicorn so there is only one Python process and
one port (:8000) for the whole stack.

Static-asset cache busting:
  * ``.js`` / ``.css`` get ``?v=<8-hex content hash>``, ``Cache-Control:
    public, max-age=31536000, immutable``
  * ``.webmanifest`` / ``.png`` / ``.ico`` get a day of cache
  * ``index.html`` itself is ``no-cache, must-revalidate`` so the page
    always picks up the latest hashed asset URLs after a deploy
"""

from __future__ import annotations

import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from src.static_versioning import (
    compute_asset_hashes,
    fleet_hash_of,
    rewrite_js_imports,
)
from src.webapp_config import load_webapp_config
from src.webauthn_gate import WebAuthnGate

from .middleware import BearerTokenMiddleware
from .routers import auth, code_usage, diagnostics, fleet_maintenance, fleet_placement, glossary, hosts, hub, machines, misc, models, playground, roles, services, startup_profile, telemetry, version, webauthn
from .routers._helpers import STATIC_DIR

_log = logging.getLogger(__name__)

_LONG_CACHE = "public, max-age=31536000, immutable"
_DAY_CACHE = "public, max-age=86400"
_HASHED_SUFFIXES = {".js", ".css"}
_DAY_CACHE_SUFFIXES = {".webmanifest", ".png", ".ico", ".svg"}


class _VersionedStatic(StaticFiles):
    """Static mount that stamps Cache-Control + rewrites JS imports.

    JS files get their ``import './foo.js'`` calls rewritten to
    ``import './foo.js?v=<hash>'`` at serve time. Hashed assets get
    a year-long immutable cache; icons and manifest get a day; anything
    else falls back to defaults.
    """

    def __init__(self, *, directory: str, asset_hashes: Dict[str, str]) -> None:
        super().__init__(directory=directory)
        self._asset_hashes = asset_hashes

    def file_response(
        self,
        full_path: os.PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        path = Path(full_path)
        suffix = path.suffix.lower()

        if suffix == ".js":
            try:
                body = path.read_text(encoding="utf-8")
            except OSError:
                return super().file_response(full_path, stat_result, scope, status_code)
            try:
                rel_parent = path.resolve().relative_to(STATIC_DIR.resolve()).parent
            except ValueError:
                rel_parent = Path(".")
            from_dir = "" if rel_parent == Path(".") else rel_parent.as_posix()
            rewritten = rewrite_js_imports(body, self._asset_hashes, from_dir)
            media_type, _ = mimetypes.guess_type(str(path))
            return Response(
                content=rewritten,
                status_code=status_code,
                media_type=media_type or "text/javascript",
                headers={"Cache-Control": _LONG_CACHE},
            )

        response = super().file_response(full_path, stat_result, scope, status_code)
        if suffix in _HASHED_SUFFIXES:
            response.headers["Cache-Control"] = _LONG_CACHE
        elif suffix in _DAY_CACHE_SUFFIXES:
            response.headers["Cache-Control"] = _DAY_CACHE
        return response


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    webapp_cfg = load_webapp_config()

    auth.ensure_log_handler()

    app = FastAPI(
        title="Local LLM Hub — admin",
        version="0.1.0",
        lifespan=_lifespan,
        # Hide swagger for the sub-app — the parent hub still has its own
        # docs for the public /v1 surface; we don't want a duplicate.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.add_middleware(
        BearerTokenMiddleware,
        get_token=lambda: getattr(app.state.webapp_config, "auth_token", ""),
    )

    app.state.webapp_config = webapp_cfg
    app.state.webauthn_gate = WebAuthnGate()

    asset_hashes = compute_asset_hashes(STATIC_DIR)
    app.state.asset_hashes = asset_hashes
    app.state.asset_fleet_hash = fleet_hash_of(asset_hashes)
    if asset_hashes:
        _log.info(
            "ℹ️ /admin assets stamped at fleet hash %s (%d files)",
            app.state.asset_fleet_hash,
            len(asset_hashes),
        )

    if STATIC_DIR.exists():
        app.mount(
            "/static",
            _VersionedStatic(directory=str(STATIC_DIR), asset_hashes=asset_hashes),
            name="static",
        )

    app.include_router(misc.router)
    app.include_router(version.router)
    app.include_router(auth.router)
    app.include_router(webauthn.router)
    app.include_router(hub.router)
    app.include_router(models.router)
    app.include_router(startup_profile.router)
    app.include_router(fleet_placement.router)
    app.include_router(fleet_maintenance.router)
    app.include_router(roles.router)
    app.include_router(glossary.router)
    app.include_router(playground.router)
    app.include_router(services.router)
    app.include_router(hosts.router)
    app.include_router(machines.router)
    app.include_router(diagnostics.router)
    app.include_router(telemetry.router)
    app.include_router(code_usage.router, prefix="/api/code")

    return app
