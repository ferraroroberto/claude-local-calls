"""Resolve the base URL of a model's *owning* host when it's not this one.

A model row can declare ``host: <host-id>`` (see ``Model.host`` in
``src/model_registry.py``) to mark which host spawns/manages its process.
When the active host is a *different* one, requests for that model must be
forwarded to the owning host's own hub instead of dispatched locally. This
module is the single place that turns "which host owns this model" into
"what URL do I forward to" — every dispatch path (chat, audio, admin model
control) shares it rather than re-deriving the address.
"""

from __future__ import annotations

import os
from typing import Optional

from . import remote_stats
from .host_profile import get_host, hub_port
from .host_profile import resolve as resolve_host
from .model_registry import Model


def remote_base_url_for_host(host_id: Optional[str]) -> Optional[str]:
    """Hub base URL of ``host_id`` (e.g. ``http://192.168.1.11:8000``, no
    trailing slash, no ``/v1``) when it is remote relative to the active host —
    ``None`` when ``host_id`` is empty, names the active host, or names a host
    with no dialable address configured.

    Host-level analogue of :func:`remote_base_url` for admin calls addressed to
    a *host* rather than a single model (e.g. the startup-profile API, #352).

    The address is resolved through ``remote_stats.dial_address`` (#396): the
    LAN ``address:`` while it answers, the host's ``tailscale:`` MagicDNS name
    when the LAN path is dead. Hosts with no ``tailscale:`` name never probe
    and behave exactly as before; when both paths are down the LAN URL is
    returned so connect errors surface unchanged.
    """
    if not host_id:
        return None
    active = resolve_host()
    if host_id == active.id:
        return None
    owner = get_host(host_id)
    if owner is None:
        return None
    address = remote_stats.dial_address(owner)
    if not address:
        return None
    return f"http://{address}:{hub_port()}"


def remote_base_url(model: Model) -> Optional[str]:
    """Owning host's hub base URL (e.g. ``http://192.168.1.11:8000``, no
    trailing slash, no ``/v1``) when ``model`` is remote relative to the
    active host — ``None`` when it's local (no owning host, or it matches
    the active host) or when the owning host has no ``address`` configured.

    Ownership is the *effective* owner (#342): for a single-host row that is
    statically ``model.host`` exactly as before; for a multi-host chain it is
    whichever candidate the failover tracker currently holds, so requests
    follow the model when it fails over/back.
    """
    from .model_failover import effective_owner

    return remote_base_url_for_host(effective_owner(model))


def remote_auth_token(owning_host_id: str) -> Optional[str]:
    """Optional bearer token to send when calling a remote host's hub.

    Env-var-driven, one var per target host id (``LOCAL_LLM_HUB_TOKEN_<ID>``,
    id uppercased with ``-`` -> ``_``) — unset by default. The primary
    cross-host auth story is the receiving hub's own ``extra_allowlist``
    (trusted LAN IP, no token exchange needed); this is a defensive extra
    for setups that don't rely on IP allowlisting.
    """
    env_name = "LOCAL_LLM_HUB_TOKEN_" + owning_host_id.upper().replace("-", "_")
    return os.environ.get(env_name) or None


def remote_auth_token_for_model(model: Model) -> Optional[str]:
    """Bearer token for the host that *currently* owns ``model`` (#342) — the
    per-model companion to :func:`remote_auth_token`, so every dispatch path
    sends the token matching the hub the request is actually proxied to.
    ``None`` for locally-served models and token-less setups.
    """
    from .model_failover import effective_owner

    owner = effective_owner(model)
    return remote_auth_token(owner) if owner else None
