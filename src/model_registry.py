"""Load models.yaml and expose typed entries filtered by the active host.

One source of truth for: which models exist, how to launch them, which
port they listen on, and how the hub should route by model-name alias.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .host_profile import HostProfile, _load_config, all_hosts, resolve as resolve_host

# ``startup:`` policy values (#422) — see the field docstring on ``Model``.
STARTUP_EAGER = "eager"
STARTUP_ON_DEMAND = "on_demand"


@dataclass(frozen=True)
class Model:
    id: str                         # short key in YAML ("qwen", "glm", "claude")
    display_name: str               # name the client sends in the `model` field
    backend: str                    # "claude" | "openai" | "gemini" | "whisper" | "tts"
    aliases: List[str] = field(default_factory=list)
    engine: Optional[str] = None
    # For backend == "gemini": when true this row is an image-*generation*
    # model (agy's built-in Imagen tool), routed through
    # POST /v1/images/generations rather than the text chat paths. There is
    # no picker entry for it — the image tool is hosted inside an ordinary
    # Gemini text session (see src/gemini_cli._IMAGE_HOST_MODEL).
    image_gen: bool = False
    # For backend == "tts" (engine "tts-server"): which synthesis engine
    # the shim loads — chatterbox, kokoro, orpheus, piper. See src/tts_engines/.
    tts_engine: Optional[str] = None
    port: Optional[int] = None
    hf_repo: Optional[str] = None
    hf_pattern: Optional[str] = None
    model_path: Optional[str] = None
    args: List[str] = field(default_factory=list)
    # Lazy-loaded engines (e.g. whisper-server-lazy) wrap the real
    # backend in a proxy that owns the child process. ``port`` stays the
    # external contract; ``internal_port`` is where the wrapped child
    # actually binds (loopback only). ``idle_seconds`` is how long the
    # child stays loaded after the last request before it's torn down.
    internal_port: Optional[int] = None
    idle_seconds: Optional[int] = None
    # A *virtual* model is an alias of another backend: it shares an existing
    # backend's ``port`` (so ``url`` already points at the running process) and
    # has no engine / weights of its own. It is never launched, downloaded, or
    # offered as a controllable process in the admin UI — it only exists to
    # route the chat shape with an ``inject_extra`` overlay. See the
    # ``qwen35_4b_nothink`` row in config/models.yaml.
    virtual: bool = False
    # Server-side defaults folded into the upstream OpenAI ``extra`` payload on
    # every /v1/chat/completions for this id (caller-sent fields win). Used by
    # the no-think alias to deliver ``chat_template_kwargs={enable_thinking:
    # false}`` to clients that can't send it themselves (e.g. Home Assistant's
    # extended_openai_conversation).
    inject_extra: Optional[Dict[str, Any]] = None
    # *Preferred* owning host — spawns/manages this model's process. Unset
    # means "whichever host resolves this row locally" (the pre-#178
    # behavior — every model was implicitly local). When set and it differs
    # from the active host, this row is a *remote* model: the active host
    # never spawns/health-checks it locally and instead proxies requests to
    # the owning host's own hub. See src/remote_proxy.py.
    #
    # Since #342 this is always ``hosts[0]`` — the first entry of the
    # ordered preference chain below. Every pre-#342 consumer that read
    # ``model.host`` as "the owner" keeps meaning "the *preferred* owner";
    # the *effective* owner (which may differ when a preferred host is down
    # and the model failed over down-chain) lives in
    # ``src.model_failover.effective_owner``.
    host: Optional[str] = None
    # Ordered host-preference chain (#342): a YAML row may declare either a
    # bare ``host: <id>`` (parsed as a one-element chain — fully backward
    # compatible) or ``hosts: [a, b, c]`` — the model's owner is the first
    # candidate that is reachable and enabled there; when it dies the model
    # fails over down the chain. Empty means "unowned / local everywhere"
    # exactly as an unset ``host:`` always did.
    hosts: List[str] = field(default_factory=list)
    # Subset of ``hosts`` flagged ``cpu: true`` in the YAML chain (#342):
    # on these hosts the model runs as a *degraded CPU offload* last resort
    # (llama-server ``-ngl 0`` / whisper ``-ng`` / tts ``--device cpu``) —
    # "up but slower" rather than "must match GPU perf". The arg rewrite is
    # applied by ``all_models()`` for the active host only.
    cpu_hosts: List[str] = field(default_factory=list)
    # Rough static GPU-VRAM footprint in MB (#375) — enough for the fleet
    # placement grid to flag a host overcommit (sum of a host's placed models
    # vs its ``HostProfile.vram_mb`` ceiling), NOT live telemetry. CPU-only /
    # off-GPU / virtual rows are 0; subscription rows leave it None. A None
    # value contributes 0 to the sum. See config/models.yaml for the estimates.
    est_vram_mb: Optional[int] = None
    # Process-lifecycle policy (#422). ``eager`` (the default — today's
    # behavior) means the model may be autostarted with the hub and kept
    # resident by the reconcile/failover loops. ``on_demand`` means nothing
    # starts it eagerly: the first request that routes to it spawns the
    # backend (the request waits for readiness — same pattern as the lazy
    # whisper proxy), and ``src.on_demand``'s idle watchdog stops it again
    # after ``idle_unload_minutes`` without traffic. Any unknown value is
    # normalized to ``eager`` so a typo can't silently disable a model.
    startup: str = STARTUP_EAGER
    # Only meaningful for ``startup: on_demand`` — minutes without a request
    # after which the idle watchdog unloads the backend. ``None`` disables
    # idle unload (the model stays up once demanded until stopped by hand).
    idle_unload_minutes: Optional[int] = None

    @property
    def host_chain(self) -> List[str]:
        """Ordered owning-host candidates — ``hosts`` (already normalized so a
        bare ``host:`` row yields a one-element chain). Empty for unowned rows.
        """
        return list(self.hosts)

    @property
    def all_names(self) -> List[str]:
        # The registry ``id`` is the canonical handle used by tools that
        # walk the YAML (the SPA Playground dropdown, swap-model, etc.).
        # Include it so /v1/models lists every name a client can send.
        names = [self.id, self.display_name, *self.aliases]
        return [n for n in dict.fromkeys(names) if n]

    @property
    def url(self) -> Optional[str]:
        return f"http://127.0.0.1:{self.port}/v1" if self.port else None


# ``_load_config()`` (imported above) is host_profile's cached YAML loader
# for config/models.yaml — both modules read the same file, so there is one
# cache rather than two kept in sync by convention.


def _parse_host_chain(row: Dict) -> Tuple[List[str], List[str]]:
    """Normalize a YAML row's host declaration into ``(hosts, cpu_hosts)`` (#342).

    Accepted shapes, in precedence order:

    * ``hosts: [a, b, {id: c, cpu: true}]`` — the ordered preference chain;
      entries are bare host-id strings or ``{id: <host>, cpu: true}`` dicts
      (``cpu`` flags a degraded CPU-offload tier, e.g. the tower last resort).
    * ``host: a`` — the pre-#342 single owner, parsed as a one-element chain
      so every existing row behaves exactly as before.
    * neither — an unowned row: empty chain, local everywhere.

    A row that declares both keeps ``hosts:`` (the superset) and ignores the
    bare ``host:`` — they'd otherwise be two sources of truth for entry one.
    """
    raw = row.get("hosts")
    if isinstance(raw, list) and raw:
        hosts: List[str] = []
        cpu_hosts: List[str] = []
        for entry in raw:
            if isinstance(entry, dict):
                host_id = str(entry.get("id") or "").strip()
                if not host_id:
                    continue
                hosts.append(host_id)
                if entry.get("cpu"):
                    cpu_hosts.append(host_id)
            elif entry:
                hosts.append(str(entry).strip())
        # De-dup preserving order — a repeated host id can't make the
        # failover engine "retry" the same dead host twice.
        seen: set = set()
        hosts = [h for h in hosts if not (h in seen or seen.add(h))]
        return hosts, [h for h in cpu_hosts if h in set(hosts)]
    single = row.get("host")
    if single:
        return [str(single)], []
    return [], []


def _parse_startup(row: Dict) -> str:
    """Normalize a row's ``startup:`` to a known policy (#422).

    Anything other than the literal ``on_demand`` reads as ``eager`` — a
    typo'd value must degrade to today's always-on behavior, never to a
    model that silently refuses to start.
    """
    raw = str(row.get("startup") or STARTUP_EAGER).strip().lower()
    return STARTUP_ON_DEMAND if raw == STARTUP_ON_DEMAND else STARTUP_EAGER


def _row_to_model(model_id: str, row: Dict) -> Model:
    hosts, cpu_hosts = _parse_host_chain(row)
    return Model(
        id=model_id,
        display_name=str(row.get("display_name") or model_id),
        backend=str(row.get("backend", "openai")),
        aliases=list(row.get("aliases", []) or []),
        engine=row.get("engine"),
        image_gen=bool(row.get("image_gen", False)),
        tts_engine=row.get("tts_engine"),
        port=int(row["port"]) if row.get("port") is not None else None,
        hf_repo=row.get("hf_repo"),
        hf_pattern=row.get("hf_pattern"),
        model_path=row.get("model_path"),
        args=list(row.get("args", []) or []),
        internal_port=int(row["internal_port"]) if row.get("internal_port") is not None else None,
        idle_seconds=int(row["idle_seconds"]) if row.get("idle_seconds") is not None else None,
        virtual=bool(row.get("virtual", False)),
        inject_extra=row.get("inject_extra") or None,
        host=hosts[0] if hosts else None,
        hosts=hosts,
        cpu_hosts=cpu_hosts,
        est_vram_mb=int(row["est_vram_mb"]) if row.get("est_vram_mb") is not None else None,
        startup=_parse_startup(row),
        idle_unload_minutes=(
            int(row["idle_unload_minutes"])
            if row.get("idle_unload_minutes") is not None else None
        ),
    )


# Engine-aware CPU-offload arg rewrites (#342): a host flagged ``cpu: true``
# in a model's chain runs it degraded-but-up. Kept as pure list-in/list-out
# so it is directly unit-testable and shared by every spawn path (the
# transform is baked into ``Model.args`` in ``all_models()``, so
# ``backend_process.build_command``, the lazy whisper proxy's child command,
# and ``run_backend`` all inherit it without knowing it exists).
_LLAMA_GPU_LAYER_FLAGS = ("-ngl", "--n-gpu-layers", "--gpu-layers")


def cpu_offload_args(engine: Optional[str], args: List[str]) -> List[str]:
    """Rewrite launch ``args`` so the backend runs CPU-only (#342)."""
    out = list(args or [])
    if engine == "llama-server":
        for i, a in enumerate(out):
            if a in _LLAMA_GPU_LAYER_FLAGS and i + 1 < len(out):
                out[i + 1] = "0"
                return out
        return [*out, "-ngl", "0"]
    if engine in ("whisper-server", "whisper-server-lazy"):
        if "-ng" in out or "--no-gpu" in out:
            return out
        return [*out, "-ng"]
    if engine == "tts-server":
        for i, a in enumerate(out):
            if a == "--device" and i + 1 < len(out):
                out[i + 1] = "cpu"
                return out
        return [*out, "--device", "cpu"]
    return out


def all_models(*, apply_cpu_offload: bool = True) -> List[Model]:
    """Every configured row.

    ``apply_cpu_offload`` (default True) bakes the active host's ``cpu: true``
    arg rewrite in — what every *spawn/args* consumer wants. Pass False to get
    the rows exactly as declared in YAML, for callers that must reason about
    other hosts: the rewrite is active-host-only, so rewritten args describe
    this box and must not be attributed fleet-wide (#405 — the placement grid
    read ``-ng`` off the rewritten row and labelled whisper "cpu" on *every*
    chain member, not just the flagged degraded tier).
    """
    cfg = _load_config()
    rows: Dict = cfg.get("models") or {}
    models = [_row_to_model(mid, row) for mid, row in rows.items()]
    # Bake the CPU-offload rewrite into rows whose chain flags the *active*
    # host ``cpu: true`` (#342) — one choke point every args consumer shares.
    # Host resolution can legitimately fail in hostless tooling contexts
    # (e.g. enumerating rows on a machine with no matching profile); the
    # rewrite is skipped there — args only matter on the host that spawns.
    try:
        active_id: Optional[str] = resolve_host().id
    except Exception:  # noqa: BLE001
        active_id = None
    if active_id and apply_cpu_offload:
        models = [
            dataclasses.replace(m, args=cpu_offload_args(m.engine, m.args))
            if active_id in m.cpu_hosts else m
            for m in models
        ]
    return models


def enabled_models(host: Optional[HostProfile] = None) -> List[Model]:
    """Return models the active host is configured to serve.

    Claude and Gemini are always enabled (subscription/CLI paths don't
    cost disk or VRAM) — openai-backed local models must appear in the
    host's `enabled` list.
    """
    profile = host or resolve_host()
    whitelist = set(profile.enabled)
    result: List[Model] = []
    for m in all_models():
        if m.backend in ("claude", "gemini") or m.id in whitelist:
            result.append(m)
    return result


def local_models(host: Optional[HostProfile] = None) -> List[Model]:
    """``enabled_models()`` filtered to rows this host may actually run —
    excludes any row whose owning chain doesn't include this host. The
    generalization of "give me the models I might spawn, manage, download
    weights for, or health-check" — every place a remote-owned row
    (cross-enabled so it *resolves* here, but proxied rather than run here)
    would otherwise be mistaken for a local one: install checks,
    port-liveness checks, spawn/inherit loops, tray menus.

    Since #342 membership is chain-based: a host anywhere in a model's
    ``hosts:`` preference list is a *candidate* runner — it must pre-stage
    the weights (install), may spawn the process (failover), and must be
    able to inherit/stop it. For a bare single-``host:`` row the chain is
    exactly ``[host]``, so this is byte-identical to the pre-#342 filter.
    Whether the model is *currently served* here is a different, dynamic
    question — ``src.model_failover.effective_owner``.
    """
    profile = host or resolve_host()
    return [m for m in enabled_models(host) if not m.hosts or profile.id in m.hosts]


# Backends that own a spawnable local process. `claude` / `gemini` are
# subscription/CLI paths with nothing to launch; a `virtual` row shares
# another model's process. Everything a `run_backend <id>` can actually
# start is one of these three engines and non-virtual.
_SPAWNABLE_BACKENDS = ("openai", "whisper", "tts")


def launchable_local_ids(host: Optional[HostProfile] = None) -> List[str]:
    """Ids of models this host can actually spawn as its own local process.

    ``local_models()`` (enabled ∧ owned-here) narrowed to rows with a
    spawnable backend, dropping virtual aliases (which share another row's
    process). This is exactly the set ``run_backend <id>`` starts without
    erroring — the single source the bulk launchers enumerate so they can
    never drift from the active host's ``enabled:`` contract.
    """
    return [
        m.id for m in local_models(host)
        if m.backend in _SPAWNABLE_BACKENDS and not m.virtual
    ]


def hub_peer_ids(active_id: Optional[str] = None) -> List[str]:
    """Ids of every declared host — other than ``active_id`` — that runs its
    own hub (issue #372).

    "Runs a hub" reuses the exact same test ``app_web/routers/fleet_placement.py``
    already applies per host row (``runs_hub = bool(launchable_local_ids(profile))``)
    rather than re-deriving it: a host with at least one launchable local
    model spawns a hub process, so a managed-only machine with an empty
    ``enabled:`` list (e.g. ``openclaw``) is correctly excluded. This is the
    peer set the Services card's Wake/Sync rows enumerate — mac-mini-m4 and
    gaming today, any future hub-running satellite automatically once it
    declares a non-empty ``enabled:``.
    """
    active = active_id if active_id is not None else resolve_host().id
    return [
        h.id for h in all_hosts()
        if h.id != active and launchable_local_ids(h)
    ]


def preferred_owner(model: Model) -> Optional[str]:
    """The first chain host that exists and enables ``model`` — the static
    preferred owner (#342), independent of live reachability. ``None`` for an
    unowned row (empty chain) or a chain with no eligible member. The dynamic
    counterpart (where the model is *currently* served after a failover) is
    ``src.model_failover.effective_owner``.
    """
    from .host_profile import get_host

    for host_id in model.hosts:
        profile = get_host(host_id)
        if profile is not None and model.id in profile.enabled:
            return host_id
    return None


def desired_model_ids(host: Optional[HostProfile] = None) -> List[str]:
    """The models ``host`` should be running right now, derived from the
    registry (#430) — no separate desired-state file.

    A model belongs to a host's desired running set when it is launchable
    there (enabled ∧ chain-member ∧ spawnable ∧ non-virtual, exactly
    ``launchable_local_ids``), declares ``startup: eager`` (#422 —
    ``on_demand`` rows are never desired; ``src.on_demand`` owns their
    lifecycle), and is either unowned (an empty chain means "local
    everywhere", the pre-#178 behavior) or preferred-owned by this host
    (``preferred_owner`` — the first *eligible* chain member, so a chain
    whose head names an unknown/non-enabling host falls through to the next
    member). A degraded ``cpu: true`` tier deeper in a chain is just a
    non-preferred member: it pre-stages weights and can inherit via
    failover, but the model is never *desired* there.

    This one derivation feeds both hub autostart on boot
    (``server_lifecycle._autostart_configured_backends`` — how a satellite
    that was offline applies its models when it powers up) and the fleet
    reconcile loop (``fleet_reconcile.reconcile_once`` via
    :func:`desired_placement`), replacing the retired
    ``config/fleet_placement.json`` + ``startup_profile.json`` ``models``
    lists (#430) — the registry's ``hosts:`` chains + ``startup:`` policy
    are the sole source of placement truth.
    """
    profile = host or resolve_host()
    launchable = set(launchable_local_ids(profile))
    desired: List[str] = []
    for m in all_models(apply_cpu_offload=False):
        if m.id not in launchable or m.startup != STARTUP_EAGER:
            continue
        if m.hosts and preferred_owner(m) != profile.id:
            continue
        desired.append(m.id)
    return desired


def desired_placement() -> Dict[str, List[str]]:
    """Fleet-wide ``{host_id: [model_id, ...]}`` desired state, derived from
    the registry (#430) — :func:`desired_model_ids` per declared host. Hosts
    with an empty desired set are omitted (nothing to run means no reason for
    the reconcile loop to probe or wake them)."""
    placement: Dict[str, List[str]] = {}
    for h in all_hosts():
        ids = desired_model_ids(h)
        if ids:
            placement[h.id] = ids
    return placement


def cpu_resident_map() -> Dict[str, set]:
    """``{host_id: {model_id, ...}}`` — rows that hold **no GPU VRAM** on that
    host (#431). The single source every VRAM-capacity sum keys off (the fleet
    summary card, the per-row host budgets, the on-demand headroom check): a
    CPU-resident row must never count against a GPU ceiling.

    Two independent ways a row lands on CPU:

    * **Always, on every host** — piper's shim hardcodes CPU unconditionally
      (``src/tts_engines/piper.py``, #371) and a ``whisper-server`` row that
      *declares* ``-ng`` never touches the GPU (see ``whisper_translate``).
    * **On one host only** — a failover chain's degraded last-resort tier
      (``{id: tower, cpu: true}``, #342): GPU on the preferred members,
      CPU-offloaded on the flagged one.

    Reads ``all_models(apply_cpu_offload=False)`` deliberately: the registry
    bakes the CPU rewrite in for the *active* host, so the default view would
    show ``-ng`` on this box's row and smear that verdict across every other
    chain member (#405). Deliberately **not** ``est_vram_mb == 0`` alone —
    ``parakeet`` is also 0 but runs on the Mac's ANE via CoreML, a real (if
    non-discrete-VRAM) device, not "cpu".
    """
    out: Dict[str, set] = {h.id: set() for h in all_hosts()}
    for m in all_models(apply_cpu_offload=False):
        always_cpu = m.tts_engine == "piper" or (
            m.engine == "whisper-server" and "-ng" in m.args
        )
        for host_id, ids in out.items():
            if always_cpu or host_id in m.cpu_hosts:
                ids.add(m.id)
    return out


def resolve(name: str, host: Optional[HostProfile] = None) -> Optional[Model]:
    """Look up a model by any of its names — registry id, display_name, or alias.

    Accepting ``id`` matters for tools that drive the hub off the YAML
    directly: the SPA Playground dropdown sends ``m.id`` (the YAML key),
    swap-model references ids when rewiring roles, and the hub's own
    ``run_backend`` picks the entry by id. Without this, every model
    whose id is not also listed in ``aliases`` 400'd on the Playground.
    """
    name = name.strip()
    for m in enabled_models(host):
        if name == m.id or name == m.display_name or name in m.aliases:
            return m
    return None


def resolve_any(name: str) -> Optional[Model]:
    """Look up a model by any of its names across **every** configured row,
    ignoring the active host's ``enabled:`` whitelist.

    The host-blind companion to :func:`resolve`. Answers a different question:
    not "can I serve this here?" but "is this name one of *our* model ids at
    all?". The audio proxy needs exactly that distinction to keep an explicit
    ``model=`` strict (issue #412): a configured id that can't be served here
    must fail loudly rather than fall through to the role chain and be
    answered by a different model, while a name the registry has never heard
    of (a client-side placeholder like OpenAI's ``whisper-1``) is not a model
    request at all and still addresses the role.

    Matching is **case-insensitive** (and whitespace-tolerant), unlike
    :func:`resolve`. That asymmetry is deliberate: this function answers "did
    the caller *name* one of our models?", and someone who typed ``Whisper``
    named whisper — treating it as an unknown id would drop them onto the role
    chain and hand back a 200 produced by a different model, exactly the
    silence #412 abolishes. :func:`resolve` stays exact because its callers
    (dispatch, ``/v1/models``, the admin model control) key off the canonical
    name; pair the two by resolving *this* row's ``id`` there.
    """
    key = (name or "").strip().lower()
    if not key:
        return None
    for m in all_models():
        if key in {n.lower() for n in m.all_names}:
            return m
    return None


def audio_role_chain(role_key: str) -> List[str]:
    """Ordered model-id chain for an audio role (issue #348).

    Reads ``roles.audio.<role_key>`` from models.yaml and returns
    ``[model_id, *fallback]`` — the primary followed by the ordered failover
    models the audio proxy tries in turn when the primary's backend is
    unavailable. ``role_key`` is ``"transcribe"`` / ``"translate"`` / ``"speech"``.

    A role with only ``model_id`` (no ``fallback``) yields a one-element chain —
    identical to the pre-#348 single-target behaviour. Returns ``[]`` when the
    role is not configured, so the caller can fall back to its own heuristic.
    Duplicate ids are collapsed (order-preserving) so a config that repeats the
    primary in ``fallback`` never makes the proxy retry the same dead backend.
    """
    cfg = _load_config()
    role = ((cfg.get("roles") or {}).get("audio") or {}).get(role_key)
    if not isinstance(role, dict):
        return []
    chain: List[str] = []
    primary = role.get("model_id")
    if primary:
        chain.append(str(primary))
    fallback = role.get("fallback")
    if isinstance(fallback, list):
        chain.extend(str(x) for x in fallback if x)
    seen: set[str] = set()
    return [mid for mid in chain if not (mid in seen or seen.add(mid))]

