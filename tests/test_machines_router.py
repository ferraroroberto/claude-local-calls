"""Unit tests for the Machines console (#309) — src/machine_console.py +
app_web/routers/machines.py.

Exercises the pure capability/inventory logic (host enrollment, which
actions each machine may offer, RDP generation) and the router's contract:
the status/self endpoint shapes, the RDP download, and the destructive
power-action guard (the active hub host and SSH-less peers are refused).
Network/SSH-touching probes are monkeypatched so the suite stays hermetic.
"""

from __future__ import annotations

import asyncio
import dataclasses

from fastapi.testclient import TestClient

from src import machine_console as mc
from src import server as server_mod
from src import ssh_exec
from src.host_profile import HostProfile, all_hosts, get_host, resolve


def _client() -> TestClient:
    return TestClient(server_mod.app)


def _run(coro):
    """Run a coroutine on a fresh worker-thread loop (see test_services_router)."""
    import threading

    bucket: dict = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            bucket["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised in caller
            bucket["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "error" in bucket:
        raise bucket["error"]
    return bucket.get("value")


# ------------------------------------------------------------ inventory / config


def test_all_hosts_enrolls_managed_machines():
    ids = {h.id for h in all_hosts()}
    assert {"tower", "mac-mini-m4", "openclaw", "gaming"} <= ids


def test_openclaw_has_ssh_and_rdp():
    h = get_host("openclaw")
    assert h is not None
    assert h.can_ssh is True  # address + ssh_user
    assert h.rdp and h.rdp["address"] == "192.168.0.11"
    assert h.tailscale == "asus-linux.tail1121fd.ts.net"  # tailnet node (#335)
    assert h.dormant is False


def test_gaming_is_live_ssh_host():
    """The old dormant `tower` node is now `gaming`, a live Linux satellite
    (#323): SSH power actions on, and its own tailnet node name (#332 corrected
    this from the historical `tower` alias to `gaming-linux`)."""
    h = get_host("gaming")
    assert h is not None
    assert h.dormant is False
    assert h.can_ssh is True  # address + ssh_user → power actions
    assert h.rdp and h.rdp["address"] == "192.168.0.16"
    assert h.tailscale == "gaming-linux.tail1121fd.ts.net"  # its own tailnet node (#332)


def test_tower_is_the_hub_box_not_the_old_satellite():
    """The `tower` id is now the hub box (renamed from `pc-cuda`, #335), not the
    old dormant node that became `gaming` (#323) — the id was recycled. Guard
    that `tower` resolves to the win32 hub and the satellite is still `gaming`,
    so the two never get conflated again."""
    h = get_host("tower")
    assert h is not None
    assert h.platform == "win32"                      # the hub box, not the linux satellite
    assert h.default is True                           # resolves as the active win32 host
    assert h.tailscale == "tower.tail1121fd.ts.net"    # owns the tower magic-DNS + Langfuse
    assert get_host("gaming") is not None              # the satellite kept its own id


# ------------------------------------------------------------ actions capability


def test_actions_host_offers_nothing():
    host = get_host(resolve().id)
    acts = mc._actions_for(host, is_host=True)
    assert acts == {"reboot": False, "shutdown": False, "rdp": False, "ssh_terminal": False, "wake": False}


def test_actions_ssh_peer_offers_power_and_terminal():
    """A confirmed-reachable peer (``reachable=True``) gets the full set."""
    acts = mc._actions_for(get_host("mac-mini-m4"), is_host=False, reachable=True)
    assert acts["reboot"] and acts["shutdown"] and acts["ssh_terminal"]


def test_actions_gaming_offers_power_terminal_and_rdp():
    acts = mc._actions_for(get_host("gaming"), is_host=False, reachable=True)
    assert acts == {"reboot": True, "shutdown": True, "ssh_terminal": True, "rdp": True, "wake": True}


# ------------------------------------------------------- actions: offline gating (#388)


def test_actions_unreachable_peer_only_wake():
    """A peer the TCP probe just found down (``reachable=False``) must gray
    out every SSH/RDP action — they can only fail against a box nothing is
    listening on. Wake survives, since that's the one action meant to reach
    a box that's down. gaming has both SSH+RDP configured and a mac row, so
    it exercises every field at once."""
    acts = mc._actions_for(get_host("gaming"), is_host=False, reachable=False)
    assert acts == {"reboot": False, "shutdown": False, "ssh_terminal": False, "rdp": False, "wake": True}


def test_actions_unreachable_peer_without_mac_has_no_wake_either():
    """openclaw has no `mac:` row — offline gating still leaves wake False,
    matching 'only Wake enabled (when it has a mac)' from the issue."""
    acts = mc._actions_for(get_host("openclaw"), is_host=False, reachable=False)
    assert acts == {"reboot": False, "shutdown": False, "ssh_terminal": False, "rdp": False, "wake": False}


def test_actions_reachable_none_gates_same_as_false():
    """``reachable=None`` (dormant / no probe path / a probe error) must gate
    identically to a confirmed-down peer — anything short of a confirmed-up
    TCP probe disables the SSH/RDP actions. Also the default, so callers
    that omit ``reachable`` entirely stay gated rather than fail open."""
    acts_none = mc._actions_for(get_host("gaming"), is_host=False, reachable=None)
    acts_default = mc._actions_for(get_host("gaming"), is_host=False)
    acts_false = mc._actions_for(get_host("gaming"), is_host=False, reachable=False)
    assert acts_none == acts_default == acts_false


def test_actions_wake_true_only_for_mac_equipped_peers():
    """openclaw has no `mac:` row (laptop, no wired NIC) — no wake action;
    mac-mini-m4 and gaming both have one configured (#356)."""
    assert mc._actions_for(get_host("openclaw"), is_host=False)["wake"] is False
    assert mc._actions_for(get_host("mac-mini-m4"), is_host=False)["wake"] is True
    assert mc._actions_for(get_host("gaming"), is_host=False)["wake"] is True


def test_actions_wake_false_for_hub_host_even_with_mac():
    """`tower` has a `mac:` row configured but is the active hub host in this
    test environment — waking the box the hub already runs on is meaningless
    (#356), so `wake` must be False when `is_host=True` regardless of MAC."""
    host = get_host(resolve().id)
    assert host.mac  # tower does carry a mac row
    assert mc._actions_for(host, is_host=True)["wake"] is False


# ------------------------------------------------------------ card runs_hub


def test_card_runs_hub_true_for_model_hosts():
    """tower and mac-mini-m4 each run their own hub (non-empty `enabled`) —
    the Machines-tab footnote (#337) points a viewer at a peer's own /admin
    only when this is true."""
    for host_id in ("tower", "mac-mini-m4"):
        card = mc._card_base(get_host(host_id), is_host=False)
        assert card["runs_hub"] is True, host_id


def test_card_runs_hub_false_for_managed_only_machines():
    """openclaw is managed-only (#309) — no `enabled` models, so it never runs
    this hub. gaming graduated to a hub-running voice-pair satellite (#323)."""
    assert mc._card_base(get_host("openclaw"), is_host=False)["runs_hub"] is False
    assert mc._card_base(get_host("gaming"), is_host=False)["runs_hub"] is True


# ------------------------------------------------------------- card mac (#397)


def test_card_base_surfaces_configured_mac():
    """The static `mac:` config field is on the card itself now, not just
    used internally to gate the Wake action (#397's acceptance criterion)."""
    assert mc._card_base(get_host("gaming"), is_host=False)["mac"] == get_host("gaming").mac


def test_card_base_mac_none_for_host_without_one():
    """openclaw has no `mac:` row (no wired NIC) — the card field is None,
    not a missing key, so the SPA can render conditionally without a KeyError."""
    assert mc._card_base(get_host("openclaw"), is_host=False)["mac"] is None


# -------------------------------------------------------------- card ip (#408)


def test_card_base_surfaces_configured_ip():
    """The static `address:` config field is surfaced on the card as `ip`."""
    assert mc._card_base(get_host("gaming"), is_host=False)["ip"] == get_host("gaming").address


def test_probe_machine_self_ip_uses_configured_address(monkeypatch):
    """When the active host's own config row has an `address:` (as every row
    does today), the self card's `ip` is that value — no lan_ip() lookup
    needed."""
    monkeypatch.setattr(mc.system_stats, "gpu_stats", lambda: [])

    def _boom():
        raise AssertionError("lan_ip() should not be called when host.address is set")

    monkeypatch.setattr(mc, "lan_ip", _boom)
    host = get_host("gaming")
    card = _run(mc._probe_machine(host, host.id))
    assert card["ip"] == host.address


def test_probe_machine_self_ip_falls_back_to_lan_ip(monkeypatch):
    """A host row with no configured `address:` (host_profile.py's documented
    case — nothing dials it) still gets an IP on its own card, via the same
    lan_ip() UDP-connect trick the Hub tab's LAN URL already uses."""
    monkeypatch.setattr(mc.system_stats, "gpu_stats", lambda: [])
    monkeypatch.setattr(mc, "lan_ip", lambda: "10.20.30.40")
    host = dataclasses.replace(get_host("gaming"), address=None)
    card = _run(mc._probe_machine(host, host.id))
    assert card["ip"] == "10.20.30.40"


# --------------------------------------------------- _probe_machine card wiring (#388)


def test_probe_machine_offline_peer_grays_every_action_but_wake(monkeypatch):
    """End-to-end through `_probe_machine`: a peer the TCP probe reports down
    gets `state: "down"` and every SSH/RDP action disabled — only wake (mac
    is configured on gaming) survives."""
    async def _unreachable(host):
        return None  # located_address: no candidate address answered (#396)

    monkeypatch.setattr(mc.remote_stats, "located_address", _unreachable)
    card = _run(mc._probe_machine(get_host("gaming"), "tower"))
    assert card["state"] == "down"
    assert card["actions"] == {
        "reboot": False, "shutdown": False, "ssh_terminal": False, "rdp": False, "wake": True,
    }
    assert card["network"] is None and card["flaky"] is None  # (#397) no probe on a down peer


def test_probe_machine_online_peer_keeps_full_actions(monkeypatch):
    """An online peer is unaffected by the new gating — full action set."""
    async def _reachable(host):
        return host.address  # located_address: the LAN path answered (#396)

    async def _fake_collect(host):
        return {"uptime_seconds": 42}  # no "network" key — an older/unsupported probe result

    monkeypatch.setattr(mc.remote_stats, "located_address", _reachable)
    monkeypatch.setattr(mc.remote_stats, "collect", _fake_collect)
    card = _run(mc._probe_machine(get_host("gaming"), "tower"))
    assert card["state"] == "up"
    assert card["via_tailscale"] is False  # reached on the LAN — no badge (#396)
    assert card["actions"] == {
        "reboot": True, "shutdown": True, "ssh_terminal": True, "rdp": True, "wake": True,
    }
    assert card["network"] is None  # (#397) stats.get("network") degrades cleanly when absent


def test_probe_machine_online_peer_carries_network_and_flaky(monkeypatch):
    """The reachable branch wires the live `network` block and the
    connection-health flag straight through onto the card (#397)."""
    async def _reachable(host):
        return host.address

    net = {"iface": "wlo1", "mac": "0c:7a:15:c0:0b:16", "wireless": True,
           "ssid": "MOVISTAR_9CC0", "signal_dbm": -86.0}

    async def _fake_collect(host):
        return {"uptime_seconds": 42, "network": net}

    monkeypatch.setattr(mc.remote_stats, "located_address", _reachable)
    monkeypatch.setattr(mc.remote_stats, "collect", _fake_collect)
    monkeypatch.setattr(mc.remote_stats, "connection_flaky", lambda host: True)
    card = _run(mc._probe_machine(get_host("openclaw"), "tower"))
    assert card["network"] == net
    assert card["flaky"] is True


def test_probe_machine_dormant_peer_grays_every_action_but_wake():
    """A dormant node is never live-probed but must gate identically to a
    confirmed-down peer — only wake, when mac-equipped."""
    host = HostProfile(
        id="sleeper", platform="linux", enabled=[], dormant=True, mac="aa:bb:cc:dd:ee:ff",
    )
    card = _run(mc._probe_machine(host, "tower"))
    assert card["state"] == "dormant"
    assert card["actions"] == {
        "reboot": False, "shutdown": False, "ssh_terminal": False, "rdp": False, "wake": True,
    }
    assert card["network"] is None and card["flaky"] is None  # (#397) never probed while dormant


def test_probe_machine_this_machine_unchanged(monkeypatch):
    """The active hub host still offers nothing, regardless of #388."""
    monkeypatch.setattr(mc.system_stats, "gpu_stats", lambda: [])  # skip nvidia-smi
    active_id = resolve().id
    card = _run(mc._probe_machine(get_host(active_id), active_id))
    assert card["state"] == "self"
    assert card["actions"] == {
        "reboot": False, "shutdown": False, "rdp": False, "ssh_terminal": False, "wake": False,
    }
    assert card["network"] is None and card["flaky"] is None  # (#397) scoped to peers only


# ------------------------------------------------------------------------ RDP


def test_rdp_file_generated_for_peer_with_target():
    generated = mc.rdp_file("openclaw")
    assert generated is not None
    filename, content = generated
    assert filename == "openclaw.rdp"
    assert "full address:s:192.168.0.11" in content
    assert "username:s:openclaw" in content


def test_rdp_file_none_for_host_without_target():
    assert mc.rdp_file(resolve().id) is None
    assert mc.rdp_file("does-not-exist") is None


# --------------------------------------------------------------- self snapshot


def test_self_snapshot_shape(monkeypatch):
    monkeypatch.setattr(mc.system_stats, "gpu_stats", lambda: [])  # skip nvidia-smi
    snap = _run(mc.self_snapshot())
    for key in ("cpu", "ram", "gpus", "disk", "uptime_seconds"):
        assert key in snap, snap
    assert isinstance(snap["uptime_seconds"], float)
    assert "version" not in snap  # build identity is not shown on machine cards


# ------------------------------------------------------- remote stats parsing


def test_remote_stats_parse_linux():
    from src import remote_stats

    raw = (
        "uptime 70254\ncpu 2\nmem_total_mb 15800\nmem_used_mb 1239\n"
        "disk_total_kb 95536548\ndisk_used_kb 40927072\n"
        "gpu_name NVIDIA GeForce MX250\ngpu_used_mb 4\ngpu_total_mb 2048\ngpu_util 0\n"
    )
    s = remote_stats._parse(raw)
    assert s["uptime_seconds"] == 70254
    assert s["cpu"] == {"percent": 2.0}
    assert s["ram"]["total_gb"] == 15.43 and s["ram"]["percent"] == 7.8
    assert s["disk"]["percent"] == 42.8
    assert s["gpus"][0]["name"] == "NVIDIA GeForce MX250"
    assert s["gpus"][0]["total_mb"] == 2048.0 and s["gpus"][0]["vram_percent"] == 0.2


def test_remote_stats_parse_darwin_no_gpu():
    from src import remote_stats

    raw = "uptime 1636838\ncpu 2.69\nmem_total_mb 16384\nmem_used_mb 8983\ndisk_total_kb 239362496\ndisk_used_kb 17383264\n"
    s = remote_stats._parse(raw)
    assert s["cpu"] == {"percent": 2.69}
    assert s["ram"]["percent"] == 54.8
    assert s["gpus"] == []  # macOS has no nvidia-smi — no GPU gauge


def test_remote_stats_parse_network_wired():
    """A wired peer (gaming) reports iface/mac and wireless=False, no
    SSID/signal (#397)."""
    from src import remote_stats

    raw = "uptime 34782\ncpu 0\nnet_iface enp4s0\nnet_mac d4:5d:64:d6:7e:a0\nnet_wireless 0\n"
    s = remote_stats._parse(raw)
    assert s["network"] == {
        "iface": "enp4s0", "mac": "d4:5d:64:d6:7e:a0",
        "wireless": False, "ssid": None, "signal_dbm": None,
    }


def test_remote_stats_parse_network_wireless_with_ssid_and_signal():
    """A Wi-Fi peer (openclaw) reports SSID + signal alongside wireless=True."""
    from src import remote_stats

    raw = (
        "uptime 34787\ncpu 0\nnet_iface wlo1\nnet_mac 0c:7a:15:c0:0b:16\n"
        "net_wireless 1\nnet_ssid MOVISTAR_9CC0\nnet_signal_dbm -86\n"
    )
    s = remote_stats._parse(raw)
    assert s["network"] == {
        "iface": "wlo1", "mac": "0c:7a:15:c0:0b:16",
        "wireless": True, "ssid": "MOVISTAR_9CC0", "signal_dbm": -86.0,
    }


def test_remote_stats_parse_network_wireless_without_iw():
    """A Wi-Fi peer with no `iw` installed still reports wireless=True — just
    without SSID/signal (degrade gracefully, never a blank/broken chip)."""
    from src import remote_stats

    raw = "uptime 1\ncpu 0\nnet_iface wlan0\nnet_mac aa:bb:cc:dd:ee:ff\nnet_wireless 1\n"
    s = remote_stats._parse(raw)
    assert s["network"] == {
        "iface": "wlan0", "mac": "aa:bb:cc:dd:ee:ff",
        "wireless": True, "ssid": None, "signal_dbm": None,
    }


def test_remote_stats_parse_network_absent_when_probe_yields_nothing():
    """No `net_iface` line at all (probe failed / platform unsupported) —
    `network` is None, same degrade contract as `ram`/`disk`/`gpus`."""
    from src import remote_stats

    raw = "uptime 1636838\ncpu 2.69\nmem_total_mb 16384\nmem_used_mb 8983\n"
    s = remote_stats._parse(raw)
    assert s["network"] is None


def test_darwin_stats_cmd_survives_missing_sudo():
    """The macOS Wi-Fi/RSSI probe rides `sudo -n wdutil info`; the guarding
    `if`/`else` structure must still exit clean when sudo has no drop-in
    (matches the Linux `if … then … fi` exit-0 contract, #397)."""
    from src import remote_stats

    cmd = remote_stats._DARWIN_STATS_CMD
    assert "sudo -n wdutil info" in cmd
    assert cmd.rstrip().endswith("fi")


def test_linux_stats_cmd_survives_missing_nvidia_smi():
    """A Linux peer with no nvidia-smi must still yield CPU/mem/disk, not a
    blank card (#329). ``_run_ssh`` drops all stdout when the remote command
    exits non-zero, so the optional GPU probe must never poison the exit code.

    The GPU probe must be the exit-0 ``if … then … fi`` form: with nvidia-smi
    absent the condition is false and the ``if`` yields 0, so the earlier
    gauges survive. The old ``command -v nvidia-smi && nvidia-smi`` chain made
    the whole script exit non-zero and blanked the card — assert it is gone.
    (Real end-to-end exit-0 confirmed live over SSH against the gaming box.)"""
    from src import remote_stats

    cmd = remote_stats._LINUX_STATS_CMD
    assert "if command -v nvidia-smi >/dev/null 2>&1; then" in cmd
    assert cmd.rstrip().endswith("fi")
    # the old, bug-causing exit-propagating form must be gone
    assert "nvidia-smi >/dev/null 2>&1 &&" not in cmd


def test_remote_stats_reachable_false_without_address():
    from src import remote_stats

    # A host with no LAN address is not reachable via the TCP probe. Every
    # enrolled host now has an address, so this uses a synthetic address-less
    # profile to keep the guard covered (#323).
    addressless = HostProfile(id="nowhere", platform="linux", enabled=[])
    assert remote_stats.reachable(addressless) is False


def test_reachable_warms_up_on_idle_first_syn(monkeypatch):
    """An idled peer drops the first SYN and answers the second — the warm-up
    retry must report it up, not down (#333). The first port scan fails, the
    retry succeeds; sleep is stubbed so the test stays fast."""
    from src import remote_stats

    calls = {"n": 0}

    def fake_scan(address):
        calls["n"] += 1
        return calls["n"] >= 2  # first pass fails, retry succeeds

    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", fake_scan)
    monkeypatch.setattr(remote_stats.time, "sleep", lambda *_: None)
    host = HostProfile(id="idle", platform="linux", address="10.0.0.9", enabled=[])
    assert remote_stats.reachable(host) is True
    assert calls["n"] == 2  # exactly one warm-up retry, no more


def test_reachable_false_when_both_passes_fail(monkeypatch):
    """A genuinely-off box fails both passes — still down, and the warm-up does
    not loop forever (#333)."""
    from src import remote_stats

    calls = {"n": 0}

    def always_fail(address):
        calls["n"] += 1
        return False

    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", always_fail)
    monkeypatch.setattr(remote_stats.time, "sleep", lambda *_: None)
    host = HostProfile(id="off", platform="linux", address="10.0.0.9", enabled=[])
    assert remote_stats.reachable(host) is False
    assert calls["n"] == 2  # one initial pass + one warm-up retry, then give up


# --------------------------------------------------- connection health (#397)


def test_connection_flaky_none_before_any_successful_locate():
    """A host that has never been located successfully in this process
    reports flaky=None (not True/False) — genuinely unknown, not "clean"."""
    from src import remote_stats

    host = HostProfile(id="never-seen-397", platform="linux", address="10.0.0.9", enabled=[])
    assert remote_stats.connection_flaky(host) is None


def test_connection_flaky_true_when_first_pass_missed(monkeypatch):
    """The #333 warm-up retry answering (first pass missed) marks the peer
    flaky for its next card render — the whole point of the signal."""
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_liveness_cache", {})
    monkeypatch.setattr(remote_stats, "_last_needed_retry", {})
    calls = {"n": 0}

    def fake_scan(address):
        calls["n"] += 1
        return calls["n"] >= 2  # first pass fails, retry succeeds

    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", fake_scan)
    monkeypatch.setattr(remote_stats.time, "sleep", lambda *_: None)
    host = HostProfile(id="flaky-397", platform="linux", address="10.0.0.9", enabled=[])
    assert remote_stats.locate(host) == "10.0.0.9"
    assert remote_stats.connection_flaky(host) is True


def test_connection_flaky_false_when_first_pass_answers(monkeypatch):
    """A clean first-pass answer marks the peer NOT flaky — no false alarm on
    a rock-solid link."""
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_liveness_cache", {})
    monkeypatch.setattr(remote_stats, "_last_needed_retry", {})
    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", lambda address: True)
    host = HostProfile(id="solid-397", platform="linux", address="10.0.0.9", enabled=[])
    assert remote_stats.locate(host) == "10.0.0.9"
    assert remote_stats.connection_flaky(host) is False


def test_connection_flaky_unchanged_on_a_failed_locate(monkeypatch):
    """A subsequent failed locate() (box now genuinely down) must not erase
    the last known health reading — flaky is a proxy computed only when a
    probe actually succeeds."""
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_liveness_cache", {})
    monkeypatch.setattr(remote_stats, "_last_needed_retry", {})
    monkeypatch.setattr(remote_stats.time, "sleep", lambda *_: None)
    host = HostProfile(id="was-solid-397", platform="linux", address="10.0.0.9", enabled=[])
    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", lambda address: True)
    assert remote_stats.locate(host) == "10.0.0.9"
    assert remote_stats.connection_flaky(host) is False
    monkeypatch.setattr(remote_stats, "_probe_liveness_ports", lambda address: False)
    assert remote_stats.locate(host) is None
    assert remote_stats.connection_flaky(host) is False  # unchanged, not cleared


# ----------------------------------- liveness cache + concurrent port probe (#369)


def test_is_reachable_caches_per_host_within_ttl(monkeypatch):
    """The Machines tab and the fleet-placement grid both probe the same hosts
    on the same page load — within the TTL the second call must be served from
    the cache (no re-probe), and the cache is keyed per host id (#369)."""
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_liveness_cache", {})
    calls = {"n": 0}

    def fake_locate(host):
        calls["n"] += 1
        return host.address  # the probe seam is locate() since #396

    monkeypatch.setattr(remote_stats, "locate", fake_locate)
    a = HostProfile(id="peer-a", platform="linux", address="10.0.0.9", enabled=[])
    b = HostProfile(id="peer-b", platform="linux", address="10.0.0.10", enabled=[])
    assert _run(remote_stats.is_reachable(a)) is True
    assert _run(remote_stats.is_reachable(a)) is True  # cache hit — no re-probe
    assert calls["n"] == 1
    assert _run(remote_stats.is_reachable(b)) is True  # different id — own probe
    assert calls["n"] == 2


def test_is_reachable_reprobes_after_ttl_expiry(monkeypatch):
    """After the TTL lapses the cached answer must not be trusted — the next
    call re-probes and surfaces the new state (#369)."""
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_liveness_cache", {})
    calls = {"n": 0}

    def fake_locate(host):
        calls["n"] += 1
        # up on the first probe, down on the re-probe (locate seam, #396)
        return host.address if calls["n"] == 1 else None

    monkeypatch.setattr(remote_stats, "locate", fake_locate)
    clock = {"now": 1000.0}
    monkeypatch.setattr(remote_stats.time, "monotonic", lambda: clock["now"])
    host = HostProfile(id="flappy", platform="linux", address="10.0.0.9", enabled=[])
    assert _run(remote_stats.is_reachable(host)) is True
    clock["now"] += remote_stats._LIVENESS_CACHE_TTL_S + 0.1
    assert _run(remote_stats.is_reachable(host)) is False  # expired — re-probed
    assert calls["n"] == 2


def test_probe_liveness_ports_true_when_one_port_accepts(monkeypatch):
    """The concurrent pass is still an OR over the liveness ports — one
    accepting port (RDP here, SSH refused) reads as up (#369)."""
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_probe_port", lambda address, port: port == 3389)
    assert remote_stats._probe_liveness_ports("10.0.0.9") is True


def test_probe_liveness_ports_false_when_none_accept(monkeypatch):
    from src import remote_stats

    monkeypatch.setattr(remote_stats, "_probe_port", lambda address, port: False)
    assert remote_stats._probe_liveness_ports("10.0.0.9") is False


# --------------------------------------------------------------------- endpoints


def test_machines_status_endpoint_shape(monkeypatch):
    async def _canned():
        return {
            "active_id": "tower",
            "machines": [{"id": "tower", "state": "self", "actions": {}}],
        }

    monkeypatch.setattr(mc, "machines_status", _canned)
    r = _client().get("/admin/api/machines/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "active_id" in body and isinstance(body["machines"], list)


def test_machines_self_endpoint_shape(monkeypatch):
    monkeypatch.setattr(mc.system_stats, "gpu_stats", lambda: [])
    r = _client().get("/admin/api/machines/self")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cpu" in body and "disk" in body and "uptime_seconds" in body


def test_rdp_endpoint_downloads_for_peer():
    r = _client().get("/admin/api/machines/openclaw/rdp")
    assert r.status_code == 200, r.text
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "full address:s:192.168.0.11" in r.text


def test_rdp_endpoint_404_for_host():
    r = _client().get(f"/admin/api/machines/{resolve().id}/rdp")
    assert r.status_code == 404, r.text


# ------------------------------------------------------------ power-action guard


def test_reboot_refuses_active_host():
    r = _client().post(f"/admin/api/machines/{resolve().id}/reboot")
    assert r.status_code == 400, r.text
    assert "hub host" in r.json()["detail"]


def test_reboot_404_unknown_machine():
    r = _client().post("/admin/api/machines/nope/reboot")
    assert r.status_code == 404, r.text


def test_shutdown_refuses_ssh_less_peer(monkeypatch):
    # A peer with an rdp target but no ssh_user has no power channel. No real
    # host is SSH-less anymore (#323), so inject a synthetic one at the router's
    # host lookup.
    import app_web.routers.machines as machines_router

    ghost = HostProfile(id="ghost", platform="linux", enabled=[],
                        rdp={"address": "10.0.0.9", "user": "x"})
    monkeypatch.setattr(machines_router, "get_host",
                        lambda hid: ghost if hid == "ghost" else get_host(hid))
    r = _client().post("/admin/api/machines/ghost/shutdown")
    assert r.status_code == 400, r.text
    assert "SSH" in r.json()["detail"]


def test_reboot_success_path(monkeypatch):
    from src import remote_bootstrap

    async def _ok(host_id):
        return {"ok": True, "detail": f"reboot scheduled on {host_id}"}

    monkeypatch.setattr(remote_bootstrap, "reboot_host", _ok)
    r = _client().post("/admin/api/machines/mac-mini-m4/reboot")
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


def test_reboot_failure_bubbles_502(monkeypatch):
    from src import remote_bootstrap

    async def _fail(host_id):
        return {"ok": False, "detail": "ssh unreachable"}

    monkeypatch.setattr(remote_bootstrap, "reboot_host", _fail)
    r = _client().post("/admin/api/machines/mac-mini-m4/reboot")
    assert r.status_code == 502, r.text


# --------------------------------------------------------------- wake (#356)


def test_wake_refuses_hub_host():
    r = _client().post(f"/admin/api/machines/{resolve().id}/wake")
    assert r.status_code == 400, r.text
    assert "hub host" in r.json()["detail"]


def test_wake_404_unknown_machine():
    r = _client().post("/admin/api/machines/nope/wake")
    assert r.status_code == 404, r.text


def test_wake_400_for_mac_less_host():
    # openclaw has no `mac:` row configured (laptop, no wired NIC).
    r = _client().post("/admin/api/machines/openclaw/wake")
    assert r.status_code == 400, r.text
    assert "MAC" in r.json()["detail"]


def test_wake_success_sends_configured_mac(monkeypatch):
    import app_web.routers.machines as machines_router

    captured = {}

    def _fake_send_wake(mac, *args, **kwargs):
        captured["mac"] = mac

    monkeypatch.setattr(machines_router, "send_wake", _fake_send_wake)
    r = _client().post("/admin/api/machines/mac-mini-m4/wake")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "sent": True}
    assert captured["mac"] == get_host("mac-mini-m4").mac


def test_wake_send_failure_yields_clean_error_not_500(monkeypatch):
    import app_web.routers.machines as machines_router
    from src.wake_on_lan import WakeOnLanError

    def _boom(mac, *args, **kwargs):
        raise WakeOnLanError(f"failed to send wake packet for {mac!r}: [Errno 101] Network is unreachable")

    monkeypatch.setattr(machines_router, "send_wake", _boom)
    r = _client().post("/admin/api/machines/mac-mini-m4/wake")
    assert r.status_code == 502, r.text
    assert "failed to send wake packet" in r.json()["detail"]


# ------------------------------------------------------------- terminal status


# ---------------------------------------------------- general-SSH power channel


class _FakeCompleted:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_power_command_uses_general_ssh_not_forced_command_key(monkeypatch):
    """reboot/shutdown must ride the hub user's own SSH (#311) — no ``-i``
    forced-command key, and the real ``sudo shutdown`` verb, targeting
    ``ssh_user@address`` of the peer."""
    from src import remote_bootstrap

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(ssh_exec.subprocess, "run", _fake_run)
    result = remote_bootstrap._run_power_command("mac-mini-m4", "-r")

    assert result["ok"] is True
    cmd = captured["cmd"]
    assert cmd[0] == "ssh"
    assert "-i" not in cmd  # NOT the LOCAL_LLM_HUB_SSH_KEY forced-command key
    assert "roberto@192.168.0.14" in cmd  # peer ssh_user@address
    remote = cmd[-1]
    assert "sudo -n /sbin/shutdown -r now" in remote
    assert "nohup" in remote  # detached so the SSH command returns cleanly


def test_power_command_flags_map_reboot_and_shutdown(monkeypatch):
    """``reboot_host`` sends ``-r``; ``shutdown_host`` sends ``-h``."""
    from src import remote_bootstrap

    seen = []

    def _fake_run(cmd, **kwargs):
        seen.append(cmd[-1])
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(ssh_exec.subprocess, "run", _fake_run)
    r1 = _run(remote_bootstrap.reboot_host("openclaw"))
    r2 = _run(remote_bootstrap.shutdown_host("openclaw"))

    assert r1["ok"] and "reboot scheduled on openclaw" in r1["detail"]
    assert r2["ok"] and "shutdown scheduled on openclaw" in r2["detail"]
    assert "sudo -n /sbin/shutdown -r now" in seen[0]
    assert "sudo -n /sbin/shutdown -h now" in seen[1]


def test_power_command_guards_missing_ssh_target(monkeypatch):
    """A host with no address/ssh_user (here, an unknown id) is rejected before
    any ssh call — the router guards at the endpoint, but the layer guards
    itself too."""
    from src import remote_bootstrap

    def _boom(cmd, **kwargs):  # pragma: no cover — must never run
        raise AssertionError("ssh should not be invoked for an unroutable host")

    monkeypatch.setattr(ssh_exec.subprocess, "run", _boom)
    result = remote_bootstrap._run_power_command("no-such-host", "-r")
    assert result["ok"] is False
    assert "address/ssh_user" in result["error"]


def test_power_command_ssh_failure_surfaces_error(monkeypatch):
    """A non-zero ssh exit is reported, not swallowed."""
    from src import remote_bootstrap

    def _fake_run(cmd, **kwargs):
        return _FakeCompleted(returncode=255, stderr="Connection refused")

    monkeypatch.setattr(ssh_exec.subprocess, "run", _fake_run)
    result = remote_bootstrap._run_power_command("mac-mini-m4", "-h")
    assert result["ok"] is False
    assert "ssh exit 255" in result["error"] and "Connection refused" in result["error"]


def test_terminal_status_endpoint_shape(monkeypatch):
    from src import ssh_terminal

    async def _canned():
        return {"available": False, "reason": "not reachable", "session_host": "http://127.0.0.1:8446"}

    monkeypatch.setattr(ssh_terminal, "terminal_status", _canned)
    r = _client().get("/admin/api/machines/terminal/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"available", "reason", "session_host"}
