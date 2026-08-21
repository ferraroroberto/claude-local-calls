"""The gitignored identity overlay that keeps a public repo out of the
business of mapping a private network (#525).

``config/models.yaml`` is committed and carries no addressing; the real
``address`` / ``tailscale`` / ``mac`` / ``ssh_user`` / ``rdp`` values live in
``config/machines.local.yaml``, which ``host_profile._load_config()`` merges
onto the matching ``hosts:`` rows. The overlay is *optional* by design, so the
contract worth pinning here is as much about its absence as its presence.
"""

from __future__ import annotations

import pytest

from src import host_profile
from src.host_profile import IDENTITY_FIELDS, all_hosts, get_host

BASE = {
    "hosts": {
        "alpha": {"platform": "linux", "enabled": [], "display_name": "Alpha"},
        "beta": {"platform": "linux", "enabled": []},
    },
    "models": {},
}


def test_public_config_ships_no_machine_identity():
    """The committed config must never regain an identity field — this is the
    regression guard for the leak itself, not just for the loader."""
    cfg_path = host_profile.PROJECT_ROOT / "config" / "models.yaml"
    import yaml

    rows = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("hosts", {})
    assert rows, "models.yaml should still declare hosts"
    leaked = {
        host_id: [f for f in IDENTITY_FIELDS if row.get(f) is not None]
        for host_id, row in rows.items()
    }
    assert not any(leaked.values()), f"identity fields in the public config: {leaked}"


def test_overlay_merges_identity_onto_matching_host(write_config):
    write_config(
        BASE,
        machines={
            "hosts": {
                "alpha": {
                    "address": "192.168.1.10",
                    "tailscale": "alpha.example-tailnet.ts.net",
                    "mac": "aa:bb:cc:dd:ee:ff",
                    "ssh_user": "youruser",
                    "rdp": {"address": "192.168.1.10", "user": "youruser"},
                }
            }
        },
    )
    alpha = get_host("alpha")
    assert alpha.address == "192.168.1.10"
    assert alpha.tailscale == "alpha.example-tailnet.ts.net"
    assert alpha.mac == "aa:bb:cc:dd:ee:ff"
    assert alpha.ssh_user == "youruser"
    assert alpha.rdp == {"address": "192.168.1.10", "user": "youruser"}
    assert alpha.can_ssh is True
    # The public row's own fields survive the merge untouched.
    assert alpha.display_name == "Alpha"


def test_no_overlay_leaves_identity_unset_and_features_inert(write_config):
    """A fresh public clone has no overlay: the hub must still load, and every
    peer-dependent feature must simply have nothing to act on."""
    write_config(BASE)  # no machines= -> no overlay written
    for host in all_hosts():
        assert all(getattr(host, f) is None for f in IDENTITY_FIELDS)
        assert host.can_ssh is False


def test_overlay_only_touches_hosts_it_names(write_config):
    write_config(BASE, machines={"hosts": {"alpha": {"address": "192.168.1.10"}}})
    assert get_host("alpha").address == "192.168.1.10"
    assert get_host("beta").address is None


def test_overlay_ignores_unknown_host_ids(write_config):
    """An overlay entry for a host the public config doesn't declare is not an
    error — machines get retired from models.yaml without anyone remembering
    to prune the local file."""
    write_config(BASE, machines={"hosts": {"ghost": {"address": "192.168.1.99"}}})
    assert {h.id for h in all_hosts()} == {"alpha", "beta"}


@pytest.mark.parametrize(
    "bad",
    ["hosts: [not, a, mapping]", "*** not: valid: yaml: ["],
    ids=["hosts-not-a-mapping", "unparseable"],
)
def test_broken_overlay_never_stops_the_hub_booting(write_config, capsys, bad):
    """A dead overlay degrades to 'no identity', loudly — never an exception on
    the boot path, and never a silent one either."""
    cfg = write_config(BASE)
    (cfg.parent / "machines.local.yaml").write_text(bad, encoding="utf-8")
    host_profile._CONFIG_CACHE.clear()

    hosts = all_hosts()  # must not raise

    assert {h.id for h in hosts} == {"alpha", "beta"}
    assert all(h.address is None for h in hosts)
    assert "machines.local.yaml" in capsys.readouterr().err


def test_shipped_example_overlay_is_valid(config_with_example_identity):
    """config/machines.local.example.yaml is what a new clone copies — if it
    stops parsing or stops matching the real host ids, that breaks silently for
    everyone but is invisible here unless asserted."""
    named = [h for h in all_hosts() if h.tailscale]
    assert named, "the example overlay should supply identity for real host ids"
    for host in named:
        assert host.tailscale.endswith(".ts.net")
        assert "example" in host.tailscale
