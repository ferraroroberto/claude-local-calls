# Fleet maintenance (drain marker)

`src/fleet_reconcile.py` (#353) and `src/model_failover.py` (#342) are both
always-on background loops that hold independent opinions about a down peer
hub:

- **Reconcile** treats an unreachable-but-placed host as drift and SSH-wakes
  it — a pass on boot plus every `FLEET_RECONCILE_INTERVAL_S` (default 300s).
- **Failover** treats the same unreachable host as a dead owner and moves
  model ownership once it has been continuously down for `fail_after_s`
  (default 90s).

Left alone, reconcile almost always wins that race — a deliberately-stopped
peer gets SSH-resurrected within seconds, long before failover's window can
be satisfied. `src/fleet_maintenance.py` closes the gap: an operator arms a
host-scoped drain window, and reconcile skips that host entirely (no wake, no
SSH bootstrap, no profile write-through, no start) until the window expires
or is cleared. This replaces the old workaround of hand-editing
`LOCAL_LLM_HUB_FLEET_RECONCILE_INTERVAL_S` / `..._BOOT_DELAY_S` in `.env` on
the tower, which still left the boot-pass racing a raised interval.

Maintenance is **tower-local**, exactly like `fleet_placement.json` — it lives
in `config/fleet_maintenance.json` next to it (gitignored, no committed
example) and is only ever consulted by the tower's own reconcile loop.
Explicit un-placement (`PATCH /admin/api/fleet-placement`) is **not** gated —
that's a separate, deliberate action, not the additive resurrection this
exists to pause.

## API

All requests below target the **tower** (the host running reconcile), not
the peer being drained.

```
GET    /admin/api/fleet-maintenance            # active (non-expired) drains
POST   /admin/api/fleet-maintenance/{host_id}   # {"duration_s"?: 900, "reason"?: "..."}
DELETE /admin/api/fleet-maintenance/{host_id}   # clears + runs one immediate reconcile pass
```

`duration_s` defaults to 900s (15 min) and is clamped to a 3600s (1h) ceiling
— long enough to cover the default `fail_after_s` with margin, short enough
that a forgotten toggle self-heals within the hour.

## Running a failover drill

1. **Arm maintenance on the tower** for the peer you're about to stop:
   ```
   curl -X POST http://<tower>:8000/admin/api/fleet-maintenance/gaming \
        -H "Content-Type: application/json" \
        -d '{"duration_s": 900, "reason": "failover drill"}'
   ```
2. **Stop the peer's hub** — on the peer itself:
   ```
   curl -X POST http://gaming:8000/admin/api/hub/stop
   ```
3. Observe `model_failover.py`'s probe/decide/act cycle move ownership after
   `fail_after_s` — the peer now stays down for the drain window instead of
   being resurrected within seconds.
4. **Clear maintenance on the tower** once the drill is done (or let the
   window expire):
   ```
   curl -X DELETE http://<tower>:8000/admin/api/fleet-maintenance/gaming
   ```
   This runs one reconcile pass immediately, so the peer's resurrection (and
   failover's `failback_after_s`-gated hand-back) is observable right away
   rather than waiting up to `FLEET_RECONCILE_INTERVAL_S`.

See `src/fleet_reconcile.py` and `src/model_failover.py` module docstrings
for the full mechanics of each loop.
