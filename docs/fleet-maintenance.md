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
SSH bootstrap, no start) until the window expires or is cleared. This
replaces the old workaround of hand-editing
`LOCAL_LLM_HUB_FLEET_RECONCILE_INTERVAL_S` / `..._BOOT_DELAY_S` in `.env` on
the tower, which still left the boot-pass racing a raised interval.

Maintenance is **tower-local** — it lives in `config/fleet_maintenance.json`
(gitignored, no committed example) and is only ever consulted by the tower's
own reconcile loop. Since #430 the desired state reconcile converges on is
derived from `config/models.yaml` (`hosts:` chains + `startup:` policy);
de-provisioning a model permanently is a registry edit, not something this
drain marker does.

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
4. **Verify with a backend-specific probe, never a role-level request** (#412).
   A plain `POST /v1/audio/transcriptions` with no `model` addresses the
   *transcribe role*, whose whole job is to fall back across models — a 200
   there proves only that *something* transcribed, not that the model under
   drill moved. That is what made the #405 drill unreadable: `model=whisper`
   came back 200 in 794 ms from a host with no whisper-server bound at all —
   a legitimate proxy hop to whisper's *owner*, but the record had no field
   that said which machine answered, so it was indistinguishable from a
   substitution. Prove the model itself, in this order:
   - **Name the model explicitly** — `-F model=whisper`. Since #412 an
     explicit model id is strict: it is either served by that model or it
     fails naming it, and can never be answered by a different one.
   - **Read back who served it, and where** — the response carries
     `X-Hub-Served-Model` and `X-Hub-Served-Host` (plus
     `X-Hub-Requested-Model`); the same trio shows up as
     `requested → served @host` in the admin Hub/Telemetry request rows and in
     `GET /admin/api/hub/requests/recent` (`served_model`, `served_host`). The
     host is the field that answers "did it move?" — a 200 from `@gaming` and
     a 200 from `@mac-mini-m4` are different outcomes.
   - **Confirm the process on the new owner** — `Get-NetTCPConnection
     -LocalPort 8090` (Windows) / `lsof -nP -iTCP:8090 -sTCP:LISTEN` (macOS,
     Linux) on the host that took ownership, so a real `whisper-server` is
     bound there rather than the request having been proxied elsewhere.
   ```
   curl -sS -D - -o /dev/null -F file=@clip.wav -F model=whisper \
        http://<owner>:8000/v1/audio/transcriptions | grep -i x-hub-
   ```
5. **Clear maintenance on the tower** once the drill is done (or let the
   window expire):
   ```
   curl -X DELETE http://<tower>:8000/admin/api/fleet-maintenance/gaming
   ```
   This runs one reconcile pass immediately, so the peer's resurrection (and
   failover's `failback_after_s`-gated hand-back) is observable right away
   rather than waiting up to `FLEET_RECONCILE_INTERVAL_S`.

See `src/fleet_reconcile.py` and `src/model_failover.py` module docstrings
for the full mechanics of each loop.
