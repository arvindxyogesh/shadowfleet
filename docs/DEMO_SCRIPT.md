# Demo Walkthrough Script

A shot-by-shot script for recording a ~3-minute walkthrough video of
ShadowFleet. Follows `scripts/demo.py`'s steps — run it in one terminal
while narrating, or run it once beforehand and walk through the output.

**Setup before recording:**
```bash
make export-model
make up
```
Wait for `docker compose ps` to show all five services running, then open
two browser tabs: `http://localhost:3000` (Grafana) and
`http://localhost:8001/docs` (control plane's auto-generated API docs).

---

### 1. The pitch (15s)

> "This is ShadowFleet — a scaled-down recreation of the fleet data engine
> loop that companies like Tesla use to keep a deployed computer vision
> model improving after launch. The detection model itself is ordinary —
> the point is the closed-loop system around it: telemetry, hard-example
> mining, auto-retraining, canary rollout, and drift-triggered rollback.
> Everything here runs on free-tier or self-hosted tools."

Show the architecture diagram in the root `README.md`.

### 2. Send inference traffic (30s)

```bash
make demo
```

> "This script sends a batch of synthetic frames through the fleet node,
> the way a real dashcam feed would."

Let the first `=== Sending N inference requests ===` block print. Point
out the per-frame latency in the output.

### 3. Fleet status + hard examples (20s)

> "The control plane consumed that telemetry off a Redis stream and
> upserted this node's state — model version, last-seen time."

Show the `=== Fleet status ===` block, then switch to Grafana
(`http://localhost:3000`) and point at the "Fleet Nodes" table and the
confidence/latency time series panels updating live.

> "Any frame the model was unsure about, or where a shadow candidate model
> disagreed with production, gets flagged here as a hard example — that's
> the seed data for the next retrain."

Show the "Hard Examples by Status" pie chart.

### 4. Canary rollout (60s)

> "Now let's roll out a new model version."

Let the script's `=== Starting a canary rollout ===` block run. While it
polls:

> "The rollout manager pushed this version to a percentage of the fleet in
> shadow mode — running alongside production, logged but not served.
> Every poll cycle it compares that canary group's confidence against the
> rest of the fleet using a Welch's t-test. If nothing looks wrong once
> the evaluation window elapses, it promotes atomically."

Show the `/rollouts/{id}` response in the terminal, or the Rollouts table
in Grafana, as status flips from `shadow` to `completed`.

> "If it *had* detected a regression, this same mechanism would have
> auto-rolled-back instead — that path is exercised in the test suite
> against a simulated multi-node fleet, since this one-node demo doesn't
> have a control group to compare against."

### 5. Audit log + wrap-up (20s)

Show the `=== Audit log ===` output or the Grafana audit-log table.

> "Every automatic and manual action — rollout started, promoted, paused,
> rolled back — is logged here with an actor and timestamp. That's the
> full loop: telemetry in, hard examples mined, a model rolled out and
> verified safe, all without touching the running fleet by hand."

Close on the root `README.md`'s Status section showing all six milestones
complete.

---

## Notes for re-recording

- The fake/placeholder model weights `make export-model` downloads are real
  YOLOv8n COCO weights (via Ultralytics) — detections will look reasonable
  on any photo, not just dashcam frames, if you want a more visual demo
  than the synthetic frames `scripts/demo.py` generates.
- To show an actual drift-triggered rollback on camera, scale the fleet
  (`docker compose up --scale edge_agent=4`, though see `infra/README.md`'s
  caveat about `SHADOWFLEET_SELF_BASE_URL` needing distinguishable
  hostnames per replica first) or just narrate over
  `control_plane/tests/test_rollout.py::test_evaluate_rollout_rolls_back_on_detected_drift`.
