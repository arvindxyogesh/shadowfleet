# Demo Walkthrough Script

A shot-by-shot script for recording a ~5-minute walkthrough video of
ShadowFleet. Every step below has been run for real (not simulated) —
see the PR's Validation section and the M6 commit history for the
actual output this script describes.

**Setup before recording:**
```bash
make export-model
docker compose -f infra/docker-compose.yml up --build -d
docker compose -f infra/docker-compose.yml ps   # all 6 services Up (healthy)
```
Open two browser tabs: `http://localhost:3000` (Grafana) and
`http://localhost:8001/docs` (control plane's auto-generated API docs).

---

### 1. The pitch (15s)

> "This is ShadowFleet — a scaled-down recreation of the fleet data engine
> loop that companies like Tesla use to keep a deployed computer vision
> model improving after launch. The detection model itself is ordinary —
> the point is the closed-loop system around it: telemetry, hard-example
> mining, auto-retraining, canary rollout, and drift-triggered rollback.
> Everything here runs on free-tier or self-hosted tools, and every part
> of this walkthrough is a real run, not a mock."

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

### 4. A real training run (60s)

> "Hard examples feed the training pipeline. This isn't a stubbed-out
> script — it's a real Ultralytics training loop."

```bash
cd training_pipeline
python scripts/train.py --control-plane-url http://localhost:8001 \
  --base-dataset coco128.yaml --data-version demo-v1 --epochs 3
```

Let a couple of epochs print live (fast on a GPU — a few seconds each).

> "Real gradient steps, real validation, real mAP. This one run got
> 0.629 mAP50 on COCO128 in 3 epochs on an H200. At the end it registers
> a new model version in the training pipeline's own registry."

Point at the final `Registered model version ... (promoted=True)` line —
that's the artifact a canary rollout will actually deploy.

### 5. Canary rollout — the happy path (45s)

> "Now let's roll that new version out to the fleet."

```bash
curl -X POST http://localhost:8001/rollouts -H 'Content-Type: application/json' -d '{
  "model_version": "v2",
  "model_path": "/app/models/yolov8n.onnx",
  "target_percentage": 50,
  "evaluation_window_seconds": 60
}'
```

> "The rollout manager pushed this version to half the fleet in shadow
> mode — running alongside production, logged but not served. Every poll
> cycle it compares that canary node's *shadow* confidence against the
> rest of the fleet's production confidence using a Welch's t-test. If
> nothing looks wrong once the evaluation window elapses, it promotes
> atomically across every canary node — not just some of them."

Send a little traffic to both `:8000/infer` and `:8002/infer`, then show
the `/rollouts/{id}` response (or the Rollouts table in Grafana) as
status flips from `shadow` to `completed`.

### 6. Canary rollout — the drift path (60s)

> "And here's what happens when the candidate is actually bad."

```bash
python edge_agent/scripts/export_model.py --weights yolov8n.yaml \
  --output edge_agent/models/v2-bad.onnx   # untrained -- random weights

docker compose -f infra/docker-compose.yml \
  -f infra/docker-compose.drift-demo.override.yml up -d edge_agent

curl -X POST http://localhost:8001/rollouts -H 'Content-Type: application/json' -d '{
  "model_version": "v2-bad",
  "model_path": "/app/models/v2-bad.onnx",
  "target_percentage": 50,
  "evaluation_window_seconds": 120
}'
```

Send traffic to both nodes again, then poll `/rollouts/{id}`.

> "This candidate never got promoted — the drift detector caught the
> confidence regression and rolled it back automatically, on live
> traffic, no human in the loop."

Show the rollout's `"status": "rolled_back"` and its `"reason"` field
naming the detected regression.

```bash
docker compose -f infra/docker-compose.yml up -d edge_agent   # revert the demo threshold
```

### 7. Audit log + wrap-up (20s)

Show the `GET /audit-log` output or the Grafana audit-log table —
`rollout_started`, `rollout_completed`, and this run's `rollback` entry
should all be visible.

> "Every automatic and manual action is logged here with an actor and
> timestamp. That's the full loop: telemetry in, hard examples mined, a
> real model trained and rolled out, and a bad one automatically caught
> and rolled back — all without touching the running fleet by hand."

Close on the root `README.md`'s Status section showing all six milestones
complete, and its Validation section with the real numbers from this
walkthrough.

---

## Notes for re-recording

- Section 4 (training) and section 6 (drift) both take a few real minutes
  of wall-clock time even on a fast GPU, once model downloads and
  container startup are included — consider recording them separately
  and cutting to the interesting parts (the final printed lines) rather
  than running everything live in one take.
- Section 6 needs `edge_agent/scripts/export_model.py`'s `--weights
  yolov8n.yaml` variant, which builds an architecture from scratch
  (random weights) rather than loading pretrained COCO weights — that's
  what makes the candidate genuinely bad. The lowered confidence
  threshold in `infra/docker-compose.drift-demo.override.yml` is needed
  too, since an untrained detector's confidence is too low to register
  as a detection at all at the default 0.25 threshold — see that file's
  comments for why.
- `docs/DEMO_SCRIPT.md`'s numbered sections map roughly to `scripts/
  demo.py`'s printed steps for sections 1-3 and 5; sections 4 and 6 are
  manual since they're not part of the scripted one-node demo.
