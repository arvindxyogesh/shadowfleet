# infra

Deployment glue for running the fleet locally.

## `docker-compose.yml`

Brings up the full stack: Redis (telemetry bus), Postgres (control-plane +
dashboard storage), `control_plane`, two `edge_agent` nodes (`edge_agent`
on :8000, `edge_agent_2` on :8002 — enough for a canary rollout to have a
real control group), and Grafana (dashboard).

```bash
# one-time: produce the ONNX weights the edge_agent container mounts
pip install ultralytics
python ../edge_agent/scripts/export_model.py --output ../edge_agent/models/yolov8n.onnx

docker compose up --build
```

- edge_agent: http://localhost:8000 (`/health`, `/infer`)
- control_plane: http://localhost:8001 (`/health`, `/fleet/nodes`,
  `/fleet/nodes/{id}/telemetry`, `/hard-examples`)
- Grafana: http://localhost:3000 (fleet-overview dashboard, provisioned
  automatically — see `../dashboard/README.md`)

Send some traffic through `edge_agent` (`curl -X POST
http://localhost:8000/infer -F "file=@image.jpg"`) and watch the dashboard
update on its own within a few seconds.

## Trying a canary rollout end-to-end

1. Stage a second ONNX artifact the node can load, e.g.
   `edge_agent/models/v2.onnx` (any valid export — even the same weights
   under a new name works for exercising the mechanism).
2. Start a rollout:
   ```bash
   curl -X POST http://localhost:8001/rollouts -H 'Content-Type: application/json' -d '{
     "model_version": "v2",
     "model_path": "/app/models/v2.onnx",
     "target_percentage": 100,
     "evaluation_window_seconds": 60
   }'
   ```
3. Keep sending `/infer` traffic (real fleet metrics only exist once
   frames are actually processed) and watch `GET
   http://localhost:8001/rollouts/{id}` — after the evaluation window
   with no detected drift, its canary node is promoted and the rollout
   moves to `completed`. `GET /audit-log` shows every step.
4. With `target_percentage: 100` above, both nodes go to canary and there's
   no control group. Use e.g. `"target_percentage": 50` with traffic sent
   to *both* `:8000/infer` and `:8002/infer` to get a real canary/control
   split — that's what lets FR-9's drift detection (Welch's t-test on
   canary vs. control confidence) actually have something to compare, not
   just the fake multi-node fleet in `control_plane/tests/test_rollout.py`.
   Drift is measured on the canary group's *shadow* confidence against
   the control group's *prod* confidence, so the model staged in step 2
   needs to actually run as the canary's shadow model (which starting a
   rollout does automatically) — a candidate that's merely different,
   not worse, will just promote cleanly with no drift, same as any other
   rollout.

## Demonstrating a real drift-triggered rollback

To see FR-9 actually fire (not just the no-drift/promote path above), the
candidate needs to be genuinely worse, and canary/control need real
traffic:

```bash
# A genuinely bad candidate: untrained YOLOv8n (random weights).
python edge_agent/scripts/export_model.py --weights yolov8n.yaml \
  --output edge_agent/models/v2-bad.onnx

# A fresh detector's objectness bias is initialized very negative, so an
# untrained model produces *no* detections above the default confidence
# threshold on most inputs, rather than low-confidence ones -- and a
# model with zero detections contributes nothing to the drift statistic.
# This override lowers just the canary node's threshold so its
# near-random signal actually registers as (bad) detections.
docker compose -f docker-compose.yml -f docker-compose.drift-demo.override.yml up -d edge_agent

curl -X POST http://localhost:8001/rollouts -H 'Content-Type: application/json' -d '{
  "model_version": "v2-bad",
  "model_path": "/app/models/v2-bad.onnx",
  "target_percentage": 50,
  "evaluation_window_seconds": 120
}'
# send traffic to both :8000/infer and :8002/infer, then poll
# GET /rollouts/{id} -- expect status "rolled_back" within one
# rollout_check_interval_seconds (default 30s) of enough samples landing.

# revert the canary node to its normal threshold afterward:
docker compose -f docker-compose.yml up -d edge_agent
```

A GitHub Actions deploy workflow lands in a later milestone — see
`../docs/SRS.md` §10 for the roadmap and §6.7 for the free-tier cost
constraint every piece here must respect.
