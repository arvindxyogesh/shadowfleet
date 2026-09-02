# infra

Deployment glue for running the fleet locally.

## `docker-compose.yml`

Brings up the full stack through M4: Redis (telemetry bus), Postgres
(control-plane + dashboard storage), `control_plane`, one `edge_agent`
node, and Grafana (dashboard).

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
4. With only one `edge_agent` replica in this compose file, there's no
   separate control group to compare against, so drift-triggered rollback
   won't fire on its own here — it's exercised in
   `control_plane/tests/test_rollout.py` against a fake multi-node fleet.
   You can still try `POST /rollouts/{id}/rollback` manually at any point.

Scaling the simulated fleet to multiple distinguishable nodes (today's
`SHADOWFLEET_SELF_BASE_URL` maps every replica to the same compose service
name) and a GitHub Actions deploy workflow land in a later milestone — see
`../docs/SRS.md` §10 for the roadmap and §6.7 for the free-tier cost
constraint every piece here must respect.
