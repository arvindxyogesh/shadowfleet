# infra

Deployment glue for running the fleet locally.

## `docker-compose.yml`

Brings up the full M2 stack: Redis (telemetry bus), Postgres (control-plane
storage), `control_plane`, and one `edge_agent` node.

```bash
# one-time: produce the ONNX weights the edge_agent container mounts
pip install ultralytics
python ../edge_agent/scripts/export_model.py --output ../edge_agent/models/yolov8n.onnx

docker compose up --build
```

- edge_agent: http://localhost:8000 (`/health`, `/infer`)
- control_plane: http://localhost:8001 (`/health`, `/fleet/nodes`, `/fleet/nodes/{id}/telemetry`)

Scaling the simulated fleet to multiple nodes (`docker compose up --scale
edge_agent=N`) and a GitHub Actions deploy workflow land in a later
milestone — see `../docs/SRS.md` §10 for the roadmap and §6.7 for the
free-tier cost constraint every piece here must respect.
