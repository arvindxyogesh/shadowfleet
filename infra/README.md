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

Scaling the simulated fleet to multiple nodes (`docker compose up --scale
edge_agent=N`) and a GitHub Actions deploy workflow land in a later
milestone — see `../docs/SRS.md` §10 for the roadmap and §6.7 for the
free-tier cost constraint every piece here must respect.
