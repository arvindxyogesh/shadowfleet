# control_plane

Central service: consumes fleet telemetry from Redis Streams, tracks fleet
node state, and exposes it over a REST API. Implements FR-3 (telemetry
streaming, consumer side) from `../docs/SRS.md`.

Hard-example mining, retrain triggering, canary rollout, and drift detection
(FR-4 through FR-9) land in M3+.

## Stack

FastAPI + SQLAlchemy (Postgres in deployment, SQLite in tests) + a Redis
Streams consumer running as an in-process background task.

## Endpoints

- `GET /health`
- `GET /fleet/nodes` — every node seen so far, its current model versions,
  and whether it's reported telemetry within `node_stale_after_seconds`
- `GET /fleet/nodes/{node_id}/telemetry?limit=50` — recent telemetry events
  for one node (404 for a node that's never reported in)

## How the consumer works

On startup, a background task polls the `telemetry:events` Redis Stream
(`XREAD`) and, for each message, upserts the publishing node's row in
`fleet_nodes` and inserts a row in `telemetry_events`. It tracks its own
read cursor in memory rather than using a Redis consumer group — this
system runs a single control-plane instance, so consumer-group semantics
(multiple competing readers) aren't needed yet.

## Setup

```bash
pip install -r requirements.txt
SHADOWFLEET_CP_DATABASE_URL=sqlite:///./shadowfleet.db \
SHADOWFLEET_CP_REDIS_URL=redis://localhost:6379/0 \
uvicorn app.main:app --reload --port 8001
```

Requires a running Redis instance to consume from (see `../infra/docker-compose.yml`
for the full stack including `edge_agent`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests -v
```

No real Postgres or Redis needed: the consumer is tested against
`fakeredis` and an in-memory SQLite database, and the API layer is tested
by overriding the DB session dependency with pre-seeded in-memory data.

## Docker

```bash
docker build -t shadowfleet-control-plane .
docker run -p 8001:8001 \
  -e SHADOWFLEET_CP_REDIS_URL=redis://host.docker.internal:6379/0 \
  -e SHADOWFLEET_CP_DATABASE_URL=postgresql+psycopg2://user:pass@host/db \
  shadowfleet-control-plane
```
