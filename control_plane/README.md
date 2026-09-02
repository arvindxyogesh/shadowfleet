# control_plane

Central service: consumes fleet telemetry from Redis Streams, tracks fleet
node state, mines hard examples for labeling, auto-triggers retraining,
and drives canary rollouts with drift-based automatic rollback — all
exposed over a REST API. Implements FR-3 through FR-9 and FR-11 from
`../docs/SRS.md`.

## Stack

FastAPI + SQLAlchemy (Postgres in deployment, SQLite in tests) + a Redis
Streams consumer running as an in-process background task.

## Endpoints

- `GET /health`
- `GET /fleet/nodes` — every node seen so far, its current model versions,
  and whether it's reported telemetry within `node_stale_after_seconds`
- `GET /fleet/nodes/{node_id}/telemetry?limit=50` — recent telemetry events
  for one node (404 for a node that's never reported in)
- `GET /hard-examples?status=pending|labeled&limit=50` — inputs flagged by
  mining (see below), most recently flagged first
- `POST /hard-examples/{input_id}/label` — attach a label (`{"label": {...}}`)
  to a flagged input and mark it `labeled`, ready for `training_pipeline` to
  pick up
- `POST /rollouts` — start a canary rollout (`model_version`, `model_path`,
  `target_percentage`, optional `evaluation_window_seconds`,
  `previous_model_path`, `actor`)
- `GET /rollouts` / `GET /rollouts/{id}` — list rollouts / one rollout's
  detail including its per-node canary/control assignments
- `POST /rollouts/{id}/pause` / `/resume` / `/rollback` — manual override
  (FR-11); rollback accepts `{"actor": ..., "reason": ...}`
- `GET /audit-log?limit=50` — every automatic and manual rollout/retrain
  action, most recent first
- `GET /retrain-triggers?limit=50` — history of FR-5 firings

## Hard-example mining (FR-4)

Every persisted telemetry event is checked against two thresholds
(`hard_example_conf_threshold`, `hard_example_disagreement_threshold`):
an event is flagged when its weakest kept detection falls below the
confidence threshold, or prod/shadow disagreement exceeds the disagreement
threshold (`app/mining.py`). This milestone's MVP has no live labeling UI —
a human or script calls `POST /hard-examples/{id}/label` with whatever
ground truth they've produced (per the SRS's documented simulation of
labeling via a held-out set, see §2.6).

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

No real Postgres or Redis needed: the consumer (including hard-example
mining) is tested against `fakeredis` and an in-memory SQLite database, and
the API layer is tested by overriding the DB session dependency with
pre-seeded in-memory data. `RolloutManager` is tested against a
`FakeNodeClient` that records calls instead of making real HTTP requests.

## Closed loop (FR-5, FR-8, FR-9, FR-11)

A background task in the same process re-checks every
`rollout_check_interval_seconds` (default 30s):

1. **Retrain trigger** (`app/retrain.py`): once the count of labeled, not-
   yet-used hard examples crosses `retrain_trigger_threshold`, a
   `RetrainTrigger` row is written and those examples are marked used.
   Actually invoking `training_pipeline` is pluggable
   (`app/retrain_dispatch.py`): the default `LoggingRetrainDispatcher`
   just logs and records the trigger — the trigger row itself is the
   auditable "a retrain should happen now" signal; wiring a real
   `GitHubActionsRetrainDispatcher` needs a repo + token this project's
   test suite and free-tier demo never require.
2. **Rollout evaluation** (`app/rollout.py`): for every rollout in
   `shadow` status, compares canary vs. control node confidence
   (`app/drift.py`, a one-sided Welch's t-test) since the rollout started.
   Detected regression → automatic rollback (FR-9). No regression and the
   evaluation window has elapsed → all canary nodes promoted to
   production atomically (FR-8), rollout marked `completed`.

**OTA push**: canary/control assignment (`app/canary.py`) is deterministic
given the same fleet and percentage. Pushing a model version to a node
means calling that node's `edge_agent` `/admin/model` endpoint
(`app/node_client.py`) — a node with no known `base_url` (never reported
one in telemetry) is skipped and logged as unreachable rather than
failing the whole rollout. A single `docker compose` fleet (see
`../infra/docker-compose.yml`) only has one edge_agent replica today, so
every node maps to the same `base_url`; scaling to a distinguishable
multi-node fleet is a documented stretch goal (SRS §10 Phase 3).

## Docker

```bash
docker build -t shadowfleet-control-plane .
docker run -p 8001:8001 \
  -e SHADOWFLEET_CP_REDIS_URL=redis://host.docker.internal:6379/0 \
  -e SHADOWFLEET_CP_DATABASE_URL=postgresql+psycopg2://user:pass@host/db \
  shadowfleet-control-plane
```
