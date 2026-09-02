# dashboard

Grafana provisioning for the fleet-overview dashboard (FR-10 from
`../docs/SRS.md`): fleet node status, telemetry trends, and hard-example
mining counts.

## How it's wired

Grafana connects **directly to `control_plane`'s Postgres database** as a
native datasource — no extra metrics-exporter code needed, since the data
the dashboard needs (`fleet_nodes`, `telemetry_events`, `hard_examples`)
already lands there via the consumer (M2/M3). Panels are plain SQL against
those tables, using Grafana's `$__timeFilter`/`$__timeGroup` macros so they
respect the dashboard's time-range picker and auto-refresh.

- `provisioning/datasources/postgres.yml` — points Grafana at the
  `control_plane` Postgres instance
- `provisioning/dashboards/dashboards.yml` — tells Grafana to load any
  dashboard JSON found under `dashboards/`
- `dashboards/fleet-overview.json` — the dashboard itself: node status
  table, avg confidence/latency/disagreement time series per node, and
  hard-example counts by status and reason. Refreshes every 10s.

## Running it

Part of the full stack — see `../infra/docker-compose.yml`:

```bash
cd ../infra
docker compose up --build
```

Then open http://localhost:3000 — anonymous viewer access is enabled for
local convenience (`GF_AUTH_ANONYMOUS_ENABLED`), or log in as
`admin`/`shadowfleet` to edit. The dashboard is provisioned automatically;
no manual datasource or import step required.

Panels will show a "relation does not exist" error until `control_plane`
has started at least once and created its schema — normal on a very first
`docker compose up` if Grafana's container finishes starting first.

## What's not here yet

Rollout progress and rollback events (also part of FR-10) have nothing to
show until M5 adds canary rollout and drift detection — those panels get
added to this same dashboard then, against the same Postgres datasource.
