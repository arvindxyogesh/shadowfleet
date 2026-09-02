# control_plane

Central service: fleet registry, hard-example mining, retrain triggering,
canary rollout state machine, drift detection, and the REST API consumed by
the dashboard and CLI.

Implements: FR-4 through FR-9, FR-11. See `../docs/SRS.md` §5.2 for the API
surface and §7.2 for the telemetry schema it consumes.

Planned stack: FastAPI + PostgreSQL, Redis Streams consumer.
