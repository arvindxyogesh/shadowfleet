# dashboard

Grafana provisioning (dashboards + datasources) and any custom panels needed
to visualize fleet state, per-node model version, rollout progress, and
drift/rollback events sourced from the control plane's metrics endpoint.

Implements: FR-10. See `../docs/SRS.md` §5.1.

Planned stack: Grafana + Prometheus, both self-hosted via Docker.
