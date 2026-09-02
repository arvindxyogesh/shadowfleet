# ShadowFleet

A closed-loop **fleet data engine** for computer vision, scaled down from
production systems (Tesla Autopilot/FSD, Waymo) to run entirely on free-tier /
self-hosted tools.

The CV task (object detection on dashcam frames) is deliberately ordinary — the
point of this project is the software system around the model: fleet telemetry,
shadow-mode evaluation, hard-example mining, auto-triggered retraining, canary
rollout, and drift-triggered rollback.

**Start here:** [`docs/SRS.md`](docs/SRS.md) — full Software Requirements
Specification (architecture, functional/non-functional requirements, data
contracts, roadmap).

## Status

Per the SRS §10 roadmap:

- ✅ **M1** — `edge_agent` inference core (FastAPI + ONNX Runtime, YOLOv8n)
- ✅ **M2** — Shadow-mode serving, telemetry streaming (Redis Streams), and
  the `control_plane` fleet registry (FastAPI + Postgres/SQLite)
- ✅ **M3** — Hard-example mining (`control_plane`) + a manually-triggered
  `training_pipeline` (dataset merge, promotion evaluation, versioned
  model registry)
- ⬜ M4 — Dashboard v1 (Grafana)
- ⬜ M5 — Closed loop: auto-trigger, canary rollout, drift detection, rollback
- ⬜ M6 — Portfolio polish (`make demo`, recorded walkthrough, hosted demo)

Run the current stack locally via `infra/docker-compose.yml` — see
[`infra/README.md`](infra/README.md).

## Repository Layout

| Path | Purpose |
|---|---|
| `docs/` | SRS and supporting design docs |
| `edge_agent/` | Fleet node inference service (prod + shadow model serving) |
| `control_plane/` | Fleet registry, rollout manager, drift detection, REST API |
| `training_pipeline/` | Hard-example consumption, retraining, model registration |
| `dashboard/` | Grafana provisioning + fleet/rollout visualization |
| `infra/` | Docker Compose, k8s manifests, CI/CD workflows |

## Tech Stack (all free-tier / self-hostable)

YOLOv8n/YOLO11n · FastAPI · ONNX Runtime · Redis Streams · PostgreSQL · DVC ·
MLflow · Docker Compose / k3d · GitHub Actions · Grafana + Prometheus

See SRS §3.4 for the full mapping of each tool to its free-tier basis.
