# Software Requirements Specification

## ShadowFleet — A Closed-Loop Fleet Data Engine for Computer Vision

**Version:** 0.1.0
**Status:** Draft
**Standard followed:** ISO/IEC/IEEE 29148:2018 (structure), IEEE 830 (naming conventions)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for **ShadowFleet**, a software system that
reproduces — at laptop/free-tier scale — the *data engine* loop that production
computer-vision fleets (e.g. Tesla Autopilot/FSD, Waymo, Cruise) use to keep a deployed
model improving after launch: fleet telemetry → shadow-mode evaluation → hard-example
mining → auto-triggered retraining → canary rollout → OTA model distribution →
drift-triggered rollback.

The underlying CV task (object detection) is intentionally ordinary. The system under
specification is the **software** that surrounds the model, not the model itself. This
document exists to demonstrate requirements-engineering discipline for an ML-SWE /
MLE-infra portfolio project, and to serve as the implementation contract for the
codebase in this repository.

### 1.2 Intended Audience and Reading Suggestions

- **Implementer (primary author)**: read sections 3–8 as the build spec.
- **Technical reviewers / interviewers**: sections 1–3 for scope and architecture,
  section 6 for engineering rigor (NFRs), section 9 for how correctness is verified.
- **Future contributors**: section 7 (data contracts) and Appendix B (repo layout)
  before touching code.

### 1.3 Project Scope

**In scope:**
- Simulating a fleet of N edge nodes running a CV inference service.
- Streaming inference telemetry (predictions, confidence, latency, disagreement
  signals) from nodes to a central control plane.
- Shadow-mode evaluation of candidate models against the production model with zero
  traffic impact.
- Automatic mining of "hard examples" (low confidence, high shadow/production
  disagreement, or human-flagged) for labeling.
- A retraining pipeline triggered once a labeled hard-example threshold is met.
- Model versioning, registry, and a canary rollout mechanism across the simulated
  fleet with automatic rollback on regression.
- A dashboard exposing fleet health, per-node model version, and drift metrics.

**Out of scope:**
- Physical hardware / robotics / real vehicles.
- Real-time video streaming at production bitrates (the system operates on sampled
  frames/clips, not live camera feeds).
- Human labeling UI (labeling is simulated via pre-labeled hold-out data unless a
  free third-party tool, e.g. Label Studio OSS, is wired in as a stretch goal).
- Multi-region / multi-cloud deployment.

### 1.4 Definitions, Acronyms, and Abbreviations

| Term | Meaning |
|---|---|
| Fleet node | A simulated edge device (Docker/k8s pod) running the inference service |
| Shadow mode | Running a candidate model alongside production without serving its output |
| Hard example | An input the model handles poorly (low confidence / high disagreement) |
| Canary rollout | Gradually shifting a % of fleet nodes to a new model version |
| OTA | Over-the-air — here, a control-plane-initiated model artifact push to nodes |
| Drift | Statistically significant change in input distribution or model performance |
| Control plane | The central service orchestrating fleet state, rollout, and retraining |
| MVP | Minimum viable product (Phase 1 scope, see §10) |

### 1.5 References

- ISO/IEC/IEEE 29148:2018 — Requirements engineering.
- Tesla AI Day (2021/2022) — public description of the Data Engine / fleet learning
  loop (conceptual reference only; no proprietary material used).
- Ultralytics YOLOv8/v11 documentation.
- BDD100K dataset documentation (Berkeley DeepDrive).

---

## 2. Overall Description

### 2.1 Product Perspective

ShadowFleet is a new, self-contained system with no dependency on proprietary
infrastructure. It is composed of independently deployable services connected by a
message bus and a shared model/data registry, all runnable via Docker Compose locally
and via GitHub Actions in CI.

### 2.2 Product Functions (summary)

1. Serve object-detection inference from simulated fleet nodes.
2. Stream structured telemetry from every inference call to the control plane.
3. Run one or more shadow model versions per node without affecting served output.
4. Detect and queue hard examples for labeling.
5. Trigger and run a retraining job once enough new labeled data exists.
6. Register, version, and store trained model artifacts.
7. Roll a new model version out to a configurable percentage of the fleet.
8. Detect performance/data drift and automatically roll back a bad rollout.
9. Visualize fleet state, rollout progress, and model performance over time.

### 2.3 User Classes and Characteristics

| User class | Description | Needs |
|---|---|---|
| ML Engineer (operator) | Triggers/reviews retraining, approves rollouts | Dashboard, CLI, retrain logs |
| SRE / Platform role | Monitors fleet health, rollback events | Alerts, metrics, drift dashboard |
| Reviewer / recruiter | Evaluates the project | README, architecture doc, live demo |

### 2.4 Operating Environment

- **Local development**: Docker Compose on Linux/macOS/WSL, Python 3.11+.
- **CI**: GitHub Actions (Ubuntu runners, free tier).
- **Demo hosting**: Hugging Face Spaces for the inference API + read-only
  dashboard snapshot; full multi-node fleet simulation runs locally / in CI, since
  free hosting tiers cannot sustain a multi-container fleet continuously.
  **Update**: as of this writing, HF Spaces' Docker SDK — needed to run a
  real FastAPI container — has moved off the free tier (Static/Gradio SDKs
  remain free). `deploy/huggingface-spaces/` is built and verified working
  but not actually pushed to a live Space as a result; see that
  directory's README.
- **Fleet simulation**: `kind` or `k3d` (local Kubernetes) or Docker Compose with N
  replica services — no paid cloud compute required.

### 2.5 Design and Implementation Constraints

- **C-1 (Free-tier only)**: Every component must run on a tool with a permanently
  free tier or be fully self-hostable. No requirement may depend on a paid cloud
  service to function end-to-end.
- **C-2**: Model inference must run on CPU (no GPU dependency), using a small model
  (e.g. YOLOv8n/YOLO11n) exported to ONNX for portability and speed.
- **C-3**: All inter-service communication must use open protocols (HTTP/REST, gRPC,
  or Redis Streams) — no proprietary message brokers.
- **C-4**: The system must be fully reproducible via `docker compose up` plus a
  documented seed-data step.

### 2.6 Assumptions and Dependencies

- The chosen dataset (BDD100K subset, or a smaller Roboflow public dashcam dataset
  as fallback) is available under a license permitting research/portfolio use.
- "Ground truth" for hard-example labeling in the MVP is simulated by holding out
  labeled data and revealing it on demand, rather than a live human-labeling UI.
- GitHub Actions free-tier minutes (2,000/month for private repos, unlimited for
  public repos) are sufficient for CI and scheduled retraining triggers.

---

## 3. System Architecture Overview

### 3.1 High-Level Component Diagram

```
                         ┌───────────────────────────┐
                         │        Dashboard           │
                         │   (Grafana + custom UI)    │
                         └────────────▲────────────────┘
                                      │ metrics/queries
                         ┌────────────┴────────────────┐
                         │       Control Plane          │
                         │  (FastAPI + Postgres)        │
                         │  - fleet registry             │
                         │  - rollout manager             │
                         │  - drift detector               │
                         │  - retrain trigger                │
                         └───┬───────────────▲──────────────┘
                             │ OTA push       │ telemetry (Redis Streams)
             ┌───────────────▼───┐   ┌─────────┴─────────┐
             │   Fleet Node 1     │   │   Fleet Node N     │  ... (Docker/k8s replicas)
             │ (inference svc)    │   │ (inference svc)    │
             │ prod model         │   │ prod model          │
             │ + shadow model     │   │ + shadow model       │
             └────────────────────┘   └──────────────────────┘

                         ┌───────────────────────────┐
                         │   Training Pipeline         │
                         │ (triggered by control plane) │
                         │  - pulls hard examples         │
                         │  - retrains (Ultralytics)         │
                         │  - logs to MLflow                    │
                         │  - registers new model version         │
                         └───────────────────────────┘

                         ┌───────────────────────────┐
                         │   Data & Model Registry     │
                         │ (DVC + free remote storage,   │
                         │  MLflow model registry,          │
                         │  or Hugging Face Hub)               │
                         └───────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Responsibility |
|---|---|
| Fleet Node (`edge_agent/`) | Runs production + shadow inference, emits telemetry, applies OTA model updates |
| Control Plane (`control_plane/`) | Fleet registry, rollout state machine, drift detection, retrain triggering, REST API |
| Training Pipeline (`training_pipeline/`) | Consumes hard-example queue, retrains, evaluates, registers new model versions |
| Dashboard (`dashboard/`) | Visualizes fleet state, rollout %, drift metrics, model lineage |
| Infra (`infra/`) | Docker Compose, k8s manifests, GitHub Actions workflows |

### 3.3 Data Flow (the closed loop)

1. Fleet nodes run inference on sampled input frames using the current production
   model, plus (optionally) one shadow candidate model.
2. Each inference emits a telemetry event (§7.2) to a Redis Streams topic.
3. The control plane consumes telemetry, computing per-node and fleet-wide metrics,
   and flags hard examples (low confidence, prod/shadow disagreement).
4. Hard examples are queued for labeling (simulated: revealed from a held-out
   labeled pool keyed by input ID).
5. Once the labeled hard-example count crosses a configured threshold, the control
   plane triggers the training pipeline (via a GitHub Actions `workflow_dispatch` or
   an internal job queue).
6. The training pipeline retrains, evaluates against a fixed validation set, and
   registers a new model version if it beats the current production model on the
   target metric (e.g. mAP@0.5).
7. The control plane starts a canary rollout: the new version is pushed (OTA) to a
   configurable % of nodes, running in shadow mode first, then promoted to
   production on those nodes if no regression/drift is detected.
8. Drift detection continuously compares canary vs. control-group node metrics; a
   statistically significant regression triggers automatic rollback.
9. The dashboard reflects every stage of this loop in near real time.

### 3.4 Technology Stack (all free-tier / self-hostable)

| Layer | Tool | Free-tier basis |
|---|---|---|
| Model | YOLOv8n/YOLO11n (Ultralytics), exported to ONNX | Open source, CPU inference |
| Inference serving | FastAPI + onnxruntime | Open source, self-hosted |
| Telemetry bus | Redis Streams | Open source, self-hosted via Docker |
| Control plane DB | PostgreSQL | Open source; free tier via Supabase/Neon for hosted demo |
| Data/model versioning | DVC + Google Drive remote (15 GB free) or Hugging Face Hub | Free storage tier |
| Experiment tracking | MLflow (self-hosted) | Open source |
| Fleet orchestration | Docker Compose / `k3d` (local Kubernetes) | Free, local |
| CI/CD & retrain trigger | GitHub Actions | Free for public repos |
| Dashboard | Grafana + Prometheus (self-hosted via Docker) | Open source |
| Demo hosting | Hugging Face Spaces | Docker SDK moved off the free tier — see §2.4 |
| Language | Python 3.11 (all services); optional Rust/Go edge agent as stretch goal | — |

---

## 4. Functional Requirements

Each requirement has an ID, priority (MoSCoW: Must/Should/Could/Won't for MVP), and
acceptance criteria.

### FR-1: Fleet Node Inference Service
**Priority:** Must
The system shall run an inference service per fleet node that accepts an input
frame, runs the current production model, and returns detections with per-box
confidence scores within 200ms on CPU for a 640×640 input.
*Acceptance:* `POST /infer` on a node returns a valid detection response for a
sample frame in under 200ms (p95) on the reference CPU environment.

### FR-2: Shadow Mode Evaluation
**Priority:** Must
The system shall allow a candidate model version to run on the same inputs as the
production model on a subset of nodes, logging its predictions without serving them
to any consumer.
*Acceptance:* Given a shadow model assigned to a node, telemetry events for that
node include both `prod_prediction` and `shadow_prediction` fields, and only
`prod_prediction` is returned to the caller.

### FR-3: Telemetry Streaming
**Priority:** Must
Every inference call shall emit a structured telemetry event (§7.2) to the central
bus within 1 second.
*Acceptance:* An event appears in the control plane's telemetry store within 1s of
the corresponding inference call, verified via integration test.

### FR-4: Hard-Example Mining
**Priority:** Must
The control plane shall flag an input as a hard example when (a) production
confidence falls below a configurable threshold, or (b) production and shadow
predictions disagree beyond a configurable IoU/class threshold.
*Acceptance:* Given synthetic telemetry with known confidence/disagreement values,
the flagged hard-example set matches the expected set exactly in a unit test.

### FR-5: Retrain Triggering
**Priority:** Must
The control plane shall automatically trigger the training pipeline once the count
of newly labeled hard examples exceeds a configurable threshold.
*Acceptance:* Crossing the threshold in a test environment results in a recorded
"retrain triggered" event and an invoked pipeline run (verified by trigger log).

### FR-6: Model Training & Evaluation
**Priority:** Must
The training pipeline shall retrain the detection model on the union of the base
dataset and newly labeled hard examples, evaluate on a fixed hold-out validation
set, and record metrics (mAP@0.5, per-class AP, inference latency) to the
experiment tracker.
*Acceptance:* A completed run produces a logged MLflow entry with all required
metrics and a model artifact.

### FR-7: Model Registry & Versioning
**Priority:** Must
Every trained model that improves on (or matches within tolerance) the current
production model's target metric shall be registered as a new version with
immutable lineage (training data version, hyperparameters, metrics).
*Acceptance:* Querying the registry for a version returns its data version, config,
and metrics; versions are never overwritten.

### FR-8: Canary Rollout
**Priority:** Must
The control plane shall support rolling a new model version out to a configurable
percentage of fleet nodes, in shadow mode first, then promoting to production per
node if no regression is observed after a configurable evaluation window.
*Acceptance:* Initiating a rollout at 20% assigns the new version to shadow mode on
~20% of nodes; after the evaluation window with no regression, those nodes serve
the new version as production.

### FR-9: Drift Detection & Automatic Rollback
**Priority:** Must
The control plane shall continuously compare canary-group vs. control-group node
metrics (confidence distribution, class distribution, latency) and automatically
revert canary nodes to the previous production model if a statistically significant
regression is detected.
*Acceptance:* Given synthetic canary metrics injected below the regression
threshold, the system emits a rollback event and canary nodes' active version
reverts within one polling interval.

### FR-10: Fleet & Rollout Dashboard
**Priority:** Must
The system shall provide a dashboard showing per-node model version, fleet-wide
confidence/latency trends, rollout progress, and rollback events.
*Acceptance:* Dashboard panels render live data from the control plane's metrics
endpoint with no manual refresh required (Grafana auto-refresh or equivalent).

### FR-11: Manual Override & Audit Log
**Priority:** Should
An operator shall be able to manually pause/resume a rollout or force a rollback via
the control plane API, with every action recorded in an audit log.
*Acceptance:* A manual rollback call is reflected in fleet state within one polling
interval and appears in the audit log with timestamp and actor.

### FR-12: Human-in-the-loop Labeling UI
**Priority:** Could (stretch)
Integrate an open-source labeling tool (e.g. Label Studio OSS) so hard examples can
be labeled by a real human rather than revealed from a held-out set.

---

## 5. External Interface Requirements

### 5.1 User Interfaces
- Grafana dashboard (read-only, embeddable) for fleet/rollout/drift visualization.
- A minimal operator CLI (`fleetctl`) for triggering rollouts, viewing fleet status,
  and forcing rollback — wraps the control-plane REST API.

### 5.2 API Interfaces (Control Plane, REST)

| Endpoint | Method | Purpose |
|---|---|---|
| `/fleet/nodes` | GET | List nodes and their current model versions |
| `/fleet/nodes/{id}/telemetry` | GET | Recent telemetry for a node |
| `/models` | GET | List registered model versions and metrics |
| `/models/{version}/rollout` | POST | Start a canary rollout for a version |
| `/models/{version}/rollback` | POST | Force rollback of a version |
| `/hard-examples` | GET | List currently queued hard examples |
| `/retrain/trigger` | POST | Manually trigger the training pipeline |
| `/drift/status` | GET | Current drift metrics per active rollout |

### 5.3 Software Interfaces
- **Redis Streams**: telemetry ingestion bus (`telemetry:events` stream).
- **PostgreSQL**: fleet state, model registry metadata, audit log.
- **MLflow tracking server**: experiment/run metadata and metrics.
- **DVC remote (Google Drive/S3-compatible free tier)**: dataset and model artifact
  storage.

### 5.4 Communication Interfaces
- Fleet nodes ↔ Control plane: HTTP/REST for OTA push and registration; Redis
  Streams for telemetry (pub, not RPC, to decouple node availability from ingestion).
- Control plane ↔ Training pipeline: triggered via GitHub Actions
  `repository_dispatch` API call, or an internal Redis-backed job queue for local
  runs.

---

## 6. Non-Functional Requirements

### 6.1 Performance
- NFR-1: p95 inference latency ≤ 200ms per frame on a 2-vCPU CPU-only node.
- NFR-2: Telemetry event end-to-end latency (node → control plane store) ≤ 1s.
- NFR-3: Drift detection polling interval ≤ 30s during an active rollout.

### 6.2 Scalability
- NFR-4: The fleet simulation shall support at least 20 concurrent simulated nodes
  on a single developer machine (16GB RAM) via Docker Compose.
- NFR-5: The architecture shall not hard-code node count; horizontal scaling is
  limited only by host resources or k3d cluster size.

### 6.3 Reliability & Availability
- NFR-6: Loss of a single fleet node shall not affect telemetry or rollout state for
  other nodes (no shared in-memory state between nodes).
- NFR-7: An automatic rollback shall complete (all canary nodes reverted) within one
  drift-detection polling interval of the regression being detected.

### 6.4 Security
- NFR-8: All inter-service API calls shall require an API key or service token; no
  unauthenticated write endpoints on the control plane.
- NFR-9: Model artifacts pushed OTA shall be checksum-verified by the receiving node
  before being loaded.
- NFR-10: No secrets (API keys, DB credentials) committed to the repository; managed
  via `.env` (gitignored) and GitHub Actions secrets.

### 6.5 Maintainability
- NFR-11: Each service (`edge_agent`, `control_plane`, `training_pipeline`,
  `dashboard`) is independently testable and independently deployable.
- NFR-12: All services expose a `/health` endpoint used by Docker Compose/k8s
  liveness checks.

### 6.6 Portability
- NFR-13: The entire system shall start via `docker compose up` with no manual
  per-service setup beyond a documented `.env`.

### 6.7 Cost Constraint
- NFR-14: The system shall incur $0 infrastructure cost when run locally or within
  the free-tier limits of GitHub Actions, Hugging Face Spaces, and the chosen DVC
  remote.

### 6.8 Observability
- NFR-15: Every service shall emit structured (JSON) logs and expose Prometheus
  metrics for request rate, error rate, and latency.

---

## 7. Data Requirements

### 7.1 Dataset
- **Primary**: BDD100K (Berkeley DeepDrive) subset — dashcam images with object
  detection labels (vehicles, pedestrians, traffic signs), free for research use.
- **Fallback**: a smaller public dashcam/street-scene dataset from Roboflow
  Universe if BDD100K's size is impractical for free-tier storage/compute.
- A fixed hold-out validation split is used both for model evaluation and for
  simulating "ground truth" reveal during hard-example labeling.

### 7.2 Telemetry Event Schema

```json
{
  "event_id": "uuid",
  "node_id": "string",
  "timestamp": "ISO-8601",
  "input_id": "string",
  "prod_model_version": "string",
  "prod_prediction": { "boxes": [...], "scores": [...], "classes": [...] },
  "shadow_model_version": "string | null",
  "shadow_prediction": { "boxes": [...], "scores": [...], "classes": [...] } ,
  "confidence_min": "float",
  "disagreement_score": "float | null",
  "latency_ms": "float"
}
```

### 7.3 Data Retention
- Raw telemetry retained 30 days (configurable) in Postgres; aggregated metrics
  retained indefinitely for the dashboard's historical trend views.
- Hard-example inputs and their eventual labels are retained permanently as part of
  the growing training set (versioned via DVC).

---

## 8. Feature Prioritization (MoSCoW / MVP Definition)

| Phase | Requirements included |
|---|---|
| **MVP (Phase 1)** | FR-1, FR-2, FR-3, FR-4, FR-6, FR-7, FR-10 (single-node shadow eval + manual retrain trigger + basic dashboard) |
| **Phase 2** | FR-5, FR-8, FR-9, FR-11 (full closed loop: auto-trigger, canary rollout, drift rollback) |
| **Phase 3 (stretch)** | FR-12, multi-region simulation, Rust/Go edge agent rewrite |

---

## 9. Verification & Validation Plan

| Requirement class | Verification method |
|---|---|
| Functional requirements (FR-x) | Unit tests per component + integration tests across the Docker Compose stack in CI |
| Performance (NFR-1–3) | Load-test script (e.g. `locust` or `k6`, both free/OSS) run in CI against the inference and telemetry endpoints |
| Reliability (NFR-6–7) | Chaos test: kill a node container mid-run, assert other nodes unaffected; inject synthetic regression, assert rollback timing |
| Security (NFR-8–10) | Automated secret-scanning in CI (e.g. `gitleaks`, free/OSS); auth tests for unauthenticated requests |
| Cost constraint (NFR-14) | Manual audit checklist before each release: confirm no paid service is a hard dependency |

**Definition of done for the portfolio milestone:** a `docker compose up` from a
clean clone starts the full stack, a scripted demo (`make demo`) feeds sample
frames through 5+ simulated nodes, triggers a retrain, runs a canary rollout, and
the Grafana dashboard visibly reflects each stage — recordable as a demo video for
applications.

---

## 10. Roadmap / Milestones

1. **M1 — Inference core**: single fleet node serving YOLOv8n via FastAPI/ONNX
   Runtime, dockerized, unit-tested. (Maps to FR-1)
2. **M2 — Telemetry + control plane skeleton**: Redis Streams ingestion, Postgres
   schema, `/fleet/nodes` endpoint. (FR-2, FR-3)
3. **M3 — Hard-example mining + manual retrain**: mining logic, MLflow-tracked
   training pipeline runnable on demand. (FR-4, FR-6, FR-7)
4. **M4 — Dashboard v1**: Grafana wired to control-plane metrics. (FR-10)
5. **M5 — Closed loop**: auto-trigger, canary rollout state machine, drift
   detection, automatic rollback. (FR-5, FR-8, FR-9)
6. **M6 — Polish for portfolio**: `make demo` script, recorded walkthrough, README
   architecture diagram, CI badge, Hugging Face Spaces demo of the inference API.

---

## Appendix A: Glossary

See §1.4.

## Appendix B: Repository Structure

```
newpro/
├── docs/
│   └── SRS.md                # this document
├── edge_agent/                # fleet node inference service
├── control_plane/             # fleet registry, rollout manager, drift detection
├── training_pipeline/         # retraining + evaluation + model registration
├── dashboard/                 # Grafana provisioning + custom panels
├── infra/                     # docker-compose.yml, k8s manifests, GitHub Actions
└── README.md
```
