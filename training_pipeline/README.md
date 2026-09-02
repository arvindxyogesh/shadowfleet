# training_pipeline

Consumes newly labeled hard examples, retrains the detection model, evaluates
against a fixed hold-out set, logs to MLflow, and registers a new model
version if it improves on the current production model.

Implements: FR-6, FR-7. See `../docs/SRS.md` for evaluation criteria and
model registry requirements.

Planned stack: Ultralytics (YOLOv8/11 training) + MLflow tracking + DVC for
data/model versioning, triggered via GitHub Actions `repository_dispatch`.
