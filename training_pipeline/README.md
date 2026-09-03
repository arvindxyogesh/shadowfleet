# training_pipeline

Consumes newly labeled hard examples from the control plane, retrains the
detection model, evaluates it, and registers a new model version if it
improves on (or matches within tolerance) the current production model.
Implements FR-6 (training/evaluation) and FR-7 (versioned registry with
immutable lineage) from `../docs/SRS.md`.

This milestone is a **manual-trigger** pipeline — a developer or CI job
runs it on demand. Auto-triggering off the hard-example count (FR-5) is a
later milestone.

## Package layout

- `app/dataset.py` — merges the base dataset with newly labeled hard
  examples into one training manifest (pure functions)
- `app/evaluation.py` — `should_promote()`: decides whether a candidate
  model's mAP@0.5 beats production within a configurable tolerance
- `app/registry.py` — `ModelVersion` table (SQLAlchemy) recording every
  trained version's data version, hyperparameters, metrics, and promotion
  outcome — nothing is ever overwritten, so training history is auditable
- `app/pipeline.py` — ties the above together: registers a trained run and
  marks it promoted or not
- `scripts/train.py` — the actual training entrypoint: pulls labeled hard
  examples from the control plane, trains via Ultralytics, evaluates, logs
  to MLflow, exports ONNX, and registers the result

`app/` has no dependency on `ultralytics`/`mlflow`/real training data and
is fully unit-tested. `scripts/train.py` does need all of that — it's kept
out of `requirements.txt` (see `requirements-train.txt`) the same way
`edge_agent/scripts/export_model.py` keeps `ultralytics` out of that
service's runtime dependencies.

## Running a real training job

```bash
pip install -r requirements-train.txt
python scripts/train.py \
    --control-plane-url http://localhost:8001 \
    --base-dataset data/bdd100k_subset.yaml \
    --data-version bdd100k-subset-v1 \
    --epochs 20
```

Requires a running control plane (to fetch labeled hard examples) and an
Ultralytics-format dataset YAML for the base dataset — see `docs/SRS.md`
§7.1 for the target dataset (BDD100K subset). Merging individual labeled
hard-example records into that dataset's image/label directory layout is
left to the operator; `app/dataset.py` handles the record-level merge once
inputs are in a common in-memory form.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests -v
```

All 12 tests run against pure functions and an in-memory SQLite registry —
no real model, dataset, or MLflow server required.
