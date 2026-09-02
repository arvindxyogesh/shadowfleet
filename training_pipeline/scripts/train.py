"""Retrain the detection model on the base dataset plus newly labeled hard
examples pulled from the control plane, evaluate it, log to MLflow, and
register the result in the model registry.

This is a manual-trigger script (FR-6 the "training" of this milestone;
auto-triggering off the hard-example count is FR-5, a later milestone). It
requires network access, the `ultralytics` and `mlflow` packages, and real
training data — deliberately kept out of app/requirements.txt so the
service package stays light. Run by a developer or a CI job, not imported
by the test suite (app/dataset.py, app/evaluation.py, and app/registry.py
carry the logic that *is* unit-tested).

Usage:
    pip install ultralytics mlflow requests
    python scripts/train.py \
        --control-plane-url http://localhost:8001 \
        --base-dataset data/bdd100k_subset.yaml \
        --data-version bdd100k-subset-v1 \
        --epochs 20

Merging labeled hard examples into the Ultralytics-format dataset directory
(writing their image/label files into the training split) is dataset-layout
specific and left to the operator; app/dataset.py's build_training_manifest
does the record-level merge once inputs are in a common in-memory form.
"""

import argparse
from datetime import datetime, timezone

from app.dataset import select_labeled_hard_examples
from app.evaluation import ModelMetrics
from app.pipeline import register_trained_model
from app.registry import create_session_factory


def fetch_labeled_hard_examples(control_plane_url: str) -> list[dict]:
    import requests

    resp = requests.get(f"{control_plane_url}/hard-examples", params={"status": "labeled", "limit": 1000})
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control-plane-url", default="http://localhost:8001")
    parser.add_argument("--base-dataset", required=True, help="Ultralytics data YAML for the base dataset")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--parent-version", default=None)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--registry-db-url", default="sqlite:///./model_registry.db")
    parser.add_argument("--mlflow-experiment", default="shadowfleet")
    args = parser.parse_args()

    labeled = fetch_labeled_hard_examples(args.control_plane_url)
    manifest_additions = select_labeled_hard_examples(labeled)
    print(f"Pulled {len(manifest_additions)} labeled hard examples from the control plane")

    import mlflow
    from ultralytics import YOLO

    model = YOLO(args.weights)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_weights": args.weights,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "data_version": args.data_version,
                "hard_examples_added": len(manifest_additions),
            }
        )

        model.train(data=args.base_dataset, epochs=args.epochs, imgsz=args.imgsz)
        val_metrics = model.val()
        map50 = float(val_metrics.box.map50)
        mlflow.log_metric("map50", map50)

        exported_path = model.export(format="onnx", imgsz=args.imgsz, opset=12)
        mlflow.log_artifact(str(exported_path))

    version = f"yolov8n-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    session_factory = create_session_factory(args.registry_db_url)
    with session_factory() as session:
        record = register_trained_model(
            session,
            version=version,
            data_version=args.data_version,
            hyperparameters={"epochs": args.epochs, "imgsz": args.imgsz, "base_dataset": args.base_dataset},
            metrics=ModelMetrics(map50=map50, latency_ms=0.0),
            # TODO(M5): look up the current production version's metrics
            # from the registry once canary rollout tracks one.
            production_metrics=None,
            tolerance=args.tolerance,
            parent_version=args.parent_version,
        )

    print(f"Registered model version {record.version} (promoted={record.promoted})")
    print(f"ONNX artifact: {exported_path}")


if __name__ == "__main__":
    main()
