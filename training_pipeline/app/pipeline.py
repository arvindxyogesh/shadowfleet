import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .evaluation import ModelMetrics, should_promote
from .registry import ModelVersion

logger = logging.getLogger("shadowfleet.training_pipeline")


def get_current_production_metrics(session: Session) -> ModelMetrics | None:
    """The metrics of the most recently registered promoted version — the
    baseline a new candidate must match within tolerance (FR-7). None when
    nothing has been promoted yet (the first-ever training run).
    """
    current = session.execute(
        select(ModelVersion)
        .where(ModelVersion.promoted.is_(True))
        .order_by(ModelVersion.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if current is None:
        return None
    return ModelMetrics(
        map50=current.metrics["map50"],
        latency_ms=current.metrics["latency_ms"],
        per_class_ap=current.metrics.get("per_class_ap", {}),
    )


def register_trained_model(
    session: Session,
    version: str,
    data_version: str,
    hyperparameters: dict,
    metrics: ModelMetrics,
    production_metrics: ModelMetrics | None,
    tolerance: float = 0.01,
    parent_version: str | None = None,
) -> ModelVersion:
    """Registers a trained model version with immutable lineage (FR-7).

    Marks it `promoted` only if evaluation.should_promote judges it fit to
    replace the current production model — i.e. eligible for the canary
    rollout a later milestone will drive. A non-promoted version is still
    registered, so every training run is auditable even when it regresses.
    """
    promoted = should_promote(metrics, production_metrics, tolerance)

    record = ModelVersion(
        version=version,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        data_version=data_version,
        hyperparameters=hyperparameters,
        metrics={
            "map50": metrics.map50,
            "latency_ms": metrics.latency_ms,
            "per_class_ap": metrics.per_class_ap,
        },
        promoted=promoted,
        parent_version=parent_version,
    )
    session.add(record)
    session.commit()

    logger.info("registered model version %s (promoted=%s)", version, promoted)
    return record
