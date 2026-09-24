from datetime import datetime, timedelta

from training_pipeline.app.evaluation import ModelMetrics
from training_pipeline.app.pipeline import get_current_production_metrics, register_trained_model
from training_pipeline.app.registry import ModelVersion, create_session_factory


def test_register_trained_model_promotes_when_no_prior_production_model():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        record = register_trained_model(
            session,
            version="yolov8n-v1",
            data_version="base-v1",
            hyperparameters={"epochs": 20},
            metrics=ModelMetrics(map50=0.4, latency_ms=15.0),
            production_metrics=None,
        )

    assert record.promoted is True
    assert record.version == "yolov8n-v1"

    with session_factory() as session:
        stored = session.get(ModelVersion, "yolov8n-v1")
        assert stored is not None
        assert stored.metrics["map50"] == 0.4
        assert stored.data_version == "base-v1"


def test_register_trained_model_does_not_promote_a_regression():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        record = register_trained_model(
            session,
            version="yolov8n-v2",
            data_version="base-v1-plus-hard-examples",
            hyperparameters={"epochs": 20},
            metrics=ModelMetrics(map50=0.3, latency_ms=15.0),
            production_metrics=ModelMetrics(map50=0.5, latency_ms=15.0),
            tolerance=0.01,
            parent_version="yolov8n-v1",
        )

    assert record.promoted is False
    assert record.parent_version == "yolov8n-v1"


def test_register_trained_model_persists_every_run_regardless_of_promotion():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        register_trained_model(
            session,
            version="v1",
            data_version="d1",
            hyperparameters={},
            metrics=ModelMetrics(map50=0.2, latency_ms=10.0),
            production_metrics=ModelMetrics(map50=0.9, latency_ms=10.0),
        )

    with session_factory() as session:
        stored = session.get(ModelVersion, "v1")
        assert stored is not None
        assert stored.promoted is False


def _train_run(session, version, map50, tolerance=0.01):
    """Mirrors scripts/train.py: gate a new candidate against whatever the
    registry currently holds as the promoted production version."""
    return register_trained_model(
        session,
        version=version,
        data_version="d1",
        hyperparameters={},
        metrics=ModelMetrics(map50=map50, latency_ms=15.0),
        production_metrics=get_current_production_metrics(session),
        tolerance=tolerance,
    )


def test_get_current_production_metrics_is_none_for_an_empty_registry():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        assert get_current_production_metrics(session) is None


def test_first_ever_run_promotes_against_an_empty_registry():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        record = _train_run(session, "v1", map50=0.1)

    assert record.promoted is True


def test_candidate_matching_promoted_version_within_tolerance_promotes():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _train_run(session, "v1", map50=0.5)
        matching = _train_run(session, "v2", map50=0.495, tolerance=0.01)
        better = _train_run(session, "v3", map50=0.6, tolerance=0.01)

    assert matching.promoted is True
    assert better.promoted is True


def test_candidate_regressing_beyond_tolerance_is_not_promoted():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _train_run(session, "v1", map50=0.5)
        regression = _train_run(session, "v2", map50=0.4, tolerance=0.01)
        # The rejected run must not become the new baseline: a later
        # candidate is still judged against v1's 0.5, not v2's 0.4.
        baseline_after = get_current_production_metrics(session)
        still_regressed = _train_run(session, "v3", map50=0.45, tolerance=0.01)

    assert regression.promoted is False
    assert baseline_after.map50 == 0.5
    assert still_regressed.promoted is False


def test_get_current_production_metrics_uses_latest_promoted_version():
    session_factory = create_session_factory("sqlite:///:memory:")
    now = datetime(2026, 1, 1)
    with session_factory() as session:
        for version, age_days, promoted, map50 in [
            ("old-promoted", 3, True, 0.7),
            ("current", 2, True, 0.6),
            ("newer-rejected", 1, False, 0.2),
        ]:
            session.add(
                ModelVersion(
                    version=version,
                    created_at=now - timedelta(days=age_days),
                    data_version="d1",
                    hyperparameters={},
                    metrics={"map50": map50, "latency_ms": 12.0, "per_class_ap": {"car": map50}},
                    promoted=promoted,
                )
            )
        session.commit()

        production = get_current_production_metrics(session)

    assert production == ModelMetrics(map50=0.6, latency_ms=12.0, per_class_ap={"car": 0.6})
