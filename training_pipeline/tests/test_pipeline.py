from training_pipeline.app.evaluation import ModelMetrics
from training_pipeline.app.pipeline import register_trained_model
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
