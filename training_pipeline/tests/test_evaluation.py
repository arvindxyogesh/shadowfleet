from training_pipeline.app.evaluation import ModelMetrics, should_promote


def test_promotes_when_no_production_model_exists():
    candidate = ModelMetrics(map50=0.3, latency_ms=20.0)
    assert should_promote(candidate, None) is True


def test_promotes_when_candidate_strictly_better():
    candidate = ModelMetrics(map50=0.55, latency_ms=20.0)
    production = ModelMetrics(map50=0.5, latency_ms=20.0)
    assert should_promote(candidate, production) is True


def test_promotes_when_within_tolerance():
    candidate = ModelMetrics(map50=0.495, latency_ms=20.0)
    production = ModelMetrics(map50=0.5, latency_ms=20.0)
    assert should_promote(candidate, production, tolerance=0.01) is True


def test_rejects_when_regression_exceeds_tolerance():
    candidate = ModelMetrics(map50=0.4, latency_ms=20.0)
    production = ModelMetrics(map50=0.5, latency_ms=20.0)
    assert should_promote(candidate, production, tolerance=0.01) is False


def test_boundary_exactly_at_tolerance_is_promoted():
    candidate = ModelMetrics(map50=0.49, latency_ms=20.0)
    production = ModelMetrics(map50=0.5, latency_ms=20.0)
    assert should_promote(candidate, production, tolerance=0.01) is True
