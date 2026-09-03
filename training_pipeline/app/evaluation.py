from dataclasses import dataclass, field


@dataclass
class ModelMetrics:
    map50: float
    latency_ms: float
    per_class_ap: dict[str, float] = field(default_factory=dict)


def should_promote(
    candidate: ModelMetrics,
    production: ModelMetrics | None,
    tolerance: float = 0.01,
) -> bool:
    """A candidate model is promotable (FR-7: "improves on, or matches
    within tolerance") when there is no production model yet, or its
    mAP@0.5 is at least the production model's mAP@0.5 minus `tolerance`.
    """
    if production is None:
        return True
    return candidate.map50 >= production.map50 - tolerance
