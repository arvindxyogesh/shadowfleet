import math
import statistics


def _welch_t_stat(mean_a: float, var_a: float, n_a: int, mean_b: float, var_b: float, n_b: int) -> float | None:
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return None
    return (mean_a - mean_b) / se


def detect_confidence_regression(
    canary_confidences: list[float],
    control_confidences: list[float],
    min_effect_size: float = 0.05,
    t_stat_threshold: float = 1.645,
) -> bool:
    """Flags a regression when the canary group's mean confidence is both
    meaningfully (min_effect_size) and statistically significantly
    (one-sided Welch's t-test, default threshold ~= 95% confidence) lower
    than the control group's.

    Requires at least 2 samples per group; returns False otherwise (FR-9
    should never fire a rollback off too little data).
    """
    if len(canary_confidences) < 2 or len(control_confidences) < 2:
        return False

    mean_canary = statistics.fmean(canary_confidences)
    mean_control = statistics.fmean(control_confidences)

    effect = mean_control - mean_canary
    if effect < min_effect_size:
        return False

    var_canary = statistics.variance(canary_confidences)
    var_control = statistics.variance(control_confidences)

    t_stat = _welch_t_stat(
        mean_control, var_control, len(control_confidences), mean_canary, var_canary, len(canary_confidences)
    )
    return t_stat is not None and t_stat > t_stat_threshold
