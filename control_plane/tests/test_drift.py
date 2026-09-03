from control_plane.app.drift import detect_confidence_regression


def test_no_regression_when_groups_perform_similarly():
    canary = [0.8, 0.82, 0.79, 0.81, 0.80]
    control = [0.81, 0.79, 0.80, 0.82, 0.80]
    assert detect_confidence_regression(canary, control) is False


def test_flags_regression_when_canary_confidence_drops_sharply():
    canary = [0.3, 0.32, 0.28, 0.31, 0.29, 0.30]
    control = [0.85, 0.83, 0.86, 0.84, 0.85, 0.84]
    assert detect_confidence_regression(canary, control) is True


def test_no_regression_when_canary_confidence_is_higher():
    canary = [0.9, 0.91, 0.89, 0.90, 0.92]
    control = [0.7, 0.71, 0.69, 0.70, 0.72]
    assert detect_confidence_regression(canary, control) is False


def test_small_difference_below_effect_size_is_not_flagged():
    canary = [0.80, 0.81, 0.79, 0.80, 0.80]
    control = [0.82, 0.83, 0.81, 0.82, 0.82]
    assert detect_confidence_regression(canary, control, min_effect_size=0.05) is False


def test_requires_at_least_two_samples_per_group():
    assert detect_confidence_regression([0.1], [0.9, 0.9, 0.9]) is False
    assert detect_confidence_regression([0.1, 0.1, 0.1], [0.9]) is False
    assert detect_confidence_regression([], []) is False


def test_stricter_t_stat_threshold_can_suppress_a_borderline_flag():
    canary = [0.60, 0.62, 0.58, 0.61, 0.59]
    control = [0.72, 0.70, 0.74, 0.71, 0.73]

    lenient = detect_confidence_regression(canary, control, t_stat_threshold=0.5)
    strict = detect_confidence_regression(canary, control, t_stat_threshold=50.0)

    assert lenient is True
    assert strict is False
