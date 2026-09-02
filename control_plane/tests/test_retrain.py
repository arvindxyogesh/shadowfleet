from control_plane.app.retrain import should_trigger_retrain


def test_does_not_trigger_below_threshold():
    assert should_trigger_retrain(19, threshold=20) is False


def test_triggers_at_threshold():
    assert should_trigger_retrain(20, threshold=20) is True


def test_triggers_above_threshold():
    assert should_trigger_retrain(50, threshold=20) is True


def test_zero_unused_never_triggers():
    assert should_trigger_retrain(0, threshold=1) is False
