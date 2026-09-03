from control_plane.app.mining import evaluate_hard_example


def test_no_flag_when_confident_and_agreeing():
    assert evaluate_hard_example(0.9, 0.1, conf_threshold=0.35, disagreement_threshold=0.5) is None


def test_flags_low_confidence():
    assert (
        evaluate_hard_example(0.2, 0.1, conf_threshold=0.35, disagreement_threshold=0.5)
        == "low_confidence"
    )


def test_flags_high_disagreement():
    assert (
        evaluate_hard_example(0.9, 0.8, conf_threshold=0.35, disagreement_threshold=0.5)
        == "disagreement"
    )


def test_flags_both_when_low_confidence_and_high_disagreement():
    assert evaluate_hard_example(0.1, 0.9, conf_threshold=0.35, disagreement_threshold=0.5) == "both"


def test_no_flag_when_confidence_is_none():
    # No detections in the frame (empty prediction set) -- not a mining
    # signal on its own in this milestone.
    assert evaluate_hard_example(None, 0.1, conf_threshold=0.35, disagreement_threshold=0.5) is None


def test_no_flag_when_disagreement_is_none():
    # No shadow model running on this node -- nothing to disagree with.
    assert evaluate_hard_example(0.9, None, conf_threshold=0.35, disagreement_threshold=0.5) is None


def test_boundary_values_are_not_flagged():
    # Exactly at threshold does not count as "below"/"beyond".
    assert evaluate_hard_example(0.35, 0.5, conf_threshold=0.35, disagreement_threshold=0.5) is None
