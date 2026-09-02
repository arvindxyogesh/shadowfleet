from edge_agent.app.disagreement import compute_disagreement


def test_disagreement_zero_when_both_empty():
    assert compute_disagreement([], []) == 0.0


def test_disagreement_full_when_only_one_side_has_detections():
    prod = [{"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 1}]
    assert compute_disagreement(prod, []) == 1.0


def test_disagreement_zero_when_boxes_closely_match():
    prod = [{"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 1}]
    shadow = [{"box": {"x1": 0.5, "y1": 0.5, "x2": 10.5, "y2": 10.5}, "class_id": 1}]
    assert compute_disagreement(prod, shadow) == 0.0


def test_disagreement_full_when_boxes_dont_overlap():
    prod = [{"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 1}]
    shadow = [{"box": {"x1": 100, "y1": 100, "x2": 110, "y2": 110}, "class_id": 1}]
    assert compute_disagreement(prod, shadow) == 1.0


def test_disagreement_full_when_classes_differ_despite_overlap():
    prod = [{"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 1}]
    shadow = [{"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 2}]
    assert compute_disagreement(prod, shadow) == 1.0


def test_disagreement_partial_when_only_some_detections_match():
    prod = [
        {"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 1},
        {"box": {"x1": 50, "y1": 50, "x2": 60, "y2": 60}, "class_id": 2},
    ]
    shadow = [{"box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "class_id": 1}]
    assert compute_disagreement(prod, shadow) == 0.5
