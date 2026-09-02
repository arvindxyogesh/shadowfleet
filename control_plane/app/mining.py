def evaluate_hard_example(
    confidence_min: float | None,
    disagreement_score: float | None,
    conf_threshold: float,
    disagreement_threshold: float,
) -> str | None:
    """Decide whether a telemetry event should be mined as a hard example.

    Returns the flag reason, or None if the event doesn't qualify:
    - "low_confidence": the frame's weakest kept detection fell below conf_threshold
    - "disagreement": prod/shadow predictions disagreed beyond disagreement_threshold
    - "both": both conditions triggered
    """
    low_confidence = confidence_min is not None and confidence_min < conf_threshold
    high_disagreement = disagreement_score is not None and disagreement_score > disagreement_threshold

    if low_confidence and high_disagreement:
        return "both"
    if low_confidence:
        return "low_confidence"
    if high_disagreement:
        return "disagreement"
    return None
