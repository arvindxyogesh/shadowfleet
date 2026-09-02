from datetime import datetime

from control_plane.app.db import HardExample, RetrainTrigger, create_session_factory
from control_plane.app.retrain import check_and_dispatch_retrain
from control_plane.app.retrain_dispatch import LoggingRetrainDispatcher


def _seed_labeled(session, n, used=False, status="labeled"):
    for i in range(n):
        session.add(
            HardExample(
                input_id=f"input-{status}-{used}-{i}",
                event_id=f"evt-{i}",
                node_id="node-0",
                reason="low_confidence",
                confidence_min=0.1,
                disagreement_score=None,
                flagged_at=datetime.utcnow(),
                status=status,
                used_in_training=used,
            )
        )


def test_does_not_dispatch_below_threshold():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _seed_labeled(session, 3)
        session.commit()

        result = check_and_dispatch_retrain(session, LoggingRetrainDispatcher(), threshold=5)

    assert result is None
    with session_factory() as session:
        assert session.query(RetrainTrigger).count() == 0


def test_dispatches_and_marks_examples_used_at_threshold():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _seed_labeled(session, 5)
        session.commit()

        result = check_and_dispatch_retrain(session, LoggingRetrainDispatcher(), threshold=5)

    assert result is not None
    assert result.labeled_example_count == 5
    assert result.dispatch_method == "log_only"

    with session_factory() as session:
        assert session.query(HardExample).filter_by(used_in_training=False).count() == 0
        assert session.query(RetrainTrigger).count() == 1


def test_only_counts_unused_labeled_examples():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _seed_labeled(session, 3, used=True)  # already folded into a prior run
        _seed_labeled(session, 2, used=False)
        session.commit()

        result = check_and_dispatch_retrain(session, LoggingRetrainDispatcher(), threshold=5)

    assert result is None  # only 2 unused, below threshold of 5


def test_ignores_pending_unlabeled_examples():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _seed_labeled(session, 5, status="pending")
        session.commit()

        result = check_and_dispatch_retrain(session, LoggingRetrainDispatcher(), threshold=5)

    assert result is None


def test_second_check_does_not_redispatch_the_same_examples():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        _seed_labeled(session, 5)
        session.commit()

        first = check_and_dispatch_retrain(session, LoggingRetrainDispatcher(), threshold=5)
        second = check_and_dispatch_retrain(session, LoggingRetrainDispatcher(), threshold=5)

    assert first is not None
    assert second is None
