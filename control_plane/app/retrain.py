from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .audit import log_audit
from .db import HardExample, RetrainTrigger
from .retrain_dispatch import RetrainDispatcher


def should_trigger_retrain(unused_labeled_count: int, threshold: int) -> bool:
    """FR-5: retraining is triggered once the count of labeled hard
    examples not yet folded into a training run crosses `threshold`.
    """
    return unused_labeled_count >= threshold


def check_and_dispatch_retrain(
    session: Session, dispatcher: RetrainDispatcher, threshold: int
) -> RetrainTrigger | None:
    """Checks the threshold and, if crossed, dispatches a retrain and marks
    every currently-unused labeled hard example as consumed by this batch
    -- so the next check starts counting from zero rather than re-firing
    on the same examples every poll.
    """
    unused = session.execute(
        select(HardExample).where(HardExample.status == "labeled", HardExample.used_in_training.is_(False))
    ).scalars().all()

    if not should_trigger_retrain(len(unused), threshold):
        return None

    dispatch_method, dispatch_details = dispatcher.dispatch(len(unused), threshold)

    session.execute(
        update(HardExample)
        .where(HardExample.id.in_([h.id for h in unused]))
        .values(used_in_training=True)
    )

    trigger = RetrainTrigger(
        triggered_at=datetime.now(timezone.utc).replace(tzinfo=None),
        labeled_example_count=len(unused),
        threshold=threshold,
        dispatch_method=dispatch_method,
        dispatch_status="dispatched",
        details=dispatch_details,
    )
    session.add(trigger)
    log_audit(
        session,
        actor="system",
        action="retrain_triggered",
        details={"labeled_example_count": len(unused), "threshold": threshold, "dispatch_method": dispatch_method},
    )
    session.commit()
    session.refresh(trigger)
    return trigger
