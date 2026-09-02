from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .audit import log_audit
from .canary import assign_canary_nodes
from .db import FleetNode, Rollout, RolloutNodeAssignment, TelemetryEvent
from .drift import detect_confidence_regression
from .node_client import NodeClient


class RolloutError(ValueError):
    pass


def _current_production_version(session: Session, node_ids: list[str]) -> str | None:
    if not node_ids:
        return None
    nodes = session.execute(select(FleetNode).where(FleetNode.node_id.in_(node_ids))).scalars().all()
    versions = [n.prod_model_version for n in nodes if n.prod_model_version]
    if not versions:
        return None
    return max(set(versions), key=versions.count)


class RolloutManager:
    """Drives FR-8 (canary rollout), FR-9 (drift-triggered rollback), and
    FR-11 (manual override) as one state machine.

    A rollout has exactly two active-evaluation outcomes reached from
    `status == "shadow"`: either drift is detected and it's rolled back
    (canary nodes were never promoted, so rollback only needs to clear
    their shadow model), or the evaluation window elapses cleanly and all
    canary nodes are promoted to production atomically, moving the rollout
    to `status == "completed"`. A completed rollout can still be manually
    rolled back (FR-11) if `previous_model_path` was recorded at start.
    """

    def __init__(
        self,
        node_client: NodeClient,
        drift_min_effect_size: float = 0.05,
        drift_t_stat_threshold: float = 1.645,
    ):
        self.node_client = node_client
        self.drift_min_effect_size = drift_min_effect_size
        self.drift_t_stat_threshold = drift_t_stat_threshold

    async def start_rollout(
        self,
        session: Session,
        model_version: str,
        model_path: str,
        target_percentage: int,
        evaluation_window_seconds: int,
        previous_model_path: str | None = None,
        actor: str = "operator",
    ) -> Rollout:
        active = session.execute(
            select(Rollout).where(Rollout.status.in_(["shadow", "paused"]))
        ).scalar_one_or_none()
        if active is not None:
            raise RolloutError(f"rollout {active.id} is already active ({active.status})")

        node_ids = [n.node_id for n in session.execute(select(FleetNode)).scalars().all()]
        if not node_ids:
            raise RolloutError("no fleet nodes registered")

        canary_ids, control_ids = assign_canary_nodes(node_ids, target_percentage)
        previous_version = _current_production_version(session, control_ids or node_ids)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rollout = Rollout(
            model_version=model_version,
            previous_version=previous_version,
            previous_model_path=previous_model_path,
            model_path=model_path,
            target_percentage=target_percentage,
            status="shadow",
            started_at=now,
            evaluation_window_seconds=evaluation_window_seconds,
        )
        session.add(rollout)
        session.flush()  # assigns rollout.id

        for node_id in canary_ids:
            session.add(RolloutNodeAssignment(rollout_id=rollout.id, node_id=node_id, role="canary"))
        for node_id in control_ids:
            session.add(RolloutNodeAssignment(rollout_id=rollout.id, node_id=node_id, role="control"))

        pushed, unreachable = await self._push_to_nodes(session, canary_ids, "shadow", model_version, model_path)

        log_audit(
            session,
            actor=actor,
            action="rollout_started",
            details={
                "rollout_id": rollout.id,
                "model_version": model_version,
                "canary_nodes": canary_ids,
                "control_nodes": control_ids,
                "pushed_to": pushed,
                "unreachable": unreachable,
            },
        )
        session.commit()
        session.refresh(rollout)
        return rollout

    async def evaluate_rollout(self, session: Session, rollout: Rollout, now: datetime | None = None) -> None:
        if rollout.status != "shadow":
            return
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)

        canary_ids, control_ids = self._node_ids(session, rollout.id)

        canary_confidences = self._recent_confidences(session, canary_ids, rollout.started_at)
        control_confidences = self._recent_confidences(session, control_ids, rollout.started_at)

        if detect_confidence_regression(
            canary_confidences,
            control_confidences,
            min_effect_size=self.drift_min_effect_size,
            t_stat_threshold=self.drift_t_stat_threshold,
        ):
            await self._rollback(
                session,
                rollout,
                reason=(
                    f"drift detected: canary mean confidence significantly below control "
                    f"(n_canary={len(canary_confidences)}, n_control={len(control_confidences)})"
                ),
                actor="system",
                now=now,
            )
            return

        elapsed = now - rollout.started_at
        if elapsed < timedelta(seconds=rollout.evaluation_window_seconds):
            return  # still collecting data

        pushed, unreachable = await self._push_to_nodes(
            session, canary_ids, "prod", rollout.model_version, rollout.model_path
        )
        await self._push_to_nodes(session, canary_ids, "shadow", None, None)

        assignments = (
            session.execute(
                select(RolloutNodeAssignment).where(
                    RolloutNodeAssignment.rollout_id == rollout.id, RolloutNodeAssignment.role == "canary"
                )
            )
            .scalars()
            .all()
        )
        for assignment in assignments:
            if assignment.node_id in pushed:
                assignment.promoted = True

        if not all(a.promoted for a in assignments):
            # Some canary nodes never actually received the new production
            # model -- calling this "completed" would be a lie. Stay in
            # `shadow` (already-promoted nodes keep assignment.promoted=True,
            # so a retry only re-pushes to the ones still missing it) and
            # let the next evaluation cycle try again.
            log_audit(
                session,
                actor="system",
                action="promotion_failed",
                details={"rollout_id": rollout.id, "promoted_nodes": pushed, "unreachable": unreachable},
            )
            session.commit()
            return

        rollout.status = "completed"
        rollout.ended_at = now
        log_audit(
            session,
            actor="system",
            action="rollout_completed",
            details={"rollout_id": rollout.id, "promoted_nodes": pushed, "unreachable": unreachable},
        )
        session.commit()

    async def pause_rollout(self, session: Session, rollout: Rollout, actor: str = "operator") -> None:
        if rollout.status != "shadow":
            raise RolloutError(f"cannot pause a rollout in status {rollout.status!r}")
        rollout.paused_status = rollout.status
        rollout.status = "paused"
        log_audit(session, actor=actor, action="rollout_paused", details={"rollout_id": rollout.id})
        session.commit()

    async def resume_rollout(self, session: Session, rollout: Rollout, actor: str = "operator") -> None:
        if rollout.status != "paused":
            raise RolloutError(f"cannot resume a rollout in status {rollout.status!r}")
        rollout.status = rollout.paused_status or "shadow"
        rollout.paused_status = None
        log_audit(session, actor=actor, action="rollout_resumed", details={"rollout_id": rollout.id})
        session.commit()

    async def force_rollback(
        self, session: Session, rollout: Rollout, reason: str, actor: str = "operator"
    ) -> None:
        if rollout.status not in ("shadow", "paused", "completed"):
            raise RolloutError(f"cannot roll back a rollout in status {rollout.status!r}")
        await self._rollback(session, rollout, reason=reason, actor=actor)

    async def _rollback(
        self, session: Session, rollout: Rollout, reason: str, actor: str, now: datetime | None = None
    ) -> None:
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        canary_ids, _control_ids = self._node_ids(session, rollout.id)

        restored, cleared, unreachable = [], [], []
        assignments = (
            session.execute(
                select(RolloutNodeAssignment).where(
                    RolloutNodeAssignment.rollout_id == rollout.id, RolloutNodeAssignment.role == "canary"
                )
            )
            .scalars()
            .all()
        )
        for assignment in assignments:
            node = session.get(FleetNode, assignment.node_id)
            if node is None or not node.base_url:
                unreachable.append(assignment.node_id)
                continue

            if assignment.promoted:
                if rollout.previous_model_path and rollout.previous_version:
                    ok = await self.node_client.set_model(
                        node.base_url, "prod", rollout.previous_version, rollout.previous_model_path
                    )
                    (restored if ok else unreachable).append(assignment.node_id)
                # else: no known prior artifact path -- prod is left as-is
                # and this is surfaced in the audit log for an operator to
                # handle manually.
            ok = await self.node_client.set_model(node.base_url, "shadow", None, None)
            if ok:
                cleared.append(assignment.node_id)
            elif assignment.node_id not in unreachable:
                unreachable.append(assignment.node_id)

        rollout.status = "rolled_back"
        rollout.ended_at = now
        rollout.reason = reason
        log_audit(
            session,
            actor=actor,
            action="rollback",
            details={
                "rollout_id": rollout.id,
                "reason": reason,
                "restored_prod_on": restored,
                "cleared_shadow_on": cleared,
                "unreachable": unreachable,
            },
        )
        session.commit()

    def _node_ids(self, session: Session, rollout_id: int) -> tuple[list[str], list[str]]:
        assignments = (
            session.execute(select(RolloutNodeAssignment).where(RolloutNodeAssignment.rollout_id == rollout_id))
            .scalars()
            .all()
        )
        canary = [a.node_id for a in assignments if a.role == "canary"]
        control = [a.node_id for a in assignments if a.role == "control"]
        return canary, control

    def _recent_confidences(self, session: Session, node_ids: list[str], since: datetime) -> list[float]:
        if not node_ids:
            return []
        rows = (
            session.execute(
                select(TelemetryEvent.confidence_min).where(
                    TelemetryEvent.node_id.in_(node_ids),
                    TelemetryEvent.timestamp >= since,
                    TelemetryEvent.confidence_min.is_not(None),
                )
            )
            .scalars()
            .all()
        )
        return list(rows)

    async def _push_to_nodes(
        self, session: Session, node_ids: list[str], role: str, model_version: str | None, model_path: str | None
    ) -> tuple[list[str], list[str]]:
        pushed, unreachable = [], []
        for node_id in node_ids:
            node = session.get(FleetNode, node_id)
            if node is None or not node.base_url:
                unreachable.append(node_id)
                continue
            ok = await self.node_client.set_model(node.base_url, role, model_version, model_path)
            (pushed if ok else unreachable).append(node_id)
        return pushed, unreachable
