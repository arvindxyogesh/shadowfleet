import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .consumer import TelemetryConsumer
from .db import (
    AuditLogEntry,
    FleetNode,
    HardExample,
    RetrainTrigger,
    Rollout,
    RolloutNodeAssignment,
    TelemetryEvent,
    create_session_factory,
)
from .node_client import HTTPNodeClient
from .retrain import check_and_dispatch_retrain
from .retrain_dispatch import LoggingRetrainDispatcher
from .rollout import RolloutError, RolloutManager
from .schemas import (
    ActorPayload,
    AuditLogEntryOut,
    FleetNodeOut,
    HardExampleOut,
    HealthResponse,
    LabelPayload,
    RetrainTriggerOut,
    RollbackPayload,
    RolloutDetailOut,
    RolloutNodeAssignmentOut,
    RolloutOut,
    StartRolloutRequest,
    TelemetryEventOut,
)

logger = logging.getLogger("shadowfleet.control_plane")


async def _consume_forever(consumer: TelemetryConsumer, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await consumer.poll_once(block_ms=settings.poll_block_ms)
        except Exception:
            logger.exception("telemetry poll failed; will retry")
            await asyncio.sleep(settings.poll_error_backoff_seconds)


async def _evaluate_rollouts_forever(app: FastAPI, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            session = app.state.session_factory()
            try:
                check_and_dispatch_retrain(
                    session, app.state.retrain_dispatcher, settings.retrain_trigger_threshold
                )
                active = session.execute(select(Rollout).where(Rollout.status == "shadow")).scalars().all()
                for rollout in active:
                    await app.state.rollout_manager.evaluate_rollout(session, rollout)
            finally:
                session.close()
        except Exception:
            logger.exception("rollout evaluation loop failed; will retry")
        await asyncio.sleep(settings.rollout_check_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session_factory = create_session_factory(settings.database_url)
    app.state.redis = redis.from_url(settings.redis_url)
    app.state.consumer = TelemetryConsumer(
        app.state.redis,
        app.state.session_factory,
        settings.telemetry_stream,
        hard_example_conf_threshold=settings.hard_example_conf_threshold,
        hard_example_disagreement_threshold=settings.hard_example_disagreement_threshold,
    )
    app.state.rollout_manager = RolloutManager(
        HTTPNodeClient(),
        drift_min_effect_size=settings.drift_min_effect_size,
        drift_t_stat_threshold=settings.drift_t_stat_threshold,
    )
    app.state.retrain_dispatcher = LoggingRetrainDispatcher()

    stop_event = asyncio.Event()
    consumer_task = asyncio.create_task(_consume_forever(app.state.consumer, stop_event))
    rollout_task = asyncio.create_task(_evaluate_rollouts_forever(app, stop_event))

    yield

    stop_event.set()
    consumer_task.cancel()
    rollout_task.cancel()
    await app.state.redis.aclose()


app = FastAPI(title="ShadowFleet Control Plane", version="0.1.0", lifespan=lifespan)


def get_session(request: Request) -> Session:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def _get_rollout_or_404(session: Session, rollout_id: int) -> Rollout:
    rollout = session.get(Rollout, rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail=f"unknown rollout: {rollout_id}")
    return rollout


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/fleet/nodes", response_model=list[FleetNodeOut])
def list_nodes(session: Session = Depends(get_session)) -> list[FleetNodeOut]:
    nodes = session.execute(select(FleetNode).order_by(FleetNode.node_id)).scalars().all()
    now = datetime.utcnow()
    stale_after = timedelta(seconds=settings.node_stale_after_seconds)
    return [
        FleetNodeOut(
            node_id=node.node_id,
            last_seen=node.last_seen,
            prod_model_version=node.prod_model_version,
            shadow_model_version=node.shadow_model_version,
            online=(now - node.last_seen) <= stale_after,
        )
        for node in nodes
    ]


@app.get("/fleet/nodes/{node_id}/telemetry", response_model=list[TelemetryEventOut])
def node_telemetry(
    node_id: str, limit: int = 50, session: Session = Depends(get_session)
) -> list[TelemetryEventOut]:
    events = (
        session.execute(
            select(TelemetryEvent)
            .where(TelemetryEvent.node_id == node_id)
            .order_by(TelemetryEvent.timestamp.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    if not events and session.get(FleetNode, node_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown node: {node_id}")
    return events


@app.get("/hard-examples", response_model=list[HardExampleOut])
def list_hard_examples(
    status: str | None = None, limit: int = 50, session: Session = Depends(get_session)
) -> list[HardExampleOut]:
    query = select(HardExample).order_by(HardExample.flagged_at.desc()).limit(limit)
    if status is not None:
        query = query.where(HardExample.status == status)
    return session.execute(query).scalars().all()


@app.post("/hard-examples/{input_id}/label", response_model=HardExampleOut)
def label_hard_example(
    input_id: str, payload: LabelPayload, session: Session = Depends(get_session)
) -> HardExampleOut:
    example = session.execute(
        select(HardExample).where(HardExample.input_id == input_id)
    ).scalar_one_or_none()
    if example is None:
        raise HTTPException(status_code=404, detail=f"unknown hard example: {input_id}")

    example.status = "labeled"
    example.label = payload.label
    session.commit()
    session.refresh(example)
    return example


@app.post("/rollouts", response_model=RolloutOut, status_code=201)
async def start_rollout(
    payload: StartRolloutRequest, request: Request, session: Session = Depends(get_session)
) -> RolloutOut:
    manager: RolloutManager = request.app.state.rollout_manager
    try:
        rollout = await manager.start_rollout(
            session,
            model_version=payload.model_version,
            model_path=payload.model_path,
            target_percentage=payload.target_percentage,
            evaluation_window_seconds=(
                payload.evaluation_window_seconds or settings.rollout_evaluation_window_seconds
            ),
            previous_model_path=payload.previous_model_path,
            actor=payload.actor,
        )
    except RolloutError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return rollout


@app.get("/rollouts", response_model=list[RolloutOut])
def list_rollouts(session: Session = Depends(get_session)) -> list[RolloutOut]:
    return session.execute(select(Rollout).order_by(Rollout.started_at.desc())).scalars().all()


@app.get("/rollouts/{rollout_id}", response_model=RolloutDetailOut)
def get_rollout(rollout_id: int, session: Session = Depends(get_session)) -> RolloutDetailOut:
    rollout = _get_rollout_or_404(session, rollout_id)
    assignments = (
        session.execute(select(RolloutNodeAssignment).where(RolloutNodeAssignment.rollout_id == rollout_id))
        .scalars()
        .all()
    )
    return RolloutDetailOut(
        id=rollout.id,
        model_version=rollout.model_version,
        previous_version=rollout.previous_version,
        target_percentage=rollout.target_percentage,
        status=rollout.status,
        started_at=rollout.started_at,
        evaluation_window_seconds=rollout.evaluation_window_seconds,
        ended_at=rollout.ended_at,
        reason=rollout.reason,
        nodes=[RolloutNodeAssignmentOut.model_validate(a) for a in assignments],
    )


@app.post("/rollouts/{rollout_id}/pause", response_model=RolloutOut)
async def pause_rollout(
    rollout_id: int, payload: ActorPayload, request: Request, session: Session = Depends(get_session)
) -> RolloutOut:
    manager: RolloutManager = request.app.state.rollout_manager
    rollout = _get_rollout_or_404(session, rollout_id)
    try:
        await manager.pause_rollout(session, rollout, actor=payload.actor)
    except RolloutError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.refresh(rollout)
    return rollout


@app.post("/rollouts/{rollout_id}/resume", response_model=RolloutOut)
async def resume_rollout(
    rollout_id: int, payload: ActorPayload, request: Request, session: Session = Depends(get_session)
) -> RolloutOut:
    manager: RolloutManager = request.app.state.rollout_manager
    rollout = _get_rollout_or_404(session, rollout_id)
    try:
        await manager.resume_rollout(session, rollout, actor=payload.actor)
    except RolloutError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.refresh(rollout)
    return rollout


@app.post("/rollouts/{rollout_id}/rollback", response_model=RolloutOut)
async def rollback_rollout(
    rollout_id: int, payload: RollbackPayload, request: Request, session: Session = Depends(get_session)
) -> RolloutOut:
    manager: RolloutManager = request.app.state.rollout_manager
    rollout = _get_rollout_or_404(session, rollout_id)
    try:
        await manager.force_rollback(session, rollout, reason=payload.reason, actor=payload.actor)
    except RolloutError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.refresh(rollout)
    return rollout


@app.get("/audit-log", response_model=list[AuditLogEntryOut])
def list_audit_log(limit: int = 50, session: Session = Depends(get_session)) -> list[AuditLogEntryOut]:
    return (
        session.execute(select(AuditLogEntry).order_by(AuditLogEntry.timestamp.desc()).limit(limit))
        .scalars()
        .all()
    )


@app.get("/retrain-triggers", response_model=list[RetrainTriggerOut])
def list_retrain_triggers(
    limit: int = 50, session: Session = Depends(get_session)
) -> list[RetrainTriggerOut]:
    return (
        session.execute(select(RetrainTrigger).order_by(RetrainTrigger.triggered_at.desc()).limit(limit))
        .scalars()
        .all()
    )
