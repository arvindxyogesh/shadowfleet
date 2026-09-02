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
from .db import FleetNode, TelemetryEvent, create_session_factory
from .schemas import FleetNodeOut, HealthResponse, TelemetryEventOut

logger = logging.getLogger("shadowfleet.control_plane")


async def _consume_forever(consumer: TelemetryConsumer, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await consumer.poll_once(block_ms=settings.poll_block_ms)
        except Exception:
            logger.exception("telemetry poll failed; will retry")
            await asyncio.sleep(settings.poll_error_backoff_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session_factory = create_session_factory(settings.database_url)
    app.state.redis = redis.from_url(settings.redis_url)
    app.state.consumer = TelemetryConsumer(
        app.state.redis, app.state.session_factory, settings.telemetry_stream
    )

    stop_event = asyncio.Event()
    consumer_task = asyncio.create_task(_consume_forever(app.state.consumer, stop_event))

    yield

    stop_event.set()
    consumer_task.cancel()
    await app.state.redis.aclose()


app = FastAPI(title="ShadowFleet Control Plane", version="0.1.0", lifespan=lifespan)


def get_session(request: Request) -> Session:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


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
