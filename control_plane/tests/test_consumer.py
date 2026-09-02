import json
from datetime import datetime, timezone

import fakeredis
import pytest

from control_plane.app.consumer import TelemetryConsumer
from control_plane.app.db import FleetNode, TelemetryEvent, create_session_factory


def make_event(node_id="node-1", event_id="evt-1", disagreement=None):
    return {
        "event_id": event_id,
        "node_id": node_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_id": "input-1",
        "prod_model_version": "yolov8n-baseline",
        "prod_prediction": [],
        "shadow_model_version": None,
        "shadow_prediction": None,
        "confidence_min": 0.7,
        "disagreement_score": disagreement,
        "latency_ms": 42.0,
    }


@pytest.mark.asyncio
async def test_poll_once_returns_zero_when_no_new_messages():
    redis_client = fakeredis.FakeAsyncRedis()
    session_factory = create_session_factory("sqlite:///:memory:")
    consumer = TelemetryConsumer(redis_client, session_factory, "telemetry:events")

    assert await consumer.poll_once(block_ms=50) == 0


@pytest.mark.asyncio
async def test_poll_once_persists_event_and_upserts_node():
    redis_client = fakeredis.FakeAsyncRedis()
    session_factory = create_session_factory("sqlite:///:memory:")
    consumer = TelemetryConsumer(redis_client, session_factory, "telemetry:events")

    await redis_client.xadd("telemetry:events", {"payload": json.dumps(make_event())})

    processed = await consumer.poll_once(block_ms=50)
    assert processed == 1

    with session_factory() as session:
        node = session.get(FleetNode, "node-1")
        assert node is not None
        assert node.prod_model_version == "yolov8n-baseline"

        events = session.query(TelemetryEvent).all()
        assert len(events) == 1
        assert events[0].event_id == "evt-1"
        assert events[0].confidence_min == 0.7


@pytest.mark.asyncio
async def test_poll_once_does_not_reprocess_already_consumed_messages():
    redis_client = fakeredis.FakeAsyncRedis()
    session_factory = create_session_factory("sqlite:///:memory:")
    consumer = TelemetryConsumer(redis_client, session_factory, "telemetry:events")

    await redis_client.xadd("telemetry:events", {"payload": json.dumps(make_event(event_id="evt-1"))})
    await consumer.poll_once(block_ms=50)

    await redis_client.xadd("telemetry:events", {"payload": json.dumps(make_event(event_id="evt-2"))})
    processed = await consumer.poll_once(block_ms=50)

    assert processed == 1
    with session_factory() as session:
        assert session.query(TelemetryEvent).count() == 2


@pytest.mark.asyncio
async def test_poll_once_updates_existing_node_on_second_event():
    redis_client = fakeredis.FakeAsyncRedis()
    session_factory = create_session_factory("sqlite:///:memory:")
    consumer = TelemetryConsumer(redis_client, session_factory, "telemetry:events")

    await redis_client.xadd("telemetry:events", {"payload": json.dumps(make_event(event_id="evt-1"))})
    await consumer.poll_once(block_ms=50)

    await redis_client.xadd(
        "telemetry:events",
        {"payload": json.dumps({**make_event(event_id="evt-2"), "prod_model_version": "yolov8n-v2"})},
    )
    await consumer.poll_once(block_ms=50)

    with session_factory() as session:
        assert session.query(FleetNode).count() == 1
        node = session.get(FleetNode, "node-1")
        assert node.prod_model_version == "yolov8n-v2"
