import json
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .db import FleetNode, HardExample, TelemetryEvent
from .mining import evaluate_hard_example

logger = logging.getLogger("shadowfleet.control_plane.consumer")


class RedisLike(Protocol):
    async def xread(self, streams: dict, count: int, block: int) -> Any: ...


class TelemetryConsumer:
    """Polls the telemetry Redis Stream and persists events, fleet node
    state, and mined hard examples (FR-4).

    Uses a plain XREAD with an in-memory cursor rather than a consumer
    group: this system runs a single control-plane instance, so
    consumer-group semantics (multiple competing readers) aren't needed yet.
    """

    def __init__(
        self,
        redis_client: RedisLike,
        session_factory: sessionmaker,
        stream_name: str,
        hard_example_conf_threshold: float = 0.35,
        hard_example_disagreement_threshold: float = 0.5,
    ):
        self.redis = redis_client
        self.session_factory = session_factory
        self.stream_name = stream_name
        self.hard_example_conf_threshold = hard_example_conf_threshold
        self.hard_example_disagreement_threshold = hard_example_disagreement_threshold
        # Starts from the beginning of the stream: on a fresh process this
        # drains any backlog (e.g. events published before the consumer's
        # first poll). The cursor then advances in-memory as messages are
        # processed, so a later poll never reprocesses what an earlier one
        # already handled.
        self.last_id = "0"

    async def poll_once(self, count: int = 100, block_ms: int = 1000) -> int:
        response = await self.redis.xread({self.stream_name: self.last_id}, count=count, block=block_ms)
        if not response:
            return 0

        processed = 0
        session = self.session_factory()
        try:
            for _stream, messages in response:
                for message_id, fields in messages:
                    self.last_id = message_id.decode() if isinstance(message_id, bytes) else message_id
                    try:
                        self._persist(session, fields)
                        processed += 1
                    except Exception:
                        logger.exception("failed to persist telemetry message %s", self.last_id)
            session.commit()
        finally:
            session.close()
        return processed

    def _persist(self, session: Session, fields: dict) -> None:
        raw = fields.get(b"payload", fields.get("payload"))
        if isinstance(raw, bytes):
            raw = raw.decode()
        event = json.loads(raw)

        node_id = event["node_id"]
        timestamp = datetime.fromisoformat(event["timestamp"]).astimezone(timezone.utc).replace(tzinfo=None)

        node = session.get(FleetNode, node_id)
        if node is None:
            node = FleetNode(node_id=node_id, last_seen=timestamp)
            session.add(node)
        else:
            node.last_seen = timestamp
        node.prod_model_version = event.get("prod_model_version")
        node.shadow_model_version = event.get("shadow_model_version")
        if event.get("base_url"):
            node.base_url = event["base_url"]

        confidence_min = event.get("confidence_min")
        shadow_confidence_min = event.get("shadow_confidence_min")
        disagreement_score = event.get("disagreement_score")

        session.add(
            TelemetryEvent(
                event_id=event["event_id"],
                node_id=node_id,
                timestamp=timestamp,
                input_id=event.get("input_id"),
                prod_model_version=event.get("prod_model_version"),
                shadow_model_version=event.get("shadow_model_version"),
                confidence_min=confidence_min,
                shadow_confidence_min=shadow_confidence_min,
                disagreement_score=disagreement_score,
                latency_ms=event.get("latency_ms"),
                raw_payload=event,
            )
        )

        self._maybe_flag_hard_example(session, event, confidence_min, disagreement_score, timestamp)

    def _maybe_flag_hard_example(
        self,
        session: Session,
        event: dict,
        confidence_min: float | None,
        disagreement_score: float | None,
        timestamp: datetime,
    ) -> None:
        reason = evaluate_hard_example(
            confidence_min,
            disagreement_score,
            self.hard_example_conf_threshold,
            self.hard_example_disagreement_threshold,
        )
        if reason is None:
            return

        input_id = event.get("input_id")
        if input_id is None:
            return

        already_flagged = session.execute(
            select(HardExample).where(HardExample.input_id == input_id)
        ).scalar_one_or_none()
        if already_flagged is not None:
            return

        session.add(
            HardExample(
                input_id=input_id,
                event_id=event["event_id"],
                node_id=event["node_id"],
                reason=reason,
                confidence_min=confidence_min,
                disagreement_score=disagreement_score,
                flagged_at=timestamp,
                status="pending",
            )
        )
