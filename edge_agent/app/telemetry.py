import json
import logging
from typing import Any, Protocol

import redis.asyncio as redis

logger = logging.getLogger("shadowfleet.edge_agent.telemetry")


class RedisLike(Protocol):
    async def xadd(self, name: str, fields: dict) -> Any: ...


class TelemetryPublisher:
    """Publishes telemetry events to the fleet's Redis Stream.

    Publishing is best-effort: a telemetry bus outage must never take down
    inference serving, so failures are logged and swallowed rather than
    propagated to the caller.
    """

    def __init__(self, redis_client: RedisLike, stream_name: str):
        self.redis = redis_client
        self.stream_name = stream_name

    async def publish(self, event: dict[str, Any]) -> None:
        try:
            await self.redis.xadd(self.stream_name, {"payload": json.dumps(event)})
        except Exception as exc:
            logger.warning("failed to publish telemetry event: %s", exc)


def build_redis_client(redis_url: str) -> redis.Redis:
    return redis.from_url(redis_url)
