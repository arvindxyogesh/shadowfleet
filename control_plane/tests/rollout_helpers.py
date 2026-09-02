"""Shared test doubles for rollout/retrain tests."""

from datetime import datetime


class FakeNodeClient:
    def __init__(self, unreachable: set[str] | None = None):
        self.calls: list[tuple[str, str, str | None, str | None]] = []
        self.unreachable = unreachable or set()

    async def set_model(self, base_url, role, model_version, model_path):
        self.calls.append((base_url, role, model_version, model_path))
        return base_url not in self.unreachable


def seed_node(session, node_id, base_url="", prod_version="v1", last_seen=None):
    from control_plane.app.db import FleetNode

    session.add(
        FleetNode(
            node_id=node_id,
            last_seen=last_seen or datetime.utcnow(),
            prod_model_version=prod_version,
            shadow_model_version=None,
            base_url=base_url or f"http://{node_id}:8000",
        )
    )


def seed_telemetry(session, node_id, confidence_min, timestamp, event_id=None):
    import uuid

    from control_plane.app.db import TelemetryEvent

    session.add(
        TelemetryEvent(
            event_id=event_id or str(uuid.uuid4()),
            node_id=node_id,
            timestamp=timestamp,
            input_id=str(uuid.uuid4()),
            prod_model_version="v1",
            shadow_model_version=None,
            confidence_min=confidence_min,
            disagreement_score=None,
            latency_ms=10.0,
            raw_payload={},
        )
    )
