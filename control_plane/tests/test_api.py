from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from control_plane.app.db import FleetNode, TelemetryEvent, create_session_factory
from control_plane.app.main import app, get_session


def _seeded_session_factory():
    session_factory = create_session_factory("sqlite:///:memory:")
    with session_factory() as session:
        session.add(
            FleetNode(
                node_id="node-1",
                last_seen=datetime.utcnow(),
                prod_model_version="yolov8n-baseline",
                shadow_model_version=None,
            )
        )
        session.add(
            FleetNode(
                node_id="node-stale",
                last_seen=datetime.utcnow() - timedelta(hours=1),
                prod_model_version="yolov8n-baseline",
                shadow_model_version=None,
            )
        )
        session.add(
            TelemetryEvent(
                event_id="evt-1",
                node_id="node-1",
                timestamp=datetime.utcnow(),
                input_id="in-1",
                prod_model_version="yolov8n-baseline",
                shadow_model_version=None,
                confidence_min=0.8,
                disagreement_score=None,
                latency_ms=30.0,
                raw_payload={},
            )
        )
        session.commit()
    return session_factory


def _override_session(session_factory):
    def _get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    return _get_session


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_nodes_reports_online_and_stale_nodes():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.get("/fleet/nodes")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = {n["node_id"]: n for n in resp.json()}
    assert body["node-1"]["online"] is True
    assert body["node-stale"]["online"] is False


def test_node_telemetry_returns_events_for_known_node():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.get("/fleet/nodes/node-1/telemetry")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["event_id"] == "evt-1"


def test_node_telemetry_404_for_unknown_node():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.get("/fleet/nodes/does-not-exist/telemetry")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404
