from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from control_plane.app.db import FleetNode, HardExample, TelemetryEvent, create_session_factory
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
        session.add(
            HardExample(
                input_id="hard-1",
                event_id="evt-1",
                node_id="node-1",
                reason="low_confidence",
                confidence_min=0.2,
                disagreement_score=None,
                flagged_at=datetime.utcnow(),
                status="pending",
            )
        )
        session.add(
            HardExample(
                input_id="hard-2",
                event_id="evt-2",
                node_id="node-1",
                reason="disagreement",
                confidence_min=0.9,
                disagreement_score=0.8,
                flagged_at=datetime.utcnow(),
                status="labeled",
                label={"boxes": []},
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


def test_list_hard_examples_returns_all_by_default():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.get("/hard-examples")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    input_ids = {ex["input_id"] for ex in resp.json()}
    assert input_ids == {"hard-1", "hard-2"}


def test_list_hard_examples_filters_by_status():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.get("/hard-examples", params={"status": "pending"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["input_id"] == "hard-1"


def test_label_hard_example_marks_it_labeled():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.post("/hard-examples/hard-1/label", json={"label": {"boxes": [1, 2, 3]}})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "labeled"
    assert body["label"] == {"boxes": [1, 2, 3]}


def test_label_hard_example_404_for_unknown_input():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.post("/hard-examples/does-not-exist/label", json={"label": {}})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_start_rollout_then_list_and_get_it():
    # node-1 and node-stale both exist with no base_url, so the rollout
    # manager's OTA push is a no-op here -- this test is about the API
    # wiring and persisted state, not the network call (covered in
    # test_rollout.py against a fake node client).
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/rollouts",
                json={"model_version": "v2", "model_path": "models/v2.onnx", "target_percentage": 100},
            )
            assert start_resp.status_code == 201
            rollout_id = start_resp.json()["id"]
            assert start_resp.json()["status"] == "shadow"

            list_resp = client.get("/rollouts")
            assert len(list_resp.json()) == 1

            detail_resp = client.get(f"/rollouts/{rollout_id}")
    finally:
        app.dependency_overrides.clear()

    assert detail_resp.status_code == 200
    assert len(detail_resp.json()["nodes"]) == 2


def test_get_rollout_404_for_unknown_id():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            resp = client.get("/rollouts/9999")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_rollout_pause_resume_and_rollback_endpoints():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/rollouts",
                json={"model_version": "v2", "model_path": "models/v2.onnx", "target_percentage": 100},
            )
            rollout_id = start_resp.json()["id"]

            pause_resp = client.post(f"/rollouts/{rollout_id}/pause", json={"actor": "alice"})
            assert pause_resp.status_code == 200
            assert pause_resp.json()["status"] == "paused"

            resume_resp = client.post(f"/rollouts/{rollout_id}/resume", json={"actor": "alice"})
            assert resume_resp.status_code == 200
            assert resume_resp.json()["status"] == "shadow"

            rollback_resp = client.post(
                f"/rollouts/{rollout_id}/rollback", json={"actor": "alice", "reason": "changed my mind"}
            )
            assert rollback_resp.status_code == 200
            assert rollback_resp.json()["status"] == "rolled_back"
            assert rollback_resp.json()["reason"] == "changed my mind"

            audit_resp = client.get("/audit-log")
    finally:
        app.dependency_overrides.clear()

    actions = {entry["action"] for entry in audit_resp.json()}
    assert {"rollout_started", "rollout_paused", "rollout_resumed", "rollback"} <= actions


def test_start_rollout_with_no_fleet_nodes_returns_400():
    empty_session_factory = create_session_factory("sqlite:///:memory:")
    app.dependency_overrides[get_session] = _override_session(empty_session_factory)
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/rollouts", json={"model_version": "v2", "model_path": "models/v2.onnx"}
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 400


def test_pause_a_never_started_rollout_returns_400():
    app.dependency_overrides[get_session] = _override_session(_seeded_session_factory())
    try:
        with TestClient(app) as client:
            start_resp = client.post(
                "/rollouts",
                json={"model_version": "v2", "model_path": "models/v2.onnx", "target_percentage": 100},
            )
            rollout_id = start_resp.json()["id"]
            client.post(f"/rollouts/{rollout_id}/pause", json={})

            second_pause_resp = client.post(f"/rollouts/{rollout_id}/pause", json={})
    finally:
        app.dependency_overrides.clear()

    assert second_pause_resp.status_code == 400
