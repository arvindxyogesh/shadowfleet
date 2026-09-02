from datetime import datetime, timedelta

import pytest

from control_plane.app.db import AuditLogEntry, Rollout, RolloutNodeAssignment, create_session_factory
from control_plane.app.rollout import RolloutError, RolloutManager
from control_plane.tests.rollout_helpers import FakeNodeClient, seed_node, seed_telemetry


def _make_manager(node_client=None):
    return RolloutManager(node_client or FakeNodeClient())


@pytest.mark.asyncio
async def test_start_rollout_assigns_canary_and_pushes_shadow_model():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient()
    manager = _make_manager(node_client)

    with session_factory() as session:
        for i in range(10):
            seed_node(session, f"node-{i}")
        session.commit()

        rollout = await manager.start_rollout(
            session,
            model_version="v2",
            model_path="models/v2.onnx",
            target_percentage=20,
            evaluation_window_seconds=300,
        )

        assert rollout.status == "shadow"
        assert rollout.previous_version == "v1"

        assignments = session.query(RolloutNodeAssignment).filter_by(rollout_id=rollout.id).all()
        canary = [a for a in assignments if a.role == "canary"]
        control = [a for a in assignments if a.role == "control"]
        assert len(canary) == 2
        assert len(control) == 8

    # Every canary node got a shadow-model push, no control nodes touched.
    assert len(node_client.calls) == 2
    for _base_url, role, version, path in node_client.calls:
        assert role == "shadow"
        assert version == "v2"
        assert path == "models/v2.onnx"

    with session_factory() as session:
        audit = session.query(AuditLogEntry).filter_by(action="rollout_started").one()
        assert audit.details["model_version"] == "v2"


@pytest.mark.asyncio
async def test_start_rollout_rejects_a_second_concurrent_rollout():
    session_factory = create_session_factory("sqlite:///:memory:")
    manager = _make_manager()

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)

        with pytest.raises(RolloutError):
            await manager.start_rollout(session, "v3", "models/v3.onnx", 100, 300)


@pytest.mark.asyncio
async def test_start_rollout_rejects_empty_fleet():
    session_factory = create_session_factory("sqlite:///:memory:")
    manager = _make_manager()

    with session_factory() as session:
        with pytest.raises(RolloutError):
            await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)


@pytest.mark.asyncio
async def test_evaluate_rollout_promotes_after_window_with_no_drift():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient()
    manager = _make_manager(node_client)

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)
        node_client.calls.clear()  # drop the initial shadow push

        started_at = rollout.started_at
        # Similar confidence on both sides -- no drift.
        for i in range(5):
            seed_telemetry(session, "node-0", 0.8 + i * 0.01, started_at + timedelta(seconds=i))
        session.commit()

        past_window = started_at + timedelta(seconds=301)
        await manager.evaluate_rollout(session, rollout, now=past_window)

        session.refresh(rollout)
        assert rollout.status == "completed"
        assignment = session.query(RolloutNodeAssignment).filter_by(rollout_id=rollout.id, role="canary").one()
        assert assignment.promoted is True

    # prod push + shadow-clear push for the one canary node
    roles_called = [call[1] for call in node_client.calls]
    assert roles_called.count("prod") == 1
    assert roles_called.count("shadow") == 1


@pytest.mark.asyncio
async def test_evaluate_rollout_does_not_complete_when_ota_push_fails_everywhere():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient(unreachable={"http://node-0:8000"})
    manager = _make_manager(node_client)

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)

        past_window = rollout.started_at + timedelta(seconds=301)
        await manager.evaluate_rollout(session, rollout, now=past_window)

        session.refresh(rollout)
        # A rollout whose OTA push never actually landed on any node must
        # not be reported as "completed" -- that would be a lie the
        # dashboard and API would repeat.
        assert rollout.status == "shadow"
        assert rollout.ended_at is None

        assignment = session.query(RolloutNodeAssignment).filter_by(rollout_id=rollout.id, role="canary").one()
        assert assignment.promoted is False

        actions = [a.action for a in session.query(AuditLogEntry).all()]
        assert "promotion_failed" in actions
        assert "rollout_completed" not in actions


@pytest.mark.asyncio
async def test_evaluate_rollout_retries_promotion_on_a_later_cycle_after_failure():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient(unreachable={"http://node-0:8000"})
    manager = _make_manager(node_client)

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)

        past_window = rollout.started_at + timedelta(seconds=301)
        await manager.evaluate_rollout(session, rollout, now=past_window)
        session.refresh(rollout)
        assert rollout.status == "shadow"

        # The node comes back; the next cycle should succeed.
        node_client.unreachable.clear()
        await manager.evaluate_rollout(session, rollout, now=past_window + timedelta(seconds=30))

        session.refresh(rollout)
        assert rollout.status == "completed"


@pytest.mark.asyncio
async def test_evaluate_rollout_does_nothing_before_window_elapses():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient()
    manager = _make_manager(node_client)

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)
        node_client.calls.clear()

        still_within_window = rollout.started_at + timedelta(seconds=10)
        await manager.evaluate_rollout(session, rollout, now=still_within_window)

        session.refresh(rollout)
        assert rollout.status == "shadow"

    assert node_client.calls == []


@pytest.mark.asyncio
async def test_evaluate_rollout_rolls_back_on_detected_drift():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient()
    manager = _make_manager(node_client)

    with session_factory() as session:
        for i in range(4):
            seed_node(session, f"node-{i}")
        session.commit()

        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 50, 300)
        node_client.calls.clear()

        canary_ids, control_ids = manager._node_ids(session, rollout.id)
        started_at = rollout.started_at
        for i in range(6):
            for node_id in canary_ids:
                seed_telemetry(session, node_id, 0.2 + i * 0.01, started_at + timedelta(seconds=i))
            for node_id in control_ids:
                seed_telemetry(session, node_id, 0.85 + i * 0.01, started_at + timedelta(seconds=i))
        session.commit()

        still_within_window = started_at + timedelta(seconds=10)
        await manager.evaluate_rollout(session, rollout, now=still_within_window)

        session.refresh(rollout)
        assert rollout.status == "rolled_back"
        assert "drift" in rollout.reason

    # Rollback clears the shadow model on every canary node; nothing was
    # ever promoted, so no "prod" calls should occur.
    roles_called = [call[1] for call in node_client.calls]
    assert "prod" not in roles_called
    assert roles_called.count("shadow") == len(canary_ids)


@pytest.mark.asyncio
async def test_pause_and_resume_rollout():
    session_factory = create_session_factory("sqlite:///:memory:")
    manager = _make_manager()

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)

        await manager.pause_rollout(session, rollout, actor="alice")
        session.refresh(rollout)
        assert rollout.status == "paused"

        await manager.resume_rollout(session, rollout, actor="alice")
        session.refresh(rollout)
        assert rollout.status == "shadow"

        actions = [a.action for a in session.query(AuditLogEntry).all()]
        assert "rollout_paused" in actions
        assert "rollout_resumed" in actions


@pytest.mark.asyncio
async def test_cannot_pause_a_rollout_that_is_not_active():
    session_factory = create_session_factory("sqlite:///:memory:")
    manager = _make_manager()

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)
        await manager.pause_rollout(session, rollout)

        with pytest.raises(RolloutError):
            await manager.pause_rollout(session, rollout)


@pytest.mark.asyncio
async def test_force_rollback_on_completed_rollout_restores_prod_when_path_known():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient()
    manager = _make_manager(node_client)

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(
            session,
            "v2",
            "models/v2.onnx",
            100,
            300,
            previous_model_path="models/v1.onnx",
        )
        await manager.evaluate_rollout(session, rollout, now=rollout.started_at + timedelta(seconds=301))
        session.refresh(rollout)
        assert rollout.status == "completed"

        node_client.calls.clear()
        await manager.force_rollback(session, rollout, reason="bad in prod", actor="bob")

        session.refresh(rollout)
        assert rollout.status == "rolled_back"
        assert rollout.reason == "bad in prod"

    roles_and_versions = [(c[1], c[2]) for c in node_client.calls]
    assert ("prod", "v1") in roles_and_versions
    assert ("shadow", None) in roles_and_versions


@pytest.mark.asyncio
async def test_evaluate_rollout_is_a_noop_for_non_shadow_status():
    session_factory = create_session_factory("sqlite:///:memory:")
    node_client = FakeNodeClient()
    manager = _make_manager(node_client)

    with session_factory() as session:
        seed_node(session, "node-0")
        session.commit()
        rollout = await manager.start_rollout(session, "v2", "models/v2.onnx", 100, 300)
        await manager.pause_rollout(session, rollout)
        node_client.calls.clear()

        await manager.evaluate_rollout(session, rollout, now=datetime.utcnow() + timedelta(days=1))

    assert node_client.calls == []
