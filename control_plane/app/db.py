from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class FleetNode(Base):
    """A fleet node's latest known state, upserted on every telemetry event."""

    __tablename__ = "fleet_nodes"

    node_id: Mapped[str] = mapped_column(String, primary_key=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime)
    prod_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    shadow_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    # Where the rollout manager reaches this node's /admin/model endpoint
    # for OTA pushes (FR-8). Null until the node reports one in telemetry.
    base_url: Mapped[str | None] = mapped_column(String, nullable=True)


class TelemetryEvent(Base):
    """One inference call's telemetry, per the SRS §7.2 event schema."""

    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    input_id: Mapped[str | None] = mapped_column(String, nullable=True)
    prod_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    shadow_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    disagreement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_payload: Mapped[dict] = mapped_column(JSON)


class HardExample(Base):
    """An input flagged by mining.evaluate_hard_example for labeling (FR-4)."""

    __tablename__ = "hard_examples"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    input_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    event_id: Mapped[str] = mapped_column(String, index=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    reason: Mapped[str] = mapped_column(String)
    confidence_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    disagreement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    flagged_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String, default="pending")
    label: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Set once this example has been folded into a RetrainTrigger's batch,
    # so should_trigger_retrain() only counts freshly labeled examples.
    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)


class RetrainTrigger(Base):
    """A record of FR-5 firing: enough labeled hard examples accumulated to
    warrant a retrain. `dispatch_status` reflects whether the pipeline was
    actually invoked (see app/retrain_dispatch.py).
    """

    __tablename__ = "retrain_triggers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime)
    labeled_example_count: Mapped[int] = mapped_column(Integer)
    threshold: Mapped[int] = mapped_column(Integer)
    dispatch_method: Mapped[str] = mapped_column(String)
    dispatch_status: Mapped[str] = mapped_column(String)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Rollout(Base):
    """A canary rollout of one model version across some percentage of the
    fleet (FR-8), including its outcome (FR-9/FR-11).
    """

    __tablename__ = "rollouts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String)
    previous_version: Mapped[str | None] = mapped_column(String, nullable=True)
    # Optional: lets a manual rollback (FR-11) restore the exact prior
    # artifact on already-promoted nodes. Without it, rollback still clears
    # the shadow model everywhere, but can't undo a completed promotion.
    previous_model_path: Mapped[str | None] = mapped_column(String, nullable=True)
    model_path: Mapped[str] = mapped_column(String)
    target_percentage: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String)  # shadow | promoting | completed | rolled_back | paused
    started_at: Mapped[datetime] = mapped_column(DateTime)
    evaluation_window_seconds: Mapped[int] = mapped_column(Integer)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    paused_status: Mapped[str | None] = mapped_column(String, nullable=True)


class RolloutNodeAssignment(Base):
    """One fleet node's role (canary/control) within a rollout."""

    __tablename__ = "rollout_node_assignments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    rollout_id: Mapped[int] = mapped_column(Integer, index=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)  # canary | control
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditLogEntry(Base):
    """Every automatic and manual rollout/retrain action (FR-11)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    actor: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


def create_session_factory(database_url: str) -> sessionmaker:
    """All timestamps are stored as naive UTC datetimes by convention, so the
    same schema works unmodified against SQLite (tests) and Postgres (real
    deployments), which differ in timezone-aware column support.
    """
    engine_kwargs: dict = {}
    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in database_url:
            engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(database_url, connect_args=connect_args, **engine_kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
