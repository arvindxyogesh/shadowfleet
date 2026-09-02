from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, create_engine
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
