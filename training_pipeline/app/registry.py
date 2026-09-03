from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    """Immutable lineage record for a trained model (FR-7)."""

    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    data_version: Mapped[str] = mapped_column(String)
    hyperparameters: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    parent_version: Mapped[str | None] = mapped_column(String, nullable=True)


def create_session_factory(database_url: str) -> sessionmaker:
    engine_kwargs: dict = {}
    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in database_url:
            engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(database_url, connect_args=connect_args, **engine_kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
