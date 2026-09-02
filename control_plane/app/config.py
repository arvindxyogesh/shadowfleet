from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Control-plane configuration, overridable via SHADOWFLEET_CP_* env vars."""

    model_config = SettingsConfigDict(env_prefix="SHADOWFLEET_CP_", protected_namespaces=())

    database_url: str = "sqlite:///./shadowfleet.db"
    redis_url: str = "redis://localhost:6379/0"
    telemetry_stream: str = "telemetry:events"
    poll_block_ms: int = 1000
    poll_error_backoff_seconds: float = 1.0
    node_stale_after_seconds: int = 120


settings = Settings()
