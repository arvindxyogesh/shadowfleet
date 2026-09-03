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

    # FR-4: a telemetry event is mined as a hard example when its weakest
    # kept detection falls below this confidence, or prod/shadow
    # disagreement exceeds this threshold.
    hard_example_conf_threshold: float = 0.35
    hard_example_disagreement_threshold: float = 0.5

    # FR-5: retraining is triggered once this many labeled hard examples
    # have accumulated since the last trigger.
    retrain_trigger_threshold: int = 20

    # FR-8/FR-9: how long a canary stays in shadow mode before promotion,
    # how often the background loop re-evaluates active rollouts, and the
    # drift thresholds that decide an automatic rollback.
    rollout_evaluation_window_seconds: int = 300
    rollout_check_interval_seconds: float = 30.0
    drift_min_effect_size: float = 0.05
    drift_t_stat_threshold: float = 1.645


settings = Settings()
