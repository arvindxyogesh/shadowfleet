from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Node configuration, overridable via SHADOWFLEET_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="SHADOWFLEET_", protected_namespaces=())

    node_id: str = "node-local"
    model_path: str = "models/yolov8n.onnx"
    model_version: str = "yolov8n-baseline"
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300

    # Shadow-mode candidate model: runs alongside production, logged but
    # never served. Unset by default (no shadow evaluation).
    shadow_model_path: str | None = None
    shadow_model_version: str = "shadow-candidate"

    redis_url: str = "redis://localhost:6379/0"
    telemetry_stream: str = "telemetry:events"

    # Reachable base URL for this node, reported in telemetry so the
    # control plane knows where to send OTA model updates (FR-8). Unset
    # in single-node/local dev; required for a node to receive rollouts.
    self_base_url: str | None = None


settings = Settings()
