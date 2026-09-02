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


settings = Settings()
