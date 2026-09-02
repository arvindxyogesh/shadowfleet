from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    box: BoundingBox
    score: float
    class_id: int
    class_name: str


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    node_id: str
    model_version: str
    detections: list[Detection]
    latency_ms: float
    image_width: int
    image_height: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    node_id: str
    model_version: str
    model_loaded: bool
