from pydantic import BaseModel, ConfigDict, Field


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


class SetModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    role: str  # "prod" | "shadow"
    model_version: str | None = None
    model_path: str | None = None
    # NFR-9: hex SHA-256 of the artifact at model_path. Required whenever
    # model_path is set; the node refuses to load a file that doesn't match.
    model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


class SetModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    node_id: str
    model_version: str
    shadow_model_version: str | None
