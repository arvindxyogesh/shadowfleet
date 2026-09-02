from datetime import datetime

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str


class FleetNodeOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    node_id: str
    last_seen: datetime
    prod_model_version: str | None
    shadow_model_version: str | None
    online: bool


class TelemetryEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    event_id: str
    node_id: str
    timestamp: datetime
    input_id: str | None
    prod_model_version: str | None
    shadow_model_version: str | None
    confidence_min: float | None
    disagreement_score: float | None
    latency_ms: float | None


class HardExampleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    input_id: str
    event_id: str
    node_id: str
    reason: str
    confidence_min: float | None
    disagreement_score: float | None
    flagged_at: datetime
    status: str
    label: dict | None


class LabelPayload(BaseModel):
    label: dict
