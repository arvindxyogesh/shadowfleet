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
    shadow_confidence_min: float | None
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


class StartRolloutRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    model_path: str
    target_percentage: int = 20
    evaluation_window_seconds: int | None = None
    previous_model_path: str | None = None
    actor: str = "operator"


class ActorPayload(BaseModel):
    actor: str = "operator"


class RollbackPayload(BaseModel):
    actor: str = "operator"
    reason: str = "manual override"


class RolloutOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    model_version: str
    previous_version: str | None
    target_percentage: int
    status: str
    started_at: datetime
    evaluation_window_seconds: int
    ended_at: datetime | None
    reason: str | None


class RolloutNodeAssignmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    node_id: str
    role: str
    promoted: bool


class RolloutDetailOut(RolloutOut):
    nodes: list[RolloutNodeAssignmentOut]


class AuditLogEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    timestamp: datetime
    actor: str
    action: str
    details: dict


class RetrainTriggerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    triggered_at: datetime
    labeled_example_count: int
    threshold: int
    dispatch_method: str
    dispatch_status: str
