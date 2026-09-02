import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import settings
from .disagreement import compute_disagreement
from .inference import ONNXModel
from .schemas import HealthResponse, InferenceResponse
from .telemetry import TelemetryPublisher, build_redis_client

logger = logging.getLogger("shadowfleet.edge_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    app.state.shadow_model = None
    try:
        app.state.model = ONNXModel(settings.model_path, settings.input_size)
    except Exception as exc:  # model file missing or invalid at this node
        app.state.model_error = str(exc)
        logger.warning("failed to load production model from %s: %s", settings.model_path, exc)

    if settings.shadow_model_path:
        try:
            app.state.shadow_model = ONNXModel(settings.shadow_model_path, settings.input_size)
        except Exception as exc:
            logger.warning("failed to load shadow model from %s: %s", settings.shadow_model_path, exc)

    app.state.telemetry = TelemetryPublisher(
        build_redis_client(settings.redis_url), settings.telemetry_stream
    )

    yield

    await app.state.telemetry.redis.aclose()


app = FastAPI(title="ShadowFleet Edge Agent", version="0.1.0", lifespan=lifespan)


def get_model(request: Request) -> ONNXModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        error = getattr(request.app.state, "model_error", "model not initialized")
        raise HTTPException(status_code=503, detail=f"model not loaded: {error}")
    return model


def get_shadow_model(request: Request) -> ONNXModel | None:
    return getattr(request.app.state, "shadow_model", None)


def get_telemetry(request: Request) -> TelemetryPublisher:
    return request.app.state.telemetry


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = getattr(request.app.state, "model", None)
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        node_id=settings.node_id,
        model_version=settings.model_version,
        model_loaded=model is not None,
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    model: ONNXModel = Depends(get_model),
    shadow_model: ONNXModel | None = Depends(get_shadow_model),
    telemetry: TelemetryPublisher = Depends(get_telemetry),
) -> InferenceResponse:
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc

    detections, latency_ms, orig_w, orig_h = model.predict(
        image, settings.conf_threshold, settings.iou_threshold, settings.max_detections
    )

    shadow_detections = None
    shadow_model_version = None
    disagreement_score = None
    if shadow_model is not None:
        shadow_detections, _, _, _ = shadow_model.predict(
            image, settings.conf_threshold, settings.iou_threshold, settings.max_detections
        )
        shadow_model_version = settings.shadow_model_version
        disagreement_score = compute_disagreement(detections, shadow_detections)

    confidence_min = min((d["score"] for d in detections), default=None)

    event = {
        "event_id": str(uuid4()),
        "node_id": settings.node_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_id": str(uuid4()),
        "prod_model_version": settings.model_version,
        "prod_prediction": detections,
        "shadow_model_version": shadow_model_version,
        "shadow_prediction": shadow_detections,
        "confidence_min": confidence_min,
        "disagreement_score": disagreement_score,
        "latency_ms": latency_ms,
    }
    await telemetry.publish(event)

    # Only the production prediction is ever returned to the caller — the
    # shadow model's output exists solely in telemetry (FR-2).
    return InferenceResponse(
        node_id=settings.node_id,
        model_version=settings.model_version,
        detections=detections,
        latency_ms=latency_ms,
        image_width=orig_w,
        image_height=orig_h,
    )
