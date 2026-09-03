import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import settings
from .disagreement import compute_disagreement
from .inference import ONNXModel
from .schemas import HealthResponse, InferenceResponse, SetModelRequest, SetModelResponse
from .telemetry import TelemetryPublisher, build_redis_client

logger = logging.getLogger("shadowfleet.edge_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    app.state.model_version = settings.model_version
    app.state.shadow_model = None
    app.state.shadow_model_version = None

    try:
        app.state.model = ONNXModel(settings.model_path, settings.input_size)
    except Exception as exc:  # model file missing or invalid at this node
        app.state.model_error = str(exc)
        logger.warning("failed to load production model from %s: %s", settings.model_path, exc)

    if settings.shadow_model_path:
        try:
            app.state.shadow_model = ONNXModel(settings.shadow_model_path, settings.input_size)
            app.state.shadow_model_version = settings.shadow_model_version
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


def get_model_version(request: Request) -> str:
    return request.app.state.model_version


def get_shadow_model_version(request: Request) -> str | None:
    return request.app.state.shadow_model_version


def get_telemetry(request: Request) -> TelemetryPublisher:
    return request.app.state.telemetry


def get_model_loader() -> Callable[[str, int], ONNXModel]:
    """DI seam so tests can hot-swap in a fake model without a real ONNX
    file on disk — the admin endpoint never constructs ONNXModel directly.
    """
    return ONNXModel


@app.get("/health", response_model=HealthResponse)
def health(
    request: Request, model_version: str = Depends(get_model_version)
) -> HealthResponse:
    model = getattr(request.app.state, "model", None)
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        node_id=settings.node_id,
        model_version=model_version,
        model_loaded=model is not None,
    )


@app.post("/admin/model", response_model=SetModelResponse)
def set_model(
    payload: SetModelRequest,
    request: Request,
    model_loader: Callable[[str, int], ONNXModel] = Depends(get_model_loader),
) -> SetModelResponse:
    """OTA hot-swap endpoint the control plane's rollout manager calls to
    push a model version onto this node (FR-8), without restarting the
    process or dropping in-flight requests.
    """
    if payload.role == "prod":
        if not payload.model_path or not payload.model_version:
            raise HTTPException(status_code=400, detail="prod role requires model_path and model_version")
        try:
            new_model = model_loader(payload.model_path, settings.input_size)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to load model: {exc}") from exc
        request.app.state.model = new_model
        request.app.state.model_error = None
        request.app.state.model_version = payload.model_version

    elif payload.role == "shadow":
        if payload.model_path is None:
            request.app.state.shadow_model = None
            request.app.state.shadow_model_version = None
        else:
            if not payload.model_version:
                raise HTTPException(status_code=400, detail="setting a shadow model requires model_version")
            try:
                new_shadow = model_loader(payload.model_path, settings.input_size)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"failed to load shadow model: {exc}") from exc
            request.app.state.shadow_model = new_shadow
            request.app.state.shadow_model_version = payload.model_version

    else:
        raise HTTPException(status_code=400, detail=f"unknown role: {payload.role!r}")

    return SetModelResponse(
        node_id=settings.node_id,
        model_version=request.app.state.model_version,
        shadow_model_version=request.app.state.shadow_model_version,
    )


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    model: ONNXModel = Depends(get_model),
    shadow_model: ONNXModel | None = Depends(get_shadow_model),
    telemetry: TelemetryPublisher = Depends(get_telemetry),
    model_version: str = Depends(get_model_version),
    shadow_model_version: str | None = Depends(get_shadow_model_version),
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
    disagreement_score = None
    shadow_confidence_min = None
    if shadow_model is not None:
        shadow_detections, _, _, _ = shadow_model.predict(
            image, settings.conf_threshold, settings.iou_threshold, settings.max_detections
        )
        disagreement_score = compute_disagreement(detections, shadow_detections)
        shadow_confidence_min = min((d["score"] for d in shadow_detections), default=None)
    else:
        shadow_model_version = None

    confidence_min = min((d["score"] for d in detections), default=None)

    event = {
        "event_id": str(uuid4()),
        "node_id": settings.node_id,
        "base_url": settings.self_base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_id": str(uuid4()),
        "prod_model_version": model_version,
        "prod_prediction": detections,
        "shadow_model_version": shadow_model_version,
        "shadow_prediction": shadow_detections,
        "confidence_min": confidence_min,
        "shadow_confidence_min": shadow_confidence_min,
        "disagreement_score": disagreement_score,
        "latency_ms": latency_ms,
    }
    await telemetry.publish(event)

    # Only the production prediction is ever returned to the caller — the
    # shadow model's output exists solely in telemetry (FR-2).
    return InferenceResponse(
        node_id=settings.node_id,
        model_version=model_version,
        detections=detections,
        latency_ms=latency_ms,
        image_width=orig_w,
        image_height=orig_h,
    )
