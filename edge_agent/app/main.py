import io
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import settings
from .inference import ONNXModel
from .schemas import HealthResponse, InferenceResponse

logger = logging.getLogger("shadowfleet.edge_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = ONNXModel(settings.model_path, settings.input_size)
    except Exception as exc:  # model file missing or invalid at this node
        app.state.model_error = str(exc)
        logger.warning("failed to load model from %s: %s", settings.model_path, exc)
    yield


app = FastAPI(title="ShadowFleet Edge Agent", version="0.1.0", lifespan=lifespan)


def get_model(request: Request) -> ONNXModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        error = getattr(request.app.state, "model_error", "model not initialized")
        raise HTTPException(status_code=503, detail=f"model not loaded: {error}")
    return model


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

    return InferenceResponse(
        node_id=settings.node_id,
        model_version=settings.model_version,
        detections=detections,
        latency_ms=latency_ms,
        image_width=orig_w,
        image_height=orig_h,
    )
