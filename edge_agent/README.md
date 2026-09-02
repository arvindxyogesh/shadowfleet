# edge_agent

Simulated fleet node. Runs the current production detection model on input
frames and returns predictions. Implements FR-1 (inference service) from
`../docs/SRS.md`.

Shadow-mode dual-model serving (FR-2) and telemetry streaming (FR-3) land in
M2 — this milestone is the single-model inference core.

## Stack

FastAPI + ONNX Runtime (CPU), YOLOv8n by default. No GPU dependency.

## Setup

1. Export a model to ONNX (one-time, needs network + the `ultralytics` package,
   which is intentionally **not** in `requirements.txt` since the runtime
   service doesn't need it):

   ```bash
   pip install ultralytics
   python scripts/export_model.py --output models/yolov8n.onnx
   ```

2. Install runtime dependencies and run the service:

   ```bash
   pip install -r requirements.txt
   SHADOWFLEET_MODEL_PATH=models/yolov8n.onnx uvicorn app.main:app --reload
   ```

3. Try it:

   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/infer -F "file=@/path/to/image.jpg"
   ```

If no model file is found at `SHADOWFLEET_MODEL_PATH`, the service still
starts and `/health` reports `degraded` — `/infer` returns `503` rather than
crashing, which is deliberate (a fleet node with a bad/missing model artifact
should fail loudly on that one endpoint, not take the process down).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests -v
```

Tests never require real model weights: postprocessing (NMS, box decoding,
letterbox-unwarping) is tested directly against synthetic arrays, and the API
layer is tested with a `FakeModel` injected via FastAPI's dependency
override, so CI runs with zero network access and zero large downloads.

## Docker

```bash
docker build -t shadowfleet-edge-agent .
docker run -p 8000:8000 -v $(pwd)/models:/app/models shadowfleet-edge-agent
```
