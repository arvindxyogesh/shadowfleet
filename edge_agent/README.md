# edge_agent

Simulated fleet node. Runs the current production detection model on input
frames and returns predictions; optionally runs a second "shadow" candidate
model on the same input and logs (but never serves) its output. Every
inference call emits a telemetry event to the fleet's Redis Stream.
Implements FR-1, FR-2, and the producer side of FR-3 from `../docs/SRS.md`.

## Stack

FastAPI + ONNX Runtime (CPU), YOLOv8n by default, Redis Streams for
telemetry. No GPU dependency.

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
   SHADOWFLEET_MODEL_PATH=models/yolov8n.onnx \
   SHADOWFLEET_REDIS_URL=redis://localhost:6379/0 \
   uvicorn app.main:app --reload
   ```

   To also run a shadow candidate (e.g. a retrained version being
   evaluated), export a second ONNX file and set:

   ```bash
   SHADOWFLEET_SHADOW_MODEL_PATH=models/yolov8n-candidate.onnx
   SHADOWFLEET_SHADOW_MODEL_VERSION=yolov8n-candidate-v2
   ```

   Telemetry publishing is best-effort — if Redis is unreachable, `/infer`
   still serves predictions normally; the failure is only logged.

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

Tests never require real model weights or a real Redis instance:
postprocessing (NMS, box decoding, letterbox-unwarping) and disagreement
scoring are tested directly against synthetic arrays, and the API layer is
tested with `FakeModel`/`FakeTelemetryPublisher` injected via FastAPI's
dependency override, so CI runs with zero network access and zero large
downloads.

## Docker

```bash
docker build -t shadowfleet-edge-agent .
docker run -p 8000:8000 -v $(pwd)/models:/app/models shadowfleet-edge-agent
```
