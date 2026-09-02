# edge_agent

Simulated fleet node. Runs the current production detection model (and,
optionally, a shadow candidate model) on input frames, returns production
predictions to callers, and emits telemetry for every inference call.

Implements: FR-1 (inference service), FR-2 (shadow mode), FR-3 (telemetry
streaming). See `../docs/SRS.md` for full requirements.

Planned stack: FastAPI + onnxruntime, one container per simulated node.
