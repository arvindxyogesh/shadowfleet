# Multi-stage build for a standalone Hugging Face Spaces deployment of
# edge_agent. Expects `app/` and `requirements.txt` copied in alongside
# this Dockerfile (see this directory's README.md "Deploying this
# yourself" for the exact commands) -- that's how HF Spaces expects a
# Docker-SDK Space to be laid out: everything flat in one repo.

# Stage 1: export real YOLOv8n COCO weights to ONNX. ultralytics/torch are
# only needed here, at build time -- they never end up in the final image.
FROM python:3.11-slim AS export
WORKDIR /export
RUN pip install --no-cache-dir ultralytics==8.3.28
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12)"

# Stage 2: the actual lean runtime image.
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY --from=export /export/yolov8n.onnx /app/models/yolov8n.onnx

ENV SHADOWFLEET_MODEL_PATH=/app/models/yolov8n.onnx
ENV SHADOWFLEET_NODE_ID=hf-spaces-demo

# Spaces' Docker SDK expects the app on port 7860 by default (matches
# app_port in this directory's README.md frontmatter).
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
