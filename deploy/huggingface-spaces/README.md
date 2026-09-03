---
title: ShadowFleet Inference API
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# ShadowFleet — Inference API (standalone demo)

A single-container deployment of `edge_agent` — just the object-detection
inference API, with no fleet/control-plane/dashboard behind it (Spaces'
free tier is one container, and can't sustain the multi-service stack
`infra/docker-compose.yml` runs continuously). It exists so a reviewer can
hit a real `/infer` endpoint without cloning the repo.

For the full closed-loop system (telemetry, hard-example mining, canary
rollout, drift rollback), see the project root README and run it locally
via `infra/docker-compose.yml` — that's the actual point of this project;
this Space is just the one piece that's meaningfully demoable standalone.

## Try it

```bash
curl -X POST https://<your-space-url>/infer -F "file=@image.jpg"
curl https://<your-space-url>/health
```

Or use the Swagger UI at `/docs`.

## Deploying this yourself

This directory is **not** wired to auto-deploy — Spaces needs your own
Hugging Face account and push access, which this session doesn't have.
Three commands, from the repo root:

```bash
git clone https://huggingface.co/spaces/<your-username>/shadowfleet-inference space
cp deploy/huggingface-spaces/README.md deploy/huggingface-spaces/Dockerfile space/
cp -r edge_agent/app edge_agent/requirements.txt space/
cd space && git add -A && git commit -m "Deploy ShadowFleet inference API" && git push
```

The build is multi-stage (see `Dockerfile` in this directory): a build
stage installs `ultralytics` just long enough to export real YOLOv8n COCO
weights to ONNX, then the final image only carries the lean runtime
dependencies (`edge_agent/requirements.txt`) plus that one `.onnx` file —
no `ultralytics`/`torch` in the deployed image, no external model download
at container start.
