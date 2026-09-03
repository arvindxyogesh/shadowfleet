"""End-to-end scripted walkthrough of the running ShadowFleet stack.

Sends inference traffic to build up telemetry, shows hard-example mining
and fleet status filling in, then exercises a canary rollout to
completion. Meant to run against `infra/docker-compose.yml` once it's up
(`make up`), and doubles as a script to follow while recording a demo
video -- each step prints what it's doing and why, so the terminal output
alone tells the story.

Usage:
    pip install -r scripts/requirements.txt
    python scripts/demo.py
"""

import io
import sys
import time
from datetime import datetime, timezone

import requests
from PIL import Image, ImageDraw

EDGE_AGENT_URL = "http://localhost:8000"
CONTROL_PLANE_URL = "http://localhost:8001"
GRAFANA_URL = "http://localhost:3000"


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def wait_for_health(name: str, url: str, timeout_s: int = 60) -> None:
    step(f"Waiting for {name} to be healthy ({url}/health)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"{name} is up: {resp.json()}")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    print(f"Timed out waiting for {name}. Is `make up` running?", file=sys.stderr)
    sys.exit(1)


def synthetic_frame(seed: int) -> bytes:
    """A deterministic, dependency-free "dashcam" frame -- no external
    image download needed, so the demo works with zero network access
    beyond the local stack itself."""
    image = Image.new("RGB", (640, 480), color=(30, 30, 40))
    draw = ImageDraw.Draw(image)
    x = 40 + (seed * 37) % 500
    draw.rectangle([x, 200, x + 80, 260], fill=(200, 60, 60))  # a "car"
    draw.ellipse([300, 100, 340, 140], fill=(230, 230, 230))  # a "sign"
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def send_inference(n: int) -> None:
    step(f"Sending {n} inference requests to {EDGE_AGENT_URL}/infer")
    for i in range(n):
        files = {"file": (f"frame-{i}.jpg", synthetic_frame(i), "image/jpeg")}
        resp = requests.post(f"{EDGE_AGENT_URL}/infer", files=files, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        print(f"  frame {i}: {len(body['detections'])} detections, {body['latency_ms']:.1f}ms")
        time.sleep(0.2)


def show_fleet_status() -> None:
    step("Fleet status")
    nodes = requests.get(f"{CONTROL_PLANE_URL}/fleet/nodes", timeout=10).json()
    for node in nodes:
        print(f"  {node['node_id']}: prod={node['prod_model_version']} online={node['online']}")


def show_hard_examples() -> None:
    step("Hard examples mined so far")
    examples = requests.get(f"{CONTROL_PLANE_URL}/hard-examples", timeout=10).json()
    print(f"  {len(examples)} flagged")
    for ex in examples[:5]:
        print(f"  - {ex['input_id']}: {ex['reason']} (status={ex['status']})")


def try_canary_rollout(model_path: str) -> None:
    step("Starting a canary rollout")
    resp = requests.post(
        f"{CONTROL_PLANE_URL}/rollouts",
        json={
            "model_version": f"demo-{datetime.now(timezone.utc):%H%M%S}",
            "model_path": model_path,
            "target_percentage": 100,
            "evaluation_window_seconds": 20,
        },
        timeout=10,
    )
    if resp.status_code != 201:
        print(f"  could not start rollout: {resp.status_code} {resp.text}")
        return

    rollout = resp.json()
    rollout_id = rollout["id"]
    print(f"  rollout {rollout_id} started (status={rollout['status']})")

    step("Polling the rollout to completion")
    deadline = time.time() + 60
    while time.time() < deadline:
        rollout = requests.get(f"{CONTROL_PLANE_URL}/rollouts/{rollout_id}", timeout=10).json()
        print(f"  status={rollout['status']}")
        if rollout["status"] in ("completed", "rolled_back"):
            break
        time.sleep(5)

    step("Audit log")
    for entry in requests.get(f"{CONTROL_PLANE_URL}/audit-log", timeout=10).json()[:10]:
        print(f"  {entry['timestamp']} [{entry['actor']}] {entry['action']}")


def main() -> None:
    wait_for_health("edge_agent", EDGE_AGENT_URL)
    wait_for_health("control_plane", CONTROL_PLANE_URL)

    send_inference(15)
    show_fleet_status()
    show_hard_examples()
    # Reuses the node's already-loaded weights under a new version name --
    # any valid ONNX export works here, since the point is exercising the
    # rollout mechanism, not showing an actual model improvement.
    try_canary_rollout(model_path="/app/models/yolov8n.onnx")

    step("Done")
    print(f"Dashboard: {GRAFANA_URL}")
    print(f"Control plane API docs: {CONTROL_PLANE_URL}/docs")


if __name__ == "__main__":
    main()
