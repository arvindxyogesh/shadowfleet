from fastapi.testclient import TestClient
from PIL import Image

from edge_agent.app.main import app, get_model


class FakeModel:
    """Stands in for ONNXModel so API tests don't depend on real weights."""

    def predict(self, image: Image.Image, conf_threshold, iou_threshold, max_detections):
        detections = [
            {
                "box": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
                "score": 0.91,
                "class_id": 2,
                "class_name": "car",
            }
        ]
        return detections, 12.5, image.width, image.height


def test_health_reports_status_without_requiring_a_loaded_model():
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"]
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["model_loaded"], bool)


def test_infer_returns_detections_from_the_model(sample_image_bytes):
    app.dependency_overrides[get_model] = lambda: FakeModel()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/infer", files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")}
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["detections"][0]["class_name"] == "car"
    assert body["detections"][0]["score"] == 0.91
    assert body["image_width"] == 320
    assert body["image_height"] == 240
    assert body["latency_ms"] >= 0


def test_infer_rejects_non_image_upload():
    app.dependency_overrides[get_model] = lambda: FakeModel()
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/infer", files={"file": ("bad.txt", b"not an image", "text/plain")}
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 400


def test_infer_without_a_loaded_model_returns_503(sample_image_bytes):
    # No dependency override and no real model file on disk in the test
    # environment: the node should degrade gracefully, not crash.
    with TestClient(app) as client:
        resp = client.post(
            "/infer", files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")}
        )

    assert resp.status_code == 503
