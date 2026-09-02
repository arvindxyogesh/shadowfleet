from fastapi.testclient import TestClient
from PIL import Image

from edge_agent.app.main import app, get_model, get_shadow_model, get_shadow_model_version, get_telemetry


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


class FakeShadowModel:
    def predict(self, image: Image.Image, conf_threshold, iou_threshold, max_detections):
        detections = [
            {
                "box": {"x1": 200.0, "y1": 200.0, "x2": 250.0, "y2": 250.0},
                "score": 0.7,
                "class_id": 5,
                "class_name": "bus",
            }
        ]
        return detections, 15.0, image.width, image.height


class FakeTelemetryPublisher:
    def __init__(self):
        self.events: list[dict] = []

    async def publish(self, event: dict) -> None:
        self.events.append(event)


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
    app.dependency_overrides[get_telemetry] = lambda: FakeTelemetryPublisher()
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
    app.dependency_overrides[get_telemetry] = lambda: FakeTelemetryPublisher()
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


def test_infer_publishes_telemetry_with_no_shadow_model(sample_image_bytes):
    fake_telemetry = FakeTelemetryPublisher()
    app.dependency_overrides[get_model] = lambda: FakeModel()
    app.dependency_overrides[get_telemetry] = lambda: fake_telemetry
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/infer", files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")}
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert len(fake_telemetry.events) == 1
    event = fake_telemetry.events[0]
    assert event["prod_prediction"][0]["class_name"] == "car"
    assert event["shadow_prediction"] is None
    assert event["shadow_model_version"] is None
    assert event["disagreement_score"] is None
    assert event["confidence_min"] == 0.91


def test_infer_with_shadow_model_logs_disagreement_but_never_serves_it(sample_image_bytes):
    fake_telemetry = FakeTelemetryPublisher()
    app.dependency_overrides[get_model] = lambda: FakeModel()
    app.dependency_overrides[get_shadow_model] = lambda: FakeShadowModel()
    app.dependency_overrides[get_shadow_model_version] = lambda: "shadow-candidate"
    app.dependency_overrides[get_telemetry] = lambda: fake_telemetry
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/infer", files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")}
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    # FR-2 acceptance: the API response carries only the production result.
    assert "shadow_prediction" not in resp.json()
    assert resp.json()["detections"][0]["class_name"] == "car"

    assert len(fake_telemetry.events) == 1
    event = fake_telemetry.events[0]
    assert event["shadow_prediction"][0]["class_name"] == "bus"
    assert event["shadow_model_version"] == "shadow-candidate"
    # prod=car@(10,10,50,50), shadow=bus@(200,200,250,250): no overlap, different class
    assert event["disagreement_score"] == 1.0
