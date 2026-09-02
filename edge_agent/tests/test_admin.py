from fastapi.testclient import TestClient

from edge_agent.app.main import app, get_model_loader


class FakeSwappedModel:
    def __init__(self, path: str, input_size: int):
        self.path = path
        self.input_size = input_size

    def predict(self, image, conf_threshold, iou_threshold, max_detections):
        return [], 1.0, image.width, image.height


def test_set_prod_model_updates_version_and_is_reflected_in_health():
    app.dependency_overrides[get_model_loader] = lambda: FakeSwappedModel
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/model",
                json={"role": "prod", "model_version": "yolov8n-v2", "model_path": "models/v2.onnx"},
            )
            assert resp.status_code == 200
            assert resp.json()["model_version"] == "yolov8n-v2"

            health_resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert health_resp.json()["model_version"] == "yolov8n-v2"
    assert health_resp.json()["model_loaded"] is True


def test_set_shadow_model_then_clear_it():
    app.dependency_overrides[get_model_loader] = lambda: FakeSwappedModel
    try:
        with TestClient(app) as client:
            set_resp = client.post(
                "/admin/model",
                json={"role": "shadow", "model_version": "candidate-v1", "model_path": "models/c1.onnx"},
            )
            assert set_resp.status_code == 200
            assert set_resp.json()["shadow_model_version"] == "candidate-v1"

            clear_resp = client.post(
                "/admin/model", json={"role": "shadow", "model_version": None, "model_path": None}
            )
    finally:
        app.dependency_overrides.clear()

    assert clear_resp.status_code == 200
    assert clear_resp.json()["shadow_model_version"] is None


def test_set_prod_model_requires_model_path_and_version():
    with TestClient(app) as client:
        resp = client.post("/admin/model", json={"role": "prod"})
    assert resp.status_code == 400


def test_set_prod_model_rejects_unloadable_model_without_disrupting_current_state():
    # No override: the real ONNXModel loader will fail on a bogus path.
    with TestClient(app) as client:
        health_before = client.get("/health").json()

        resp = client.post(
            "/admin/model",
            json={"role": "prod", "model_version": "bad-version", "model_path": "/no/such/file.onnx"},
        )
        assert resp.status_code == 400

        health_after = client.get("/health").json()

    assert health_before["model_version"] == health_after["model_version"]


def test_set_model_rejects_unknown_role():
    with TestClient(app) as client:
        resp = client.post("/admin/model", json={"role": "bogus"})
    assert resp.status_code == 400
