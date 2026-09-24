from fastapi.testclient import TestClient

import hashlib

import pytest

from edge_agent.app.config import settings
from edge_agent.app.main import app, get_model_loader

AUTH_HEADERS = {"X-Service-Token": settings.service_token}


def _artifact(tmp_path, name: str, content: bytes = b"fake onnx bytes") -> tuple[str, str]:
    """A real file on disk plus its SHA-256 -- /admin/model checksum-verifies
    the artifact (NFR-9) before handing its path to the model loader."""
    path = tmp_path / name
    path.write_bytes(content)
    return str(path), hashlib.sha256(content).hexdigest()


class FakeSwappedModel:
    def __init__(self, path: str, input_size: int):
        self.path = path
        self.input_size = input_size

    def predict(self, image, conf_threshold, iou_threshold, max_detections):
        return [], 1.0, image.width, image.height


def test_set_prod_model_updates_version_and_is_reflected_in_health(tmp_path):
    path, sha256 = _artifact(tmp_path, "v2.onnx")
    app.dependency_overrides[get_model_loader] = lambda: FakeSwappedModel
    try:
        with TestClient(app, headers=AUTH_HEADERS) as client:
            resp = client.post(
                "/admin/model",
                json={"role": "prod", "model_version": "yolov8n-v2", "model_path": path, "model_sha256": sha256},
            )
            assert resp.status_code == 200
            assert resp.json()["model_version"] == "yolov8n-v2"

            health_resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert health_resp.json()["model_version"] == "yolov8n-v2"
    assert health_resp.json()["model_loaded"] is True


def test_set_shadow_model_then_clear_it(tmp_path):
    path, sha256 = _artifact(tmp_path, "c1.onnx")
    app.dependency_overrides[get_model_loader] = lambda: FakeSwappedModel
    try:
        with TestClient(app, headers=AUTH_HEADERS) as client:
            set_resp = client.post(
                "/admin/model",
                json={"role": "shadow", "model_version": "candidate-v1", "model_path": path, "model_sha256": sha256},
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
    with TestClient(app, headers=AUTH_HEADERS) as client:
        resp = client.post("/admin/model", json={"role": "prod"})
    assert resp.status_code == 400


def test_set_prod_model_rejects_unloadable_model_without_disrupting_current_state(tmp_path):
    # No override: the checksum matches, but the real ONNXModel loader
    # fails on bytes that aren't a valid ONNX graph.
    path, sha256 = _artifact(tmp_path, "corrupt.onnx", b"not an onnx model")
    with TestClient(app, headers=AUTH_HEADERS) as client:
        health_before = client.get("/health").json()

        resp = client.post(
            "/admin/model",
            json={"role": "prod", "model_version": "bad-version", "model_path": path, "model_sha256": sha256},
        )
        assert resp.status_code == 400

        health_after = client.get("/health").json()

    assert health_before["model_version"] == health_after["model_version"]


def test_set_model_rejects_unknown_role():
    with TestClient(app, headers=AUTH_HEADERS) as client:
        resp = client.post("/admin/model", json={"role": "bogus"})
    assert resp.status_code == 400


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong-token"}])
def test_set_model_rejects_missing_or_wrong_service_token(headers, tmp_path):
    path, sha256 = _artifact(tmp_path, "v2.onnx")
    app.dependency_overrides[get_model_loader] = lambda: FakeSwappedModel
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/admin/model",
                json={"role": "prod", "model_version": "yolov8n-v2", "model_path": path, "model_sha256": sha256},
                headers=headers,
            )
            health_resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 401
    assert health_resp.json()["model_version"] != "yolov8n-v2"


def test_set_model_rejects_everything_when_no_token_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "service_token", "")
    with TestClient(app) as client:
        resp = client.post(
            "/admin/model", json={"role": "shadow", "model_version": None, "model_path": None},
            headers={"X-Service-Token": ""},
        )
    assert resp.status_code == 401


class RecordingLoader:
    """Fails the test if the node ever tries to load an artifact that
    didn't pass checksum verification."""

    loaded: list[str] = []

    def __init__(self, path: str, input_size: int):
        RecordingLoader.loaded.append(path)

    def predict(self, image, conf_threshold, iou_threshold, max_detections):
        return [], 1.0, image.width, image.height


def _post_unverifiable(client, role, path, sha256):
    body = {"role": role, "model_version": "tampered", "model_path": path}
    if sha256 is not None:
        body["model_sha256"] = sha256
    return client.post("/admin/model", json=body)


@pytest.mark.parametrize("role", ["prod", "shadow"])
@pytest.mark.parametrize(
    "case", ["mismatched_checksum", "missing_checksum", "missing_file"]
)
def test_set_model_rejects_unverified_artifact_without_loading_it(tmp_path, role, case):
    path, sha256 = _artifact(tmp_path, "v2.onnx", b"the bytes that were actually published")
    if case == "mismatched_checksum":
        # Same path, different bytes on disk than the checksum describes --
        # what a corrupted transfer or tampered artifact looks like.
        (tmp_path / "v2.onnx").write_bytes(b"tampered bytes")
    elif case == "missing_checksum":
        sha256 = None
    else:
        path = str(tmp_path / "does-not-exist.onnx")

    RecordingLoader.loaded = []
    app.dependency_overrides[get_model_loader] = lambda: RecordingLoader
    try:
        with TestClient(app, headers=AUTH_HEADERS) as client:
            health_before = client.get("/health").json()
            resp = _post_unverifiable(client, role, path, sha256)
            health_after = client.get("/health").json()
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 400
    assert RecordingLoader.loaded == []
    assert health_after["model_version"] == health_before["model_version"]


def test_set_model_accepts_uppercase_checksum(tmp_path):
    path, sha256 = _artifact(tmp_path, "v2.onnx")
    app.dependency_overrides[get_model_loader] = lambda: FakeSwappedModel
    try:
        with TestClient(app, headers=AUTH_HEADERS) as client:
            resp = client.post(
                "/admin/model",
                json={"role": "prod", "model_version": "v2", "model_path": path, "model_sha256": sha256.upper()},
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200


def test_set_model_rejects_malformed_checksum(tmp_path):
    path, _sha256 = _artifact(tmp_path, "v2.onnx")
    with TestClient(app, headers=AUTH_HEADERS) as client:
        resp = client.post(
            "/admin/model",
            json={"role": "prod", "model_version": "v2", "model_path": path, "model_sha256": "not-a-sha256"},
        )

    assert resp.status_code == 422


def test_clearing_shadow_model_needs_no_checksum():
    with TestClient(app, headers=AUTH_HEADERS) as client:
        resp = client.post("/admin/model", json={"role": "shadow", "model_version": None, "model_path": None})

    assert resp.status_code == 200
