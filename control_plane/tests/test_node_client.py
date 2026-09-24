import httpx

from control_plane.app import node_client
from control_plane.app.node_client import HTTPNodeClient


def _patch_transport(monkeypatch, handler):
    real_async_client = httpx.AsyncClient

    def _client_with_mock_transport(*args, **kwargs):
        return real_async_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(node_client.httpx, "AsyncClient", _client_with_mock_transport)


async def test_set_model_sends_service_token_to_node(monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    _patch_transport(monkeypatch, handler)

    ok = await HTTPNodeClient("shared-secret").set_model(
        "http://node-1:8000", "shadow", "v2", "models/v2.onnx"
    )

    assert ok is True
    assert len(requests) == 1
    assert str(requests[0].url) == "http://node-1:8000/admin/model"
    assert requests[0].headers["X-Service-Token"] == "shared-secret"


async def test_set_model_reports_rejected_token_as_failure(monkeypatch):
    _patch_transport(monkeypatch, lambda request: httpx.Response(401))

    ok = await HTTPNodeClient("wrong-secret").set_model(
        "http://node-1:8000", "prod", "v2", "models/v2.onnx"
    )

    assert ok is False
