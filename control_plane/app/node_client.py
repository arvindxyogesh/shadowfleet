import logging
from typing import Protocol

import httpx

logger = logging.getLogger("shadowfleet.control_plane.node_client")


class NodeClient(Protocol):
    async def set_model(
        self,
        base_url: str,
        role: str,
        model_version: str | None,
        model_path: str | None,
        model_sha256: str | None,
    ) -> bool: ...


class HTTPNodeClient:
    """Calls a fleet node's /admin/model endpoint to push a model version
    (FR-8's OTA mechanism). Failures are logged and reported as False
    rather than raised — one unreachable node must not abort a rollout
    affecting the rest of the fleet.

    Every push carries the shared service token (NFR-8), since the node's
    /admin/model endpoint rejects unauthenticated writes, and the
    artifact's SHA-256, which the node verifies before loading (NFR-9).
    """

    def __init__(self, service_token: str, timeout: float = 5.0):
        self.service_token = service_token
        self.timeout = timeout

    async def set_model(
        self,
        base_url: str,
        role: str,
        model_version: str | None,
        model_path: str | None,
        model_sha256: str | None,
    ) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{base_url}/admin/model",
                    json={
                        "role": role,
                        "model_version": model_version,
                        "model_path": model_path,
                        "model_sha256": model_sha256,
                    },
                    headers={"X-Service-Token": self.service_token},
                )
                resp.raise_for_status()
            return True
        except Exception:
            logger.exception("failed to push model to node at %s", base_url)
            return False
