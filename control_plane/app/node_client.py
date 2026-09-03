import logging
from typing import Protocol

import httpx

logger = logging.getLogger("shadowfleet.control_plane.node_client")


class NodeClient(Protocol):
    async def set_model(
        self, base_url: str, role: str, model_version: str | None, model_path: str | None
    ) -> bool: ...


class HTTPNodeClient:
    """Calls a fleet node's /admin/model endpoint to push a model version
    (FR-8's OTA mechanism). Failures are logged and reported as False
    rather than raised — one unreachable node must not abort a rollout
    affecting the rest of the fleet.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def set_model(
        self, base_url: str, role: str, model_version: str | None, model_path: str | None
    ) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{base_url}/admin/model",
                    json={"role": role, "model_version": model_version, "model_path": model_path},
                )
                resp.raise_for_status()
            return True
        except Exception:
            logger.exception("failed to push model to node at %s", base_url)
            return False
