import secrets

from fastapi import Header, HTTPException

from .config import settings


def require_service_token(x_service_token: str | None = Header(default=None)) -> None:
    """Shared-secret check for write endpoints (NFR-8). An unset
    configured token rejects every request rather than accepting an empty
    header, so a node deployed without the secret fails closed.
    """
    expected = settings.service_token
    if not expected or x_service_token is None or not secrets.compare_digest(
        x_service_token.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="missing or invalid service token")
