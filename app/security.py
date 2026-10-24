import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject requests without the right API key. A constant-time compare avoids timing leaks."""
    if settings.api_key is None:
        return

    if provided is None or not secrets.compare_digest(provided.encode(), settings.api_key.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
