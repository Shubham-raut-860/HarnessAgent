"""FastAPI dependency injection for HarnessAgent."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from harness.core.config import get_config

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


# ---------------------------------------------------------------------------
# Infrastructure dependencies
# ---------------------------------------------------------------------------


async def get_redis(request: Request) -> Any:
    """Return the Redis client stored in app state.

    Raises:
        HTTPException 503 if Redis is not available.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis not available",
        )
    return redis


async def get_runner(request: Request) -> Any:
    """Return the singleton AgentRunner stored in app state."""
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent runner not initialised",
        )
    return runner


async def get_memory_manager(request: Request) -> Any:
    """Return the singleton MemoryManager stored in app state."""
    mm = getattr(request.app.state, "memory_manager", None)
    if mm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not initialised",
        )
    return mm


async def get_event_bus(request: Request) -> Any:
    """Return the singleton EventBus stored in app state."""
    bus = getattr(request.app.state, "event_bus", None)
    if bus is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event bus not initialised",
        )
    return bus


async def get_hitl_manager(request: Request) -> Any:
    """Return the HITLManager from app state."""
    hitl = getattr(request.app.state, "hitl_manager", None)
    if hitl is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HITL manager not initialised",
        )
    return hitl


async def get_prompt_manager(request: Request) -> Any:
    """Return the PromptManager from app state."""
    pm = getattr(request.app.state, "prompt_manager", None)
    if pm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prompt manager not initialised",
        )
    return pm


async def get_hermes_loop(request: Request) -> Any:
    """Return the HermesLoop from app state."""
    hermes = getattr(request.app.state, "hermes_loop", None)
    if hermes is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hermes loop not initialised",
        )
    return hermes


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------


def _decode_jwt(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException 401 on invalid or expired token.
    """
    try:
        from jose import JWTError, jwt  # type: ignore

        cfg = get_config()
        payload = jwt.decode(
            token,
            cfg.jwt_secret_key,
            algorithms=["HS256"],
        )
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_tenant(
    token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """Extract the tenant_id from the JWT bearer token.

    Returns "default" if no token is provided in dev mode.

    Args:
        token: JWT bearer token (optional).

    Returns:
        tenant_id string.

    Raises:
        HTTPException 401 if token is invalid.
    """
    cfg = get_config()
    if not token:
        if cfg.environment == "dev":
            return "default"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_jwt(token)
    tenant_id = payload.get("tenant_id") or payload.get("sub")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing tenant_id claim",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(tenant_id)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> dict:
    """Extract the full user info from the JWT bearer token.

    Returns a minimal user dict if no token is provided in dev mode.

    Args:
        token: JWT bearer token (optional).

    Returns:
        User payload dict with at minimum: sub, tenant_id.

    Raises:
        HTTPException 401 if token is invalid.
    """
    cfg = get_config()
    if not token:
        if cfg.environment == "dev":
            return {"sub": "dev-user", "tenant_id": "default", "role": "admin"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _decode_jwt(token)
