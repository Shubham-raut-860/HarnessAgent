"""Custom middleware for Codex Harness API."""

from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every request and response.

    If the client sends an X-Request-ID header, it is honoured; otherwise
    a new UUID4 hex string is generated.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        # Attach to request state so downstream handlers can log it
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TenantMiddleware(BaseHTTPMiddleware):
    """Extract the tenant_id from the JWT Bearer token and attach it to request.state.

    This middleware is best-effort: it does NOT raise on missing/invalid tokens
    (that is the responsibility of the ``get_current_tenant`` dependency).
    It simply pre-populates ``request.state.tenant_id`` when a valid token is
    present so that logging and observability layers can read it without
    repeating JWT decoding.

    Paths that match ``_PUBLIC_PATHS`` skip token extraction entirely.
    """

    _PUBLIC_PATHS = {"/health", "/health/ready", "/health/live", "/metrics", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.tenant_id = "anonymous"

        path = request.url.path
        if path in self._PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            tenant_id = self._extract_tenant(token)
            if tenant_id:
                request.state.tenant_id = tenant_id

        return await call_next(request)

    @staticmethod
    def _extract_tenant(token: str) -> str | None:
        """Try to decode the JWT and return tenant_id without raising."""
        try:
            from jose import jwt  # type: ignore
            from harness.core.config import get_config

            cfg = get_config()
            payload = jwt.decode(token, cfg.jwt_secret_key, algorithms=["HS256"])
            return payload.get("tenant_id") or payload.get("sub")
        except Exception:
            return None
