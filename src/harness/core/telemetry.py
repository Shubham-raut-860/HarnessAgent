"""OpenTelemetry tracing and Prometheus metrics setup for Codex Harness."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — set on first successful setup call
# ---------------------------------------------------------------------------
_tracer_provider: Any = None
_metrics_configured: bool = False


def setup_tracing(
    service_name: str,
    otlp_endpoint: str,
    *,
    force: bool = False,
) -> Any:
    """Configure and return an OTLP-exporting TracerProvider (idempotent)."""
    global _tracer_provider
    if _tracer_provider is not None and not force:
        return _tracer_provider

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": "0.1.0",
            }
        )
        provider = TracerProvider(resource=resource)

        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        logger.info(
            "OTel tracing configured: service=%s endpoint=%s",
            service_name,
            otlp_endpoint,
        )
    except Exception as exc:
        logger.warning("Failed to configure OTel tracing: %s", exc)
        # Fall back to no-op provider so the rest of the codebase can still
        # call get_tracer() without crashing.
        try:
            from opentelemetry import trace

            _tracer_provider = trace.get_tracer_provider()
        except Exception:
            _tracer_provider = object()  # sentinel

    return _tracer_provider


def setup_metrics(
    port: int = 8000,
    *,
    force: bool = False,
) -> None:
    """Start Prometheus metrics HTTP server (idempotent)."""
    global _metrics_configured
    if _metrics_configured and not force:
        return

    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        _metrics_configured = True
        logger.info("Prometheus metrics server started on port %d", port)
    except OSError as exc:
        # Port already in use is OK in tests / multiprocess envs
        if "Address already in use" in str(exc):
            _metrics_configured = True
            logger.debug("Prometheus metrics server already running on port %d", port)
        else:
            logger.warning("Failed to start Prometheus metrics server: %s", exc)
    except Exception as exc:
        logger.warning("Failed to configure Prometheus metrics: %s", exc)


def get_tracer(name: str) -> Any:
    """Return a named tracer from the configured provider (or no-op tracer)."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return _NoOpTracer()


class _NoOpTracer:
    """Minimal no-op tracer used when OTel is not available."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        """Return a no-op context manager."""
        return _NoOpSpanContext()

    def start_span(self, name: str, **kwargs: Any) -> Any:
        """Return a no-op span."""
        return _NoOpSpan()


class _NoOpSpan:
    """No-op span that satisfies the basic trace API."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op attribute setter."""

    def record_exception(self, exc: BaseException, **kwargs: Any) -> None:
        """No-op exception recorder."""

    def set_status(self, status: Any) -> None:
        """No-op status setter."""

    def end(self) -> None:
        """No-op span end."""

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpSpanContext:
    """No-op context manager returned by start_as_current_span."""

    def __enter__(self) -> "_NoOpSpan":
        return _NoOpSpan()

    def __exit__(self, *args: Any) -> None:
        pass
