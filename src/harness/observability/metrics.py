"""Prometheus metrics definitions for HarnessAgent."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import: prometheus_client may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (  # type: ignore[import]
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )

    # Stub classes so the rest of the module works without prometheus_client
    class _NoOpMetric:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def set(self, value: float) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

        def time(self) -> Any:
            import contextlib

            return contextlib.nullcontext()

    Counter = Gauge = Histogram = _NoOpMetric  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Singleton metric instances
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_metrics: dict[str, Any] = {}


def _get_or_create(factory: type, name: str, doc: str, **kwargs: Any) -> Any:
    """Return an existing metric by name or create it (idempotent)."""
    with _lock:
        if name not in _metrics:
            _metrics[name] = factory(name, doc, **kwargs)
        return _metrics[name]


# ------ Counters ------

agent_steps_total = _get_or_create(
    Counter,
    "harness_agent_steps_total",
    "Total agent steps executed",
    labelnames=["agent_type", "tenant_id", "status"],
)

agent_tokens_total = _get_or_create(
    Counter,
    "harness_agent_tokens_total",
    "Total tokens consumed by agents",
    labelnames=["agent_type", "provider", "token_type"],
)

tool_calls_total = _get_or_create(
    Counter,
    "harness_tool_calls_total",
    "Total tool invocations",
    labelnames=["tool_name", "agent_type", "status"],
)

safety_blocks_total = _get_or_create(
    Counter,
    "harness_safety_blocks_total",
    "Total safety guard blocks",
    labelnames=["guard", "agent_type", "stage"],
)

hermes_patches_total = _get_or_create(
    Counter,
    "harness_hermes_patches_total",
    "Total Hermes self-healing patches proposed/applied",
    labelnames=["agent_type", "status"],
)

llm_cache_hits_total = _get_or_create(
    Counter,
    "harness_llm_cache_hits_total",
    "LLM response cache hits",
    labelnames=["provider"],
)

inter_agent_messages_total = _get_or_create(
    Counter,
    "harness_inter_agent_messages_total",
    "Messages exchanged between agents",
    labelnames=["sender", "recipient", "message_type"],
)

failure_total = _get_or_create(
    Counter,
    "harness_failure_total",
    "Total classified failures",
    labelnames=["failure_class", "agent_type"],
)

cost_usd_total = _get_or_create(
    Counter,
    "harness_cost_usd_total",
    "Total LLM cost in USD",
    labelnames=["tenant_id", "model"],
)

# ------ Histograms ------

agent_latency_seconds = _get_or_create(
    Histogram,
    "harness_agent_latency_seconds",
    "End-to-end agent run latency",
    labelnames=["agent_type"],
    buckets=[1, 5, 15, 30, 60, 120, 300],
)

tool_latency_seconds = _get_or_create(
    Histogram,
    "harness_tool_latency_seconds",
    "Per-tool execution latency",
    labelnames=["tool_name"],
    buckets=[0.1, 0.5, 1, 5, 15, 30],
)

graph_rag_hops = _get_or_create(
    Histogram,
    "harness_graph_rag_hops",
    "Number of graph hops in GraphRAG traversal",
    labelnames=["agent_type"],
    buckets=[1, 2, 3, 4, 5],
)

# ------ Gauges ------

dlq_depth = _get_or_create(
    Gauge,
    "harness_dlq_depth",
    "Current dead-letter queue depth",
    labelnames=["tenant_id"],
)

active_runs = _get_or_create(
    Gauge,
    "harness_active_runs",
    "Currently active agent runs",
    labelnames=["agent_type", "tenant_id"],
)

circuit_breaker_state = _get_or_create(
    Gauge,
    "harness_circuit_breaker_state",
    "Circuit breaker state: 0=closed, 1=open, 2=half_open",
    labelnames=["service"],
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_prometheus_metrics() -> dict[str, Any]:
    """Return all registered metrics as a named dict."""
    return {
        "agent_steps_total": agent_steps_total,
        "agent_tokens_total": agent_tokens_total,
        "agent_latency_seconds": agent_latency_seconds,
        "tool_calls_total": tool_calls_total,
        "tool_latency_seconds": tool_latency_seconds,
        "safety_blocks_total": safety_blocks_total,
        "hermes_patches_total": hermes_patches_total,
        "graph_rag_hops": graph_rag_hops,
        "llm_cache_hits_total": llm_cache_hits_total,
        "inter_agent_messages_total": inter_agent_messages_total,
        "failure_total": failure_total,
        "dlq_depth": dlq_depth,
        "active_runs": active_runs,
        "circuit_breaker_state": circuit_breaker_state,
        "cost_usd_total": cost_usd_total,
    }


def setup_prometheus_server(port: int = 9090) -> None:
    """Start Prometheus HTTP metrics server on the given port."""
    if not _PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not installed — cannot start metrics server.")
        return
    try:
        start_http_server(port)
        logger.info("Prometheus metrics server started on port %d", port)
    except OSError as exc:
        logger.warning("Could not start Prometheus server on port %d: %s", port, exc)
