"""Unit tests for CircuitBreaker."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.core.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from harness.core.errors import CircuitOpenError


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _always_succeed():
    return "ok"


async def _always_fail():
    raise RuntimeError("Service unavailable")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_state_allows_calls():
    """A fresh circuit breaker in CLOSED state should allow calls through."""
    cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)
    result = await cb.call(_always_succeed)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_open_state_after_threshold_failures():
    """Circuit should open after failure_threshold consecutive failures."""
    cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=60.0)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(_always_fail)

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_raises_circuit_open_error():
    """Open circuit should raise CircuitOpenError without calling the function."""
    cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=60.0)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_always_fail)

    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        await cb.call(_always_succeed)


@pytest.mark.asyncio
async def test_half_open_after_recovery_timeout():
    """Circuit should move to HALF_OPEN after recovery_timeout elapses."""
    cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.05, success_threshold=1)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_always_fail)

    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.1)

    # After timeout, HALF_OPEN: one success closes with success_threshold=1
    result = await cb.call(_always_succeed)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_closes_after_success_threshold():
    """Circuit in HALF_OPEN state should close after success_threshold successes."""
    cb = CircuitBreaker(
        name="test",
        failure_threshold=2,
        recovery_timeout=0.05,
        success_threshold=2,
    )

    # Open it
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(_always_fail)

    await asyncio.sleep(0.1)

    # First success: HALF_OPEN
    await cb.call(_always_succeed)
    # Second success: should close
    await cb.call(_always_succeed)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_resets_on_close():
    """Failure counter should reset when circuit closes."""
    cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=0.05)

    # One failure, then recovery
    with pytest.raises(RuntimeError):
        await cb.call(_always_fail)

    assert cb.failure_count == 1

    # Simulate time passing and successful call
    await asyncio.sleep(0.1)
    await cb.call(_always_succeed)

    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_concurrent_failures_thread_safe():
    """Multiple concurrent failures should all be counted correctly."""
    cb = CircuitBreaker(name="test", failure_threshold=5, recovery_timeout=60.0)

    async def fail():
        with pytest.raises(RuntimeError):
            await cb.call(_always_fail)

    await asyncio.gather(*[fail() for _ in range(5)])
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_cb_registry_manages_multiple_breakers():
    """CircuitBreakerRegistry should maintain independent state per service."""
    registry = CircuitBreakerRegistry()

    cb1 = registry.get_or_create("service-a", failure_threshold=2)
    cb2 = registry.get_or_create("service-b", failure_threshold=2)

    # Fail service-a
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb1.call(_always_fail)

    # service-a open, service-b still closed
    assert cb1.state == CircuitState.OPEN
    assert cb2.state == CircuitState.CLOSED

    # Same name returns same instance
    cb1_again = registry.get_or_create("service-a")
    assert cb1_again is cb1


@pytest.mark.asyncio
async def test_decorator_usage():
    """CircuitBreaker can be used as a decorator on async functions."""
    cb = CircuitBreaker(name="decorator-test", failure_threshold=2, recovery_timeout=60.0)

    call_count = 0

    @cb.protect
    async def fragile_service():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("down")
        return "recovered"

    with pytest.raises(RuntimeError):
        await fragile_service()
    with pytest.raises(RuntimeError):
        await fragile_service()

    assert cb.state == CircuitState.OPEN
