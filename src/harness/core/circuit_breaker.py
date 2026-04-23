"""Async circuit breaker implementation for Codex Harness."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, TypeVar

from harness.core.errors import CircuitOpenError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class CircuitState(str, Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Thread-safe async circuit breaker with configurable thresholds and callbacks."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.on_state_change = on_state_change

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: float | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Return True when the circuit is in OPEN state."""
        return self._state == CircuitState.OPEN

    @property
    def failure_count(self) -> int:
        """Current failure count (resets when circuit closes)."""
        return self._failure_count

    @property
    def protect(self) -> "Callable[[Callable[..., Coroutine[Any, Any, _T]]], Callable[..., Coroutine[Any, Any, _T]]]":
        """Decorator that protects an async function with this circuit breaker.

        Usage::

            @cb.protect
            async def fragile():
                ...
        """
        def _decorator(fn: "Callable[..., Coroutine[Any, Any, _T]]") -> "Callable[..., Coroutine[Any, Any, _T]]":
            @functools.wraps(fn)
            async def _wrapper(*args: Any, **kwargs: Any) -> "_T":
                async with _CircuitBreakerCall(self):
                    return await fn(*args, **kwargs)
            return _wrapper
        return _decorator

    def _transition(self, new_state: CircuitState) -> None:
        """Apply a state transition and fire the on_state_change callback."""
        old_state = self._state
        self._state = new_state
        logger.info(
            "Circuit breaker '%s' state transition: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )
        if self.on_state_change is not None:
            try:
                self.on_state_change(self.name, old_state, new_state)
            except Exception:
                logger.exception("on_state_change callback raised an error")

    async def _check_half_open(self) -> None:
        """Transition OPEN -> HALF_OPEN if recovery_timeout has elapsed."""
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.recovery_timeout
        ):
            self._success_count = 0
            self._transition(CircuitState.HALF_OPEN)

    async def record_success(self) -> None:
        """Record a successful call; may close the circuit."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._failure_count = 0
                    self._success_count = 0
                    self._opened_at = None
                    self._transition(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed call; may open the circuit."""
        async with self._lock:
            await self._check_half_open()
            self._failure_count += 1
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                if self._failure_count >= self.failure_threshold:
                    self._opened_at = time.monotonic()
                    self._transition(CircuitState.OPEN)

    async def can_proceed(self) -> bool:
        """Return True if a call may proceed through the circuit."""
        async with self._lock:
            await self._check_half_open()
            if self._state == CircuitState.OPEN:
                return False
            return True

    def call(
        self,
        fn: "Callable[..., Coroutine[Any, Any, _T]] | None" = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Any":
        """Guard a call through the circuit breaker.

        Usage:
            await cb.call(async_fn, *args, **kwargs)  # direct call
            async with cb.call():                      # context manager
        """
        if fn is not None:
            return self._call_fn(fn, *args, **kwargs)
        return _CircuitBreakerCall(self)

    async def _call_fn(
        self,
        fn: "Callable[..., Coroutine[Any, Any, _T]]",
        *args: Any,
        **kwargs: Any,
    ) -> "_T":
        async with _CircuitBreakerCall(self):
            return await fn(*args, **kwargs)


class _CircuitBreakerCall:
    """Async context manager that records success/failure on the circuit breaker."""

    def __init__(self, cb: CircuitBreaker) -> None:
        self._cb = cb

    async def __aenter__(self) -> "_CircuitBreakerCall":
        allowed = await self._cb.can_proceed()
        if not allowed:
            raise CircuitOpenError(
                f"Circuit breaker '{self._cb.name}' is OPEN — call rejected",
                service_name=self._cb.name,
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        if exc_type is None:
            await self._cb.record_success()
        else:
            await self._cb.record_failure()
        return False  # never suppress exceptions


def circuit_breaker_call(cb: CircuitBreaker) -> Callable[..., Any]:
    """Decorator that wraps an async function with circuit breaker protection."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, _T]],
    ) -> Callable[..., Coroutine[Any, Any, _T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            async with cb.call():
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """Central registry for named CircuitBreaker instances."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> CircuitBreaker:
        """Return an existing breaker or create a new one with the given config."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
                on_state_change=on_state_change,
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Return an existing breaker by name, or None."""
        return self._breakers.get(name)

    def all_states(self) -> dict[str, CircuitState]:
        """Return a snapshot of all breaker states."""
        return {name: cb.state for name, cb in self._breakers.items()}

    async def reset(self, name: str) -> None:
        """Forcibly close a breaker by name."""
        cb = self._breakers.get(name)
        if cb is not None:
            async with cb._lock:
                cb._failure_count = 0
                cb._success_count = 0
                cb._opened_at = None
                cb._transition(CircuitState.CLOSED)


# Module-level default registry
default_registry: CircuitBreakerRegistry = CircuitBreakerRegistry()
