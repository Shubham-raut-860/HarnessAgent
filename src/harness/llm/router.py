"""Health-aware, context-window-aware LLM provider router for Codex Harness."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from harness.core.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from harness.core.context import LLMResponse
from harness.core.errors import CircuitOpenError, FailureClass, LLMError
from harness.core.protocols import LLMProvider

logger = logging.getLogger(__name__)

# Errors that should trigger fallback to the next provider
_RETRYABLE = frozenset({
    FailureClass.LLM_RATE_LIMIT,
    FailureClass.LLM_TIMEOUT,
    FailureClass.LLM_ERROR,
})


@dataclass
class ProviderEntry:
    """A provider registered with the router."""
    priority: int
    provider: LLMProvider
    context_window: int = 200_000
    enabled: bool = True


@dataclass
class LLMRouterConfig:
    """Configuration for LLMRouter."""
    providers: list[ProviderEntry] = field(default_factory=list)
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    circuit_success_threshold: int = 2


class LLMRouter:
    """Routes LLM completion requests across multiple providers with circuit breaking."""

    def __init__(
        self,
        config: LLMRouterConfig | None = None,
        registry: CircuitBreakerRegistry | None = None,
    ) -> None:
        self._config = config or LLMRouterConfig()
        self._registry = registry or CircuitBreakerRegistry()
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(
        self,
        provider: LLMProvider,
        priority: int = 0,
        context_window: int = 200_000,
    ) -> None:
        """Add a provider to the router."""
        self._config.providers.append(
            ProviderEntry(priority=priority, provider=provider, context_window=context_window)
        )
        self._config.providers.sort(key=lambda e: e.priority)

    def _get_breaker(self, provider: LLMProvider) -> CircuitBreaker:
        """Return or create the circuit breaker for a provider (synchronous)."""
        key = f"{provider.provider_name}:{provider.model}"
        if key not in self._breakers:
            self._breakers[key] = self._registry.get_or_create(
                name=key,
                failure_threshold=self._config.circuit_failure_threshold,
                recovery_timeout=self._config.circuit_recovery_timeout,
                success_threshold=self._config.circuit_success_threshold,
            )
        return self._breakers[key]

    def _sorted_providers(self) -> list[ProviderEntry]:
        return [e for e in sorted(self._config.providers, key=lambda e: e.priority) if e.enabled]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        required_context: int = 0,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Route a completion request, falling back on retryable errors."""
        last_exc: Exception | None = None

        for entry in self._sorted_providers():
            provider = entry.provider

            # Skip providers whose context window is too small
            if required_context > 0 and required_context > entry.context_window:
                logger.debug(
                    "Skipping %s:%s — required_context %d > window %d",
                    provider.provider_name, provider.model,
                    required_context, entry.context_window,
                )
                continue

            # Skip providers that fail a health check
            try:
                healthy = await provider.health_check()
                if not healthy:
                    logger.debug("Skipping %s:%s — health check failed",
                                 provider.provider_name, provider.model)
                    continue
            except Exception as exc:
                logger.debug("Health check error for %s:%s: %s",
                             provider.provider_name, provider.model, exc)
                continue

            breaker = self._get_breaker(provider)

            try:
                async with breaker.call():
                    response = await provider.complete(
                        messages,
                        max_tokens=max_tokens,
                        system=system,
                        tools=tools,
                        **kwargs,
                    )
                    return response
            except CircuitOpenError as exc:
                logger.warning("Circuit open for %s:%s, trying next",
                               provider.provider_name, provider.model)
                last_exc = exc
                continue
            except LLMError as exc:
                if exc.failure_class in _RETRYABLE:
                    logger.warning(
                        "Retryable error from %s:%s (%s), trying next",
                        provider.provider_name, provider.model, exc.failure_class,
                    )
                    last_exc = exc
                    continue
                raise

        raise LLMError(
            f"All providers exhausted. Last error: {last_exc}",
            failure_class=FailureClass.LLM_ERROR,
        ) from last_exc

    async def stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from the first available provider."""
        for entry in self._sorted_providers():
            provider = entry.provider
            try:
                healthy = await provider.health_check()
                if not healthy:
                    continue
            except Exception:
                continue

            breaker = self._get_breaker(provider)
            try:
                async with breaker.call():
                    async for token in provider.stream(messages, **kwargs):
                        yield token
                    return
            except (CircuitOpenError, LLMError):
                continue

        raise LLMError("No providers available for streaming", failure_class=FailureClass.LLM_ERROR)

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered providers concurrently."""
        async def _check(entry: ProviderEntry) -> tuple[str, bool]:
            key = f"{entry.provider.provider_name}:{entry.provider.model}"
            try:
                return key, await entry.provider.health_check()
            except Exception:
                return key, False

        results = await asyncio.gather(*[_check(e) for e in self._sorted_providers()])
        return dict(results)
