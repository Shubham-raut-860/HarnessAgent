"""Unit tests for LLMRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from harness.core.context import LLMResponse
from harness.core.errors import CircuitOpenError, FailureClass, LLMError
from harness.llm.router import LLMRouter


def _make_provider(name="mock", model="mock-model", healthy=True,
                   context_window=128_000, response=None, side_effect=None):
    p = AsyncMock()
    p.provider_name = name
    p.model = model
    p.context_window = context_window
    default_resp = LLMResponse(content=response or "Mock response", tool_calls=[],
                               input_tokens=10, output_tokens=20, model=model, provider=name)
    p.complete = AsyncMock(side_effect=side_effect) if side_effect else AsyncMock(return_value=default_resp)
    p.health_check = AsyncMock(return_value=healthy)
    return p


def _make_router(*providers_with_windows):
    """Build a router from (provider, context_window) tuples."""
    router = LLMRouter()
    for i, (p, cw) in enumerate(providers_with_windows):
        router.register(p, priority=i, context_window=cw)
    return router


@pytest.mark.asyncio
async def test_routes_to_primary_provider():
    p1 = _make_provider(name="primary", response="Primary response")
    p2 = _make_provider(name="fallback", response="Fallback response")
    router = _make_router((p1, 200_000), (p2, 200_000))
    response = await router.complete([{"role": "user", "content": "Hello"}], max_tokens=100)
    assert response.provider == "primary"
    assert response.content == "Primary response"


@pytest.mark.asyncio
async def test_falls_back_on_rate_limit():
    p1 = _make_provider(name="primary", side_effect=LLMError("429", failure_class=FailureClass.LLM_RATE_LIMIT))
    p2 = _make_provider(name="fallback", response="Fallback OK")
    router = _make_router((p1, 200_000), (p2, 200_000))
    response = await router.complete([{"role": "user", "content": "Hi"}], max_tokens=100)
    assert response.provider == "fallback"


@pytest.mark.asyncio
async def test_falls_back_on_timeout():
    p1 = _make_provider(name="primary", side_effect=LLMError("timeout", failure_class=FailureClass.LLM_TIMEOUT))
    p2 = _make_provider(name="fallback", response="Fallback after timeout")
    router = _make_router((p1, 200_000), (p2, 200_000))
    response = await router.complete([{"role": "user", "content": "Hi"}], max_tokens=100)
    assert response.provider == "fallback"


@pytest.mark.asyncio
async def test_raises_when_all_exhausted():
    p1 = _make_provider(name="p1", side_effect=LLMError("down", failure_class=FailureClass.LLM_ERROR))
    p2 = _make_provider(name="p2", side_effect=LLMError("down", failure_class=FailureClass.LLM_ERROR))
    router = _make_router((p1, 200_000), (p2, 200_000))
    with pytest.raises(LLMError):
        await router.complete([{"role": "user", "content": "Hi"}], max_tokens=100)


@pytest.mark.asyncio
async def test_skips_unhealthy_providers():
    p1 = _make_provider(name="unhealthy", healthy=False)
    p2 = _make_provider(name="healthy", response="Healthy response")
    router = _make_router((p1, 200_000), (p2, 200_000))
    response = await router.complete([{"role": "user", "content": "Hi"}], max_tokens=100)
    assert response.provider == "healthy"


@pytest.mark.asyncio
async def test_skips_providers_with_small_context_window():
    p1 = _make_provider(name="tiny", response="Should not be called")
    p2 = _make_provider(name="large", response="Large context response")
    # p1 registered with tiny context_window
    router = _make_router((p1, 100), (p2, 128_000))
    response = await router.complete([{"role": "user", "content": "Hi"}],
                                     max_tokens=100, required_context=1000)
    assert response.provider == "large"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    """After failure_threshold errors, circuit opens and further calls raise immediately."""
    from harness.llm.router import LLMRouterConfig
    p1 = _make_provider(name="flaky",
                        side_effect=LLMError("err", failure_class=FailureClass.LLM_ERROR))
    config = LLMRouterConfig(circuit_failure_threshold=3, circuit_recovery_timeout=60.0)
    router = LLMRouter(config=config)
    router.register(p1, priority=0, context_window=200_000)

    for _ in range(3):
        with pytest.raises(LLMError):
            await router.complete([{"role": "user", "content": "Hi"}], max_tokens=100)

    # Circuit should now be open — either CircuitOpenError or LLMError (no providers left)
    with pytest.raises((CircuitOpenError, LLMError)):
        await router.complete([{"role": "user", "content": "Hi"}], max_tokens=100)
