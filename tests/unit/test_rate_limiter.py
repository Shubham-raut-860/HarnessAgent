"""Unit tests for RateLimiter using fakeredis."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from harness.core.rate_limiter import RateLimiter


@pytest_asyncio.fixture
async def rate_limiter(redis_client):
    """Rate limiter backed by fakeredis, limit of 5 req/60s."""
    return RateLimiter(redis_client=redis_client, default_rpm=5, window_seconds=60)


@pytest.mark.asyncio
async def test_allows_requests_under_limit(rate_limiter):
    for i in range(5):
        result = await rate_limiter.check(tenant_id="tenant-a", resource="api")
        assert result.allowed is True, f"Request {i} should be allowed"
        assert result.retry_after == 0.0


@pytest.mark.asyncio
async def test_blocks_over_limit(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-b", resource="api")

    result = await rate_limiter.check(tenant_id="tenant-b", resource="api")
    assert result.allowed is False
    assert result.retry_after > 0


@pytest.mark.asyncio
async def test_sliding_window_clears_old_requests(redis_client):
    rl = RateLimiter(redis_client=redis_client, default_rpm=3, window_seconds=1)

    for _ in range(3):
        await rl.check(tenant_id="tenant-window", resource="api")

    result = await rl.check(tenant_id="tenant-window", resource="api")
    assert result.allowed is False

    await asyncio.sleep(1.1)

    result = await rl.check(tenant_id="tenant-window", resource="api")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_different_tenants_independent(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-x", resource="api")

    result = await rate_limiter.check(tenant_id="tenant-x", resource="api")
    assert result.allowed is False

    result = await rate_limiter.check(tenant_id="tenant-y", resource="api")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_different_resources_independent(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-r", resource="search")

    result = await rate_limiter.check(tenant_id="tenant-r", resource="search")
    assert result.allowed is False

    result = await rate_limiter.check(tenant_id="tenant-r", resource="ingest")
    assert result.allowed is True


@pytest.mark.asyncio
async def test_returns_retry_after_when_blocked(rate_limiter):
    for _ in range(5):
        await rate_limiter.check(tenant_id="tenant-retry", resource="llm")

    result = await rate_limiter.check(tenant_id="tenant-retry", resource="llm")
    assert result.allowed is False
    assert isinstance(result.retry_after, float)
    assert 0.0 < result.retry_after <= 60.0
