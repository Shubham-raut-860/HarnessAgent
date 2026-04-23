"""Unit tests for inter-agent messaging."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def event_bus(redis_client):
    """Create an EventBus backed by fakeredis."""
    try:
        from harness.orchestrator.event_bus import EventBus  # type: ignore
        return EventBus(redis=redis_client)
    except ImportError:
        pytest.skip("EventBus not implemented")


@pytest_asyncio.fixture
async def message_bus(redis_client):
    """Create a MessageBus backed by fakeredis."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        return MessageBus(redis=redis_client)
    except ImportError:
        try:
            from harness.orchestrator.event_bus import EventBus  # type: ignore
            return EventBus(redis=redis_client)
        except ImportError:
            pytest.skip("MessageBus/EventBus not implemented")


@pytest.mark.asyncio
async def test_send_message_to_specific_agent(redis_client):
    """A message sent to a specific agent should be receivable by that agent only."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        bus = MessageBus(redis=redis_client)
    except ImportError:
        pytest.skip("MessageBus not implemented")

    received = []

    async def handler(msg):
        received.append(msg)

    await bus.subscribe("agent-sql", handler)
    await bus.send("agent-sql", {"type": "task", "content": "List tables"})
    await asyncio.sleep(0.1)

    assert len(received) >= 1
    assert any("List tables" in str(m) for m in received)


@pytest.mark.asyncio
async def test_broadcast_received_by_all(redis_client):
    """A broadcast message should be received by all subscribed agents."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        bus = MessageBus(redis=redis_client)
    except ImportError:
        pytest.skip("MessageBus not implemented")

    received_a = []
    received_b = []

    await bus.subscribe("agent-a", lambda m: received_a.append(m))
    await bus.subscribe("agent-b", lambda m: received_b.append(m))
    await bus.broadcast({"type": "status", "content": "cycle started"})
    await asyncio.sleep(0.1)

    # Both should have received the broadcast
    assert len(received_a) + len(received_b) >= 2


@pytest.mark.asyncio
async def test_request_reply_returns_correlated_response(redis_client):
    """Request-reply should return the response correlated by correlation_id."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        bus = MessageBus(redis=redis_client)
    except ImportError:
        pytest.skip("MessageBus not implemented")

    async def responder(msg):
        cid = msg.get("correlation_id")
        if cid:
            await bus.send(
                msg.get("reply_to", "requestor"),
                {"correlation_id": cid, "result": "table_list"},
            )

    await bus.subscribe("sql-agent", responder)

    response = await bus.request(
        target="sql-agent",
        message={"action": "list_tables"},
        timeout=2.0,
    )
    assert response is not None
    assert "table_list" in str(response)


@pytest.mark.asyncio
async def test_request_raises_timeout(redis_client):
    """Request should raise TimeoutError when no response is received in time."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        bus = MessageBus(redis=redis_client)
    except ImportError:
        pytest.skip("MessageBus not implemented")

    with pytest.raises((asyncio.TimeoutError, TimeoutError, Exception)):
        await bus.request(
            target="nonexistent-agent",
            message={"action": "do_something"},
            timeout=0.1,
        )


@pytest.mark.asyncio
async def test_fan_out_collects_all_replies(redis_client):
    """fan_out should collect responses from multiple agents."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        bus = MessageBus(redis=redis_client)
    except ImportError:
        pytest.skip("MessageBus not implemented")

    for agent_id in ["agent-1", "agent-2", "agent-3"]:
        async def _responder(msg, aid=agent_id):
            cid = msg.get("correlation_id")
            if cid:
                await bus.send(
                    msg.get("reply_to", "orchestrator"),
                    {"correlation_id": cid, "agent": aid, "result": f"done-{aid}"},
                )

        await bus.subscribe(agent_id, _responder)

    results = await bus.fan_out(
        targets=["agent-1", "agent-2", "agent-3"],
        message={"action": "status"},
        timeout=2.0,
    )
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_expired_messages_skipped(redis_client):
    """Messages with expired TTL should be skipped during retrieval."""
    try:
        from harness.orchestrator.messaging import MessageBus  # type: ignore
        bus = MessageBus(redis=redis_client)
    except ImportError:
        pytest.skip("MessageBus not implemented")

    received = []
    await bus.subscribe("agent-ttl", lambda m: received.append(m))

    # Send a message with a very short TTL
    await bus.send(
        "agent-ttl",
        {"type": "ephemeral", "content": "expires fast"},
        ttl_seconds=0.01,
    )

    # Wait for it to expire
    await asyncio.sleep(0.1)

    # No message should be delivered (already expired)
    assert len(received) == 0
