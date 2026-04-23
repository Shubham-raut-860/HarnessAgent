"""Redis Pub/Sub event bus for real-time step streaming."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from harness.core.context import StepEvent

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "harness:events:"


class EventBus:
    """
    Redis Pub/Sub event bus for streaming StepEvents to SSE endpoints.

    Publishers call ``publish(run_id, event)``; subscribers iterate over
    ``subscribe(run_id)`` which yields StepEvent instances in real time.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pub_client: Any = None
        self._sub_clients: list[Any] = []

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _get_publisher(self) -> Any:
        if self._pub_client is None:
            import redis.asyncio as aioredis

            self._pub_client = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._pub_client

    async def _create_subscriber(self) -> Any:
        """Create a dedicated Redis client for a single subscription."""
        import redis.asyncio as aioredis

        client = aioredis.from_url(self._redis_url, decode_responses=True)
        self._sub_clients.append(client)
        return client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, run_id: str, event: StepEvent) -> None:
        """
        Serialise and publish a StepEvent to the run's Pub/Sub channel.
        """
        channel = f"{_CHANNEL_PREFIX}{run_id}"
        payload = {
            "run_id": event.run_id,
            "step": event.step,
            "event_type": event.event_type,
            "payload": event.payload,
            "timestamp": event.timestamp.isoformat(),
        }
        try:
            publisher = await self._get_publisher()
            await publisher.publish(channel, json.dumps(payload, default=str))
        except Exception as exc:
            logger.warning("EventBus publish failed for run %s: %s", run_id, exc)

    async def subscribe(self, run_id: str) -> AsyncIterator[StepEvent]:
        """
        Async generator that yields StepEvents published for ``run_id``.

        Automatically unsubscribes when the generator is closed (GeneratorExit).
        Each subscriber uses its own dedicated Redis connection.
        """
        channel = f"{_CHANNEL_PREFIX}{run_id}"
        client = await self._create_subscriber()
        pubsub = client.pubsub()

        try:
            await pubsub.subscribe(channel)
            logger.debug("EventBus: subscribed to channel %s", channel)

            async for message in pubsub.listen():
                if message is None:
                    continue

                msg_type = message.get("type")
                if msg_type != "message":
                    # Skip subscribe/unsubscribe confirmation messages
                    continue

                data = message.get("data")
                if not data:
                    continue

                try:
                    payload = json.loads(data)
                    ts_raw = payload.get("timestamp")
                    ts = (
                        datetime.fromisoformat(ts_raw)
                        if ts_raw
                        else datetime.now(timezone.utc)
                    )
                    step_event = StepEvent(
                        run_id=payload.get("run_id", run_id),
                        step=int(payload.get("step", 0)),
                        event_type=payload.get("event_type", "unknown"),
                        payload=payload.get("payload", {}),
                        timestamp=ts,
                    )
                    yield step_event
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    logger.debug("EventBus: malformed message skipped: %s", exc)
                    continue

        except GeneratorExit:
            logger.debug("EventBus: subscriber for run %s closed", run_id)
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()
                await client.aclose()
                if client in self._sub_clients:
                    self._sub_clients.remove(client)
            except Exception as exc:
                logger.debug("EventBus cleanup error: %s", exc)

    async def close(self) -> None:
        """Close all subscriber and publisher connections."""
        for client in list(self._sub_clients):
            try:
                await client.aclose()
            except Exception:
                pass
        self._sub_clients.clear()

        if self._pub_client is not None:
            try:
                await self._pub_client.aclose()
            except Exception:
                pass
            self._pub_client = None
