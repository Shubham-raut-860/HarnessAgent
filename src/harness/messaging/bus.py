"""AgentMessageBus: Redis Streams-backed inter-agent messaging."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from harness.core.errors import InterAgentTimeout
from harness.messaging.schema import AgentMessage

logger = logging.getLogger(__name__)

_STREAM_PREFIX = "harness:stream:"
_BROADCAST_STREAM = "harness:stream:broadcast"
_MSG_INDEX_KEY = "harness:msg_index"
_BLOCK_MS = 100  # XREAD block timeout in milliseconds


class AgentMessageBus:
    """
    Redis Streams-backed message bus for inter-agent communication.

    Each agent has a dedicated stream: ``harness:stream:{agent_id}``
    A shared broadcast stream: ``harness:stream:broadcast``

    Message routing:
    - ``send(msg)``          → XADD to recipient or broadcast stream
    - ``subscribe(agent_id)``→ XREAD blocking on agent + broadcast streams
    - ``request(...)``       → send + await correlated reply
    - ``fan_out(...)``       → send to N agents, gather replies
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: Any = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                max_connections=30,
            )
        return self._client

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def send(self, msg: AgentMessage) -> str:
        """
        Publish a message to the appropriate Redis stream.

        Returns the stream entry ID assigned by Redis.
        Also registers the message in the TTL index.
        """
        client = await self._get_client()
        stream_key = (
            _BROADCAST_STREAM
            if msg.is_broadcast()
            else f"{_STREAM_PREFIX}{msg.recipient_id}"
        )
        serialized = json.dumps(msg.to_dict(), default=str)

        try:
            entry_id: str = await client.xadd(
                stream_key,
                {"data": serialized},
                maxlen=10_000,
                approximate=True,
            )

            # Register in TTL index (score = expire timestamp)
            expire_ts = time.time() + msg.ttl_seconds
            await client.zadd(_MSG_INDEX_KEY, {msg.id: expire_ts})

            return entry_id
        except Exception as exc:
            logger.error("AgentMessageBus.send failed: %s", exc)
            raise

    async def subscribe(
        self,
        agent_id: str,
        message_types: list[str] | None = None,
        last_id: str = "$",
    ) -> AsyncIterator[AgentMessage]:
        """
        Async generator yielding messages for ``agent_id``.

        Reads from both:
        - ``harness:stream:{agent_id}`` (direct messages)
        - ``harness:stream:broadcast`` (broadcast messages)

        Filters by ``message_types`` if specified.
        Expired messages are silently skipped.
        Uses XREAD with BLOCK=100ms to avoid busy-waiting.
        """
        client = await self._get_client()
        agent_stream = f"{_STREAM_PREFIX}{agent_id}"

        # Maintain per-stream cursor IDs
        cursors: dict[str, str] = {
            agent_stream: last_id,
            _BROADCAST_STREAM: last_id,
        }

        try:
            while True:
                streams_to_read = list(cursors.keys())
                ids_to_read = list(cursors.values())

                try:
                    results = await client.xread(
                        streams={k: v for k, v in zip(streams_to_read, ids_to_read)},
                        block=_BLOCK_MS,
                        count=50,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("XREAD failed: %s — retrying", exc)
                    await asyncio.sleep(0.5)
                    continue

                if not results:
                    continue

                for stream_name, entries in results:
                    # Update cursor
                    if entries:
                        cursors[stream_name] = entries[-1][0]

                    for entry_id, data in entries:
                        raw = data.get("data")
                        if not raw:
                            continue
                        try:
                            msg_dict = json.loads(raw)
                            msg = AgentMessage.from_dict(msg_dict)
                        except (json.JSONDecodeError, KeyError, TypeError) as exc:
                            logger.debug("Message deserialisation failed: %s", exc)
                            continue

                        # Skip expired messages
                        if msg.is_expired():
                            continue

                        # Filter by message type
                        if message_types and msg.message_type not in message_types:
                            continue

                        yield msg

        except GeneratorExit:
            pass

    async def request(
        self,
        sender_id: str,
        recipient_id: str,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> AgentMessage:
        """
        Send a query and await a correlated result or error reply.

        Raises InterAgentTimeout if no reply arrives within ``timeout`` seconds.
        """
        import uuid as _uuid

        correlation_id = _uuid.uuid4().hex
        query_msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type="query",
            payload=payload,
            correlation_id=correlation_id,
        )
        await self.send(query_msg)

        deadline = asyncio.get_running_loop().time() + timeout
        async for reply in self.subscribe(
            sender_id, message_types=["result", "error"]
        ):
            if reply.correlation_id == correlation_id:
                return reply
            if asyncio.get_running_loop().time() > deadline:
                break

        raise InterAgentTimeout(
            f"No reply from {recipient_id} within {timeout}s",
            context={"sender": sender_id, "correlation_id": correlation_id},
        )

    async def fan_out(
        self,
        sender_id: str,
        recipient_ids: list[str],
        payload: dict[str, Any],
        timeout: float = 60.0,
    ) -> list[AgentMessage]:
        """
        Send the same payload to multiple recipients and collect replies.

        Returns partial results — if some recipients time out their replies
        are simply omitted from the list.
        """
        import uuid as _uuid

        correlation_id = _uuid.uuid4().hex
        pending: set[str] = set(recipient_ids)
        replies: list[AgentMessage] = []

        # Send to all recipients in parallel
        send_tasks = [
            self.send(
                AgentMessage(
                    sender_id=sender_id,
                    recipient_id=rid,
                    message_type="task",
                    payload=payload,
                    correlation_id=correlation_id,
                )
            )
            for rid in recipient_ids
        ]
        await asyncio.gather(*send_tasks, return_exceptions=True)

        # Collect replies
        deadline = asyncio.get_running_loop().time() + timeout
        async for reply in self.subscribe(
            sender_id, message_types=["result", "error"]
        ):
            if reply.correlation_id == correlation_id and reply.sender_id in pending:
                replies.append(reply)
                pending.discard(reply.sender_id)
                if not pending:
                    break
            if asyncio.get_running_loop().time() > deadline:
                break

        return replies

    async def broadcast(
        self,
        sender_id: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Publish a broadcast message to all subscribers."""
        msg = AgentMessage(
            sender_id=sender_id,
            recipient_id=None,
            message_type=message_type,  # type: ignore[arg-type]
            payload=payload,
        )
        return await self.send(msg)

    async def cleanup_expired(self) -> int:
        """
        Remove expired message IDs from the TTL index.

        Returns the number of entries removed.
        """
        client = await self._get_client()
        try:
            now = time.time()
            removed: int = await client.zremrangebyscore(
                _MSG_INDEX_KEY, "-inf", now
            )
            if removed:
                logger.debug("DLQ cleanup: removed %d expired message index entries", removed)
            return removed
        except Exception as exc:
            logger.warning("cleanup_expired failed: %s", exc)
            return 0

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
