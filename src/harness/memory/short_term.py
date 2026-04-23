"""Short-term conversation memory backed by Redis."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError

from harness.core.errors import FailureClass, HarnessError
from harness.memory.schemas import ConversationMessage

logger = logging.getLogger(__name__)

_CONV_PREFIX = "harness:conv:"
_SCRATCH_PREFIX = "harness:scratch:"


class ShortTermMemory:
    """
    Redis-backed short-term memory for per-run conversation history and scratch pad.

    Conversation messages are stored as JSON strings in a Redis List (LPUSH),
    so index 0 is always the most recent message (reversed order).
    The scratch pad uses a Redis Hash scoped to the run_id with a TTL.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._pool = aioredis.ConnectionPool.from_url(
                self._redis_url,
                max_connections=20,
                decode_responses=True,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
        return self._client

    async def close(self) -> None:
        """Release Redis connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conv_key(self, run_id: str) -> str:
        return f"{_CONV_PREFIX}{run_id}"

    def _scratch_key(self, run_id: str) -> str:
        return f"{_SCRATCH_PREFIX}{run_id}"

    def _serialize_message(self, msg: ConversationMessage) -> str:
        return json.dumps(
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "tokens": msg.tokens,
            }
        )

    def _deserialize_message(self, raw: str) -> ConversationMessage:
        data = json.loads(raw)
        ts_raw = data.get("timestamp")
        ts = (
            datetime.fromisoformat(ts_raw)
            if ts_raw
            else datetime.now(timezone.utc)
        )
        return ConversationMessage(
            role=data["role"],
            content=data["content"],
            timestamp=ts,
            tokens=data.get("tokens", 0),
        )

    async def _safe_exec(self, coro: Any) -> Any:
        """Execute a Redis coroutine and translate connection errors to HarnessError."""
        try:
            return await coro
        except RedisConnectionError as exc:
            logger.warning("Redis connection error: %s", exc)
            raise HarnessError(
                f"Redis unavailable: {exc}",
                failure_class=FailureClass.MEMORY_REDIS,
                context={"redis_url": self._redis_url},
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def push_message(
        self,
        run_id: str,
        role: str,
        content: str,
        tokens: int = 0,
    ) -> None:
        """Prepend a new conversation message to the Redis list for run_id."""
        client = await self._get_client()
        msg = ConversationMessage(
            role=role,  # type: ignore[arg-type]
            content=content,
            tokens=tokens,
        )
        serialized = self._serialize_message(msg)
        await self._safe_exec(client.lpush(self._conv_key(run_id), serialized))

    async def get_history(
        self,
        run_id: str,
        last_n: int = 20,
    ) -> list[ConversationMessage]:
        """
        Retrieve the most recent ``last_n`` messages for this run.
        Returns newest-first order is reversed to chronological order.
        """
        client = await self._get_client()
        # Messages are LPUSH'd so index 0 = newest; LRANGE 0..last_n-1 gives newest→oldest
        raw_items: list[str] = await self._safe_exec(
            client.lrange(self._conv_key(run_id), 0, last_n - 1)
        )
        # Reverse so we return chronological order (oldest first)
        messages = [self._deserialize_message(raw) for raw in reversed(raw_items)]
        return messages

    async def set_scratch(
        self,
        run_id: str,
        key: str,
        value: Any,
        ttl: int = 3600,
    ) -> None:
        """Store an arbitrary JSON-serialisable value in the scratch hash."""
        client = await self._get_client()
        scratch_key = self._scratch_key(run_id)
        serialized_value = json.dumps(value, default=str)
        await self._safe_exec(client.hset(scratch_key, key, serialized_value))
        await self._safe_exec(client.expire(scratch_key, ttl))

    async def get_scratch(self, run_id: str, key: str) -> Any | None:
        """Retrieve a value from the scratch hash; returns None if missing."""
        client = await self._get_client()
        raw = await self._safe_exec(
            client.hget(self._scratch_key(run_id), key)
        )
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def clear(self, run_id: str) -> None:
        """Delete conversation list and scratch hash for the given run_id."""
        client = await self._get_client()
        await self._safe_exec(
            client.delete(self._conv_key(run_id), self._scratch_key(run_id))
        )

    async def get_context_size(self, run_id: str) -> int:
        """Return the number of messages stored for this run."""
        client = await self._get_client()
        length: int = await self._safe_exec(
            client.llen(self._conv_key(run_id))
        )
        return length
