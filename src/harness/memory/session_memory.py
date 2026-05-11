"""SessionMemory — Redis-backed cross-run memory for persistent agent state.

Unlike ShortTermMemory (per-run conversation history), SessionMemory
stores durable facts keyed by (tenant_id, session_id) that survive
across multiple runs. Typical use cases:

- User preferences learned in run 1, recalled in run 5
- Decisions made in a multi-day research project
- Accumulated errors or rules across Hermes improvement cycles
- Agent persona / learned behaviour for a specific user

Storage layout
--------------
harness:session:{tenant_id}:{session_id}          HASH  — fact key → JSON value
harness:session:{tenant_id}:{session_id}:history  LIST  — append-only event log
harness:session_index:{tenant_id}                  ZSET  — session_ids by last_used

Default TTL: 7 days (configurable).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_PFX = "harness:session:"
_SESSION_IDX_PFX = "harness:session_index:"
_DEFAULT_TTL = 604_800          # 7 days
_HISTORY_MAX = 500              # max events per session history


class SessionMemory:
    """
    Persistent cross-run memory for a (tenant_id, session_id) pair.

    Usage
    -----
    session = SessionMemory(redis_client, tenant_id="acme", session_id="user-42")

    # Remember a fact
    await session.remember("preferred_language", "Python")
    await session.remember("last_topic", "SQL optimisation")

    # Recall all facts
    facts = await session.recall_all()
    # → {"preferred_language": "Python", "last_topic": "SQL optimisation"}

    # Recall one fact
    lang = await session.recall("preferred_language")   # → "Python"

    # Append to history log
    await session.append_event("run_completed", {"run_id": "abc", "success": True})

    # Get recent history
    events = await session.get_history(last_n=20)
    """

    def __init__(
        self,
        redis: Any,
        tenant_id: str,
        session_id: str,
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        self._redis = redis
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._ttl = ttl

    # ── Keys ─────────────────────────────────────────────────────────────

    @property
    def _hash_key(self) -> str:
        return f"{_SESSION_PFX}{self._tenant_id}:{self._session_id}"

    @property
    def _history_key(self) -> str:
        return f"{_SESSION_PFX}{self._tenant_id}:{self._session_id}:history"

    @property
    def _index_key(self) -> str:
        return f"{_SESSION_IDX_PFX}{self._tenant_id}"

    # ── Core API ──────────────────────────────────────────────────────────

    async def remember(self, key: str, value: Any) -> None:
        """Store a fact. Value is JSON-serialised so any type is supported."""
        serialized = json.dumps(value, default=str)
        await self._redis.hset(self._hash_key, key, serialized)
        now = time.time()
        await self._redis.expire(self._hash_key, self._ttl)
        await self._redis.zadd(self._index_key, {self._session_id: now})
        await self._redis.expire(self._index_key, self._ttl)

    async def recall(self, key: str, default: Any = None) -> Any:
        """Return the stored fact for ``key``, or ``default`` if not found."""
        raw = await self._redis.hget(self._hash_key, key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def recall_all(self) -> dict[str, Any]:
        """Return all stored facts as a dict."""
        raw_map = await self._redis.hgetall(self._hash_key)
        result: dict[str, Any] = {}
        for k, v in raw_map.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result

    async def forget(self, key: str) -> None:
        """Remove a specific fact."""
        await self._redis.hdel(self._hash_key, key)

    async def clear(self) -> None:
        """Delete all facts and history for this session."""
        await self._redis.delete(self._hash_key, self._history_key)
        await self._redis.zrem(self._index_key, self._session_id)

    # ── History log ───────────────────────────────────────────────────────

    async def append_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append an event to the session history log."""
        entry = json.dumps({
            "event_type": event_type,
            "payload": payload or {},
            "ts": datetime.now(timezone.utc).isoformat(),
        }, default=str)
        await self._redis.lpush(self._history_key, entry)
        await self._redis.ltrim(self._history_key, 0, _HISTORY_MAX - 1)
        await self._redis.expire(self._history_key, self._ttl)

    async def get_history(self, last_n: int = 20) -> list[dict[str, Any]]:
        """Return the most recent ``last_n`` events (newest first)."""
        raw_items = await self._redis.lrange(self._history_key, 0, last_n - 1)
        events: list[dict[str, Any]] = []
        for raw in raw_items:
            try:
                events.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass
        return events

    # ── Context summary ───────────────────────────────────────────────────

    async def build_context_summary(self, max_facts: int = 20) -> str:
        """
        Return a plain-text summary of stored facts for injection into an
        agent system prompt or context window.

        Example output
        --------------
        [Session memory — user-42]
        preferred_language: Python
        last_topic: SQL optimisation
        user_skill_level: intermediate
        """
        facts = await self.recall_all()
        if not facts:
            return ""
        lines = [f"[Session memory — {self._session_id}]"]
        for k, v in list(facts.items())[:max_facts]:
            lines.append(f"{k}: {v}")
        return "\n".join(lines)


# ── Session registry helper (MemoryManager integration) ──────────────────────

class SessionMemoryRegistry:
    """
    Manages SessionMemory instances across multiple (tenant, session) pairs.

    Typically one registry per application process, stored in app.state.
    """

    def __init__(self, redis: Any, default_ttl: int = _DEFAULT_TTL) -> None:
        self._redis = redis
        self._ttl = default_ttl
        self._sessions: dict[str, SessionMemory] = {}

    def get(self, tenant_id: str, session_id: str = "default") -> SessionMemory:
        """Return or create a SessionMemory for the given (tenant, session)."""
        key = f"{tenant_id}:{session_id}"
        if key not in self._sessions:
            self._sessions[key] = SessionMemory(
                self._redis, tenant_id, session_id, self._ttl
            )
        return self._sessions[key]

    async def list_sessions(self, tenant_id: str) -> list[str]:
        """Return all active session IDs for a tenant (by last-used recency)."""
        index_key = f"{_SESSION_IDX_PFX}{tenant_id}"
        try:
            members = await self._redis.zrevrange(index_key, 0, 99, withscores=False)
            return [m if isinstance(m, str) else m.decode() for m in members]
        except Exception:
            return []
