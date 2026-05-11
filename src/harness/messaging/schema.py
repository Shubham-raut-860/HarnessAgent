"""AgentMessage dataclass for inter-agent communication."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class AgentMessage:
    """
    A typed message exchanged between agents via the AgentMessageBus.

    ``recipient_id=None`` indicates a broadcast message.
    ``correlation_id`` links replies to original requests.
    ``ttl_seconds`` controls how long this message remains valid.
    """

    sender_id: str
    message_type: Literal["task", "result", "error", "query", "status", "heartbeat"]
    payload: dict[str, Any] = field(default_factory=dict)
    recipient_id: str | None = None
    correlation_id: str | None = None
    parent_run_id: str | None = None
    traceparent: str | None = None   # W3C TraceContext header for span propagation
    trace_id: str | None = None      # harness AgentContext.trace_id
    ttl_seconds: float = 300.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_broadcast(self) -> bool:
        """Return True if this message has no specific recipient."""
        return self.recipient_id is None

    def is_expired(self) -> bool:
        """Return True if the message TTL has elapsed."""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "parent_run_id": self.parent_run_id,
            "traceparent": self.traceparent,
            "trace_id": self.trace_id,
            "ttl_seconds": self.ttl_seconds,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentMessage":
        d = dict(d)
        ts = d.get("timestamp")
        if isinstance(ts, str):
            d["timestamp"] = datetime.fromisoformat(ts)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
