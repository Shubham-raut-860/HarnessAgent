"""Append-only audit log for compliance and security tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """
    Immutable record of a significant harness action.

    Sensitive payload data is stored only as a SHA-256 hash (``payload_hash``);
    the raw data is never persisted by this logger.
    """

    run_id: str
    tenant_id: str
    agent_type: str
    actor: str
    action: Literal[
        "tool_call",
        "memory_write",
        "memory_read",
        "llm_call",
        "agent_start",
        "agent_complete",
        "hitl_request",
        "hitl_resolve",
        "safety_block",
        "patch_apply",
    ]
    resource: str
    payload_hash: str  # SHA-256 of sensitive payload data
    decision: Literal["allowed", "blocked", "escalated"]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @staticmethod
    def hash_payload(payload: Any) -> str:
        """Return the SHA-256 hex digest of the JSON-serialised payload."""
        try:
            raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        except (TypeError, ValueError):
            raw = str(payload).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "agent_type": self.agent_type,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "payload_hash": self.payload_hash,
            "decision": self.decision,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuditEvent":
        d = dict(d)
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AuditLogger:
    """
    Append-only audit logger that writes to:
    1. A JSONL file at ``{workspace_base}/audit/{tenant_id}/{date}.jsonl``
    2. A Redis stream ``harness:audit`` for real-time consumption.

    Events are never modified or deleted (append-only by design).
    """

    _STREAM_KEY = "harness:audit"

    def __init__(
        self,
        workspace_base: str | Path,
        redis_client: Any = None,
    ) -> None:
        self._workspace_base = Path(workspace_base)
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def log(self, event: AuditEvent) -> None:
        """
        Persist an audit event to file and Redis stream.

        File write is synchronous but wrapped in asyncio executor for
        production use; here we use direct write for simplicity and reliability.
        """
        event_dict = event.to_dict()
        event_json = json.dumps(event_dict, ensure_ascii=False)

        # 1. Append to JSONL file
        try:
            log_path = self._audit_path(event.tenant_id, event.timestamp)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(event_json + "\n")
        except Exception as exc:
            logger.error("Audit file write failed: %s", exc)

        # 2. Publish to Redis stream
        if self._redis is not None:
            try:
                await self._redis.xadd(
                    self._STREAM_KEY,
                    {"data": event_json},
                    maxlen=50_000,
                    approximate=True,
                )
            except Exception as exc:
                logger.warning("Audit Redis publish failed: %s", exc)

    async def query(
        self,
        tenant_id: str,
        start: datetime,
        end: datetime,
        action: str | None = None,
    ) -> list[AuditEvent]:
        """
        Query audit events for a tenant within a time range.

        Reads from the JSONL files on disk.  For high-volume use cases,
        this should be replaced with a database query.
        """
        events: list[AuditEvent] = []
        audit_dir = self._workspace_base / "audit" / tenant_id

        if not audit_dir.exists():
            return events

        # Collect all relevant date files in range
        for log_file in sorted(audit_dir.glob("*.jsonl")):
            try:
                with log_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            event = AuditEvent.from_dict(data)
                            if event.timestamp < start or event.timestamp > end:
                                continue
                            if action and event.action != action:
                                continue
                            events.append(event)
                        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                            continue
            except Exception as exc:
                logger.warning("Audit query file read failed (%s): %s", log_file, exc)

        # Sort by timestamp ascending
        events.sort(key=lambda e: e.timestamp)
        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _audit_path(self, tenant_id: str, dt: datetime) -> Path:
        """Return the JSONL file path for the given tenant and date."""
        date_str = dt.strftime("%Y-%m-%d")
        return self._workspace_base / "audit" / tenant_id / f"{date_str}.jsonl"
