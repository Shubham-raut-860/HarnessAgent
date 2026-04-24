"""PromptVersion dataclass and related schemas for HarnessAgent."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PromptVersion:
    """A versioned prompt snapshot for a specific agent type.

    Attributes:
        version_id:     UUID string uniquely identifying this version.
        agent_type:     Which agent this prompt belongs to (e.g. "sql", "code").
        content:        The full prompt text.
        version_number: Auto-incrementing integer per agent_type (1-based).
        active:         If True, this is the currently active prompt.
        score:          Eval score set after running evaluation (0-1 or None).
        patch_id:       ID of the Hermes patch that created this version, if any.
        created_by:     Who created this version ("human", "hermes", etc.).
        created_at:     UTC datetime when this version was created.
        tags:           Classification tags (e.g. ["v2", "improved"]).
        metadata:       Arbitrary extra data dict.
    """

    version_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_type: str = ""
    content: str = ""
    version_number: int = 1
    active: bool = False
    score: Optional[float] = None
    patch_id: Optional[str] = None
    created_by: str = "human"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict."""
        return {
            "version_id": self.version_id,
            "agent_type": self.agent_type,
            "content": self.content,
            "version_number": self.version_number,
            "active": self.active,
            "score": self.score,
            "patch_id": self.patch_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "PromptVersion":
        """Deserialise from a dict (e.g. loaded from Redis)."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)

        return cls(
            version_id=data.get("version_id", uuid.uuid4().hex),
            agent_type=data.get("agent_type", ""),
            content=data.get("content", ""),
            version_number=int(data.get("version_number", 1)),
            active=bool(data.get("active", False)),
            score=data.get("score"),
            patch_id=data.get("patch_id"),
            created_by=data.get("created_by", "human"),
            created_at=created_at,
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PromptVersion":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(raw))

    def __repr__(self) -> str:
        return (
            f"PromptVersion(agent_type={self.agent_type!r}, "
            f"version_number={self.version_number}, active={self.active}, "
            f"version_id={self.version_id[:8]}...)"
        )
