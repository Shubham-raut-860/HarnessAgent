"""Memory-related dataclasses for Codex Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class MemoryEntry:
    """A single retrieved memory entry from any tier."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["short", "long", "graph"] = "long"


@dataclass
class ConversationMessage:
    """A single turn in a conversation history."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: int = 0


@dataclass
class GraphFact:
    """A subject-predicate-object triple stored in the knowledge graph."""

    subject: str
    predicate: str
    object_: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Combined result from graph + vector retrieval."""

    graph_paths: list[Any]  # list[GraphPath]
    graph_context: str
    vector_hits: list[Any]  # list[VectorHit]
    vector_context: list[str]
    total_tokens_estimate: int
    strategy: Literal[
        "graph_primary", "vector_primary", "hybrid", "vector_fallback"
    ] = "hybrid"


@dataclass
class ContextWindow:
    """A fitted context window ready for LLM consumption."""

    messages: list[ConversationMessage]
    total_tokens: int
    truncated: bool
    summary: str | None = None
