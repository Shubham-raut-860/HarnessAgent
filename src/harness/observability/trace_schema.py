"""TraceSpan schema — canonical span model for HarnessAgent traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    RUN = "run"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    GUARDRAIL = "guardrail"
    MEMORY = "memory"
    HANDOFF = "handoff"
    EVAL = "eval"


class SpanStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"


@dataclass
class TraceSpan:
    """
    One node in the agent trace tree.

    Hierarchy
    ---------
    run_span
     └── agent_span
          ├── llm_span        (one per LLM call)
          ├── tool_span       (one per tool execution)
          ├── guardrail_span  (one per safety check)
          ├── memory_span     (one per retrieval)
          └── handoff_span    (one per inter-agent message)
    """

    # Identity
    trace_id: str
    span_id: str
    run_id: str
    kind: SpanKind
    name: str                        # e.g. "llm:claude-3-5-sonnet", "tool:execute_sql"
    status: SpanStatus

    # Timing
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: float | None = None

    # Hierarchy
    parent_span_id: str | None = None

    # Payload (truncated to 500 chars for storage efficiency)
    input_preview: str = ""
    output_preview: str = ""
    error: str | None = None

    # Resource usage
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    # Run context
    agent_type: str = ""
    tenant_id: str = ""
    step: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def finish(
        self,
        status: SpanStatus = SpanStatus.OK,
        output_preview: str = "",
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> TraceSpan:
        """Finalise the span in-place and return self."""
        now = datetime.now(UTC)
        self.end_time = now
        self.duration_ms = (now - self.start_time).total_seconds() * 1000
        self.status = status
        if output_preview:
            self.output_preview = output_preview[:500]
        self.error = error
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.cached = cached
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        d["start_time"] = self.start_time.isoformat()
        d["end_time"] = self.end_time.isoformat() if self.end_time else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceSpan:
        d = dict(d)
        d["kind"] = SpanKind(d["kind"])
        d["status"] = SpanStatus(d["status"])
        d["start_time"] = datetime.fromisoformat(d["start_time"])
        if d.get("end_time"):
            d["end_time"] = datetime.fromisoformat(d["end_time"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TraceView:
    """Full trace for a single run — returned by the API."""

    trace_id: str
    run_id: str
    agent_type: str
    status: SpanStatus
    start_time: datetime
    end_time: datetime | None
    duration_ms: float | None
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    span_count: int
    spans: list[TraceSpan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "span_count": self.span_count,
            "spans": [s.to_dict() for s in self.spans],
        }
