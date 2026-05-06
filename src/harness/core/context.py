"""AgentContext and core data-transfer dataclasses for HarnessAgent."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.core.errors import BudgetExceeded, FailureClass


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolCall."""

    data: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Return True when the result represents a failure."""
        return self.error is not None

    def to_text(self) -> str:
        """Serialise the result to a plain string for LLM consumption."""
        if self.is_error:
            return f"ERROR: {self.error}"
        try:
            return json.dumps(self.data, default=str)
        except (TypeError, ValueError):
            return str(self.data)


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    cached: bool = False


@dataclass
class StepEvent:
    """Structured event emitted at each step of agent execution."""

    run_id: str
    step: int
    event_type: str
    payload: dict[str, Any]
    timestamp: datetime

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def started(cls, ctx: AgentContext) -> StepEvent:
        """Build a 'started' event from the given context."""
        return cls(
            run_id=ctx.run_id,
            step=ctx.step_count,
            event_type="started",
            payload={"task": ctx.task, "agent_type": ctx.agent_type},
            timestamp=datetime.now(UTC),
        )

    @classmethod
    def llm_called(cls, ctx: AgentContext, response: LLMResponse) -> StepEvent:
        """Build an 'llm_call' event after receiving an LLM response."""
        return cls(
            run_id=ctx.run_id,
            step=ctx.step_count,
            event_type="llm_call",
            payload={
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cached": response.cached,
                "tool_calls": len(response.tool_calls),
            },
            timestamp=datetime.now(UTC),
        )

    @classmethod
    def tool_called(
        cls, ctx: AgentContext, call: ToolCall, result: ToolResult
    ) -> StepEvent:
        """Build a 'tool_call' event after executing a tool."""
        return cls(
            run_id=ctx.run_id,
            step=ctx.step_count,
            event_type="tool_call",
            payload={
                "tool_id": call.id,
                "tool_name": call.name,
                "args": call.args,
                "is_error": result.is_error,
                "error": result.error,
            },
            timestamp=datetime.now(UTC),
        )

    @classmethod
    def completed(cls, ctx: AgentContext, output: str) -> StepEvent:
        """Build a 'completed' event when the agent finishes successfully."""
        return cls(
            run_id=ctx.run_id,
            step=ctx.step_count,
            event_type="completed",
            payload={"output": output, "elapsed_seconds": ctx.elapsed_seconds},
            timestamp=datetime.now(UTC),
        )

    @classmethod
    def failed(cls, ctx: AgentContext, error: str) -> StepEvent:
        """Build a 'failed' event when the agent terminates with an error."""
        return cls(
            run_id=ctx.run_id,
            step=ctx.step_count,
            event_type="failed",
            payload={
                "error": error,
                "failure_class": ctx.failure_class,
                "elapsed_seconds": ctx.elapsed_seconds,
            },
            timestamp=datetime.now(UTC),
        )


@dataclass
class AgentResult:
    """Final result returned after an agent run completes or fails."""

    run_id: str
    output: str
    steps: int
    tokens: int
    success: bool
    failure_class: str | None = None
    error_message: str | None = None
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0
    mlflow_run_id: str | None = None
    tool_calls: int = 0
    tool_errors: int = 0
    guardrail_hits: int = 0
    handoff_count: int = 0
    cache_hits: int = 0
    cache_read_tokens: int = 0


@dataclass
class AgentContext:
    """Mutable per-run execution context shared across all agent components."""

    run_id: str
    tenant_id: str
    agent_type: str
    task: str
    memory: Any  # MemoryManager — forward ref to avoid circular imports
    workspace_path: Path
    max_steps: int = 50
    max_tokens: int = 100_000
    timeout_seconds: float = 300.0
    step_count: int = 0
    token_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    failed: bool = False
    failure_class: str | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        tenant_id: str,
        agent_type: str,
        task: str,
        memory: Any,
        workspace_path: Path,
        *,
        max_steps: int = 50,
        max_tokens: int = 100_000,
        timeout_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentContext:
        """Construct a new AgentContext with auto-generated run_id and trace_id."""
        return cls(
            run_id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            agent_type=agent_type,
            task=task,
            memory=memory,
            workspace_path=workspace_path,
            max_steps=max_steps,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def is_budget_ok(self) -> bool:
        """Return True if the run has not exceeded any budget limit."""
        if self.step_count >= self.max_steps:
            return False
        if self.token_count >= self.max_tokens:
            return False
        if self.elapsed_seconds >= self.timeout_seconds:
            return False
        return True

    def tick(self, tokens: int = 0) -> None:
        """Advance step counter and accumulate token usage; raise on budget exceeded."""
        self.step_count += 1
        self.token_count += tokens

        if self.step_count > self.max_steps:
            self.failed = True
            self.failure_class = FailureClass.BUDGET_STEPS.value
            raise BudgetExceeded(
                f"Step budget exceeded: {self.step_count}/{self.max_steps}",
                failure_class=FailureClass.BUDGET_STEPS,
                context={"run_id": self.run_id, "step_count": self.step_count},
            )
        if self.token_count > self.max_tokens:
            self.failed = True
            self.failure_class = FailureClass.BUDGET_TOKENS.value
            raise BudgetExceeded(
                f"Token budget exceeded: {self.token_count}/{self.max_tokens}",
                failure_class=FailureClass.BUDGET_TOKENS,
                context={"run_id": self.run_id, "token_count": self.token_count},
            )
        if self.elapsed_seconds > self.timeout_seconds:
            self.failed = True
            self.failure_class = FailureClass.BUDGET_TIME.value
            raise BudgetExceeded(
                f"Time budget exceeded: {self.elapsed_seconds:.1f}s / {self.timeout_seconds}s",
                failure_class=FailureClass.BUDGET_TIME,
                context={"run_id": self.run_id, "elapsed_seconds": self.elapsed_seconds},
            )

    @property
    def elapsed_seconds(self) -> float:
        """Return wall-clock seconds elapsed since this run started."""
        return (datetime.now(UTC) - self.started_at).total_seconds()
