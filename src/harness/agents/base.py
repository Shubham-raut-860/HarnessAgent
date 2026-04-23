"""BaseAgent: the full agent lifecycle with all production features."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from harness.core.context import (
    AgentContext,
    AgentResult,
    LLMResponse,
    StepEvent,
    ToolCall,
    ToolResult,
)
from harness.core.errors import (
    BudgetExceeded,
    FailureClass,
    HarnessError,
    HITLRejected,
    SafetyViolation,
    ToolError,
)

logger = logging.getLogger(__name__)

# How often (in steps) to save a checkpoint
_CHECKPOINT_INTERVAL = 10

# Maximum history messages to keep in context
_MAX_HISTORY_MESSAGES = 40


class BaseAgent:
    """Production-grade base agent with full lifecycle management.

    Subclasses override:
    - agent_type (class attribute)
    - build_system_prompt(ctx)
    - Optionally override run() to add pre/post processing
    """

    agent_type: str = "base"

    def __init__(
        self,
        llm_router: Any,
        memory_manager: Any,
        tool_registry: Any,
        safety_pipeline: Any,
        step_tracer: Any,
        mlflow_tracer: Any,
        failure_tracker: Any,
        audit_logger: Any,
        event_bus: Any,
        cost_tracker: Any,
        checkpoint_manager: Any,
        message_bus: Any | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._memory = memory_manager
        self._tool_registry = tool_registry
        self._safety_pipeline = safety_pipeline
        self._step_tracer = step_tracer
        self._mlflow_tracer = mlflow_tracer
        self._failure_tracker = failure_tracker
        self._audit_logger = audit_logger
        self._event_bus = event_bus
        self._cost_tracker = cost_tracker
        self._checkpoint_manager = checkpoint_manager
        self._message_bus = message_bus

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Execute the full agent lifecycle and return an AgentResult."""
        run_start = time.monotonic()
        output: str = ""
        total_cost_usd: float = 0.0
        metrics = _get_metrics()

        # 1. Emit started event and begin MLflow run
        await self._emit_event(StepEvent.started(ctx))
        mlflow_run_id: str | None = None

        if metrics is not None:
            try:
                metrics.active_runs.labels(agent_type=self.agent_type).inc()
            except Exception:
                pass

        async with self._mlflow_context(ctx) as mlflow_run_id:
            try:
                # 2. Resume from checkpoint if one exists
                await self._maybe_resume_checkpoint(ctx)

                # Main agentic loop
                history: list[dict[str, Any]] = []

                while ctx.is_budget_ok():
                    # 3a. Fit history to context window
                    history = await self._fit_history(ctx, history)

                    # 3b. Build retrieval context from memory
                    retrieval_context = await self._smart_retrieve(ctx)

                    # 3c. Build messages
                    messages = self.build_messages(ctx, history, retrieval_context)
                    system_prompt = self.build_system_prompt(ctx)

                    # 3d. LLM call with OTel span
                    async with self._llm_span(ctx):
                        response = await self._call_llm(ctx, messages, system_prompt)

                    # 3e-f. Record token usage
                    total_tokens = response.input_tokens + response.output_tokens
                    ctx.tick(tokens=total_tokens)

                    # 3g. Cost tracking
                    if self._cost_tracker is not None:
                        try:
                            run_cost = await self._cost_tracker.record(
                                run_id=ctx.run_id,
                                tenant_id=ctx.tenant_id,
                                model=response.model,
                                input_tokens=response.input_tokens,
                                output_tokens=response.output_tokens,
                            )
                            total_cost_usd += run_cost.cost_usd
                        except Exception as exc:
                            logger.warning("Cost tracking failed: %s", exc)

                    # 3h. MLflow LLM span
                    await self._log_llm_span(ctx, response)

                    # 3i. Emit llm_called event
                    await self._emit_event(StepEvent.llm_called(ctx, response))

                    # 3j. Safety check on output
                    if self._safety_pipeline is not None:
                        try:
                            guard = await _safe_call(
                                self._safety_pipeline.check_output,
                                {"content": response.content},
                            )
                            if guard is not None and getattr(guard, "blocked", False):
                                raise SafetyViolation(
                                    f"Output blocked by safety pipeline: {getattr(guard, 'reason', '')}",
                                    guard_source="output_guard",
                                    failure_class=FailureClass.SAFETY_OUTPUT,
                                )
                        except SafetyViolation:
                            raise
                        except Exception as exc:
                            logger.debug("Safety output check raised: %s", exc)

                    # Add assistant message to history
                    history.append({"role": "assistant", "content": response.content})

                    # 3k. Push assistant response to short-term memory
                    if ctx.memory is not None:
                        try:
                            await ctx.memory.push_message(
                                run_id=ctx.run_id,
                                role="assistant",
                                content=response.content,
                                tokens=response.output_tokens,
                            )
                        except Exception as exc:
                            logger.debug("Failed to push assistant message to memory: %s", exc)

                    # 3l. If no tool calls, we are done
                    if not response.tool_calls:
                        output = self.extract_final_answer(history)
                        break

                    # 3m. Execute each tool call
                    tool_results_for_history: list[dict[str, Any]] = []
                    for call in response.tool_calls:
                        # HITL check
                        await self._check_hitl(ctx, call)

                        # Execute tool
                        try:
                            result = await self._tool_registry.execute(ctx, call)
                        except ToolError as exc:
                            logger.warning("Tool '%s' failed: %s", call.name, exc)
                            result = ToolResult(
                                data=None,
                                error=str(exc),
                                metadata={"failure_class": exc.failure_class.value},
                            )
                        except SafetyViolation as exc:
                            logger.warning("Tool '%s' blocked: %s", call.name, exc)
                            result = ToolResult(
                                data=None,
                                error=f"Blocked by safety policy: {exc}",
                            )

                        # Push tool result to memory
                        if ctx.memory is not None:
                            try:
                                await ctx.memory.push_message(
                                    run_id=ctx.run_id,
                                    role="tool",
                                    content=result.to_text(),
                                    tokens=0,
                                )
                            except Exception as mem_exc:
                                logger.debug("Failed to push tool result to memory: %s", mem_exc)

                        # Emit tool_called step event
                        await self._emit_event(StepEvent.tool_called(ctx, call, result))

                        # Audit log
                        await self._audit(ctx, call, result)

                        tool_results_for_history.append(
                            {
                                "role": "tool",
                                "tool_use_id": call.id,
                                "content": result.to_text(),
                            }
                        )

                    # Append tool results to history
                    history.extend(tool_results_for_history)

                    # 3n. Checkpoint every N steps
                    if ctx.step_count % _CHECKPOINT_INTERVAL == 0:
                        await self._save_checkpoint(ctx, history)

                else:
                    # Budget exceeded — loop condition failed
                    ctx.failed = True
                    if ctx.step_count >= ctx.max_steps:
                        ctx.failure_class = FailureClass.BUDGET_STEPS.value
                    elif ctx.token_count >= ctx.max_tokens:
                        ctx.failure_class = FailureClass.BUDGET_TOKENS.value
                    else:
                        ctx.failure_class = FailureClass.BUDGET_TIME.value
                    await self._emit_event(
                        StepEvent(
                            run_id=ctx.run_id,
                            step=ctx.step_count,
                            event_type="budget_exceeded",
                            payload={"failure_class": ctx.failure_class},
                            timestamp=_utcnow(),
                        )
                    )

            except BudgetExceeded as exc:
                ctx.failed = True
                ctx.failure_class = exc.failure_class.value
                output = f"Budget exceeded: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except HITLRejected as exc:
                ctx.failed = True
                ctx.failure_class = FailureClass.INTER_AGENT_REJECT.value
                output = f"HITL rejected: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except SafetyViolation as exc:
                ctx.failed = True
                ctx.failure_class = exc.failure_class.value
                output = f"Safety violation: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except HarnessError as exc:
                ctx.failed = True
                ctx.failure_class = exc.failure_class.value
                output = f"Harness error: {exc}"
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            except asyncio.CancelledError:
                ctx.failed = True
                ctx.failure_class = FailureClass.UNKNOWN.value
                output = "Run was cancelled."
                await self._emit_event(StepEvent.failed(ctx, "cancelled"))
                raise

            except Exception as exc:
                ctx.failed = True
                ctx.failure_class = self._classify_exception(exc).value
                output = f"Unexpected error: {exc}"
                logger.exception("Unhandled exception in agent run %s", ctx.run_id)
                await self._record_failure(ctx, exc)
                await self._emit_event(StepEvent.failed(ctx, str(exc)))

            finally:
                # 8. Decrement active_runs gauge
                if metrics is not None:
                    try:
                        metrics.active_runs.labels(agent_type=self.agent_type).dec()
                        metrics.agent_runs_total.labels(
                            agent_type=self.agent_type,
                            success=str(not ctx.failed).lower(),
                        ).inc()
                    except Exception:
                        pass

        # Emit completed event if success
        if not ctx.failed:
            await self._emit_event(StepEvent.completed(ctx, output))

        elapsed = time.monotonic() - run_start

        return AgentResult(
            run_id=ctx.run_id,
            output=output,
            steps=ctx.step_count,
            tokens=ctx.token_count,
            success=not ctx.failed,
            failure_class=ctx.failure_class,
            error_message=output if ctx.failed else None,
            elapsed_seconds=elapsed,
            cost_usd=total_cost_usd,
        )

    # ------------------------------------------------------------------
    # Overridable methods
    # ------------------------------------------------------------------

    def build_system_prompt(self, ctx: AgentContext) -> str:
        """Return the system prompt for this agent. Override in subclasses."""
        return (
            f"You are a helpful AI agent of type '{self.agent_type}'. "
            f"Task: {ctx.task}\n"
            "Use the available tools to complete the task. "
            "When you have a final answer, respond without requesting any more tools."
        )

    def build_messages(
        self,
        ctx: AgentContext,
        history: list[dict[str, Any]],
        retrieval_context: str,
    ) -> list[dict[str, Any]]:
        """Assemble the messages list for the LLM call."""
        messages: list[dict[str, Any]] = []

        # If we have retrieval context, prepend it as a user message
        if retrieval_context and not history:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Relevant context from memory:\n{retrieval_context}\n\n"
                        f"Task: {ctx.task}"
                    ),
                }
            )
        elif not history:
            messages.append({"role": "user", "content": ctx.task})

        messages.extend(history)

        # Ensure first message is from user
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": ctx.task})

        return messages

    def extract_final_answer(self, history: list[dict[str, Any]]) -> str:
        """Extract the final answer text from agent history."""
        # Walk history in reverse to find the last assistant message
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Anthropic block content
                    text_parts = [
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                        if not (isinstance(b, dict) and b.get("type") == "tool_use")
                    ]
                    text = " ".join(text_parts).strip()
                    if text:
                        return text
                elif isinstance(content, str) and content.strip():
                    return content.strip()
        return "Task completed."

    def _classify_exception(self, e: Exception) -> FailureClass:
        """Map an exception to a FailureClass."""
        if isinstance(e, BudgetExceeded):
            return e.failure_class
        if isinstance(e, ToolError):
            return e.failure_class
        if isinstance(e, SafetyViolation):
            return e.failure_class
        if isinstance(e, HITLRejected):
            return FailureClass.INTER_AGENT_REJECT
        if isinstance(e, HarnessError):
            return e.failure_class
        if isinstance(e, asyncio.TimeoutError):
            return FailureClass.BUDGET_TIME
        return FailureClass.UNKNOWN

    # ------------------------------------------------------------------
    # HITL
    # ------------------------------------------------------------------

    async def _check_hitl(self, ctx: AgentContext, call: ToolCall) -> None:
        """Check if human-in-the-loop approval is required for this tool call."""
        # Policy-based check: look for hitl manager in metadata
        hitl_manager = ctx.metadata.get("hitl_manager")
        policy = ctx.metadata.get("policy")

        if hitl_manager is None or policy is None:
            return

        if not policy.requires_hitl(call.name):
            return

        logger.info(
            "HITL required for tool '%s' in run '%s'", call.name, ctx.run_id
        )

        # Create approval request
        request = await hitl_manager.request_approval(
            ctx=ctx,
            tool_name=call.name,
            tool_args=call.args,
        )

        # Await the decision
        decision = await hitl_manager.await_decision(
            request_id=request.request_id,
            timeout=3600.0,
        )

        if decision == "rejected":
            raise HITLRejected(
                f"Human reviewer rejected tool call '{call.name}'",
                request_id=request.request_id,
            )
        if decision == "expired":
            raise HITLRejected(
                f"HITL approval for '{call.name}' expired",
                request_id=request.request_id,
            )

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        ctx: AgentContext,
        messages: list[dict[str, Any]],
        system: str,
    ) -> LLMResponse:
        """Call the LLM router with tools from the registry."""
        tools = (
            self._tool_registry.to_anthropic_format()
            if self._tool_registry is not None
            else []
        )

        remaining_tokens = ctx.max_tokens - ctx.token_count
        max_tokens = min(4096, max(256, remaining_tokens // 2))

        return await self._llm_router.complete(
            messages=messages,
            system=system,
            tools=tools if tools else None,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _smart_retrieve(self, ctx: AgentContext) -> str:
        """Retrieve relevant context from long-term memory for the current task."""
        if ctx.memory is None:
            return ""
        try:
            if hasattr(ctx.memory, "smart_retrieve"):
                result = await ctx.memory.smart_retrieve(
                    query=ctx.task,
                    run_id=ctx.run_id,
                    k=5,
                )
                if result:
                    if isinstance(result, str):
                        return result
                    if hasattr(result, "formatted"):
                        return result.formatted
                    return str(result)
        except Exception as exc:
            logger.debug("smart_retrieve failed: %s", exc)
        return ""

    async def _fit_history(
        self,
        ctx: AgentContext,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Trim history to fit within the context window budget."""
        if len(history) <= _MAX_HISTORY_MESSAGES:
            return history
        # Keep system-level messages and recent messages
        # Always keep the last N messages
        return history[-_MAX_HISTORY_MESSAGES:]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    async def _maybe_resume_checkpoint(self, ctx: AgentContext) -> None:
        """Attempt to load a checkpoint and resume from a previous run."""
        if self._checkpoint_manager is None:
            return
        try:
            checkpoint = await _safe_call(
                self._checkpoint_manager.load, ctx.run_id
            )
            if checkpoint is not None:
                ctx.step_count = checkpoint.get("step_count", 0)
                ctx.token_count = checkpoint.get("token_count", 0)
                logger.info(
                    "Resumed run %s from checkpoint at step %d",
                    ctx.run_id,
                    ctx.step_count,
                )
        except Exception as exc:
            logger.debug("Checkpoint load failed (will start fresh): %s", exc)

    async def _save_checkpoint(
        self, ctx: AgentContext, history: list[dict[str, Any]]
    ) -> None:
        """Persist a checkpoint for potential run resumption."""
        if self._checkpoint_manager is None:
            return
        try:
            await _safe_call(
                self._checkpoint_manager.save,
                ctx.run_id,
                {
                    "step_count": ctx.step_count,
                    "token_count": ctx.token_count,
                    "history_len": len(history),
                },
            )
        except Exception as exc:
            logger.debug("Checkpoint save failed: %s", exc)

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------

    async def _emit_event(self, event: StepEvent) -> None:
        """Publish a StepEvent to the event bus and message bus."""
        if self._event_bus is not None:
            try:
                await _safe_call(self._event_bus.publish, event)
            except Exception as exc:
                logger.debug("event_bus.publish failed: %s", exc)
        if self._message_bus is not None:
            try:
                await _safe_call(self._message_bus.publish, event)
            except Exception as exc:
                logger.debug("message_bus.publish failed: %s", exc)

    async def _record_failure(self, ctx: AgentContext, exc: Exception) -> None:
        """Record a failure in the failure tracker and potentially DLQ."""
        if self._failure_tracker is None:
            return
        try:
            failure_class = self._classify_exception(exc)
            await _safe_call(
                self._failure_tracker.record,
                run_id=ctx.run_id,
                step_number=ctx.step_count,
                failure_class=failure_class,
                message=str(exc),
                agent_type=self.agent_type,
            )
        except Exception as track_exc:
            logger.debug("failure_tracker.record failed: %s", track_exc)

    async def _audit(
        self, ctx: AgentContext, call: ToolCall, result: ToolResult
    ) -> None:
        """Fire-and-forget audit log entry for a tool call."""
        if self._audit_logger is None:
            return
        try:
            await _safe_call(
                self._audit_logger.log,
                event_type="tool_call",
                run_id=ctx.run_id,
                tenant_id=ctx.tenant_id,
                payload={
                    "tool_name": call.name,
                    "tool_id": call.id,
                    "is_error": result.is_error,
                },
            )
        except Exception as exc:
            logger.debug("audit_logger.log failed: %s", exc)

    async def _log_llm_span(self, ctx: AgentContext, response: LLMResponse) -> None:
        """Record an MLflow LLM span for this response."""
        if self._mlflow_tracer is None:
            return
        try:
            await _safe_call(
                self._mlflow_tracer.log_llm_call,
                run_id=ctx.run_id,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except Exception as exc:
            logger.debug("mlflow_tracer.log_llm_call failed: %s", exc)

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _mlflow_context(self, ctx: AgentContext):  # type: ignore[return]
        """Start an MLflow agent run if tracer is available."""
        mlflow_run_id = None
        if self._mlflow_tracer is not None:
            try:
                async with self._mlflow_tracer.agent_run(ctx) as run:
                    yield getattr(run, "info", {}).get("run_id") if run else None
                    return
            except Exception as exc:
                logger.debug("MLflow agent_run context failed: %s", exc)
        yield mlflow_run_id

    @asynccontextmanager
    async def _llm_span(self, ctx: AgentContext):  # type: ignore[return]
        """Open an OTel span for the LLM call."""
        if self._step_tracer is not None:
            try:
                async with self._step_tracer.span(
                    name="llm_call",
                    run_id=ctx.run_id,
                    trace_id=ctx.trace_id,
                ) as span:
                    yield span
                    return
            except Exception as exc:
                logger.debug("StepTracer.span failed: %s", exc)
        yield None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs); await if coroutine. Never raises."""
    import asyncio
    try:
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception as exc:
        logger.debug("_safe_call(%s) raised: %s", getattr(fn, "__name__", fn), exc)
        return None


def _get_metrics() -> Any:
    """Return the HarnessMetrics instance without failing if unavailable."""
    try:
        from harness.observability.metrics import get_prometheus_metrics
        return get_prometheus_metrics()
    except Exception:
        return None


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
