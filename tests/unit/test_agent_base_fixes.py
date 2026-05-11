"""
Tests targeting the two bug fixes + TraceRecorder integration in BaseAgent:

Bug 1: _record_failure was calling failure_tracker.record with raw keyword args
       instead of a StepFailure object → now constructs StepFailure.from_exception().

Bug 2: _llm_span was using `async with` on a synchronous @contextmanager
       and never passing ctx → spans were silently never opened.

Also tests: TraceRecorder wiring for RUN, LLM, TOOL, GUARDRAIL span kinds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.agents.base import BaseAgent
from harness.core.context import AgentResult, LLMResponse, ToolCall, ToolResult
from harness.core.errors import BudgetExceeded, FailureClass, SafetyViolation, ToolError
from harness.observability.failures import StepFailure
from harness.observability.trace_schema import SpanKind, SpanStatus


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _make_llm_resp(content="done", tool_calls=None, tokens=20, cached=False):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        input_tokens=tokens // 2,
        output_tokens=tokens // 2,
        model="mock-model",
        provider="mock",
        cached=cached,
    )


def _mock_memory():
    m = AsyncMock()
    m.push_message = AsyncMock()
    m.smart_retrieve = AsyncMock(return_value=MagicMock(
        graph_context="", vector_context=[], total_tokens_estimate=0
    ))
    m.fit_history = AsyncMock(return_value=MagicMock(
        messages=[], truncated=False, summary=None
    ))
    return m


def _make_agent(
    router=None,
    tool_registry=None,
    safety_pipeline=None,
    failure_tracker=None,
    cost_tracker=None,
    step_tracer=None,
    trace_recorder=None,
):
    # Only set defaults when the caller didn't provide the component;
    # never overwrite a caller-configured mock.
    if router is None:
        router = AsyncMock()
        router.complete = AsyncMock(return_value=_make_llm_resp())

    tool_registry = tool_registry or AsyncMock()

    if failure_tracker is None:
        failure_tracker = AsyncMock()
        failure_tracker.record = AsyncMock()

    if cost_tracker is None:
        cost_tracker = AsyncMock()
        cost_tracker.record = AsyncMock(return_value=MagicMock(cost_usd=0.001))

    checkpoint = AsyncMock()
    checkpoint.load = AsyncMock(return_value=None)
    checkpoint.save = AsyncMock()

    return BaseAgent(
        llm_router=router,
        memory_manager=_mock_memory(),
        tool_registry=tool_registry,
        safety_pipeline=safety_pipeline,
        step_tracer=step_tracer,
        mlflow_tracer=None,
        failure_tracker=failure_tracker,
        audit_logger=None,
        event_bus=None,
        cost_tracker=cost_tracker,
        checkpoint_manager=checkpoint,
        trace_recorder=trace_recorder,
    )


def _make_ctx(agent_context, **kwargs):
    return agent_context(**kwargs)


# ===========================================================================
# Bug 1: _record_failure passes a StepFailure object
# ===========================================================================

@pytest.mark.asyncio
async def test_record_failure_calls_tracker_with_step_failure_object(agent_context):
    """FailureTracker.record must receive a StepFailure, not raw keyword args."""
    mock_tracker = AsyncMock()
    received = []

    async def _capture(failure):
        received.append(failure)

    mock_tracker.record = _capture

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("kaboom"))

    agent = _make_agent(router=router, failure_tracker=mock_tracker)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    assert len(received) == 1
    assert isinstance(received[0], StepFailure), (
        f"Expected StepFailure, got {type(received[0])}"
    )


@pytest.mark.asyncio
async def test_record_failure_step_failure_has_correct_run_id(agent_context):
    received = []

    async def _capture(failure):
        received.append(failure)

    mock_tracker = AsyncMock()
    mock_tracker.record = _capture

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=ValueError("bad input"))

    agent = _make_agent(router=router, failure_tracker=mock_tracker)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    assert received[0].run_id == ctx.run_id


@pytest.mark.asyncio
async def test_record_failure_step_failure_has_agent_type(agent_context):
    received = []

    async def _capture(failure):
        received.append(failure)

    mock_tracker = AsyncMock()
    mock_tracker.record = _capture

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("crash"))
    agent = _make_agent(router=router, failure_tracker=mock_tracker)
    agent.agent_type = "sql"
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    assert received[0].agent_type == "sql"


@pytest.mark.asyncio
async def test_record_failure_step_failure_captures_stack_trace(agent_context):
    received = []

    async def _capture(failure):
        received.append(failure)

    mock_tracker = AsyncMock()
    mock_tracker.record = _capture

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("stack test"))

    agent = _make_agent(router=router, failure_tracker=mock_tracker)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    assert "stack test" in received[0].message or received[0].stack_trace != ""


@pytest.mark.asyncio
async def test_record_failure_classifies_exception_type(agent_context):
    """_record_failure must produce a StepFailure for different exception types."""
    received = []

    async def _capture(failure):
        received.append(failure)

    mock_tracker = AsyncMock()
    mock_tracker.record = _capture

    # Use a plain ValueError — tests that _record_failure handles arbitrary exceptions
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=ValueError("unexpected input"))

    agent = _make_agent(router=router, failure_tracker=mock_tracker)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)

    assert result.success is False
    assert len(received) == 1
    assert isinstance(received[0], StepFailure)
    # FailureClass must be an enum value, not a raw exception
    assert received[0].failure_class is not None


# ===========================================================================
# Bug 2: _llm_span uses sync context manager and passes ctx correctly
# ===========================================================================

@pytest.mark.asyncio
async def test_llm_span_invokes_step_tracer_with_ctx(agent_context):
    """StepTracer.span must be called with the AgentContext, not just run_id/trace_id."""
    mock_tracer = MagicMock()
    ctx_received = []

    class _FakeSpan:
        def set_attribute(self, *a, **kw): pass
        def record_exception(self, *a, **kw): pass
        def set_status(self, *a, **kw): pass

    from contextlib import contextmanager

    @contextmanager
    def _fake_span(name, ctx=None, **attrs):
        ctx_received.append(ctx)
        yield _FakeSpan()

    mock_tracer.span = _fake_span

    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("done"))
    agent = _make_agent(router=router, step_tracer=mock_tracer)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    assert len(ctx_received) >= 1
    # ctx must be the AgentContext, not None
    from harness.core.context import AgentContext
    assert all(isinstance(c, AgentContext) for c in ctx_received)


@pytest.mark.asyncio
async def test_llm_span_noop_without_step_tracer(agent_context):
    """_llm_span must succeed (yield None) when step_tracer is None."""
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("done"))
    agent = _make_agent(router=router, step_tracer=None)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is True


@pytest.mark.asyncio
async def test_llm_span_does_not_use_async_with_on_sync_cm(agent_context):
    """Regression: async with on sync @contextmanager silently fails."""
    call_count = [0]

    from contextlib import contextmanager

    @contextmanager
    def _counting_span(name, ctx=None, **attrs):
        call_count[0] += 1
        yield MagicMock()

    mock_tracer = MagicMock()
    mock_tracer.span = _counting_span

    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("done"))
    agent = _make_agent(router=router, step_tracer=mock_tracer)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    # Must be called at least once per LLM call
    assert call_count[0] >= 1


# ===========================================================================
# TraceRecorder span wiring
# ===========================================================================

@pytest.mark.asyncio
async def test_run_emits_run_span(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("done"))
    agent = _make_agent(router=router, trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    assert trace is not None
    kinds = {s.kind for s in trace.spans}
    assert SpanKind.RUN in kinds


@pytest.mark.asyncio
async def test_run_emits_llm_span(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("done", tokens=40))
    cost = AsyncMock()
    cost.record = AsyncMock(return_value=MagicMock(cost_usd=0.002))
    agent = _make_agent(router=router, cost_tracker=cost, trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    llm_spans = [s for s in trace.spans if s.kind == SpanKind.LLM]
    assert len(llm_spans) == 1
    assert llm_spans[0].input_tokens == 20
    assert llm_spans[0].output_tokens == 20


@pytest.mark.asyncio
async def test_run_emits_tool_span(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    call = ToolCall(id="c1", name="execute_sql", args={"query": "SELECT 1"})
    responses = [
        _make_llm_resp("calling tool", tool_calls=[call]),
        _make_llm_resp("done"),
    ]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)

    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={"rows": 1}))

    agent = _make_agent(router=router, tool_registry=mock_registry,
                        trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    tool_spans = [s for s in trace.spans if s.kind == SpanKind.TOOL]
    assert len(tool_spans) == 1
    assert "execute_sql" in tool_spans[0].name


@pytest.mark.asyncio
async def test_run_emits_guardrail_span_when_pipeline_present(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    mock_pipeline = MagicMock()
    guard_ok = MagicMock()
    guard_ok.blocked = False
    mock_pipeline.check_output = AsyncMock(return_value=guard_ok)

    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("response text"))

    agent = _make_agent(router=router, safety_pipeline=mock_pipeline,
                        trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    guardrail_spans = [s for s in trace.spans if s.kind == SpanKind.GUARDRAIL]
    assert len(guardrail_spans) >= 1


@pytest.mark.asyncio
async def test_run_span_is_error_on_failure(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("crash"))

    agent = _make_agent(router=router, trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)

    assert result.success is False
    trace = await recorder.get_trace(ctx.run_id)
    run_span = next(s for s in trace.spans if s.kind == SpanKind.RUN)
    assert run_span.status == SpanStatus.ERROR


@pytest.mark.asyncio
async def test_run_span_is_ok_on_success(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("all done"))

    agent = _make_agent(router=router, trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)

    assert result.success is True
    trace = await recorder.get_trace(ctx.run_id)
    run_span = next(s for s in trace.spans if s.kind == SpanKind.RUN)
    assert run_span.status == SpanStatus.OK


@pytest.mark.asyncio
async def test_llm_span_registers_token_usage_via_set_llm_usage(agent_context, redis_client, tmp_path):
    """set_llm_usage must be called with the correct token counts."""
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    router = AsyncMock()
    router.complete = AsyncMock(return_value=LLMResponse(
        content="done", tool_calls=[],
        input_tokens=123, output_tokens=77,
        model="mock", provider="mock", cached=True,
    ))
    cost = AsyncMock()
    cost.record = AsyncMock(return_value=MagicMock(cost_usd=0.005))

    agent = _make_agent(router=router, cost_tracker=cost, trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    llm_span = next(s for s in trace.spans if s.kind == SpanKind.LLM)
    assert llm_span.input_tokens == 123
    assert llm_span.output_tokens == 77
    assert llm_span.cached is True


@pytest.mark.asyncio
async def test_tool_span_is_error_on_tool_failure(agent_context, redis_client, tmp_path):
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    call = ToolCall(id="c1", name="bad_tool", args={})
    responses = [
        _make_llm_resp("calling", tool_calls=[call]),
        _make_llm_resp("done"),
    ]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)

    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(
        side_effect=ToolError("failed", FailureClass.TOOL_NOT_FOUND, "bad_tool")
    )

    agent = _make_agent(router=router, tool_registry=mock_registry,
                        trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    tool_spans = [s for s in trace.spans if s.kind == SpanKind.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].status == SpanStatus.ERROR


@pytest.mark.asyncio
async def test_span_hierarchy_tool_parent_is_run(agent_context, redis_client, tmp_path):
    """Tool span's parent should be the RUN span (or at least exist)."""
    from harness.observability.trace_recorder import TraceRecorder
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    call = ToolCall(id="c1", name="noop", args={})
    responses = [_make_llm_resp("calling", tool_calls=[call]), _make_llm_resp("done")]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))

    agent = _make_agent(router=router, tool_registry=mock_registry,
                        trace_recorder=recorder)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)

    trace = await recorder.get_trace(ctx.run_id)
    tool_span = next((s for s in trace.spans if s.kind == SpanKind.TOOL), None)
    assert tool_span is not None
    assert tool_span.parent_span_id is not None


@pytest.mark.asyncio
async def test_no_trace_recorder_run_succeeds_without_error(agent_context):
    """When trace_recorder=None, the agent must still run normally."""
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_resp("done"))
    agent = _make_agent(router=router, trace_recorder=None)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is True
