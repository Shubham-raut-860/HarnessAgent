"""Unit tests for BaseAgent lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.agents.base import BaseAgent
from harness.core.context import AgentResult, LLMResponse, ToolCall, ToolResult
from harness.core.errors import FailureClass


def _make_llm_response(content="Done", tool_calls=None, tokens=30):
    return LLMResponse(
        content=content, tool_calls=tool_calls or [],
        input_tokens=tokens // 2, output_tokens=tokens // 2,
        model="mock-model", provider="mock",
    )


def _mock_memory():
    m = AsyncMock()
    m.push_message = AsyncMock()
    m.get_history = AsyncMock(return_value=[])
    m.fit_history = AsyncMock(return_value=MagicMock(messages=[], truncated=False, summary=None))
    m.smart_retrieve = AsyncMock(return_value=MagicMock(
        graph_context="", vector_context=[], total_tokens_estimate=0
    ))
    return m


def _make_agent(router=None, tool_registry=None, memory=None,
                failure_tracker=None, cost_tracker=None, checkpoint_manager=None):
    router = router or AsyncMock()
    if not hasattr(router.complete, 'return_value') or router.complete.return_value is None:
        router.complete = AsyncMock(return_value=_make_llm_response())

    tool_registry = tool_registry or AsyncMock()
    memory = memory or _mock_memory()

    failure_tracker = failure_tracker or AsyncMock()
    failure_tracker.record = AsyncMock()

    cost_tracker = cost_tracker or AsyncMock()
    cost_tracker.record = AsyncMock(return_value=MagicMock(cost_usd=0.001))

    checkpoint_manager = checkpoint_manager or AsyncMock()
    checkpoint_manager.exists = AsyncMock(return_value=False)
    checkpoint_manager.load = AsyncMock(return_value=None)   # None = no checkpoint to restore
    checkpoint_manager.save = AsyncMock()

    return BaseAgent(
        llm_router=router,
        memory_manager=memory,
        tool_registry=tool_registry,
        safety_pipeline=None,
        step_tracer=None,
        mlflow_tracer=None,
        failure_tracker=failure_tracker,
        audit_logger=None,
        event_bus=None,
        cost_tracker=cost_tracker,
        checkpoint_manager=checkpoint_manager,
        message_bus=None,
    )


def _make_ctx(agent_context, memory=None):
    mem = memory or _mock_memory()
    return agent_context(memory=mem)


@pytest.mark.asyncio
async def test_run_completes_with_no_tool_calls(agent_context):
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("Final answer"))
    agent = _make_agent(router=router)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert isinstance(result, AgentResult)
    assert result.success is True


@pytest.mark.asyncio
async def test_run_executes_tool_calls(agent_context):
    call = ToolCall(id="c1", name="echo", args={"message": "hi"})
    responses = [
        _make_llm_response("calling tool", tool_calls=[call]),
        _make_llm_response("Done"),
    ]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={"echo": "hi"}))
    agent = _make_agent(router=router, tool_registry=mock_registry)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is True
    mock_registry.execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_respects_step_budget(agent_context):
    call = ToolCall(id="c1", name="noop", args={})
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("looping", tool_calls=[call]))
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))
    agent = _make_agent(router=router, tool_registry=mock_registry)
    ctx = _make_ctx(agent_context)
    ctx.max_steps = 3
    result = await agent.run(ctx)
    assert result.success is False


@pytest.mark.asyncio
async def test_run_respects_token_budget(agent_context):
    call = ToolCall(id="c1", name="noop", args={})
    router = AsyncMock()
    router.complete = AsyncMock(return_value=LLMResponse(
        content="looping", tool_calls=[call],
        input_tokens=2500, output_tokens=2500, model="mock", provider="mock",
    ))
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))
    agent = _make_agent(router=router, tool_registry=mock_registry)
    ctx = _make_ctx(agent_context)
    ctx.max_tokens = 6000
    result = await agent.run(ctx)
    assert result.success is False


@pytest.mark.asyncio
async def test_safety_violation_stops_run(agent_context):
    from guardrail.result import Decision, GuardResult
    mock_pipeline = MagicMock()
    blocked = GuardResult(decision=Decision.BLOCK, reason="PII", source="pii")
    allowed = GuardResult.allow(source="test")
    mock_pipeline.check_input = MagicMock(return_value=(allowed, {}))
    mock_pipeline.check_step = MagicMock(return_value=(allowed, {}))
    mock_pipeline.check_output = MagicMock(return_value=(blocked, {}))
    agent = _make_agent()
    agent._safety_pipeline = mock_pipeline
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is False


@pytest.mark.asyncio
async def test_checkpoint_saved_every_10_steps(agent_context):
    call = ToolCall(id="c1", name="noop", args={})
    responses = [_make_llm_response("step", tool_calls=[call])] * 10 + [_make_llm_response("done")]
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=responses)
    mock_registry = AsyncMock()
    mock_registry.execute = AsyncMock(return_value=ToolResult(data={}))
    mock_checkpoint = AsyncMock()
    mock_checkpoint.exists = AsyncMock(return_value=False)
    mock_checkpoint.save = AsyncMock()
    agent = _make_agent(router=router, tool_registry=mock_registry, checkpoint_manager=mock_checkpoint)
    ctx = _make_ctx(agent_context)
    ctx.max_steps = 15
    await agent.run(ctx)
    assert mock_checkpoint.save.call_count >= 1


@pytest.mark.asyncio
async def test_resume_from_checkpoint(agent_context):
    mock_checkpoint = AsyncMock()
    mock_checkpoint.exists = AsyncMock(return_value=True)
    mock_checkpoint.load = AsyncMock(return_value=MagicMock(
        step_count=5, token_count=500, history_snapshot=[]
    ))
    mock_checkpoint.save = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("resumed"))
    agent = _make_agent(router=router, checkpoint_manager=mock_checkpoint)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)
    mock_checkpoint.load.assert_called_once()


@pytest.mark.asyncio
async def test_failure_recorded_on_exception(agent_context):
    mock_failure_tracker = AsyncMock()
    mock_failure_tracker.record = AsyncMock()
    router = AsyncMock()
    router.complete = AsyncMock(side_effect=RuntimeError("Unexpected crash"))
    agent = _make_agent(router=router, failure_tracker=mock_failure_tracker)
    ctx = _make_ctx(agent_context)
    result = await agent.run(ctx)
    assert result.success is False
    mock_failure_tracker.record.assert_called_once()


@pytest.mark.asyncio
async def test_cost_tracked_per_llm_call(agent_context):
    mock_cost = AsyncMock()
    mock_cost.record = AsyncMock(return_value=MagicMock(cost_usd=0.005))
    router = AsyncMock()
    router.complete = AsyncMock(return_value=_make_llm_response("done"))
    agent = _make_agent(router=router, cost_tracker=mock_cost)
    ctx = _make_ctx(agent_context)
    await agent.run(ctx)
    mock_cost.record.assert_called()
