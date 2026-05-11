"""Unit tests for TraceRecorder — span lifecycle, nesting, Redis storage, JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.core.context import AgentContext
from harness.observability.trace_recorder import TraceRecorder, _new_span_id
from harness.observability.trace_schema import SpanKind, SpanStatus, TraceSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(run_id: str = "run-001", tmp_path: Path = None) -> AgentContext:
    return AgentContext(
        run_id=run_id,
        tenant_id="tenant-test",
        agent_type="sql",
        task="test task",
        memory=None,
        workspace_path=tmp_path or Path("/tmp"),
    )


def _make_recorder(redis_client, tmp_path: Path) -> TraceRecorder:
    rec = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    rec._client = redis_client
    return rec


# ---------------------------------------------------------------------------
# span_id generation
# ---------------------------------------------------------------------------

def test_new_span_id_is_16_hex_chars():
    sid = _new_span_id()
    assert len(sid) == 16
    assert all(c in "0123456789abcdef" for c in sid)


def test_new_span_id_is_unique():
    ids = {_new_span_id() for _ in range(200)}
    assert len(ids) == 200


# ---------------------------------------------------------------------------
# start_span
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_span_returns_span_id(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    assert len(sid) == 16


@pytest.mark.asyncio
async def test_start_span_first_has_no_parent(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    span = await rec.get_span(sid)
    assert span.parent_span_id is None


@pytest.mark.asyncio
async def test_start_span_child_gets_parent_from_stack(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    root_id = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    child_id = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    child = await rec.get_span(child_id)
    assert child.parent_span_id == root_id


@pytest.mark.asyncio
async def test_start_span_sets_status_running(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.TOOL, "tool:sql", ctx)
    span = await rec.get_span(sid)
    assert span.status == SpanStatus.RUNNING


@pytest.mark.asyncio
async def test_start_span_injects_ctx_fields(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-ctx-test", tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx,
                                input_preview="hello world")
    span = await rec.get_span(sid)
    assert span.run_id == "run-ctx-test"
    assert span.agent_type == "sql"
    assert span.tenant_id == "tenant-test"
    assert span.trace_id == ctx.trace_id
    assert span.input_preview == "hello world"


@pytest.mark.asyncio
async def test_start_span_without_ctx(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    sid = await rec.start_span("run-no-ctx", SpanKind.TOOL, "tool:x")
    span = await rec.get_span(sid)
    assert span.agent_type == ""
    assert span.tenant_id == ""
    assert span.run_id == "run-no-ctx"


# ---------------------------------------------------------------------------
# end_span
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_span_sets_ok_status(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    span = await rec.end_span(ctx.run_id, sid, status=SpanStatus.OK)
    assert span.status == SpanStatus.OK


@pytest.mark.asyncio
async def test_end_span_sets_duration_ms(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    span = await rec.end_span(ctx.run_id, sid)
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0


@pytest.mark.asyncio
async def test_end_span_sets_end_time(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.TOOL, "tool:x", ctx)
    span = await rec.end_span(ctx.run_id, sid)
    assert span.end_time is not None


@pytest.mark.asyncio
async def test_end_span_error_records_error_message(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.TOOL, "tool:x", ctx)
    span = await rec.end_span(ctx.run_id, sid, status=SpanStatus.ERROR, error="timeout")
    assert span.status == SpanStatus.ERROR
    assert span.error == "timeout"


@pytest.mark.asyncio
async def test_end_span_sets_output_preview(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    span = await rec.end_span(ctx.run_id, sid, output_preview="The answer is 42")
    assert span.output_preview == "The answer is 42"


@pytest.mark.asyncio
async def test_end_span_returns_none_for_unknown_span(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    result = await rec.end_span("run-x", "nonexistent-span-id")
    assert result is None


@pytest.mark.asyncio
async def test_end_span_pops_from_stack(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    root = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    child = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    assert rec._stacks[ctx.run_id] == [root, child]
    await rec.end_span(ctx.run_id, child)
    assert rec._stacks[ctx.run_id] == [root]


# ---------------------------------------------------------------------------
# set_llm_usage / pending usage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_llm_usage_applied_on_end(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    rec.set_llm_usage(sid, input_tokens=150, output_tokens=75, cost_usd=0.002, cached=True)
    span = await rec.end_span(ctx.run_id, sid)
    assert span.input_tokens == 150
    assert span.output_tokens == 75
    assert span.cost_usd == pytest.approx(0.002)
    assert span.cached is True


@pytest.mark.asyncio
async def test_set_llm_usage_cleared_after_end(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    rec.set_llm_usage(sid, input_tokens=10, output_tokens=5)
    await rec.end_span(ctx.run_id, sid)
    assert sid not in rec._pending_usage


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_span_context_manager_ok_path(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    async with rec.span(ctx.run_id, SpanKind.LLM, "llm:call", ctx) as sid:
        assert len(sid) == 16
    span = await rec.get_span(sid)
    assert span.status == SpanStatus.OK
    assert span.end_time is not None


@pytest.mark.asyncio
async def test_span_context_manager_exception_sets_error(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    captured_sid = None
    with pytest.raises(ValueError, match="test error"):
        async with rec.span(ctx.run_id, SpanKind.TOOL, "tool:x", ctx) as sid:
            captured_sid = sid
            raise ValueError("test error")
    span = await rec.get_span(captured_sid)
    assert span.status == SpanStatus.ERROR
    assert "test error" in span.error


@pytest.mark.asyncio
async def test_span_context_manager_yields_span_id(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    async with rec.span(ctx.run_id, SpanKind.GUARDRAIL, "guardrail:out", ctx) as sid:
        assert isinstance(sid, str)
        assert len(sid) == 16


@pytest.mark.asyncio
async def test_span_context_manager_pops_stack_on_exit(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    async with rec.span(ctx.run_id, SpanKind.RUN, "run:sql", ctx):
        pass
    assert rec._stacks.get(ctx.run_id, []) == []


# ---------------------------------------------------------------------------
# get_trace
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_trace_returns_none_for_unknown_run(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    result = await rec.get_trace("no-such-run")
    assert result is None


@pytest.mark.asyncio
async def test_get_trace_contains_all_spans(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-all-spans", tmp_path=tmp_path)
    root = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    llm = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    await rec.end_span(ctx.run_id, llm)
    await rec.end_span(ctx.run_id, root)
    trace = await rec.get_trace(ctx.run_id)
    assert trace.span_count == 2
    kinds = {s.kind for s in trace.spans}
    assert SpanKind.RUN in kinds
    assert SpanKind.LLM in kinds


@pytest.mark.asyncio
async def test_get_trace_spans_sorted_by_start_time(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-sorted", tmp_path=tmp_path)
    root = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    child1 = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:1", ctx)
    await rec.end_span(ctx.run_id, child1)
    child2 = await rec.start_span(ctx.run_id, SpanKind.TOOL, "tool:sql", ctx)
    await rec.end_span(ctx.run_id, child2)
    await rec.end_span(ctx.run_id, root)
    trace = await rec.get_trace(ctx.run_id)
    starts = [s.start_time for s in trace.spans]
    assert starts == sorted(starts)


@pytest.mark.asyncio
async def test_get_trace_aggregates_token_counts(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-tokens", tmp_path=tmp_path)
    root = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    llm1 = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:1", ctx)
    rec.set_llm_usage(llm1, input_tokens=100, output_tokens=50, cost_usd=0.001)
    await rec.end_span(ctx.run_id, llm1)
    llm2 = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:2", ctx)
    rec.set_llm_usage(llm2, input_tokens=80, output_tokens=40, cost_usd=0.0008)
    await rec.end_span(ctx.run_id, llm2)
    await rec.end_span(ctx.run_id, root)
    trace = await rec.get_trace(ctx.run_id)
    assert trace.total_input_tokens == 180
    assert trace.total_output_tokens == 90
    assert trace.total_cost_usd == pytest.approx(0.0018)


@pytest.mark.asyncio
async def test_get_trace_root_has_no_parent(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-root", tmp_path=tmp_path)
    root = await rec.start_span(ctx.run_id, SpanKind.RUN, "run:sql", ctx)
    child = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    await rec.end_span(ctx.run_id, child)
    await rec.end_span(ctx.run_id, root)
    trace = await rec.get_trace(ctx.run_id)
    root_span = next(s for s in trace.spans if s.parent_span_id is None)
    assert root_span.kind == SpanKind.RUN


# ---------------------------------------------------------------------------
# get_span
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_span_returns_none_for_unknown(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    result = await rec.get_span("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_get_span_returns_correct_span(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.GUARDRAIL, "guardrail:output", ctx,
                                input_preview="response content")
    await rec.end_span(ctx.run_id, sid, output_preview="passed")
    span = await rec.get_span(sid)
    assert span.kind == SpanKind.GUARDRAIL
    assert span.name == "guardrail:output"
    assert span.input_preview == "response content"
    assert span.output_preview == "passed"


# ---------------------------------------------------------------------------
# Multiple runs isolated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_runs_have_separate_stacks(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx_a = _make_ctx(run_id="run-a", tmp_path=tmp_path)
    ctx_b = _make_ctx(run_id="run-b", tmp_path=tmp_path)
    sid_a = await rec.start_span(ctx_a.run_id, SpanKind.RUN, "run:sql", ctx_a)
    sid_b = await rec.start_span(ctx_b.run_id, SpanKind.RUN, "run:code", ctx_b)
    span_b = await rec.get_span(sid_b)
    assert span_b.parent_span_id is None   # b's root has no parent from a's stack


@pytest.mark.asyncio
async def test_multiple_runs_traces_are_independent(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx_a = _make_ctx(run_id="iso-run-a", tmp_path=tmp_path)
    ctx_b = _make_ctx(run_id="iso-run-b", tmp_path=tmp_path)
    sid_a = await rec.start_span(ctx_a.run_id, SpanKind.RUN, "run:sql", ctx_a)
    sid_b = await rec.start_span(ctx_b.run_id, SpanKind.RUN, "run:code", ctx_b)
    await rec.end_span(ctx_a.run_id, sid_a)
    await rec.end_span(ctx_b.run_id, sid_b)
    trace_a = await rec.get_trace(ctx_a.run_id)
    trace_b = await rec.get_trace(ctx_b.run_id)
    assert trace_a.span_count == 1
    assert trace_b.span_count == 1
    assert trace_a.spans[0].span_id != trace_b.spans[0].span_id


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_span_appends_to_jsonl(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-jsonl", tmp_path=tmp_path)
    sid = await rec.start_span(ctx.run_id, SpanKind.LLM, "llm:call", ctx)
    await rec.end_span(ctx.run_id, sid, output_preview="42")
    log_path = tmp_path / "runs" / ctx.run_id / "trace.jsonl"
    assert log_path.exists()
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["span_id"] == sid
    assert lines[0]["kind"] == "llm"


@pytest.mark.asyncio
async def test_multiple_spans_each_appended_to_jsonl(redis_client, tmp_path):
    rec = _make_recorder(redis_client, tmp_path)
    ctx = _make_ctx(run_id="run-multi-jsonl", tmp_path=tmp_path)
    for i in range(3):
        sid = await rec.start_span(ctx.run_id, SpanKind.TOOL, f"tool:{i}", ctx)
        await rec.end_span(ctx.run_id, sid)
    log_path = tmp_path / "runs" / ctx.run_id / "trace.jsonl"
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# Existing round-trip test (from original file, kept for regression)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_recorder_round_trips_nested_spans(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    ctx = AgentContext(
        run_id="run-trace",
        tenant_id="tenant-a",
        agent_type="code",
        task="trace this",
        memory=None,
        workspace_path=tmp_path,
    )

    root = await recorder.start_span(ctx.run_id, SpanKind.RUN, "run:code", ctx)
    child = await recorder.start_span(
        ctx.run_id, SpanKind.LLM, "llm:call", ctx, input_preview="hello",
    )
    recorder.set_llm_usage(child, input_tokens=12, output_tokens=8, cost_usd=0.002, cached=True)
    await recorder.end_span(ctx.run_id, child, output_preview="world")
    await recorder.end_span(ctx.run_id, root, status=SpanStatus.OK)

    trace = await recorder.get_trace(ctx.run_id)

    assert trace is not None
    assert trace.trace_id == ctx.trace_id
    assert trace.span_count == 2
    assert trace.total_input_tokens == 12
    assert trace.total_output_tokens == 8
    assert trace.total_cost_usd == pytest.approx(0.002)

    llm_span = next(s for s in trace.spans if s.kind == SpanKind.LLM)
    assert llm_span.parent_span_id == root
    assert llm_span.cached is True
    assert llm_span.input_preview == "hello"
    assert llm_span.output_preview == "world"

    fetched = await recorder.get_span(child)
    assert fetched is not None
    assert fetched.kind == SpanKind.LLM
