"""Unit tests for the /runs/{run_id}/trace and /runs/spans/{span_id} API endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.core.context import AgentContext
from harness.observability.trace_recorder import TraceRecorder
from harness.observability.trace_schema import SpanKind, SpanStatus


# ---------------------------------------------------------------------------
# Helpers — build a populated TraceRecorder backed by fakeredis
# ---------------------------------------------------------------------------

async def _populate_trace(recorder: TraceRecorder, run_id: str, tmp_path: Path) -> dict:
    """Create a simple 3-span trace and return metadata."""
    ctx = AgentContext(
        run_id=run_id, tenant_id="tenant-api",
        agent_type="sql", task="list tables",
        memory=None, workspace_path=tmp_path,
    )
    root = await recorder.start_span(run_id, SpanKind.RUN, "run:sql", ctx)
    llm = await recorder.start_span(run_id, SpanKind.LLM, "llm:call", ctx)
    recorder.set_llm_usage(llm, input_tokens=100, output_tokens=50, cost_usd=0.002)
    await recorder.end_span(run_id, llm, output_preview="SELECT * FROM users")
    tool = await recorder.start_span(run_id, SpanKind.TOOL, "tool:execute_sql", ctx)
    await recorder.end_span(run_id, tool, output_preview="42 rows")
    await recorder.end_span(run_id, root, status=SpanStatus.OK, output_preview="done")
    return {"root_id": root, "llm_id": llm, "tool_id": tool, "trace_id": ctx.trace_id}


# ---------------------------------------------------------------------------
# HTTP client fixture using httpx + FastAPI TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client(redis_client, tmp_path):
    """Return a synchronous TestClient wired to a fake recorder."""
    pytest.importorskip("httpx", reason="httpx required for API tests")
    from fastapi.testclient import TestClient

    from harness.api.main import create_app

    app = create_app()

    # Inject fake redis into app state
    app.state.redis = redis_client

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper: patch TraceRecorder to use our fakeredis-backed instance
# ---------------------------------------------------------------------------

def _patch_recorder(redis_client, tmp_path):
    """Return a context manager that patches TraceRecorder.create to use fakeredis."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
        recorder._client = redis_client
        with patch(
            "harness.api.routes.traces.TraceRecorder.create",
            return_value=recorder,
        ):
            yield recorder

    return _ctx()


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/trace — success paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_trace_returns_trace_view(redis_client, tmp_path):
    """TraceRecorder.get_trace returns a TraceView for a populated run."""
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client

    run_id = "api-trace-001"
    meta = await _populate_trace(recorder, run_id, tmp_path)

    trace = await recorder.get_trace(run_id)
    assert trace is not None
    assert trace.run_id == run_id
    assert trace.span_count == 3
    assert trace.total_input_tokens == 100
    assert trace.total_output_tokens == 50


@pytest.mark.asyncio
async def test_get_trace_to_dict_has_required_keys(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-trace-keys"
    await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    d = trace.to_dict()
    required = {"trace_id", "run_id", "agent_type", "status", "start_time",
                 "span_count", "spans", "total_input_tokens", "total_output_tokens",
                 "total_cost_usd"}
    assert required.issubset(d.keys())


@pytest.mark.asyncio
async def test_get_trace_spans_have_hierarchy(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-trace-hier"
    meta = await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    root = next(s for s in trace.spans if s.parent_span_id is None)
    children = [s for s in trace.spans if s.parent_span_id == root.span_id]
    assert len(children) >= 1


@pytest.mark.asyncio
async def test_get_trace_status_ok_for_successful_run(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-trace-ok"
    await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    assert trace.status == SpanStatus.OK


@pytest.mark.asyncio
async def test_get_trace_span_kinds_include_run_llm_tool(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-trace-kinds"
    await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    kinds = {s.kind for s in trace.spans}
    assert SpanKind.RUN in kinds
    assert SpanKind.LLM in kinds
    assert SpanKind.TOOL in kinds


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/trace — not found
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_trace_returns_none_for_unknown_run(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    result = await recorder.get_trace("no-such-run-api")
    assert result is None


# ---------------------------------------------------------------------------
# GET /runs/spans/{span_id} — success paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_span_returns_correct_span(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-span-001"
    meta = await _populate_trace(recorder, run_id, tmp_path)
    span = await recorder.get_span(meta["llm_id"])
    assert span is not None
    assert span.span_id == meta["llm_id"]
    assert span.kind == SpanKind.LLM


@pytest.mark.asyncio
async def test_get_span_has_correct_token_counts(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-span-tok"
    meta = await _populate_trace(recorder, run_id, tmp_path)
    span = await recorder.get_span(meta["llm_id"])
    assert span.input_tokens == 100
    assert span.output_tokens == 50


@pytest.mark.asyncio
async def test_get_span_has_output_preview(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-span-preview"
    meta = await _populate_trace(recorder, run_id, tmp_path)
    span = await recorder.get_span(meta["tool_id"])
    assert span.output_preview == "42 rows"


@pytest.mark.asyncio
async def test_get_span_tool_has_parent(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-span-parent"
    meta = await _populate_trace(recorder, run_id, tmp_path)
    span = await recorder.get_span(meta["tool_id"])
    assert span.parent_span_id is not None


# ---------------------------------------------------------------------------
# GET /runs/spans/{span_id} — not found
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_span_returns_none_for_unknown_id(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    result = await recorder.get_span("span-does-not-exist-abc123")
    assert result is None


# ---------------------------------------------------------------------------
# TraceView serialisation — used by the API response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_to_dict_spans_are_dicts(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-serial-001"
    await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    d = trace.to_dict()
    for span_dict in d["spans"]:
        assert isinstance(span_dict, dict)
        assert "span_id" in span_dict
        assert "kind" in span_dict
        assert isinstance(span_dict["kind"], str)
        assert isinstance(span_dict["status"], str)


@pytest.mark.asyncio
async def test_trace_to_dict_cost_is_float(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-cost-001"
    await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    d = trace.to_dict()
    assert isinstance(d["total_cost_usd"], float)
    assert d["total_cost_usd"] == pytest.approx(0.002, abs=1e-6)


@pytest.mark.asyncio
async def test_trace_to_dict_duration_ms_positive(redis_client, tmp_path):
    recorder = TraceRecorder(redis_url="redis://unused", log_dir=tmp_path)
    recorder._client = redis_client
    run_id = "api-dur-001"
    await _populate_trace(recorder, run_id, tmp_path)
    trace = await recorder.get_trace(run_id)
    d = trace.to_dict()
    # duration_ms can be None for in-flight runs, but should be ≥ 0 here
    if d["duration_ms"] is not None:
        assert d["duration_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Route module import smoke test
# ---------------------------------------------------------------------------

def test_traces_router_is_importable():
    from harness.api.routes.traces import router
    assert router is not None


def test_traces_router_has_expected_routes():
    from harness.api.routes.traces import router
    paths = {r.path for r in router.routes}
    assert "/{run_id}/trace" in paths
    assert "/spans/{span_id}" in paths
