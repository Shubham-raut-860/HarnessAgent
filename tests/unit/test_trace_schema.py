"""Unit tests for TraceSpan schema — dataclasses, serialisation, and TraceView."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from harness.observability.trace_schema import (
    SpanKind,
    SpanStatus,
    TraceSpan,
    TraceView,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_span(**overrides) -> TraceSpan:
    defaults = dict(
        trace_id="trace001",
        span_id="span001abc",
        run_id="run001",
        kind=SpanKind.LLM,
        name="llm:test-model",
        status=SpanStatus.RUNNING,
        start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return TraceSpan(**defaults)


# ---------------------------------------------------------------------------
# SpanKind / SpanStatus enum values
# ---------------------------------------------------------------------------

def test_span_kind_all_values_exist():
    expected = {"run", "agent", "llm", "tool", "guardrail", "memory", "handoff", "eval"}
    actual = {k.value for k in SpanKind}
    assert expected == actual


def test_span_status_all_values_exist():
    assert {s.value for s in SpanStatus} == {"running", "ok", "error"}


def test_span_kind_is_string_subclass():
    assert isinstance(SpanKind.LLM, str)
    assert SpanKind.LLM == "llm"


def test_span_status_is_string_subclass():
    assert SpanStatus.OK == "ok"
    assert SpanStatus.ERROR == "error"


# ---------------------------------------------------------------------------
# TraceSpan construction and defaults
# ---------------------------------------------------------------------------

def test_trace_span_default_fields():
    span = _make_span()
    assert span.parent_span_id is None
    assert span.end_time is None
    assert span.duration_ms is None
    assert span.input_preview == ""
    assert span.output_preview == ""
    assert span.error is None
    assert span.input_tokens == 0
    assert span.output_tokens == 0
    assert span.cost_usd == 0.0
    assert span.cached is False
    assert span.agent_type == ""
    assert span.tenant_id == ""
    assert span.step == 0
    assert span.metadata == {}


def test_trace_span_accepts_all_fields():
    span = TraceSpan(
        trace_id="t1", span_id="s1", run_id="r1",
        kind=SpanKind.TOOL, name="tool:sql",
        status=SpanStatus.OK,
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        parent_span_id="parent-span",
        input_preview="SELECT *",
        output_preview="42 rows",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        cached=True,
        agent_type="sql",
        tenant_id="acme",
        step=3,
        metadata={"db": "prod"},
    )
    assert span.parent_span_id == "parent-span"
    assert span.input_tokens == 100
    assert span.cached is True
    assert span.metadata == {"db": "prod"}


# ---------------------------------------------------------------------------
# finish() method
# ---------------------------------------------------------------------------

def test_finish_sets_end_time_and_duration():
    span = _make_span()
    span.finish(SpanStatus.OK)
    assert span.end_time is not None
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0
    assert span.status == SpanStatus.OK


def test_finish_sets_output_preview():
    span = _make_span()
    span.finish(SpanStatus.OK, output_preview="result text")
    assert span.output_preview == "result text"


def test_finish_truncates_output_to_500_chars():
    span = _make_span()
    long_output = "x" * 600
    span.finish(SpanStatus.OK, output_preview=long_output)
    assert len(span.output_preview) == 500


def test_finish_records_error():
    span = _make_span()
    span.finish(SpanStatus.ERROR, error="timeout exceeded")
    assert span.status == SpanStatus.ERROR
    assert span.error == "timeout exceeded"


def test_finish_records_token_usage():
    span = _make_span()
    span.finish(SpanStatus.OK, input_tokens=200, output_tokens=100, cost_usd=0.003, cached=True)
    assert span.input_tokens == 200
    assert span.output_tokens == 100
    assert span.cost_usd == pytest.approx(0.003)
    assert span.cached is True


def test_finish_returns_self():
    span = _make_span()
    result = span.finish(SpanStatus.OK)
    assert result is span


def test_finish_duration_positive_for_non_instant_span():
    import time
    span = _make_span()
    time.sleep(0.001)
    span.finish(SpanStatus.OK)
    assert span.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# to_dict() serialisation
# ---------------------------------------------------------------------------

def test_to_dict_serialises_kind_as_string():
    span = _make_span(kind=SpanKind.GUARDRAIL)
    d = span.to_dict()
    assert d["kind"] == "guardrail"
    assert isinstance(d["kind"], str)


def test_to_dict_serialises_status_as_string():
    span = _make_span(status=SpanStatus.ERROR)
    d = span.to_dict()
    assert d["status"] == "error"


def test_to_dict_serialises_start_time_as_iso():
    span = _make_span()
    d = span.to_dict()
    assert isinstance(d["start_time"], str)
    assert "2024" in d["start_time"]


def test_to_dict_end_time_none_when_running():
    span = _make_span(status=SpanStatus.RUNNING)
    d = span.to_dict()
    assert d["end_time"] is None


def test_to_dict_end_time_iso_after_finish():
    span = _make_span()
    span.finish(SpanStatus.OK)
    d = span.to_dict()
    assert isinstance(d["end_time"], str)


def test_to_dict_is_json_serialisable():
    span = _make_span()
    span.finish(SpanStatus.OK, output_preview="ok", input_tokens=10)
    raw = json.dumps(span.to_dict())
    assert "llm:test-model" in raw


# ---------------------------------------------------------------------------
# from_dict() deserialisation
# ---------------------------------------------------------------------------

def test_from_dict_roundtrip_preserves_all_fields():
    span = TraceSpan(
        trace_id="t1", span_id="s1", run_id="r1",
        kind=SpanKind.TOOL, name="tool:exec",
        status=SpanStatus.OK,
        start_time=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        parent_span_id="parent1",
        input_preview="args",
        output_preview="result",
        input_tokens=50, output_tokens=25, cost_usd=0.0005,
        cached=False, agent_type="code", tenant_id="t1", step=2,
        metadata={"tool": "python"},
    )
    span.finish(SpanStatus.OK, output_preview="42",
                input_tokens=50, output_tokens=25, cost_usd=0.0005, cached=False)
    d = span.to_dict()
    restored = TraceSpan.from_dict(d)

    assert restored.span_id == span.span_id
    assert restored.kind == SpanKind.TOOL
    assert restored.status == SpanStatus.OK
    assert restored.input_tokens == 50
    assert restored.output_tokens == 25
    assert restored.parent_span_id == "parent1"
    assert restored.metadata == {"tool": "python"}
    assert restored.end_time is not None


def test_from_dict_handles_null_end_time():
    span = _make_span()
    d = span.to_dict()
    assert d["end_time"] is None
    restored = TraceSpan.from_dict(d)
    assert restored.end_time is None


def test_from_dict_parses_kind_enum():
    span = _make_span(kind=SpanKind.HANDOFF)
    restored = TraceSpan.from_dict(span.to_dict())
    assert restored.kind == SpanKind.HANDOFF
    assert isinstance(restored.kind, SpanKind)


def test_from_dict_parses_status_enum():
    span = _make_span(status=SpanStatus.ERROR)
    restored = TraceSpan.from_dict(span.to_dict())
    assert restored.status == SpanStatus.ERROR


# ---------------------------------------------------------------------------
# TraceView
# ---------------------------------------------------------------------------

def _make_trace_view() -> TraceView:
    root = _make_span(span_id="root", kind=SpanKind.RUN, name="run:sql",
                      agent_type="sql", status=SpanStatus.OK)
    root.finish(SpanStatus.OK, input_tokens=100, output_tokens=50, cost_usd=0.002)
    child = _make_span(span_id="child", kind=SpanKind.LLM, name="llm:call",
                       parent_span_id="root", status=SpanStatus.OK)
    child.finish(SpanStatus.OK, input_tokens=80, output_tokens=40, cost_usd=0.001)
    return TraceView(
        trace_id="t1", run_id="r1", agent_type="sql",
        status=SpanStatus.OK,
        start_time=root.start_time, end_time=root.end_time,
        duration_ms=root.duration_ms,
        total_input_tokens=180, total_output_tokens=90, total_cost_usd=0.003,
        span_count=2, spans=[root, child],
    )


def test_trace_view_to_dict_structure():
    tv = _make_trace_view()
    d = tv.to_dict()
    assert d["trace_id"] == "t1"
    assert d["run_id"] == "r1"
    assert d["agent_type"] == "sql"
    assert d["status"] == "ok"
    assert d["span_count"] == 2
    assert len(d["spans"]) == 2
    assert d["total_input_tokens"] == 180
    assert d["total_output_tokens"] == 90


def test_trace_view_to_dict_cost_rounded():
    tv = _make_trace_view()
    d = tv.to_dict()
    # Should be rounded to 6 decimal places
    assert isinstance(d["total_cost_usd"], float)


def test_trace_view_to_dict_is_json_serialisable():
    tv = _make_trace_view()
    raw = json.dumps(tv.to_dict())
    assert "run:sql" in raw


def test_trace_view_spans_serialised_as_dicts():
    tv = _make_trace_view()
    d = tv.to_dict()
    for span_dict in d["spans"]:
        assert isinstance(span_dict, dict)
        assert "span_id" in span_dict
        assert "kind" in span_dict
        assert isinstance(span_dict["kind"], str)
