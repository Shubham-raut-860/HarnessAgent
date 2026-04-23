"""Unit tests for framework adapters (LangGraph, AutoGen, CrewAI).

All external framework packages are mocked via ``sys.modules`` injection so
these tests run without langgraph / pyautogen / crewai installed.
"""

from __future__ import annotations

import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.core.context import AgentContext, StepEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    max_steps: int = 50,
    max_tokens: int = 100_000,
    timeout_seconds: float = 300.0,
) -> AgentContext:
    """Build a minimal AgentContext for adapter tests."""
    return AgentContext(
        run_id=uuid.uuid4().hex,
        tenant_id="test-tenant",
        agent_type="adapter-test",
        task="test task",
        memory=None,
        workspace_path=Path("/tmp"),
        max_steps=max_steps,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


async def _collect(ait: AsyncIterator[StepEvent]) -> list[StepEvent]:
    """Drain an async iterator of StepEvents into a plain list."""
    events: list[StepEvent] = []
    async for ev in ait:
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Fake langgraph module factory
# ---------------------------------------------------------------------------


def _make_fake_langgraph(stream_chunks: list[dict[str, Any]]) -> types.ModuleType:
    """Build a minimal fake ``langgraph`` module tree.

    Args:
        stream_chunks: List of dicts that ``astream()`` will yield sequentially.
                       Each dict maps node_name → node_output.
    """
    lg = types.ModuleType("langgraph")
    lg_graph = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        """Stub StateGraph — only used for the import-check in the adapter."""
        pass

    class FakeCompiledGraph:
        """Compiled graph stub with an async ``astream()`` method."""

        def __init__(self, chunks: list[dict[str, Any]]) -> None:
            self._chunks = chunks

        async def astream(self, input: dict) -> AsyncIterator[dict[str, Any]]:
            for chunk in self._chunks:
                yield chunk

    lg_graph.StateGraph = FakeStateGraph
    lg_graph.CompiledStateGraph = FakeCompiledGraph
    lg.graph = lg_graph

    return lg, lg_graph, FakeCompiledGraph


# ---------------------------------------------------------------------------
# LangGraph adapter tests
# ---------------------------------------------------------------------------


class TestLangGraphAdapter:
    """Tests for :class:`harness.adapters.langgraph.LangGraphAdapter`."""

    def _inject_langgraph(self, chunks: list[dict[str, Any]]):
        """Inject fake langgraph into sys.modules and return the compiled graph."""
        lg, lg_graph, FakeCompiledGraph = _make_fake_langgraph(chunks)
        sys.modules["langgraph"] = lg
        sys.modules["langgraph.graph"] = lg_graph
        compiled = FakeCompiledGraph(chunks)
        return compiled

    def _cleanup(self):
        sys.modules.pop("langgraph", None)
        sys.modules.pop("langgraph.graph", None)

    @pytest.mark.asyncio
    async def test_emits_step_events_for_each_node(self):
        """One StepEvent per node per chunk should be yielded."""
        chunks = [
            {"retriever": {"docs": ["doc1", "doc2"]}},
            {"generator": {"messages": ["hello world"]}},
        ]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)
            ctx = _make_ctx()

            events = await _collect(adapter.run(ctx, input={"messages": []}))

            # Two chunks × one node each = two events.
            assert len(events) == 2
            assert all(ev.event_type == "tool_call" for ev in events)
            assert events[0].payload["node"] == "retriever"
            assert events[1].payload["node"] == "generator"
            assert events[0].payload["framework"] == "langgraph"
            assert events[0].run_id == ctx.run_id
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_output_keys_included_in_payload(self):
        """``output_keys`` in the event payload must list the node output dict keys."""
        chunks = [{"planner": {"plan": "step 1", "tools": ["search"]}}]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)
            ctx = _make_ctx()

            events = await _collect(adapter.run(ctx, input={}))

            assert len(events) == 1
            assert set(events[0].payload["output_keys"]) == {"plan", "tools"}
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_respects_budget_stops_early(self):
        """Adapter must stop yielding and emit budget_exceeded when budget runs out."""
        # 5 chunks but ctx only allows 2 steps.
        chunks = [
            {"node_a": {"x": 1}},
            {"node_b": {"x": 2}},
            {"node_c": {"x": 3}},
            {"node_d": {"x": 4}},
            {"node_e": {"x": 5}},
        ]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)
            ctx = _make_ctx(max_steps=2)

            events = await _collect(adapter.run(ctx, input={}))

            # The first two tool_call events are emitted (steps 1 and 2); the
            # third iteration fails ctx.is_budget_ok() and emits budget_exceeded.
            event_types = [ev.event_type for ev in events]
            assert "budget_exceeded" in event_types
            # No more tool_call events should follow budget_exceeded.
            first_budget = event_types.index("budget_exceeded")
            assert all(
                et == "budget_exceeded"
                for et in event_types[first_budget:]
            )
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_extracts_output_from_messages_key(self):
        """``get_result()`` should read the last element of the 'messages' list."""
        chunks = [{"responder": {"messages": ["first", "final answer"]}}]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))
            result = await adapter.get_result()

            assert result.framework == "langgraph"
            assert "final answer" in result.output
            assert result.steps >= 1
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_extracts_output_from_result_key(self):
        """``get_result()`` should fall back to the 'result' key when 'messages' absent."""
        chunks = [{"worker": {"result": "computed value"}}]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))
            result = await adapter.get_result()

            assert result.output == "computed value"
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_before_run_raises(self):
        """``get_result()`` before ``run()`` must raise RuntimeError."""
        chunks: list = []
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)

            with pytest.raises(RuntimeError, match="before run\\(\\) completed"):
                await adapter.get_result()
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_import_error_raised_when_langgraph_missing(self):
        """ImportError with install hint must be raised if langgraph is absent."""
        # Remove any cached fake module first.
        sys.modules.pop("langgraph", None)
        sys.modules.pop("langgraph.graph", None)

        # Force the import to fail by injecting a broken module.
        broken = types.ModuleType("langgraph")

        class _BrokenGraph:
            async def astream(self, _):  # pragma: no cover
                return
                yield

        broken_compiled = _BrokenGraph()

        # We don't inject langgraph.graph so the import inside the adapter fails.
        with patch.dict(sys.modules, {"langgraph": None, "langgraph.graph": None}):
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=broken_compiled)
            ctx = _make_ctx()

            with pytest.raises(ImportError, match="pip install langgraph"):
                await _collect(adapter.run(ctx, input={}))

    @pytest.mark.asyncio
    async def test_step_count_tracked_in_result(self):
        """FrameworkResult.steps must equal the number of chunks streamed."""
        chunks = [
            {"a": {"v": 1}},
            {"b": {"v": 2}},
            {"c": {"v": 3}},
        ]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            adapter = LangGraphAdapter(graph=compiled)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))
            result = await adapter.get_result()

            assert result.steps == 3
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_event_bus_publish_called_per_event(self):
        """If an event_bus is supplied, publish must be called for each StepEvent."""
        chunks = [{"node1": {"k": "v"}}, {"node2": {"k": "v"}}]
        compiled = self._inject_langgraph(chunks)
        try:
            from harness.adapters.langgraph import LangGraphAdapter

            mock_bus = AsyncMock()
            mock_bus.publish = AsyncMock()
            adapter = LangGraphAdapter(graph=compiled, event_bus=mock_bus)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))

            assert mock_bus.publish.call_count == 2
        finally:
            self._cleanup()


# ---------------------------------------------------------------------------
# Fake autogen module factory
# ---------------------------------------------------------------------------


def _make_fake_autogen() -> types.ModuleType:
    """Build a minimal fake ``autogen`` module."""
    ag = types.ModuleType("autogen")

    class FakeConversableAgent:
        def __init__(self, name: str = "agent") -> None:
            self.name = name
            self._messages: list[dict] = []

        def receive(self, message, sender, request_reply=None, silent=False):
            """Default no-op receive."""
            return None

        def initiate_chat(self, recipient, message: str, max_turns: int = 10):
            """Simulate a 3-message exchange."""
            recipient.receive(message, self)
            recipient.receive("I understand your task.", self)
            recipient.receive("Here is the answer.", self)
            return MagicMock(summary="Here is the answer.")

    ag.ConversableAgent = FakeConversableAgent
    return ag


# ---------------------------------------------------------------------------
# AutoGen adapter tests
# ---------------------------------------------------------------------------


class TestAutoGenAdapter:
    """Tests for :class:`harness.adapters.autogen.AutoGenAdapter`."""

    def _inject_autogen(self):
        ag = _make_fake_autogen()
        sys.modules["autogen"] = ag
        return ag

    def _cleanup(self):
        sys.modules.pop("autogen", None)

    @pytest.mark.asyncio
    async def test_emits_message_events_for_each_captured_message(self):
        """One 'message' StepEvent should be yielded per captured receive call."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            adapter = AutoGenAdapter(
                initiator_agent=initiator,
                recipient_agent_or_groupchat=recipient,
                max_turns=5,
            )
            ctx = _make_ctx()

            events = await _collect(adapter.run(ctx, input={"task": "summarise docs"}))

            # FakeConversableAgent.initiate_chat calls receive 3 times.
            assert len(events) == 3
            assert all(ev.event_type == "message" for ev in events)
            assert all(ev.payload["framework"] == "autogen" for ev in events)
            assert all(ev.run_id == ctx.run_id for ev in events)
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_role_is_set_to_sender_name(self):
        """The 'role' payload key must equal the sender agent's name."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            adapter = AutoGenAdapter(initiator, recipient)
            ctx = _make_ctx()

            events = await _collect(adapter.run(ctx, input={"task": "go"}))

            # All messages were sent by "user_proxy" in the fake implementation.
            assert events[0].payload["role"] == "user_proxy"
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_respects_budget_stops_early(self):
        """budget_exceeded must be emitted and iteration must stop when over budget."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            adapter = AutoGenAdapter(initiator, recipient)
            # max_steps=1 → budget exhausted after first event
            ctx = _make_ctx(max_steps=1)
            # Pre-exhaust the step count so is_budget_ok() fails immediately.
            ctx.step_count = 1

            events = await _collect(adapter.run(ctx, input={"task": "go"}))

            assert len(events) == 1
            assert events[0].event_type == "budget_exceeded"
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_returns_last_message_as_output(self):
        """FrameworkResult.output must be the content of the last captured message."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            adapter = AutoGenAdapter(initiator, recipient)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={"task": "go"}))
            result = await adapter.get_result()

            assert result.framework == "autogen"
            assert result.output == "Here is the answer."
            assert result.steps == 3
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_before_run_raises(self):
        """Calling get_result() before run() must raise RuntimeError."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            adapter = AutoGenAdapter(initiator, recipient)

            with pytest.raises(RuntimeError, match="before run\\(\\) completed"):
                await adapter.get_result()
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_import_error_raised_when_autogen_missing(self):
        """ImportError with install hint must be raised if autogen is absent."""
        sys.modules.pop("autogen", None)

        with patch.dict(sys.modules, {"autogen": None}):
            from harness.adapters.autogen import AutoGenAdapter

            initiator = MagicMock()
            recipient = MagicMock()
            adapter = AutoGenAdapter(initiator, recipient)
            ctx = _make_ctx()

            with pytest.raises(ImportError, match="pip install pyautogen"):
                await _collect(adapter.run(ctx, input={"task": "go"}))

    @pytest.mark.asyncio
    async def test_original_receive_restored_after_run(self):
        """The recipient's original receive method must be restored after run()."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            # Capture the underlying function, not the transient bound-method object.
            # Each attribute access on a regular method produces a new bound-method
            # instance; comparing __func__ is the correct identity check.
            original_func = recipient.receive.__func__
            adapter = AutoGenAdapter(initiator, recipient)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={"task": "go"}))

            # After run(), the patched lambda/wrapper must be gone; the method
            # should resolve back to the original underlying function.
            assert recipient.receive.__func__ is original_func
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_message_fallback_uses_message_key(self):
        """Input dict key 'message' must be used as task when 'task' is absent."""
        ag = self._inject_autogen()
        try:
            from harness.adapters.autogen import AutoGenAdapter

            initiator = ag.ConversableAgent(name="user_proxy")
            recipient = ag.ConversableAgent(name="assistant")

            adapter = AutoGenAdapter(initiator, recipient)
            ctx = _make_ctx()

            # Should not raise — the task is taken from "message" key.
            events = await _collect(
                adapter.run(ctx, input={"message": "hello from message key"})
            )
            assert len(events) > 0
        finally:
            self._cleanup()


# ---------------------------------------------------------------------------
# Fake crewai module factory
# ---------------------------------------------------------------------------


def _make_fake_crewai(
    step_outputs: list[str],
    kickoff_result: str = "crew result",
) -> types.ModuleType:
    """Build a minimal fake ``crewai`` module.

    The ``Crew`` stub calls ``step_callback`` once per string in
    ``step_outputs`` during ``kickoff()``.
    """
    cr = types.ModuleType("crewai")

    step_outs = step_outputs
    final_result = kickoff_result

    class FakeCrew:
        def __init__(self) -> None:
            self.step_callback = None

        def kickoff(self, inputs: dict | None = None) -> str:
            for out in step_outs:
                if self.step_callback is not None:
                    self.step_callback(out)
            return final_result

    cr.Crew = FakeCrew
    return cr


# ---------------------------------------------------------------------------
# CrewAI adapter tests
# ---------------------------------------------------------------------------


class TestCrewAIAdapter:
    """Tests for :class:`harness.adapters.crewai.CrewAIAdapter`."""

    def _inject_crewai(
        self,
        step_outputs: list[str],
        kickoff_result: str = "crew result",
    ):
        cr = _make_fake_crewai(step_outputs, kickoff_result)
        sys.modules["crewai"] = cr
        return cr

    def _cleanup(self):
        sys.modules.pop("crewai", None)

    @pytest.mark.asyncio
    async def test_emits_task_events_for_each_step(self):
        """One 'tool_call' StepEvent per step_callback invocation must be yielded."""
        cr = self._inject_crewai(
            step_outputs=["step A output", "step B output", "step C output"]
        )
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            events = await _collect(adapter.run(ctx, input={"topic": "AI"}))

            assert len(events) == 3
            assert all(ev.event_type == "tool_call" for ev in events)
            assert all(ev.payload["framework"] == "crewai" for ev in events)
            assert "step A output" in events[0].payload["task_output"]
            assert "step B output" in events[1].payload["task_output"]
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_respects_budget_stops_early(self):
        """budget_exceeded must be yielded and iteration stopped when budget is exhausted."""
        cr = self._inject_crewai(
            step_outputs=["s1", "s2", "s3", "s4", "s5"],
        )
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx(max_steps=50)
            # Exhaust the budget manually so it fails on first event.
            ctx.step_count = 50

            events = await _collect(adapter.run(ctx, input={}))

            # First event should be budget_exceeded.
            assert len(events) == 1
            assert events[0].event_type == "budget_exceeded"
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_returns_kickoff_output(self):
        """FrameworkResult.output must equal the string returned by crew.kickoff()."""
        cr = self._inject_crewai(
            step_outputs=["step 1"],
            kickoff_result="The final crew answer.",
        )
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))
            result = await adapter.get_result()

            assert result.framework == "crewai"
            assert result.output == "The final crew answer."
            assert result.steps == 1
            assert result.metadata["task_outputs"] == ["step 1"]
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_get_result_before_run_raises(self):
        """Calling get_result() before run() must raise RuntimeError."""
        cr = self._inject_crewai(step_outputs=[])
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            adapter = CrewAIAdapter(crew=crew)

            with pytest.raises(RuntimeError, match="before run\\(\\) completed"):
                await adapter.get_result()
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_import_error_raised_when_crewai_missing(self):
        """ImportError with install hint must be raised if crewai is absent."""
        sys.modules.pop("crewai", None)

        with patch.dict(sys.modules, {"crewai": None}):
            from harness.adapters.crewai import CrewAIAdapter

            crew = MagicMock()
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            with pytest.raises(ImportError, match="pip install crewai"):
                await _collect(adapter.run(ctx, input={}))

    @pytest.mark.asyncio
    async def test_original_step_callback_forwarded(self):
        """Adapter must call through to a pre-existing step_callback."""
        cr = self._inject_crewai(step_outputs=["output 1", "output 2"])
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            original_calls: list[str] = []

            def _original_cb(out: Any) -> None:
                original_calls.append(str(out))

            crew.step_callback = _original_cb
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))

            # Both step outputs must have been forwarded to the original callback.
            assert original_calls == ["output 1", "output 2"]
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_original_callback_restored_after_run(self):
        """crew.step_callback must be restored to its original value after run()."""
        cr = self._inject_crewai(step_outputs=["x"])
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            sentinel = MagicMock()
            crew.step_callback = sentinel

            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))

            assert crew.step_callback is sentinel
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_task_output_truncated_in_payload(self):
        """task_output in event payload must be truncated to at most 300 characters."""
        long_output = "x" * 500
        cr = self._inject_crewai(step_outputs=[long_output])
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            events = await _collect(adapter.run(ctx, input={}))

            assert len(events[0].payload["task_output"]) == 300
        finally:
            self._cleanup()

    @pytest.mark.asyncio
    async def test_metadata_contains_all_task_outputs(self):
        """FrameworkResult.metadata['task_outputs'] must include all step strings."""
        cr = self._inject_crewai(step_outputs=["alpha", "beta", "gamma"])
        try:
            from harness.adapters.crewai import CrewAIAdapter

            crew = cr.Crew()
            adapter = CrewAIAdapter(crew=crew)
            ctx = _make_ctx()

            await _collect(adapter.run(ctx, input={}))
            result = await adapter.get_result()

            assert result.metadata["task_outputs"] == ["alpha", "beta", "gamma"]
        finally:
            self._cleanup()


# ---------------------------------------------------------------------------
# FrameworkResult dataclass tests
# ---------------------------------------------------------------------------


class TestFrameworkResult:
    """Tests for the :class:`harness.adapters.base.FrameworkResult` dataclass."""

    def test_defaults(self):
        from harness.adapters.base import FrameworkResult

        fr = FrameworkResult(framework="langgraph", output="hello", steps=3)
        assert fr.metadata == {}
        assert fr.framework == "langgraph"
        assert fr.output == "hello"
        assert fr.steps == 3

    def test_with_metadata(self):
        from harness.adapters.base import FrameworkResult

        fr = FrameworkResult(
            framework="crewai",
            output="result",
            steps=5,
            metadata={"key": "value"},
        )
        assert fr.metadata["key"] == "value"


# ---------------------------------------------------------------------------
# __init__.py public surface test
# ---------------------------------------------------------------------------


class TestAdaptersInit:
    """Verify that the public adapter surface is exported correctly."""

    def test_all_exports_importable(self):
        from harness.adapters import (
            AutoGenAdapter,
            CrewAIAdapter,
            FrameworkAdapter,
            FrameworkResult,
            LangGraphAdapter,
        )

        assert issubclass(LangGraphAdapter, FrameworkAdapter)
        assert issubclass(AutoGenAdapter, FrameworkAdapter)
        assert issubclass(CrewAIAdapter, FrameworkAdapter)
        # FrameworkResult is a dataclass, not a subclass of FrameworkAdapter.
        assert FrameworkResult is not None

    def test_framework_names(self):
        from harness.adapters import AutoGenAdapter, CrewAIAdapter, LangGraphAdapter

        assert LangGraphAdapter.framework_name == "langgraph"
        assert AutoGenAdapter.framework_name == "autogen"
        assert CrewAIAdapter.framework_name == "crewai"
