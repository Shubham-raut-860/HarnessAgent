"""Evaluation pipeline tests for single-agent and multi-agent harness runs."""

from __future__ import annotations

from harness.core.context import AgentContext, AgentResult
from harness.eval import (
    EvalCase,
    EvalDataset,
    EvalRunner,
    MultiAgentEvalCase,
    MultiAgentEvalDataset,
    classify_failure,
)
from harness.orchestrator.runner import AgentRunner


class _FakeAgent:
    """Small BaseAgent-shaped test double."""

    async def run(self, ctx: AgentContext) -> AgentResult:
        return AgentResult(
            run_id=ctx.run_id,
            output=f"{ctx.agent_type} completed: {ctx.task}",
            steps=2,
            tokens=120 + len(ctx.task),
            success=True,
            cost_usd=0.002,
            elapsed_seconds=0.05,
            tool_calls=1,
            tool_errors=0,
            guardrail_hits=0,
            handoff_count=int(ctx.metadata.get("handoff_count", 0) or 0),
            cache_hits=1,
            cache_read_tokens=40,
        )


async def test_eval_runner_uses_agent_runner_lifecycle(redis_client, tmp_path):
    runner = AgentRunner(
        redis=redis_client,
        agent_factory=lambda _agent_type: _FakeAgent(),
        workspace_base=str(tmp_path),
    )
    dataset = EvalDataset(
        name="single_smoke",
        agent_type="code",
        cases=[
            EvalCase(
                case_id="case_1",
                agent_type="code",
                task="produce ok output",
                expected_output="completed",
            )
        ],
    )

    report = await EvalRunner(runner).run(dataset)

    assert report.success_rate == 1.0
    assert report.diagnostics is not None
    assert report.diagnostics.by_agent["code"].total_tokens > 0
    records = await runner.list_runs("eval")
    assert len(records) == 1
    assert records[0].status == "completed"


async def test_multi_agent_eval_reports_subagent_costs(redis_client, tmp_path):
    runner = AgentRunner(
        redis=redis_client,
        agent_factory=lambda _agent_type: _FakeAgent(),
        workspace_base=str(tmp_path),
    )
    dataset = MultiAgentEvalDataset(
        name="multi_smoke",
        cases=[
            MultiAgentEvalCase(
                case_id="multi_1",
                task="inspect then summarize risk",
                expected_output="risk",
                subtasks=[
                    {
                        "id": "inspect",
                        "agent_type": "sql",
                        "task": "inspect data quality",
                        "depends_on": [],
                    },
                    {
                        "id": "summarize",
                        "agent_type": "code",
                        "task": "summarize release risk",
                        "depends_on": ["inspect"],
                    },
                ],
            )
        ],
    )

    report = await EvalRunner(runner).run_multi_agent(dataset)

    assert report.success_rate == 1.0
    assert report.diagnostics is not None
    assert set(report.diagnostics.by_agent) >= {"multi", "sql", "code"}
    assert report.diagnostics.by_agent["code"].total_cost_usd > 0
    assert report.diagnostics.by_agent["multi"].tool_calls == 2
    assert report.diagnostics.by_agent["code"].handoff_count == 1
    assert report.diagnostics.by_agent["multi"].cache_hits == 2


def test_classify_failure_detects_tool_stage():
    result = AgentResult(
        run_id="r1",
        output="",
        steps=1,
        tokens=0,
        success=False,
        failure_class="TOOL_SCHEMA_ERROR",
        error_message="Tool arg schema validation failed",
    )

    assert classify_failure(result) == "tool"


def test_diagnostics_surface_guardrails_and_tool_failures():
    from harness.eval.diagnostics import build_diagnostics

    result = AgentResult(
        run_id="r2",
        output="",
        steps=3,
        tokens=900,
        success=False,
        error_message="Blocked by safety policy after tool schema validation",
        tool_calls=2,
        tool_errors=1,
        guardrail_hits=1,
        handoff_count=1,
    )

    diagnostics = build_diagnostics(
        "guardrail_tool_suite",
        [
            {
                "case_id": "case_guard",
                "agent_type": "code",
                "score": 0.0,
                "result": result,
                "error": result.error_message,
            }
        ],
        pass_threshold=0.5,
    )

    summary = diagnostics.by_agent["code"]
    assert summary.tool_calls == 2
    assert summary.tool_errors == 1
    assert summary.guardrail_hits == 1
    assert summary.handoff_count == 1
    assert any("guardrail" in hint.lower() for hint in diagnostics.recommendations)
