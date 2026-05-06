"""Run a tiny single-agent and multi-agent HarnessAgent eval locally.

This example intentionally uses a fake runner so the evaluation pipeline can be
tested without API keys, Redis, or a database. Swap ``DemoRunner`` for the real
``AgentRunner`` to evaluate production agents.
"""

from __future__ import annotations

import asyncio

from harness.core.context import AgentResult
from harness.eval import (
    MULTI_AGENT_EVAL_CASES,
    EvalCase,
    EvalDataset,
    EvalRunner,
    MultiAgentEvalDataset,
)
from harness.orchestrator.runner import RunRecord


class DemoRunner:
    """Minimal AgentRunner-shaped interface accepted by EvalRunner."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}

    async def create_run(
        self,
        tenant_id: str,
        agent_type: str,
        task: str,
        metadata: dict | None = None,
    ) -> RunRecord:
        record = RunRecord(
            tenant_id=tenant_id,
            agent_type=agent_type,
            task=task,
            metadata=metadata or {},
        )
        self._records[record.run_id] = record
        return record

    async def execute_run(self, run_id: str) -> RunRecord:
        record = self._records[run_id]
        tokens = 200 + len(record.task.split()) * 12
        output = (
            f"{record.agent_type} completed for {record.tenant_id}: {record.task}\n"
            "remediation risk checklist"
        )
        result = AgentResult(
            run_id=record.run_id,
            output=output,
            steps=3,
            tokens=tokens,
            success=True,
            cost_usd=tokens * 0.0000004,
            elapsed_seconds=0.12,
            tool_calls=1,
            tool_errors=0,
            guardrail_hits=0,
            cache_hits=1,
            cache_read_tokens=80,
        )
        record.status = "completed"
        record.result = {
            "run_id": result.run_id,
            "output": result.output,
            "steps": result.steps,
            "tokens": result.tokens,
            "success": result.success,
            "cost_usd": result.cost_usd,
            "elapsed_seconds": result.elapsed_seconds,
            "tool_calls": result.tool_calls,
            "tool_errors": result.tool_errors,
            "guardrail_hits": result.guardrail_hits,
            "cache_hits": result.cache_hits,
            "cache_read_tokens": result.cache_read_tokens,
        }
        return record


async def main() -> None:
    runner = DemoRunner()
    evaluator = EvalRunner(runner)

    single = EvalDataset(
        name="demo_single_agent",
        agent_type="code",
        cases=[
            EvalCase(
                case_id="single_1",
                agent_type="code",
                task="Write a utility and summarize risk.",
                expected_output="risk",
            )
        ],
    )
    single_report = await evaluator.run(single)
    print(single_report.to_markdown())

    multi = MultiAgentEvalDataset(
        name="demo_multi_agent",
        cases=MULTI_AGENT_EVAL_CASES[:1],
    )
    multi_report = await evaluator.run_multi_agent(multi)
    print(multi_report.to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
