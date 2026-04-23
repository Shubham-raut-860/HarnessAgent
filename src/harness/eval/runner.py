"""EvalRunner — runs an agent over a dataset and produces an EvalReport."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.scorers import ScoreResult, score_exact_match, score_success_rate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Aggregated evaluation results for a dataset run.

    Attributes:
        dataset_name:        Name of the EvalDataset.
        agent_type:          The agent type evaluated.
        total_cases:         Total number of cases attempted.
        passed:              Cases that scored >= 0.5 (configurable threshold).
        failed:              Cases that scored < 0.5 or raised an exception.
        success_rate:        passed / total_cases.
        avg_steps:           Mean step count across all runs.
        avg_tokens:          Mean token count across all runs.
        avg_cost_usd:        Mean USD cost across all runs.
        avg_latency_seconds: Mean wall-clock seconds per run.
        scores:              Per-case float scores in [0, 1].
        errors:              Per-case error messages for failed cases.
        prompt_version:      Active prompt version ID at time of eval.
        run_at:              UTC timestamp when this eval was executed.
    """

    dataset_name: str
    agent_type: str
    total_cases: int
    passed: int
    failed: int
    success_rate: float
    avg_steps: float
    avg_tokens: float
    avg_cost_usd: float
    avg_latency_seconds: float
    scores: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    prompt_version: str = "unknown"
    run_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the report as a Markdown-formatted string."""
        lines = [
            f"# Eval Report: {self.dataset_name}",
            "",
            f"**Agent type:** `{self.agent_type}`  ",
            f"**Prompt version:** `{self.prompt_version}`  ",
            f"**Run at:** {self.run_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total cases | {self.total_cases} |",
            f"| Passed | {self.passed} |",
            f"| Failed | {self.failed} |",
            f"| Success rate | {self.success_rate:.1%} |",
            f"| Avg steps | {self.avg_steps:.1f} |",
            f"| Avg tokens | {self.avg_tokens:.0f} |",
            f"| Avg cost (USD) | ${self.avg_cost_usd:.4f} |",
            f"| Avg latency (s) | {self.avg_latency_seconds:.2f} |",
            "",
        ]

        if self.scores:
            lines += [
                "## Per-Case Scores",
                "",
                "| Case ID | Score | Error |",
                "|---------|-------|-------|",
            ]
            for case_id, score in sorted(self.scores.items()):
                error = self.errors.get(case_id, "")
                error_short = error[:60] + "..." if len(error) > 60 else error
                lines.append(f"| `{case_id}` | {score:.2f} | {error_short} |")
            lines.append("")

        return "\n".join(lines)

    def compare(self, other: "EvalReport") -> str:
        """Produce a diff summary comparing this report to *other*.

        Args:
            other: The baseline EvalReport to compare against.

        Returns:
            A human-readable Markdown comparison string.
        """
        delta_sr = self.success_rate - other.success_rate
        delta_steps = self.avg_steps - other.avg_steps
        delta_tokens = self.avg_tokens - other.avg_tokens
        delta_cost = self.avg_cost_usd - other.avg_cost_usd
        delta_latency = self.avg_latency_seconds - other.avg_latency_seconds

        def fmt_delta(v: float, unit: str = "", higher_is_better: bool = True) -> str:
            sign = "+" if v >= 0 else ""
            direction = ""
            if v > 0:
                direction = " (better)" if higher_is_better else " (worse)"
            elif v < 0:
                direction = " (worse)" if higher_is_better else " (better)"
            return f"{sign}{v:.4f}{unit}{direction}"

        lines = [
            f"## Comparison: {self.dataset_name} vs baseline",
            "",
            f"| Metric | Baseline | New | Delta |",
            f"|--------|----------|-----|-------|",
            f"| Success rate | {other.success_rate:.1%} | {self.success_rate:.1%} | {fmt_delta(delta_sr, '%', True)} |",
            f"| Avg steps | {other.avg_steps:.1f} | {self.avg_steps:.1f} | {fmt_delta(delta_steps, '', False)} |",
            f"| Avg tokens | {other.avg_tokens:.0f} | {self.avg_tokens:.0f} | {fmt_delta(delta_tokens, '', False)} |",
            f"| Avg cost (USD) | ${other.avg_cost_usd:.4f} | ${self.avg_cost_usd:.4f} | {fmt_delta(delta_cost, '', False)} |",
            f"| Avg latency (s) | {other.avg_latency_seconds:.2f} | {self.avg_latency_seconds:.2f} | {fmt_delta(delta_latency, '', False)} |",
            "",
        ]

        # Per-case diffs
        all_ids = sorted(set(self.scores) | set(other.scores))
        if all_ids:
            lines += [
                "## Per-Case Score Differences",
                "",
                "| Case ID | Baseline | New | Delta |",
                "|---------|----------|-----|-------|",
            ]
            for cid in all_ids:
                b_score = other.scores.get(cid, float("nan"))
                n_score = self.scores.get(cid, float("nan"))
                if b_score != b_score or n_score != n_score:  # NaN check
                    diff = "N/A"
                else:
                    d = n_score - b_score
                    diff = f"{'+' if d >= 0 else ''}{d:.2f}"
                lines.append(
                    f"| `{cid}` | {b_score:.2f} | {n_score:.2f} | {diff} |"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise report to a plain dict."""
        return {
            "dataset_name": self.dataset_name,
            "agent_type": self.agent_type,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_tokens": self.avg_tokens,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_latency_seconds": self.avg_latency_seconds,
            "scores": self.scores,
            "errors": self.errors,
            "prompt_version": self.prompt_version,
            "run_at": self.run_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

_PASS_THRESHOLD = 0.5  # Score at or above this is considered a pass


class EvalRunner:
    """Runs an agent over an EvalDataset and produces an EvalReport.

    Args:
        agent_runner: Object with an ``execute_run(run_id)`` or
                      ``run(context)`` async method, or a callable that
                      takes (tenant_id, agent_type, task) and returns an
                      AgentResult-like object.
        llm_provider: Optional LLMProvider used by ``score_llm_judge``.
    """

    def __init__(
        self,
        agent_runner: Any,
        llm_provider: Optional[Any] = None,
    ) -> None:
        self._runner = agent_runner
        self._llm = llm_provider

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        dataset: EvalDataset,
        tenant_id: str = "eval",
        concurrency: int = 3,
        scorer: Optional[Callable] = None,
        pass_threshold: float = _PASS_THRESHOLD,
        prompt_version: str = "unknown",
    ) -> EvalReport:
        """Execute all cases in the dataset and return a consolidated EvalReport.

        Args:
            dataset:        The EvalDataset to evaluate.
            tenant_id:      Tenant ID to use for all eval runs.
            concurrency:    Maximum number of concurrent agent runs.
            scorer:         Scoring function to use.  If None: uses
                            ``score_exact_match`` when expected is set,
                            otherwise checks ``result.success``.
            pass_threshold: Score >= this value counts as "passed".
            prompt_version: Label for the prompt version in the report.

        Returns:
            EvalReport with aggregated and per-case metrics.
        """
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._run_case(
                case=case,
                tenant_id=tenant_id,
                scorer=scorer,
                semaphore=semaphore,
            )
            for case in dataset.cases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate metrics
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        total_steps: list[float] = []
        total_tokens: list[float] = []
        total_cost: list[float] = []
        total_latency: list[float] = []

        for case, result in zip(dataset.cases, results):
            if isinstance(result, Exception):
                scores[case.case_id] = 0.0
                errors[case.case_id] = str(result)
                logger.warning("Case %s raised exception: %s", case.case_id, result)
            else:
                score_val, agent_result = result
                scores[case.case_id] = score_val
                if agent_result is not None:
                    total_steps.append(float(getattr(agent_result, "steps", 0)))
                    total_tokens.append(float(getattr(agent_result, "tokens", 0)))
                    total_cost.append(float(getattr(agent_result, "cost_usd", 0.0)))
                    total_latency.append(float(getattr(agent_result, "elapsed_seconds", 0.0)))
                if score_val < pass_threshold:
                    error_msg = getattr(agent_result, "error_message", "") or ""
                    if error_msg:
                        errors[case.case_id] = error_msg

        passed = sum(1 for s in scores.values() if s >= pass_threshold)
        failed = len(dataset.cases) - passed

        def _safe_avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return EvalReport(
            dataset_name=dataset.name,
            agent_type=dataset.agent_type,
            total_cases=len(dataset.cases),
            passed=passed,
            failed=failed,
            success_rate=passed / max(len(dataset.cases), 1),
            avg_steps=_safe_avg(total_steps),
            avg_tokens=_safe_avg(total_tokens),
            avg_cost_usd=_safe_avg(total_cost),
            avg_latency_seconds=_safe_avg(total_latency),
            scores=scores,
            errors=errors,
            prompt_version=prompt_version,
            run_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Patch comparison
    # ------------------------------------------------------------------

    async def compare_patches(
        self,
        dataset: EvalDataset,
        baseline_config: dict,
        patched_config: dict,
        tenant_id: str = "eval",
    ) -> tuple[EvalReport, EvalReport]:
        """Run the dataset twice with different configs and return both reports.

        The runner's config is temporarily swapped to *baseline_config* then
        *patched_config*.  If the runner exposes a ``configure(dict)`` method
        it will be called; otherwise the configs are passed as metadata.

        Args:
            dataset:         Dataset to evaluate.
            baseline_config: Config dict for the baseline run.
            patched_config:  Config dict for the patched run.
            tenant_id:       Tenant ID to use for all runs.

        Returns:
            Tuple of (baseline_report, patched_report).
        """
        logger.info(
            "compare_patches: running baseline for dataset=%s", dataset.name
        )
        baseline_report = await self._run_with_config(
            dataset, baseline_config, tenant_id, label="baseline"
        )

        logger.info(
            "compare_patches: running patched for dataset=%s", dataset.name
        )
        patched_report = await self._run_with_config(
            dataset, patched_config, tenant_id, label="patched"
        )

        return baseline_report, patched_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_case(
        self,
        case: EvalCase,
        tenant_id: str,
        scorer: Optional[Callable],
        semaphore: asyncio.Semaphore,
    ) -> tuple[float, Any]:
        """Execute a single EvalCase and return (score, agent_result)."""
        async with semaphore:
            try:
                agent_result = await self._invoke_runner(
                    tenant_id=tenant_id,
                    agent_type=case.agent_type,
                    task=case.task,
                )
            except Exception as exc:
                logger.warning(
                    "Runner raised exception for case %s: %s", case.case_id, exc
                )
                raise

            output = getattr(agent_result, "output", "") or ""
            score = await self._score(case, output, agent_result, scorer)
            return score, agent_result

    async def _invoke_runner(
        self, tenant_id: str, agent_type: str, task: str
    ) -> Any:
        """Invoke the agent runner using the most appropriate interface."""
        # Protocol 1: runner.run(tenant_id, agent_type, task)
        if hasattr(self._runner, "run") and callable(self._runner.run):
            result = self._runner.run(tenant_id, agent_type, task)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Protocol 2: runner is a coroutine function itself
        if asyncio.iscoroutinefunction(self._runner):
            return await self._runner(tenant_id=tenant_id, agent_type=agent_type, task=task)

        # Protocol 3: synchronous callable
        if callable(self._runner):
            return self._runner(tenant_id=tenant_id, agent_type=agent_type, task=task)

        raise TypeError(
            f"agent_runner does not expose a callable interface: {type(self._runner)}"
        )

    async def _score(
        self,
        case: EvalCase,
        output: str,
        agent_result: Any,
        scorer: Optional[Callable],
    ) -> float:
        """Compute a score for the given case and output."""
        if scorer is not None:
            result = scorer(output, case.expected_output)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, ScoreResult):
                return result.score
            return float(result)

        # Default scoring strategy
        if case.expected_output:
            return score_exact_match(output, case.expected_output)

        # No ground truth: use success flag
        return 1.0 if getattr(agent_result, "success", False) else 0.0

    async def _run_with_config(
        self,
        dataset: EvalDataset,
        config: dict,
        tenant_id: str,
        label: str,
    ) -> EvalReport:
        """Apply config to runner, run eval, restore prior state."""
        # Attempt to apply config if runner supports it
        if hasattr(self._runner, "configure") and callable(self._runner.configure):
            try:
                cfg_result = self._runner.configure(config)
                if asyncio.iscoroutine(cfg_result):
                    await cfg_result
            except Exception as exc:
                logger.warning("Runner.configure() raised: %s", exc)

        report = await self.run(
            dataset=dataset,
            tenant_id=tenant_id,
            prompt_version=config.get("prompt_version", label),
        )
        return report
