"""Diagnostics and optimization hints for agent evaluation reports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _value(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or mapping key from an arbitrary result object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class CaseDiagnostic:
    """Per-case evaluation diagnosis.

    The diagnosis is intentionally operational: it names the likely failure
    stage, resource use, and concrete optimization hints.
    """

    case_id: str
    agent_type: str
    score: float
    passed: bool
    failure_stage: str = "success"
    failure_class: str | None = None
    error_message: str = ""
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    guardrail_hits: int = 0
    handoff_count: int = 0
    cache_hits: int = 0
    cache_read_tokens: int = 0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "agent_type": self.agent_type,
            "score": self.score,
            "passed": self.passed,
            "failure_stage": self.failure_stage,
            "failure_class": self.failure_class,
            "error_message": self.error_message,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "latency_seconds": self.latency_seconds,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "guardrail_hits": self.guardrail_hits,
            "handoff_count": self.handoff_count,
            "cache_hits": self.cache_hits,
            "cache_read_tokens": self.cache_read_tokens,
            "recommendations": self.recommendations,
        }


@dataclass
class AgentMetricSummary:
    """Aggregated quality and resource metrics for one agent type."""

    agent_type: str
    cases: int = 0
    passed: int = 0
    total_steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    guardrail_hits: int = 0
    handoff_count: int = 0
    cache_hits: int = 0
    cache_read_tokens: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.passed / self.cases if self.cases else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.cases if self.cases else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.cases if self.cases else 0.0

    @property
    def avg_latency_seconds(self) -> float:
        return self.total_latency_seconds / self.cases if self.cases else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "cases": self.cases,
            "passed": self.passed,
            "success_rate": self.success_rate,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_tokens": self.avg_tokens,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_latency_seconds": self.avg_latency_seconds,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "guardrail_hits": self.guardrail_hits,
            "handoff_count": self.handoff_count,
            "cache_hits": self.cache_hits,
            "cache_read_tokens": self.cache_read_tokens,
            "failures": self.failures,
        }


@dataclass
class EvalDiagnostics:
    """Evaluation diagnostics across all cases in a run."""

    dataset_name: str
    cases: list[CaseDiagnostic]
    by_agent: dict[str, AgentMetricSummary]
    failure_counts: dict[str, int]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "cases": [case.to_dict() for case in self.cases],
            "by_agent": {
                agent: summary.to_dict()
                for agent, summary in sorted(self.by_agent.items())
            },
            "failure_counts": self.failure_counts,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Render a compact Markdown diagnostic section."""
        lines = [
            "## Diagnostics",
            "",
            "| Agent | Cases | Pass Rate | Avg Tokens | Avg Cost | Tools | Guards | Handoffs | Cache | Common Failures |",
            "|-------|-------|-----------|------------|----------|-------|--------|----------|-------|-----------------|",
        ]
        for agent, summary in sorted(self.by_agent.items()):
            failures = ", ".join(
                f"{stage}:{count}" for stage, count in sorted(summary.failures.items())
            ) or "-"
            lines.append(
                f"| `{agent}` | {summary.cases} | {summary.success_rate:.1%} | "
                f"{summary.avg_tokens:.0f} | ${summary.avg_cost_usd:.4f} | "
                f"{summary.tool_calls}/{summary.tool_errors} | {summary.guardrail_hits} | "
                f"{summary.handoff_count} | {summary.cache_hits} | {failures} |"
            )

        if self.failure_counts:
            lines += [
                "",
                "### Failure Stages",
                "",
                "| Stage | Count |",
                "|-------|-------|",
            ]
            for stage, count in sorted(self.failure_counts.items()):
                lines.append(f"| `{stage}` | {count} |")

        if self.recommendations:
            lines += ["", "### Optimization Hints", ""]
            lines.extend(f"- {hint}" for hint in self.recommendations)

        worst = [
            case for case in self.cases
            if not case.passed or case.failure_stage != "success"
        ][:10]
        if worst:
            lines += [
                "",
                "### Failed / Risky Cases",
                "",
                "| Case | Agent | Stage | Score | Error |",
                "|------|-------|-------|-------|-------|",
            ]
            for case in worst:
                error = case.error_message.replace("|", "\\|")[:90]
                lines.append(
                    f"| `{case.case_id}` | `{case.agent_type}` | "
                    f"`{case.failure_stage}` | {case.score:.2f} | {error} |"
                )
        return "\n".join(lines)


_STAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("tool", re.compile(r"\b(tool|schema|argument|validation|timeout)\b", re.I)),
    ("safety", re.compile(r"\b(safety|guard|blocked|pii|injection|hitl|approval)\b", re.I)),
    ("budget", re.compile(r"\b(budget|token|step|cost|rate limit|quota)\b", re.I)),
    ("llm", re.compile(r"\b(llm|model|provider|openai|anthropic|context|completion)\b", re.I)),
    ("memory", re.compile(r"\b(memory|retrieval|vector|graph|rag|redis)\b", re.I)),
    ("router", re.compile(r"\b(router|fallback|circuit|provider exhausted)\b", re.I)),
    ("communication", re.compile(r"\b(handoff|communication|message|predecessor)\b", re.I)),
    ("planner", re.compile(r"\b(plan|planner|dag|dependency|deadlock)\b", re.I)),
]


def classify_failure(result: Any, error_message: str = "") -> str:
    """Infer the most likely failure stage from an agent result and error text."""
    success = bool(_value(result, "success", False))
    if success and not error_message:
        return "success"

    failure_class = str(_value(result, "failure_class", "") or "")
    text = " ".join(
        part for part in [
            failure_class,
            str(_value(result, "error_message", "") or ""),
            error_message,
        ]
        if part
    )
    for stage, pattern in _STAGE_PATTERNS:
        if pattern.search(text):
            return stage
    if not success:
        return "quality"
    return "success"


def build_diagnostics(
    dataset_name: str,
    records: list[dict[str, Any]],
    pass_threshold: float,
) -> EvalDiagnostics:
    """Build case diagnostics, per-agent summaries, and optimization hints."""
    cases: list[CaseDiagnostic] = []
    by_agent: dict[str, AgentMetricSummary] = {}
    failure_counts: dict[str, int] = {}

    for rec in records:
        result = rec.get("result")
        score = float(rec.get("score", 0.0))
        error = str(rec.get("error", "") or _value(result, "error_message", "") or "")
        agent_type = str(rec.get("agent_type", "unknown"))
        passed = score >= pass_threshold and not error
        stage = classify_failure(result, error)
        failure_class = _value(result, "failure_class")

        diagnostic = CaseDiagnostic(
            case_id=str(rec.get("case_id", "unknown")),
            agent_type=agent_type,
            score=score,
            passed=passed,
            failure_stage=stage,
            failure_class=str(failure_class) if failure_class else None,
            error_message=error,
            steps=int(_value(result, "steps", 0) or 0),
            tokens=int(_value(result, "tokens", 0) or 0),
            cost_usd=float(_value(result, "cost_usd", 0.0) or 0.0),
            latency_seconds=float(_value(result, "elapsed_seconds", 0.0) or 0.0),
            tool_calls=int(_value(result, "tool_calls", 0) or 0),
            tool_errors=int(_value(result, "tool_errors", 0) or 0),
            guardrail_hits=int(_value(result, "guardrail_hits", 0) or 0),
            handoff_count=int(_value(result, "handoff_count", 0) or 0),
            cache_hits=int(_value(result, "cache_hits", 0) or 0),
            cache_read_tokens=int(_value(result, "cache_read_tokens", 0) or 0),
        )
        diagnostic.recommendations = _case_hints(diagnostic)
        cases.append(diagnostic)

        summary = by_agent.setdefault(agent_type, AgentMetricSummary(agent_type=agent_type))
        summary.cases += 1
        summary.passed += int(passed)
        summary.total_steps += diagnostic.steps
        summary.total_tokens += diagnostic.tokens
        summary.total_cost_usd += diagnostic.cost_usd
        summary.total_latency_seconds += diagnostic.latency_seconds
        summary.tool_calls += diagnostic.tool_calls
        summary.tool_errors += diagnostic.tool_errors
        summary.guardrail_hits += diagnostic.guardrail_hits
        summary.handoff_count += diagnostic.handoff_count
        summary.cache_hits += diagnostic.cache_hits
        summary.cache_read_tokens += diagnostic.cache_read_tokens
        if stage != "success":
            summary.failures[stage] = summary.failures.get(stage, 0) + 1
            failure_counts[stage] = failure_counts.get(stage, 0) + 1

    return EvalDiagnostics(
        dataset_name=dataset_name,
        cases=cases,
        by_agent=by_agent,
        failure_counts=failure_counts,
        recommendations=_global_hints(cases, by_agent, failure_counts),
    )


def _case_hints(case: CaseDiagnostic) -> list[str]:
    hints: list[str] = []
    if case.failure_stage == "tool":
        hints.append("Add a tool schema regression case and validate tool args before LLM handoff.")
    elif case.failure_stage == "safety":
        hints.append("Review guardrail threshold and add HITL policy coverage for this action.")
    elif case.failure_stage == "budget":
        hints.append("Reduce retrieved context, enable cache reuse, or route to a cheaper model.")
    elif case.failure_stage == "llm":
        hints.append("Add provider fallback coverage and capture the failing model response in traces.")
    elif case.failure_stage == "memory":
        hints.append("Check retrieval filters, graph expansion depth, and context packing efficiency.")
    elif case.failure_stage == "planner":
        hints.append("Add a DAG validation case and simplify dependencies between subtasks.")
    elif case.failure_stage == "communication":
        hints.append("Inspect predecessor outputs and add handoff contract assertions between agents.")
    elif case.failure_stage == "quality":
        hints.append("Use this case for prompt patch generation and compare against the baseline.")

    if case.tool_errors:
        hints.append("Tool errors detected: add schema/time-out evals and validate arguments before execution.")
    if case.guardrail_hits:
        hints.append("Guardrail activity detected: review blocked step context and HITL routing policy.")
    if case.handoff_count and not case.passed:
        hints.append("Failed after inter-agent handoff: check summary quality and required context fields.")
    if case.tokens > 12_000:
        hints.append("High token use: summarize history and cap GraphRAG/vector context for this case.")
    if case.steps > 20:
        hints.append("High step count: inspect loop detector signals and add a stop condition.")
    return hints


def _global_hints(
    cases: list[CaseDiagnostic],
    by_agent: dict[str, AgentMetricSummary],
    failure_counts: dict[str, int],
) -> list[str]:
    hints: list[str] = []
    if not cases:
        return ["No cases were evaluated; add smoke, regression, and adversarial eval sets."]

    worst_agent = max(by_agent.values(), key=lambda s: s.total_tokens, default=None)
    if worst_agent and worst_agent.total_tokens:
        hints.append(
            f"`{worst_agent.agent_type}` used the most tokens "
            f"({worst_agent.total_tokens}); inspect prompts, retrieval budget, and cache hit rate."
        )

    most_common_failure = max(failure_counts.items(), key=lambda item: item[1], default=None)
    if most_common_failure:
        stage, count = most_common_failure
        hints.append(
            f"Most common failure stage is `{stage}` ({count}); prioritize evals and telemetry there."
        )

    low_success = [s for s in by_agent.values() if s.cases and s.success_rate < 0.8]
    for summary in low_success:
        hints.append(
            f"`{summary.agent_type}` pass rate is {summary.success_rate:.1%}; gate deployments on this suite."
        )

    high_cost = [s for s in by_agent.values() if s.avg_cost_usd > 0.05]
    for summary in high_cost:
        hints.append(
            f"`{summary.agent_type}` average cost is ${summary.avg_cost_usd:.4f}; add router rules for cheap/simple tasks."
        )

    tool_error_agents = [s for s in by_agent.values() if s.tool_errors]
    for summary in tool_error_agents:
        hints.append(
            f"`{summary.agent_type}` had {summary.tool_errors}/{summary.tool_calls} tool errors; tighten schemas and add tool contract tests."
        )

    guardrail_agents = [s for s in by_agent.values() if s.guardrail_hits]
    for summary in guardrail_agents:
        hints.append(
            f"`{summary.agent_type}` triggered {summary.guardrail_hits} guardrail checks; review policy thresholds and HITL queues."
        )

    handoff_agents = [s for s in by_agent.values() if s.handoff_count and s.success_rate < 1.0]
    for summary in handoff_agents:
        hints.append(
            f"`{summary.agent_type}` receives {summary.handoff_count} handoffs; add communication contract evals for predecessor outputs."
        )

    uncached_heavy = [
        s for s in by_agent.values()
        if s.total_tokens > 10_000 and s.cache_hits == 0
    ]
    for summary in uncached_heavy:
        hints.append(
            f"`{summary.agent_type}` is token-heavy with no cache hits; enable prompt/context caching for repeated evals."
        )

    if not hints:
        hints.append("No major bottleneck detected; keep this suite as a release gate.")
    return hints
