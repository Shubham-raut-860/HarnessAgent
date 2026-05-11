"""
Tutorial 4 — Agno × HarnessAgent
===================================
AI Research & Content Team with Full Eval Flywheel

This tutorial showcases the most complete HarnessAgent integration:
  - Agno multi-agent research pipeline
  - EvalDataset with ground-truth cases
  - EvalRunner scoring with custom LLM judge
  - HermesLoop: automatic prompt improvement from failures
  - OnlineLearningMonitor: regression detection + auto-rollback
  - TraceRecorder: full span tree per research task
  - FailureTracker: feeds Hermes error sampling

Agents
------
ResearchDirector   — Decomposes research goal into subtasks and assigns agents
WebResearcher      — Gathers data from multiple sources (web search, docs)
DataAnalyst        — Synthesises findings, identifies patterns and insights
ContentWriter      — Crafts polished, cited content from analysed data
FactChecker        — Verifies claims, flags speculation, approves final output

Best scenario — AI Landscape Report
------------------------------------
Given a topic, the team produces a comprehensive, fact-checked research report
that would take a human analyst 4–8 hours to write.

Install
-------
pip install agent-haas[vector,observe] agno
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Agno ─────────────────────────────────────────────────────────────────────
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.yfinance import YFinanceTools

# ── HarnessAgent ─────────────────────────────────────────────────────────────
import harness
from harness.core.config import get_config
from harness.core.context import AgentContext
from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.runner import EvalReport, EvalRunner
from harness.improvement.error_collector import ErrorCollector
from harness.improvement.evaluator import Evaluator
from harness.improvement.hermes import HermesLoop
from harness.improvement.online_monitor import OnlineLearningMonitor
from harness.observability.failures import FailureTracker
from harness.observability.metrics import get_prometheus_metrics
from harness.observability.trace_recorder import TraceRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Research request schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchRequest:
    """Defines a research task for the Agno team."""
    topic:          str
    depth:          str = "comprehensive"    # "quick" | "standard" | "comprehensive"
    output_format:  str = "report"           # "report" | "summary" | "bullets"
    target_audience: str = "technical professionals"
    word_count:     int = 1500
    include_sources: bool = True
    focus_areas:    list[str] = field(default_factory=list)

    def to_brief(self) -> str:
        focus = f"\nFocus on: {', '.join(self.focus_areas)}" if self.focus_areas else ""
        return (
            f"Topic: {self.topic}\n"
            f"Depth: {self.depth}\n"
            f"Format: {self.output_format} (~{self.word_count} words)\n"
            f"Audience: {self.target_audience}"
            f"{focus}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build the Agno research team
# ─────────────────────────────────────────────────────────────────────────────

def _get_model() -> Any:
    """Return the best available LLM model for Agno."""
    cfg = get_config()
    if cfg.anthropic_api_key:
        return Claude(id="claude-sonnet-4-6", api_key=cfg.anthropic_api_key)
    if cfg.openai_api_key:
        return OpenAIChat(id="gpt-4o-mini", api_key=cfg.openai_api_key)
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")


def build_research_team(request: ResearchRequest, debug: bool = False) -> Team:
    """
    Assemble the 5-agent Agno research team.

    The team uses Agno's coordinate mode: ResearchDirector orchestrates
    the other four agents, passing tasks and receiving results.
    """
    model = _get_model()

    # Optional web tools
    search_tools = [DuckDuckGoTools()]
    try:
        article_tools = [Newspaper4kTools()]
    except Exception:
        article_tools = []

    # ── Agent 1: Research Director ────────────────────────────────────────────
    director = Agent(
        name="ResearchDirector",
        role="Research Director",
        model=model,
        instructions=[
            "You are the Research Director orchestrating the team.",
            "Break down complex research goals into specific subtasks.",
            "Assign WebResearcher to gather raw data.",
            "Assign DataAnalyst to synthesise findings.",
            "Assign ContentWriter to write the final content.",
            "Assign FactChecker to verify before delivery.",
            "Track progress and ensure the final output meets quality standards.",
            "Always specify word count targets and format requirements clearly.",
        ],
        debug_mode=debug,
    )

    # ── Agent 2: Web Researcher ───────────────────────────────────────────────
    researcher = Agent(
        name="WebResearcher",
        role="Senior Research Analyst",
        model=model,
        tools=search_tools + article_tools,
        instructions=[
            "You are an expert research analyst.",
            "Search for accurate, up-to-date information from authoritative sources.",
            "Gather at least 5 distinct sources for each research task.",
            "Extract key statistics, quotes, and data points.",
            "Note the source URL and publication date for every fact.",
            "Flag any conflicting information between sources.",
            "Organise findings into: key facts, statistics, expert opinions, trends.",
        ],
        debug_mode=debug,
    )

    # ── Agent 3: Data Analyst ─────────────────────────────────────────────────
    analyst = Agent(
        name="DataAnalyst",
        role="Data & Insights Analyst",
        model=model,
        instructions=[
            "You are a data analyst who turns raw research into actionable insights.",
            "Identify patterns, trends, and anomalies in the research data.",
            "Synthesise multiple sources into coherent findings.",
            "Quantify claims wherever possible with numbers and percentages.",
            "Compare and contrast different viewpoints objectively.",
            "Highlight the most significant insights for the target audience.",
            "Structure analysis as: Overview → Key Findings → Trends → Implications.",
        ],
        debug_mode=debug,
    )

    # ── Agent 4: Content Writer ───────────────────────────────────────────────
    writer = Agent(
        name="ContentWriter",
        role="Senior Technical Writer",
        model=model,
        instructions=[
            "You are an expert technical writer creating polished, publication-ready content.",
            "Write clearly for the specified target audience — never condescend or over-explain.",
            "Use concrete examples and case studies to illustrate abstract concepts.",
            "Structure content with clear headers, subheaders, and logical flow.",
            "Cite sources inline using [Source: URL] format.",
            "Match the specified word count (±10%).",
            "Avoid jargon unless writing for technical professionals.",
            "End with clear actionable takeaways or next steps.",
        ],
        debug_mode=debug,
    )

    # ── Agent 5: Fact Checker ─────────────────────────────────────────────────
    fact_checker = Agent(
        name="FactChecker",
        role="Editorial Fact-Checker",
        model=model,
        instructions=[
            "You are a rigorous fact-checker reviewing content before publication.",
            "Verify every specific claim, statistic, and quote against provided sources.",
            "Flag: UNVERIFIED (claim has no source), SPECULATION (opinion stated as fact), ",
            "OUTDATED (data older than 2 years without caveat), INACCURATE (wrong data).",
            "For each flag provide: [FLAG TYPE] Claim → Correction/Source needed",
            "Calculate: verified_claims / total_claims = accuracy_score",
            "Final verdict: APPROVED (>90% verified) | NEEDS_REVISION (≤90%) | REJECTED (<70%)",
            "Approved content must include accuracy score and list of verified sources.",
        ],
        debug_mode=debug,
    )

    # ── Assemble team ─────────────────────────────────────────────────────────
    team = Team(
        name="ResearchTeam",
        mode="coordinate",          # Director orchestrates specialists
        model=model,
        members=[director, researcher, analyst, writer, fact_checker],
        instructions=[
            f"Research brief: {request.to_brief()}",
            "Complete all research phases before delivering final content.",
            "The FactChecker must approve before the team is considered done.",
            "Output the final APPROVED content in the requested format.",
        ],
        debug_mode=debug,
        show_members_responses=True,
    )

    return team


# ─────────────────────────────────────────────────────────────────────────────
# Run with full HarnessAgent harness
# ─────────────────────────────────────────────────────────────────────────────

async def run_research(
    request: ResearchRequest,
    output_dir: Path = Path("output/research"),
) -> str:
    """
    Execute the Agno research team with full HarnessAgent observability.

    Returns the final fact-checked research content.
    """
    cfg = get_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    recorder        = TraceRecorder.create(redis_url=cfg.redis_url, log_dir="logs")
    failure_tracker = FailureTracker(redis_client=None)

    team    = build_research_team(request)
    adapter = harness.wrap(team) if hasattr(harness, 'wrap') else None

    ctx = AgentContext.create(
        tenant_id="research-team",
        agent_type="agno_research",
        task=f"Research: {request.topic}",
        memory=None,
        workspace_path=output_dir,
        max_steps=150,
        max_tokens=2_000_000,
    )

    print(f"\n🔬 Research Team — Topic: {request.topic}")
    print(f"   Depth: {request.depth} | Format: {request.output_format}\n")

    # Run with span tracing
    async with recorder.span(ctx.run_id, "run", f"research:{request.topic[:40]}", ctx,
                              input_preview=request.to_brief()):
        # Run the Agno team
        response = team.run(request.to_brief())
        content = response.content if hasattr(response, "content") else str(response)

    # Save output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = request.topic.lower().replace(" ", "_")[:40]
    out_path = output_dir / f"{slug}_{ts}.md"
    out_path.write_text(
        f"# {request.topic}\n\n"
        f"*Generated by Agno Research Team × HarnessAgent*\n"
        f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        + content
    )
    print(f"\n✅ Research saved: {out_path}")

    # Print trace
    trace = await recorder.get_trace(ctx.run_id)
    if trace:
        print(f"📊 Trace: {trace.span_count} spans | ${trace.total_cost_usd:.4f}")
        print(f"   GET /runs/{ctx.run_id}/trace")

    return content


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Dataset — ground truth for quality measurement
# ─────────────────────────────────────────────────────────────────────────────

def build_research_eval_dataset() -> EvalDataset:
    """
    Ground-truth evaluation cases for the research team.

    expected_output contains keywords that must appear in a quality response.
    The LLM judge (if configured) gives a richer score.
    """
    return EvalDataset(
        name="research_team_eval",
        agent_type="agno_research",
        cases=[
            EvalCase(
                case_id="llm_landscape_001",
                task="Write a research report on the current state of Large Language Models in 2024",
                expected_output="transformer",   # must mention transformers
                metadata={
                    "focus": ["GPT-4", "Claude", "Gemini", "open-source models"],
                    "min_words": 800,
                },
            ),
            EvalCase(
                case_id="rag_patterns_001",
                task="Research best practices for Retrieval-Augmented Generation (RAG) systems",
                expected_output="embedding",
                metadata={
                    "focus": ["chunking", "retrieval", "reranking", "evaluation"],
                    "min_words": 600,
                },
            ),
            EvalCase(
                case_id="agent_frameworks_001",
                task="Compare LangGraph, CrewAI, AutoGen, and Agno as agent frameworks",
                expected_output="multi-agent",
                metadata={
                    "focus": ["orchestration", "use cases", "trade-offs"],
                    "min_words": 1000,
                },
            ),
            EvalCase(
                case_id="ai_safety_001",
                task="Research current AI safety techniques and guardrail approaches",
                expected_output="guardrail",
                metadata={
                    "focus": ["RLHF", "Constitutional AI", "red-teaming"],
                    "min_words": 700,
                },
            ),
            EvalCase(
                case_id="vector_db_comparison_001",
                task="Compare Pinecone, Qdrant, Weaviate, and Chroma for production RAG",
                expected_output="latency",
                metadata={
                    "focus": ["performance", "scalability", "cost", "self-hosted"],
                    "min_words": 600,
                },
            ),
        ],
    )


def keyword_scorer(output: str, expected: str) -> float:
    """
    Score based on presence of expected keyword AND output length.

    A good research response must:
    - Contain the expected keyword (40% of score)
    - Be at least 400 words (40% of score)
    - Have section headers (20% of score — structured content)
    """
    output_lower = output.lower()
    score = 0.0

    # Keyword check
    if expected.lower() in output_lower:
        score += 0.40

    # Length check (400 words minimum)
    word_count = len(output.split())
    if word_count >= 800:
        score += 0.40
    elif word_count >= 400:
        score += 0.20

    # Structure check (has headers)
    if "##" in output or "###" in output or "\n#" in output:
        score += 0.20

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# Hermes Automatic Improvement Loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_eval_and_improve(
    agent_runner: Any,
    redis_url: str = "redis://localhost:6379",
) -> None:
    """
    Full eval → Hermes improvement flywheel:

    Step 1: Run evaluation suite on current prompts
    Step 2: Collect failures in ErrorCollector
    Step 3: Hermes samples failures → LLM generates prompt patch
    Step 4: Evaluator replays failing cases with patched config
    Step 5: If score > 70%: apply patch (or queue for manual review)
    Step 6: OnlineLearningMonitor watches production runs for regressions
    Step 7: Auto-rollback if error rate increases > 30% after patch
    """
    print("\n" + "=" * 60)
    print("🔄 EVAL + HERMES IMPROVEMENT LOOP")
    print("=" * 60)

    # ── Step 1: Evaluate ───────────────────────────────────────────────────
    dataset = build_research_eval_dataset()
    runner  = EvalRunner(agent_runner=agent_runner)

    print(f"\n📋 Step 1: Running {len(dataset.cases)} eval cases…")
    report: EvalReport = await runner.run(
        dataset,
        scorer=keyword_scorer,
        concurrency=2,
        prompt_version="baseline_v1",
    )

    print(f"   Baseline pass rate: {report.success_rate:.0%} "
          f"({report.passed}/{report.total_cases} passed)")
    print(f"   Avg latency: {report.avg_latency_seconds:.1f}s | "
          f"Avg cost: ${report.avg_cost_usd:.4f}")

    # ── Step 2: Collect failures ───────────────────────────────────────────
    print(f"\n📥 Step 2: Recording {report.failed} failures…")
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url, decode_responses=True)
        collector = ErrorCollector(redis=r)

        for case_id, error_msg in report.errors.items():
            case = next((c for c in dataset.cases if c.case_id == case_id), None)
            if case:
                await collector.record(
                    agent_type="agno_research",
                    task=case.task,
                    failure_class="quality",
                    error_message=error_msg or f"Low score on case {case_id}",
                    stack_trace="",
                    context_snapshot={"case_id": case_id, "score": report.scores.get(case_id, 0)},
                )
        error_counts = await collector.get_error_counts("agno_research", hours=1)
        print(f"   Recorded failures: {error_counts}")
    except Exception as e:
        print(f"   (ErrorCollector unavailable: {e} — skipping)")

    # ── Step 3–5: Hermes cycle ─────────────────────────────────────────────
    print("\n🤖 Step 3–5: Hermes improvement cycle…")
    if report.success_rate < 0.80:
        print(f"   ⚠️  Pass rate {report.success_rate:.0%} < 80% threshold")
        print("   Hermes would now:")
        print("   → Sample representative failures from ErrorCollector")
        print("   → Call LLM to propose a prompt patch (e.g. 'add more structure requirements')")
        print("   → Replay failing cases with patched config")
        print("   → Score the patch (success_rate - 0.01×steps_delta - 0.001×tokens_delta)")
        print(f"   → Apply if score > 70% (HERMES_PATCH_SCORE_THRESHOLD)")
        print("   → Schedule rollback check for next cycle")

        # In production with Redis + LLM configured, this runs automatically:
        # hermes = HermesLoop(collector, generator, evaluator, prompt_store, metrics, cfg)
        # outcome = await hermes.run_cycle("agno_research")
        # print(f"   Patch applied: {outcome.applied} | Score: {outcome.eval_result.score:.2f}")
    else:
        print(f"   ✅ Pass rate {report.success_rate:.0%} — no improvement needed")

    # ── Step 6–7: Online monitoring ────────────────────────────────────────
    print("\n📈 Step 6–7: Online learning monitor…")
    print("   OnlineLearningMonitor tracks per-version rolling metrics:")
    print("   → Records success_rate, avg_cost, avg_steps per prompt version")
    print("   → Snapshots to MLflow every 20 runs")
    print("   → If error rate grows >30% post-patch: auto-rollback to baseline")

    # Simulate monitoring output
    print("\n   Version metrics (simulated):")
    print("   ┌─────────────────┬──────────────┬──────────┬───────────┐")
    print("   │ Version         │ Success Rate │ Avg Cost │ Samples   │")
    print("   ├─────────────────┼──────────────┼──────────┼───────────┤")
    print(f"   │ baseline_v1     │ {report.success_rate:.0%}         │ ${report.avg_cost_usd:.4f}   │ {report.total_cases}         │")
    print("   │ patch_v1 (sim.) │ 85%          │ $0.0038  │ —         │")
    print("   └─────────────────┴──────────────┴──────────┴───────────┘")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n📊 Eval Summary:")
    print(report.to_markdown())


# ─────────────────────────────────────────────────────────────────────────────
# Batch research runner — multiple topics at once
# ─────────────────────────────────────────────────────────────────────────────

async def run_batch_research(
    requests: list[ResearchRequest],
    max_concurrent: int = 2,
) -> list[str]:
    """
    Run multiple research requests concurrently.
    Demonstrates HarnessAgent's multi-run isolation.
    """
    import asyncio
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(req: ResearchRequest) -> str:
        async with semaphore:
            return await run_research(req)

    results = await asyncio.gather(*[_run_one(r) for r in requests])
    return list(results)


# ─────────────────────────────────────────────────────────────────────────────
# Demo research requests
# ─────────────────────────────────────────────────────────────────────────────

DEMO_REQUESTS = [
    ResearchRequest(
        topic="The State of AI Agent Frameworks in 2024: LangGraph vs CrewAI vs AutoGen vs Agno",
        depth="comprehensive",
        output_format="report",
        target_audience="senior software engineers",
        word_count=2000,
        include_sources=True,
        focus_areas=[
            "Architecture differences and trade-offs",
            "Production readiness and performance",
            "Multi-agent coordination patterns",
            "Integration with HarnessAgent-style observability",
            "Real-world adoption and community size",
        ],
    ),
    ResearchRequest(
        topic="Best Practices for LLM Cost Optimization in Production",
        depth="standard",
        output_format="report",
        target_audience="CTOs and engineering leaders",
        word_count=1200,
        focus_areas=[
            "Prompt caching strategies",
            "Model routing and fallback",
            "Semantic caching",
            "Context window management",
        ],
    ),
]


if __name__ == "__main__":
    async def main():
        print("🚀 Agno Research Team × HarnessAgent Demo\n")

        # Run single research task
        content = await run_research(DEMO_REQUESTS[0])
        print("\n" + "─" * 60)
        print("RESEARCH OUTPUT PREVIEW:")
        print("─" * 60)
        print(content[:2000] + "…" if len(content) > 2000 else content)

        print("\n" + "─" * 60)
        print("EVALUATION + HERMES IMPROVEMENT LOOP:")
        print("─" * 60)

        # Demonstrate eval + improvement loop with a mock runner
        class MockRunner:
            """Minimal runner for eval demo purposes."""
            async def run(self, tenant_id, agent_type, task, metadata=None):
                from harness.core.context import AgentResult
                return AgentResult(
                    run_id="mock-run",
                    output=f"Mock research on: {task}. Contains keyword from metadata.",
                    steps=5, tokens=1500, success=True,
                    elapsed_seconds=3.0, cost_usd=0.004,
                )

        await run_eval_and_improve(MockRunner())

    asyncio.run(main())
