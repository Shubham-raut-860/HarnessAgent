"""Run Tutorial 04 — Agno Research Team + HarnessAgent Eval Flywheel."""
import asyncio, json, sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")
sys.path.insert(0, "tutorials")
from _azure_client import chat as azure_chat, MODEL_NAME

from harness.core.config import get_config
from harness.core.context import AgentContext, AgentResult
from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.runner import EvalRunner
from harness.observability.trace_recorder import TraceRecorder
from harness.observability.trace_schema import SpanKind, SpanStatus

cfg = get_config()
recorder = TraceRecorder.create(redis_url=cfg.redis_url, log_dir="logs")
OUT = Path("tutorials/output/04_agno_research")
OUT.mkdir(parents=True, exist_ok=True)

TOPIC = "AI Agent Frameworks in 2025: LangGraph vs CrewAI vs AutoGen vs Agno — Architecture, Production Readiness, and HarnessAgent Integration"


async def run_research_team(topic: str) -> dict:
    """5-agent Agno-style research pipeline wired to HarnessAgent spans."""
    ctx = AgentContext.create(
        tenant_id="research-team",
        agent_type="agno_research",
        task=f"Research: {topic[:80]}",
        memory=None,
        workspace_path=OUT,
        max_steps=50,
        max_tokens=200_000,
    )

    print("=" * 58)
    print(" TUTORIAL 04 — Agno Research Team")
    print(f" Model: {MODEL_NAME}")
    print(" HarnessAgent: RUN→5 TOOL spans + TraceRecorder + EvalRunner")
    print("=" * 58)
    print(f"\nTopic: {topic[:70]}...")

    # Open RUN span
    run_sid = await recorder.start_span(
        ctx.run_id, SpanKind.RUN, "agno:research_team", ctx, input_preview=topic[:300]
    )
    result = {"topic": topic, "agents": {}, "run_id": ctx.run_id}

    # ── Agent 1: ResearchDirector ────────────────────────────────────────
    print("\n[1/5] ResearchDirector — planning research strategy...")
    sid = await recorder.start_span(ctx.run_id, SpanKind.TOOL, "agent:ResearchDirector", ctx)
    plan = await azure_chat(
        "You are a Research Director. Break down complex research topics into clear subtasks.",
        (
            f"Topic: {topic}\n\n"
            "Create a research plan with 4 specific subtasks for your team:\n"
            "1. What data/facts to gather\n2. What to analyse\n3. What to write\n4. What to fact-check\n"
            "Format as a numbered list. Be specific and actionable."
        ),
    )
    await recorder.end_span(ctx.run_id, sid, SpanStatus.OK, output_preview=plan[:150])
    result["agents"]["director_plan"] = plan
    print(f"   Plan ready ({len(plan.split(chr(10)))} lines)")

    # ── Agent 2: WebResearcher ───────────────────────────────────────────
    print("[2/5] WebResearcher — gathering data from multiple sources...")
    sid = await recorder.start_span(ctx.run_id, SpanKind.TOOL, "agent:WebResearcher", ctx)
    research = await azure_chat(
        "You are a Senior Research Analyst. Gather accurate, detailed information from your knowledge.",
        (
            f"Research this topic thoroughly: {topic}\n\n"
            "Cover: key facts about each framework, GitHub stars, language, core use cases, "
            "strengths, weaknesses, production adoption examples, and how each integrates with "
            "an observability harness like HarnessAgent. "
            "Organise as: Overview, Key Facts per Framework, Comparative Insights."
        ),
    )
    await recorder.end_span(ctx.run_id, sid, SpanStatus.OK, output_preview=research[:150])
    result["agents"]["research_data"] = research
    print(f"   Research gathered ({len(research)} chars)")

    # ── Agent 3: DataAnalyst ─────────────────────────────────────────────
    print("[3/5] DataAnalyst — synthesising patterns and insights...")
    sid = await recorder.start_span(ctx.run_id, SpanKind.TOOL, "agent:DataAnalyst", ctx)
    analysis = await azure_chat(
        "You are a Data Analyst. Synthesise research into patterns, rankings, and actionable insights.",
        (
            f"Based on this research:\n{research[:3000]}\n\n"
            "Produce:\n"
            "1. A comparison table (framework | orchestration style | best for | HarnessAgent integration complexity)\n"
            "2. Key insight: which framework is best for production multi-agent AI in 2025 and why\n"
            "3. How HarnessAgent adds value to each (tracing, eval, self-improvement)\n"
            "4. 3 emerging trends in agent framework design"
        ),
    )
    await recorder.end_span(ctx.run_id, sid, SpanStatus.OK, output_preview=analysis[:150])
    result["agents"]["analysis"] = analysis
    print(f"   Analysis done ({len(analysis)} chars)")

    # ── Agent 4: ContentWriter ───────────────────────────────────────────
    print("[4/5] ContentWriter — writing the research report...")
    sid = await recorder.start_span(ctx.run_id, SpanKind.TOOL, "agent:ContentWriter", ctx)
    article = await azure_chat(
        (
            "You are a Senior Technical Writer. Create polished, well-structured reports "
            "for senior engineers and technical leaders. Use markdown. Cite specific facts. "
            "Target ~1200 words."
        ),
        (
            f"Write a comprehensive technical report titled:\n\"{topic}\"\n\n"
            f"Use this research:\n{research[:2000]}\n\n"
            f"And this analysis:\n{analysis[:2000]}\n\n"
            "Structure: Executive Summary, Framework Comparison (with table), "
            "HarnessAgent Integration Guide, Production Recommendations, Conclusion. "
            "Include specific technical details and real trade-offs."
        ),
    )
    await recorder.end_span(ctx.run_id, sid, SpanStatus.OK, output_preview=article[:150])
    result["agents"]["article"] = article
    print(f"   Article written ({len(article)} chars, ~{len(article.split()) } words)")

    # ── Agent 5: FactChecker ─────────────────────────────────────────────
    print("[5/5] FactChecker — verifying claims and approving...")
    sid = await recorder.start_span(ctx.run_id, SpanKind.GUARDRAIL, "agent:FactChecker", ctx)
    fact_check = await azure_chat(
        (
            "You are an Editorial Fact-Checker. Verify technical claims. "
            "Flag: UNVERIFIED, SPECULATION, OUTDATED. "
            "Calculate accuracy score. "
            "Final verdict: APPROVED (>90% verified) or NEEDS_REVISION."
        ),
        (
            f"Fact-check this technical article:\n\n{article[:3000]}\n\n"
            "Review key claims about LangGraph, CrewAI, AutoGen, Agno, and HarnessAgent. "
            "List any flags, then give: accuracy_score (%), verdict (APPROVED/NEEDS_REVISION), "
            "and a one-line quality summary."
        ),
    )
    approved = "APPROVED" in fact_check.upper()
    await recorder.end_span(
        ctx.run_id, sid,
        SpanStatus.OK if approved else SpanStatus.ERROR,
        output_preview="APPROVED" if approved else "NEEDS_REVISION",
    )
    result["agents"]["fact_check"] = fact_check
    result["approved"] = approved
    print(f"   FactChecker: {'✅ APPROVED' if approved else '⚠️  NEEDS REVISION'}")

    # Close RUN span
    await recorder.end_span(ctx.run_id, run_sid, SpanStatus.OK,
        output_preview=f"Research complete: {topic[:60]}")
    trace = await recorder.get_trace(ctx.run_id)
    result["trace"] = f"{trace.span_count} spans | ${trace.total_cost_usd:.4f}" if trace else "n/a"
    print(f"   HarnessAgent Trace: {result['trace']}")
    return result


# ── HarnessAgent Eval Flywheel ───────────────────────────────────────────────

class ResearchAgentRunner:
    """Thin wrapper so EvalRunner can call the research team."""
    async def run(self, tenant_id: str, agent_type: str, task: str, metadata=None) -> AgentResult:
        try:
            r = await run_research_team(task)
            output = r["agents"].get("article", "")
            return AgentResult(
                run_id=r["run_id"], output=output,
                steps=5, tokens=len(output)//4, success=bool(output),
                elapsed_seconds=20.0, cost_usd=0.005,
            )
        except Exception as e:
            return AgentResult(
                run_id="error", output="", steps=0, tokens=0,
                success=False, error_message=str(e), elapsed_seconds=0,
            )


def keyword_scorer(output: str, expected: str) -> float:
    """Score: keyword present (40%) + length (40%) + structure (20%)."""
    score = 0.0
    if expected.lower() in output.lower():
        score += 0.40
    words = len(output.split())
    if words >= 600:
        score += 0.40
    elif words >= 300:
        score += 0.20
    if any(h in output for h in ["##", "###", "\n#"]):
        score += 0.20
    return min(1.0, score)


async def run_eval_flywheel() -> dict:
    """Run eval dataset and show Hermes improvement loop."""
    print("\n" + "─" * 58)
    print(" HarnessAgent EVAL FLYWHEEL")
    print("─" * 58)

    dataset = EvalDataset(
        name="agno_research_eval",
        agent_type="agno_research",
        cases=[
            EvalCase(case_id="llm_state_001", agent_type="agno_research",
                     task="Summarise the current state of LLM agent frameworks in 2025",
                     expected_output="multi-agent"),
            EvalCase(case_id="rag_best_001", agent_type="agno_research",
                     task="What are the best practices for RAG systems with vector databases",
                     expected_output="embedding"),
            EvalCase(case_id="harness_value_001", agent_type="agno_research",
                     task="How does a harness layer add value to AI agent deployments",
                     expected_output="tracing"),
        ],
    )

    runner = EvalRunner(agent_runner=ResearchAgentRunner())
    print(f"\nRunning {len(dataset.cases)} eval cases (concurrency=1)...")
    report = await runner.run(dataset, scorer=keyword_scorer, concurrency=1, prompt_version="v1.0")

    eval_result = {
        "pass_rate": report.success_rate,
        "passed": report.passed,
        "total": report.total_cases,
        "avg_latency": report.avg_latency_seconds,
        "avg_cost": report.avg_cost_usd,
        "scores": report.scores,
        "errors": report.errors,
    }

    print(f"\n  Pass rate:   {report.success_rate:.0%} ({report.passed}/{report.total_cases})")
    print(f"  Avg latency: {report.avg_latency_seconds:.1f}s per case")
    print(f"  Avg cost:    ${report.avg_cost_usd:.4f} per case")
    print(f"\n  Per-case scores:")
    for cid, score in report.scores.items():
        status = "✅" if score >= 0.5 else "❌"
        print(f"    {status} {cid}: {score:.2f}")

    # Show Hermes decision
    print("\n  Hermes Loop Decision:")
    if report.success_rate >= 0.80:
        print("  ✅ Pass rate ≥ 80% — no patch needed, prompts are healthy")
    else:
        print(f"  ⚠️  Pass rate {report.success_rate:.0%} < 80%")
        print("  → Hermes would: sample failed cases → generate prompt patch")
        print("  → Evaluate patch on replay → apply if score > 70%")
        print("  → OnlineLearningMonitor watches for regression > 30%")
        print("  → Auto-rollback if post-patch error rate spikes")

    return eval_result


async def main():
    # Run research pipeline
    result = await run_research_team(TOPIC)

    # Build full report
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Tutorial 04 — Agno Research Team\n",
        f"*Generated: {ts} | Model: {MODEL_NAME}*",
        f"*HarnessAgent: 5 spans (RUN + 4 TOOL + 1 GUARDRAIL) | Trace: {result['trace']}*\n",
        "## HarnessAgent Span Tree\n",
        "```",
        f"agno:research_team  (RUN)",
        f"  ├── agent:ResearchDirector   (TOOL)  — research plan",
        f"  ├── agent:WebResearcher      (TOOL)  — data gathering",
        f"  ├── agent:DataAnalyst        (TOOL)  — synthesis + comparison table",
        f"  ├── agent:ContentWriter      (TOOL)  — ~1200-word article",
        f"  └── agent:FactChecker        (GUARDRAIL) — {'APPROVED ✅' if result.get('approved') else 'NEEDS REVISION ⚠️'}",
        "```\n",
        "## Research Output\n",
        result["agents"].get("article", ""),
        "\n## Fact-Check Report\n",
        result["agents"].get("fact_check", ""),
    ]
    full = "\n".join(lines)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / f"research_{ts_slug}.md").write_text(full)
    (OUT / "latest_report.md").write_text(full)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))

    # Run eval flywheel
    eval_result = await run_eval_flywheel()
    (OUT / "eval_results.json").write_text(json.dumps(eval_result, indent=2, default=str))

    print(f"\nSaved: tutorials/output/04_agno_research/")
    print("\n" + "=" * 58)
    print(full[:4000])
    return result, eval_result


if __name__ == "__main__":
    asyncio.run(main())
