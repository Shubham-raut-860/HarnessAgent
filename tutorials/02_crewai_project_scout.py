"""
Tutorial 2 — CrewAI × HarnessAgent
=====================================
Best Open-Source Project Scout

Given a tech stack, business goal, and constraints, this crew researches,
evaluates, and recommends the single best open-source project to adopt.

Agents
------
1. TechAnalyst       — Deconstructs requirements into weighted evaluation criteria
2. ProjectResearcher — Finds 5–8 candidate projects with metadata
3. QualityEvaluator  — Scores every candidate on all criteria
4. ReportCompiler    — Produces final ranked recommendation with reasoning

Tasks flow
----------
analyse_requirements → research_projects → evaluate_projects → compile_report

HarnessAgent adds
-----------------
- Full run trace with TOOL spans per agent step
- Cost tracking per crew execution
- FailureTracker → Hermes improvement signal
- Eval harness to measure recommendation quality over time

Install
-------
pip install agent-haas[vector,observe] crewai crewai-tools
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── CrewAI ───────────────────────────────────────────────────────────────────
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, WebsiteSearchTool

# ── HarnessAgent ─────────────────────────────────────────────────────────────
import harness
from harness.core.config import get_config
from harness.core.context import AgentContext
from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.runner import EvalRunner
from harness.observability.failures import FailureTracker
from harness.observability.trace_recorder import TraceRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Input schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProjectScoutRequest:
    """Input for the Project Scout crew."""
    tech_stack:   list[str]       # e.g. ["Python", "FastAPI", "PostgreSQL"]
    business_goal: str            # e.g. "real-time observability for our ML platform"
    team_size:    int = 5
    budget:       str = "open-source only"
    constraints:  list[str] = field(default_factory=list)   # e.g. ["must have Docker support"]
    evaluation_criteria: list[str] = field(default_factory=lambda: [
        "GitHub stars and activity",
        "Documentation quality",
        "Community support",
        "Python integration",
        "Production readiness",
        "License (permissive preferred)",
    ])

    def to_brief(self) -> str:
        return (
            f"Tech Stack: {', '.join(self.tech_stack)}\n"
            f"Goal: {self.business_goal}\n"
            f"Team Size: {self.team_size} engineers\n"
            f"Budget: {self.budget}\n"
            f"Constraints: {', '.join(self.constraints) or 'None'}\n"
            f"Evaluation Criteria: {', '.join(self.evaluation_criteria)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build the CrewAI crew
# ─────────────────────────────────────────────────────────────────────────────

def build_project_scout_crew(request: ProjectScoutRequest, verbose: bool = True) -> Crew:
    """
    Assemble the four-agent Project Scout crew.

    The crew runs sequentially: each agent's output becomes context
    for the next agent's task.
    """
    # Optional search tools (works without API key — agents fall back to LLM knowledge)
    search_tools = []
    if os.getenv("SERPER_API_KEY"):
        search_tools = [SerperDevTool(), WebsiteSearchTool()]

    brief = request.to_brief()

    # ── Agent 1: Tech Analyst ────────────────────────────────────────────────
    tech_analyst = Agent(
        role="Senior Technology Analyst",
        goal=(
            "Deeply understand the team's technical requirements and translate them "
            "into a precise, weighted set of evaluation criteria that will guide "
            "objective project comparison."
        ),
        backstory=(
            "You have 15 years experience evaluating open-source tooling for "
            "engineering teams. You know what makes a project maintainable at scale "
            "and how to avoid dependency regret. You are methodical and unbiased."
        ),
        tools=search_tools,
        verbose=verbose,
        allow_delegation=False,
    )

    # ── Agent 2: Project Researcher ─────────────────────────────────────────
    researcher = Agent(
        role="Open-Source Project Researcher",
        goal=(
            "Identify 5–8 strong open-source project candidates that could solve "
            "the team's problem. For each project gather: name, GitHub URL, stars, "
            "last commit date, license, core language, brief description."
        ),
        backstory=(
            "You are obsessed with the open-source ecosystem. You know the hidden "
            "gems, the deprecated traps, and the actively-maintained winners. "
            "You never recommend projects that haven't been committed to in 6 months."
        ),
        tools=search_tools,
        verbose=verbose,
        allow_delegation=False,
    )

    # ── Agent 3: Quality Evaluator ──────────────────────────────────────────
    evaluator = Agent(
        role="Technical Due-Diligence Evaluator",
        goal=(
            "Score each candidate project on every evaluation criterion (1–10). "
            "Produce a comparison matrix. Identify trade-offs clearly. "
            "Select the single best recommendation with a confidence score."
        ),
        backstory=(
            "You have reviewed hundreds of open-source projects for Fortune 500 "
            "engineering teams. You never let hype bias your scoring and always "
            "consider the team's specific context."
        ),
        tools=search_tools,
        verbose=verbose,
        allow_delegation=False,
    )

    # ── Agent 4: Report Compiler ─────────────────────────────────────────────
    report_compiler = Agent(
        role="Technical Report Writer",
        goal=(
            "Compile a clear, executive-ready recommendation report with: "
            "top recommendation, full comparison table, adoption roadmap (3 steps), "
            "and risk assessment."
        ),
        backstory=(
            "You turn complex technical analyses into crisp, actionable documents "
            "that both CTOs and individual contributors can act on immediately."
        ),
        verbose=verbose,
        allow_delegation=False,
    )

    # ── Tasks ────────────────────────────────────────────────────────────────

    task_analyse = Task(
        description=(
            f"Analyse this engineering brief and produce weighted evaluation criteria:\n\n{brief}\n\n"
            "Deliverable: A numbered list of 6–8 criteria with weight (%) and what "
            "'excellent' looks like for each. Format as Markdown."
        ),
        expected_output="Markdown document with weighted evaluation criteria table.",
        agent=tech_analyst,
    )

    task_research = Task(
        description=(
            f"Given this brief:\n{brief}\n\n"
            "Research and list 6 open-source projects that could solve the problem. "
            "For each include: name, GitHub URL (if known), stars (approx), "
            "license, last active date, one-line description, key pros, key cons."
        ),
        expected_output="Markdown table of 6 candidate projects with metadata.",
        agent=researcher,
        context=[task_analyse],
    )

    task_evaluate = Task(
        description=(
            "Using the evaluation criteria and candidate list, score each project "
            "on a 1–10 scale for every criterion. "
            "Calculate a weighted total score. Rank all projects. "
            "Select the winner and state your confidence (%)."
        ),
        expected_output=(
            "Markdown comparison matrix + ranked list + winner declaration with confidence score."
        ),
        agent=evaluator,
        context=[task_analyse, task_research],
    )

    task_report = Task(
        description=(
            "Write the final recommendation report. Include:\n"
            "1. Executive Summary (3 sentences)\n"
            "2. Winner and why\n"
            "3. Full comparison table\n"
            "4. 3-step adoption roadmap\n"
            "5. Risk Assessment (top 3 risks + mitigations)\n"
            "6. Alternatives if the winner doesn't fit"
        ),
        expected_output="Complete Markdown recommendation report ready to share with the team.",
        agent=report_compiler,
        context=[task_analyse, task_research, task_evaluate],
    )

    # ── Assemble crew ────────────────────────────────────────────────────────
    crew = Crew(
        agents=[tech_analyst, researcher, evaluator, report_compiler],
        tasks=[task_analyse, task_research, task_evaluate, task_report],
        process=Process.sequential,
        verbose=verbose,
    )

    return crew


# ─────────────────────────────────────────────────────────────────────────────
# Run with HarnessAgent
# ─────────────────────────────────────────────────────────────────────────────

async def run_project_scout(
    request: ProjectScoutRequest,
    output_dir: Path = Path("output/project_scout"),
) -> str:
    """
    Execute the Project Scout crew wrapped in the full HarnessAgent harness.

    Returns the final recommendation report as a Markdown string.
    """
    cfg = get_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    recorder  = TraceRecorder.create(redis_url=cfg.redis_url, log_dir="logs")
    failure_tracker = FailureTracker(redis_client=None)

    crew   = build_project_scout_crew(request, verbose=True)
    adapter = harness.wrap(crew)
    adapter.attach_harness(safety_pipeline=None, cost_tracker=None, audit_logger=None)

    ctx = AgentContext.create(
        tenant_id="project-scout",
        agent_type="crewai_scout",
        task=f"Find best project for: {request.business_goal}",
        memory=None,
        workspace_path=output_dir,
        max_steps=200,
        max_tokens=1_000_000,
    )

    print(f"\n🚀 Starting Project Scout crew for: {request.business_goal}")
    print(f"   Tech stack: {', '.join(request.tech_stack)}\n")

    # Run crew with harness observability
    events = []
    async for event in adapter.run_with_harness(ctx, {"request": request.to_brief()}):
        events.append(event)

    # Extract result
    result = await adapter.get_result()
    report = str(result.output) if result else "No output generated."

    # Save report
    report_path = output_dir / "recommendation_report.md"
    report_path.write_text(report)
    print(f"\n✅ Report saved: {report_path}")

    # Print trace
    trace = await recorder.get_trace(ctx.run_id)
    if trace:
        print(f"📊 Trace: {trace.span_count} spans | ${trace.total_cost_usd:.4f} total cost")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation harness — measure recommendation quality over time
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_dataset() -> EvalDataset:
    """
    Evaluation dataset for the Project Scout crew.

    Each case has a known 'best' answer so we can measure whether the crew's
    recommendations improve after Hermes prompt patches.
    """
    return EvalDataset(
        name="project_scout_eval",
        agent_type="crewai_scout",
        cases=[
            EvalCase(
                case_id="obs_001",
                task="Find best observability tool for Python microservices",
                metadata={"tech_stack": ["Python", "FastAPI"], "goal": "distributed tracing"},
                expected_output="OpenTelemetry",   # should be in the recommendation
            ),
            EvalCase(
                case_id="vector_db_001",
                task="Find best vector database for RAG applications",
                metadata={"tech_stack": ["Python", "LangChain"], "goal": "semantic search"},
                expected_output="Qdrant",
            ),
            EvalCase(
                case_id="task_queue_001",
                task="Find best task queue for Python async workloads",
                metadata={"tech_stack": ["Python", "FastAPI", "Redis"], "goal": "background jobs"},
                expected_output="Celery",
            ),
        ],
    )


def custom_scorer(output: str, expected: str) -> float:
    """
    Score: 1.0 if expected project name appears in the output, else 0.0.
    In production, use an LLM judge for richer semantic scoring.
    """
    return 1.0 if expected.lower() in output.lower() else 0.0


async def run_evaluation(agent_runner: Any) -> None:
    """Run the evaluation suite and print results."""
    dataset = build_eval_dataset()
    runner  = EvalRunner(agent_runner=agent_runner)

    print("\n📋 Running Project Scout evaluation suite…")
    report = await runner.run(dataset, scorer=custom_scorer, concurrency=2)

    print(f"\nEval Results: {report.success_rate:.0%} pass rate "
          f"({report.passed}/{report.total_cases} cases)")
    print(report.to_markdown())


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

DEMO_REQUEST = ProjectScoutRequest(
    tech_stack=["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
    business_goal=(
        "Real-time distributed tracing and metrics for our ML inference platform. "
        "We need to track LLM call latency, token usage, and error rates across "
        "10+ microservices."
    ),
    team_size=6,
    budget="open-source only",
    constraints=[
        "Must support Python OpenTelemetry SDK",
        "Must have a self-hosted option (no vendor lock-in)",
        "Docker Compose deployment required",
        "Active community (commits in last 90 days)",
    ],
)


if __name__ == "__main__":
    report = asyncio.run(run_project_scout(DEMO_REQUEST))
    print("\n" + "=" * 60)
    print(textwrap.shorten(report, width=3000, placeholder="…"))
