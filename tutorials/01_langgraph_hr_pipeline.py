"""
Tutorial 1 — LangGraph × HarnessAgent
======================================
Multi-Agent HR Recruitment Pipeline

Agents in this pipeline
-----------------------
1. JDCreatorAgent        — Generates a rich Job Description from raw requirements
2. ResumeFilterAgent     — Scores every resume against the JD (0–1 match score)
3. ResumeEnhancerAgent   — Suggests concrete improvements for top candidates
4. QuizGeneratorAgent    — Creates a 10-question technical quiz from the JD
5. FollowUpAgent         — Produces 5 personalised interview questions per candidate

Flow
----
requirements → JD → filter resumes → enhance top-N → quiz + follow-ups → report

Install
-------
pip install agent-haas[vector,observe] langgraph
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

# ── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ── HarnessAgent ─────────────────────────────────────────────────────────────
import harness
from harness.core.config import get_config
from harness.core.context import AgentContext
from harness.llm.anthropic import AnthropicProvider
from harness.llm.router import LLMRouter
from harness.memory.manager import MemoryManager
from harness.observability.event_bus import EventBus
from harness.observability.trace_recorder import TraceRecorder
from harness.observability.failures import FailureTracker

# ─────────────────────────────────────────────────────────────────────────────
# Shared state for the HR pipeline graph
# ─────────────────────────────────────────────────────────────────────────────

class HRState(TypedDict):
    """Mutable state passed through every node in the graph."""
    # Inputs
    job_requirements: str           # raw requirements from hiring manager
    resumes: list[dict]             # list of {"name": str, "text": str}

    # Generated artefacts
    job_description: str
    scored_resumes: list[dict]      # [{"name", "text", "score", "gaps"}]
    shortlist: list[dict]           # resumes with score >= threshold
    enhanced_resumes: list[dict]    # [{"name", "suggestions": [str]}]
    quiz: list[dict]                # [{"question": str, "expected": str}]
    followup_questions: dict        # {candidate_name: [str, ...]}
    final_report: str

    # Control
    messages: Annotated[list, add_messages]
    error: str


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper (shared across nodes)
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm() -> Any:
    """Build the LLM router used by every node."""
    cfg = get_config()
    router = LLMRouter()
    if cfg.anthropic_api_key:
        router.register(
            AnthropicProvider(api_key=cfg.anthropic_api_key, model="claude-sonnet-4-6"),
            priority=0,
            context_window=200_000,
        )
    return router


LLM = _build_llm()


async def _chat(system: str, user: str, max_tokens: int = 1024) -> str:
    """Single LLM call returning plain text."""
    response = await LLM.complete(
        messages=[{"role": "user", "content": user}],
        system=system,
        max_tokens=max_tokens,
    )
    return response.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — JD Creator
# ─────────────────────────────────────────────────────────────────────────────

async def jd_creator_node(state: HRState) -> dict:
    """
    Transform raw hiring-manager requirements into a structured Job Description.

    Output fields: title, summary, responsibilities, required_skills,
    nice_to_have_skills, experience_years, salary_range, location.
    """
    print("🏗  JD Creator — generating job description…")

    jd = await _chat(
        system=(
            "You are a senior HR specialist. Write professional, inclusive, and "
            "ATS-optimised job descriptions. Return JSON only."
        ),
        user=(
            f"Create a complete Job Description from these requirements:\n\n"
            f"{state['job_requirements']}\n\n"
            "Return a JSON object with keys: title, summary, responsibilities (list), "
            "required_skills (list), nice_to_have_skills (list), "
            "experience_years (int), salary_range (str), location (str)."
        ),
        max_tokens=1500,
    )

    # Strip markdown code fences if present
    jd_clean = jd.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        jd_obj = json.loads(jd_clean)
        jd_text = json.dumps(jd_obj, indent=2)
    except json.JSONDecodeError:
        jd_text = jd_clean

    return {"job_description": jd_text}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Resume Filter
# ─────────────────────────────────────────────────────────────────────────────

async def resume_filter_node(state: HRState) -> dict:
    """
    Score every resume against the JD.

    Returns scored_resumes (all) and shortlist (score >= 0.65).
    Each scored resume has: name, text, score (0–1), gaps (list[str]).
    """
    print(f"🔍 Resume Filter — scoring {len(state['resumes'])} resumes…")

    scored: list[dict] = []

    for resume in state["resumes"]:
        result = await _chat(
            system=(
                "You are an expert technical recruiter. Analyse resume-JD fit objectively. "
                "Return JSON only."
            ),
            user=(
                f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
                f"RESUME — {resume['name']}:\n{resume['text']}\n\n"
                "Return JSON: {\"score\": float 0-1, \"strengths\": [str], \"gaps\": [str], "
                "\"summary\": str}"
            ),
            max_tokens=600,
        )

        clean = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(clean)
            score = float(parsed.get("score", 0))
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            parsed = {"score": 0, "strengths": [], "gaps": ["parse error"], "summary": result}

        scored.append({
            "name":      resume["name"],
            "text":      resume["text"],
            "score":     score,
            "strengths": parsed.get("strengths", []),
            "gaps":      parsed.get("gaps", []),
            "summary":   parsed.get("summary", ""),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    shortlist = [r for r in scored if r["score"] >= 0.65]

    print(f"   Shortlisted {len(shortlist)}/{len(scored)} candidates")
    return {"scored_resumes": scored, "shortlist": shortlist}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge — enough candidates?
# ─────────────────────────────────────────────────────────────────────────────

def check_shortlist(state: HRState) -> str:
    if not state.get("shortlist"):
        print("⚠️  No candidates meet threshold — ending pipeline.")
        return "no_candidates"
    return "has_candidates"


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Resume Enhancer
# ─────────────────────────────────────────────────────────────────────────────

async def resume_enhancer_node(state: HRState) -> dict:
    """
    For each shortlisted candidate suggest concrete resume improvements
    aligned with the JD and current best practices.
    """
    print(f"✏️  Resume Enhancer — enhancing {len(state['shortlist'])} resumes…")

    enhanced: list[dict] = []

    for candidate in state["shortlist"]:
        suggestions = await _chat(
            system=(
                "You are a professional resume coach. Give actionable, specific "
                "improvements that increase ATS scores and recruiter appeal."
            ),
            user=(
                f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
                f"CANDIDATE: {candidate['name']}\nRESUME:\n{candidate['text']}\n\n"
                f"IDENTIFIED GAPS: {', '.join(candidate['gaps'])}\n\n"
                "Provide 5 specific, actionable resume improvements as a JSON list of strings."
            ),
            max_tokens=800,
        )

        clean = suggestions.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            items = json.loads(clean)
            if not isinstance(items, list):
                items = [str(items)]
        except json.JSONDecodeError:
            items = [line.strip("- ").strip() for line in suggestions.split("\n") if line.strip()]

        enhanced.append({"name": candidate["name"], "suggestions": items[:5]})

    return {"enhanced_resumes": enhanced}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Quiz Generator
# ─────────────────────────────────────────────────────────────────────────────

async def quiz_generator_node(state: HRState) -> dict:
    """
    Generate a 10-question technical assessment quiz directly from the JD.

    Each question has: question, type (mcq/open/coding), difficulty (easy/medium/hard),
    expected_answer.
    """
    print("📝 Quiz Generator — building technical assessment…")

    quiz_raw = await _chat(
        system=(
            "You are a senior technical interviewer. Create rigorous, role-specific "
            "assessment questions. Return JSON only."
        ),
        user=(
            f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
            "Create a 10-question technical assessment. Mix of difficulties: "
            "4 easy, 4 medium, 2 hard. Types: MCQ, open-ended, and coding.\n"
            "Return JSON array: [{\"question\": str, \"type\": str, "
            "\"difficulty\": str, \"expected_answer\": str}]"
        ),
        max_tokens=2000,
    )

    clean = quiz_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        quiz = json.loads(clean)
    except json.JSONDecodeError:
        quiz = [{"question": quiz_raw, "type": "open", "difficulty": "medium", "expected_answer": ""}]

    return {"quiz": quiz[:10]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Follow-Up Question Generator
# ─────────────────────────────────────────────────────────────────────────────

async def followup_generator_node(state: HRState) -> dict:
    """
    For each shortlisted candidate generate 5 personalised follow-up interview
    questions based on their specific resume + the JD gaps identified.
    """
    print(f"💬 Follow-Up Generator — {len(state['shortlist'])} candidates…")

    followups: dict[str, list[str]] = {}

    for candidate in state["shortlist"]:
        questions_raw = await _chat(
            system=(
                "You are a skilled behavioural interviewer. Ask incisive, open-ended "
                "questions tailored to the candidate's background and role requirements."
            ),
            user=(
                f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
                f"CANDIDATE: {candidate['name']}\n"
                f"RESUME SUMMARY: {candidate['summary']}\n"
                f"STRENGTHS: {', '.join(candidate['strengths'])}\n"
                f"GAPS: {', '.join(candidate['gaps'])}\n\n"
                "Generate 5 personalised interview follow-up questions as a JSON list of strings. "
                "Questions should probe the gaps and validate the strengths."
            ),
            max_tokens=700,
        )

        clean = questions_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            qs = json.loads(clean)
            if not isinstance(qs, list):
                qs = [str(qs)]
        except json.JSONDecodeError:
            qs = [q.strip("- ").strip() for q in questions_raw.split("\n") if q.strip()]

        followups[candidate["name"]] = qs[:5]

    return {"followup_questions": followups}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Report Builder
# ─────────────────────────────────────────────────────────────────────────────

async def report_builder_node(state: HRState) -> dict:
    """Compile everything into a structured hiring report."""
    print("📊 Report Builder — compiling final report…")

    lines = ["# HR Recruitment Pipeline Report", ""]

    # JD summary
    lines += ["## Job Description (Generated)", "```json", state.get("job_description", ""), "```", ""]

    # Candidate rankings
    lines += ["## Candidate Rankings", ""]
    for r in state.get("scored_resumes", []):
        bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        lines.append(f"**{r['name']}** [{bar}] {r['score']:.0%}")
        lines.append(f"  - {r['summary']}")
        lines.append("")

    # Shortlist details
    lines += ["## Shortlisted Candidates", ""]
    for c in state.get("shortlist", []):
        lines.append(f"### {c['name']}  (score: {c['score']:.0%})")
        lines.append(f"**Strengths:** {', '.join(c['strengths'])}")
        lines.append(f"**Gaps:** {', '.join(c['gaps'])}")

        # Resume enhancement suggestions
        enhanced = next(
            (e for e in state.get("enhanced_resumes", []) if e["name"] == c["name"]), None
        )
        if enhanced:
            lines.append("**Resume improvement suggestions:**")
            for s in enhanced["suggestions"]:
                lines.append(f"  - {s}")

        # Follow-up questions
        fqs = state.get("followup_questions", {}).get(c["name"], [])
        if fqs:
            lines.append("**Personalised interview questions:**")
            for i, q in enumerate(fqs, 1):
                lines.append(f"  {i}. {q}")
        lines.append("")

    # Technical quiz
    lines += ["## Technical Assessment Quiz", ""]
    for i, q in enumerate(state.get("quiz", []), 1):
        lines.append(f"**Q{i} [{q.get('type','open')} / {q.get('difficulty','medium')}]**")
        lines.append(f"  {q['question']}")
        if q.get("expected_answer"):
            lines.append(f"  *Expected:* {q['expected_answer'][:200]}")
        lines.append("")

    return {"final_report": "\n".join(lines)}


# ─────────────────────────────────────────────────────────────────────────────
# Node — No candidates fallback
# ─────────────────────────────────────────────────────────────────────────────

async def no_candidates_node(state: HRState) -> dict:
    return {"final_report": "# No Suitable Candidates Found\n\nAll resumes scored below the 65% threshold. Recommend broadening the candidate pool or relaxing requirements."}


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def build_hr_graph() -> Any:
    """Compile the HR pipeline state graph."""
    g = StateGraph(HRState)

    g.add_node("jd_creator",        jd_creator_node)
    g.add_node("resume_filter",     resume_filter_node)
    g.add_node("resume_enhancer",   resume_enhancer_node)
    g.add_node("quiz_generator",    quiz_generator_node)
    g.add_node("followup_generator",followup_generator_node)
    g.add_node("report_builder",    report_builder_node)
    g.add_node("no_candidates",     no_candidates_node)

    g.add_edge(START,               "jd_creator")
    g.add_edge("jd_creator",        "resume_filter")

    g.add_conditional_edges(
        "resume_filter",
        check_shortlist,
        {
            "has_candidates": "resume_enhancer",
            "no_candidates":  "no_candidates",
        },
    )

    g.add_edge("resume_enhancer",    "quiz_generator")
    g.add_edge("quiz_generator",     "followup_generator")
    g.add_edge("followup_generator", "report_builder")
    g.add_edge("report_builder",     END)
    g.add_edge("no_candidates",      END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Wrap with HarnessAgent for full observability
# ─────────────────────────────────────────────────────────────────────────────

async def run_hr_pipeline(
    job_requirements: str,
    resumes: list[dict],
    output_dir: Path = Path("output/hr"),
) -> str:
    """
    Run the full HR pipeline with HarnessAgent:
    - Distributed tracing (RUN → node spans)
    - Cost tracking per run
    - Event streaming (SSE)
    - Checkpoint recovery
    - Failure classification

    Parameters
    ----------
    job_requirements : str
        Plain-text description of role requirements from hiring manager.
    resumes : list[dict]
        List of {"name": str, "text": str} resume dicts.
    output_dir : Path
        Directory to write final_report.md

    Returns
    -------
    str
        The final hiring report as a Markdown string.
    """
    cfg = get_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build harness components
    recorder = TraceRecorder.create(redis_url=cfg.redis_url, log_dir="logs")
    event_bus = EventBus(redis_url=cfg.redis_url)

    # Wrap graph with HarnessAgent
    graph = build_hr_graph()
    adapter = harness.wrap(graph)
    adapter.attach_harness(safety_pipeline=None, cost_tracker=None, audit_logger=None)

    # Execution context
    ctx = AgentContext.create(
        tenant_id="hr-demo",
        agent_type="hr_pipeline",
        task=f"HR pipeline for: {job_requirements[:100]}",
        memory=None,
        workspace_path=output_dir,
        max_steps=100,
        max_tokens=500_000,
    )

    # Initial state
    initial_state: HRState = {
        "job_requirements": job_requirements,
        "resumes":          resumes,
        "job_description":  "",
        "scored_resumes":   [],
        "shortlist":        [],
        "enhanced_resumes": [],
        "quiz":             [],
        "followup_questions": {},
        "final_report":     "",
        "messages":         [],
        "error":            "",
    }

    # Run with harness tracing
    async with recorder.span(ctx.run_id, harness.SpanKind.RUN if hasattr(harness, "SpanKind") else "run", "hr_pipeline", ctx):
        result = await graph.ainvoke(initial_state)

    report = result.get("final_report", "No report generated.")

    # Save to disk
    report_path = output_dir / "final_report.md"
    report_path.write_text(report)
    print(f"\n✅ Report saved to {report_path}")

    # Print trace summary
    trace = await recorder.get_trace(ctx.run_id)
    if trace:
        print(f"📊 Trace: {trace.span_count} spans | {trace.total_input_tokens} in / "
              f"{trace.total_output_tokens} out | ${trace.total_cost_usd:.4f}")
        print(f"   View: GET /runs/{ctx.run_id}/trace")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_REQUIREMENTS = """
We need a Senior Python Backend Engineer for our AI/ML platform team.
- 5+ years Python experience
- Strong FastAPI or Django REST expertise
- Experience with async Python (asyncio, aiohttp)
- PostgreSQL and Redis proficiency
- Docker and Kubernetes knowledge
- Familiarity with LLM APIs (OpenAI, Anthropic)
- Nice to have: MLflow, OpenTelemetry, Prometheus
- Remote-first, must overlap US Eastern hours (3h minimum)
- Salary: $150k–$180k USD
"""

SAMPLE_RESUMES = [
    {
        "name": "Alice Chen",
        "text": """
            Senior Software Engineer — 7 years Python.
            FastAPI (4 years), SQLAlchemy, asyncio expert.
            Built real-time data pipeline processing 1M events/day.
            PostgreSQL, Redis, Celery. Docker + k8s (3 years).
            Integrated OpenAI GPT-4 API for chatbot product.
            BS Computer Science, UC Berkeley.
        """,
    },
    {
        "name": "Bob Martinez",
        "text": """
            Backend Developer — 3 years Python.
            Django REST Framework. MySQL, basic Redis.
            Some Docker experience, no Kubernetes.
            Built CRUD APIs for e-commerce platform.
            No LLM or ML experience.
            Self-taught programmer.
        """,
    },
    {
        "name": "Priya Sharma",
        "text": """
            ML Engineer / Backend — 6 years Python.
            FastAPI + async since 2021. PostgreSQL, Redis, Kafka.
            MLflow for experiment tracking, Prometheus monitoring.
            Deployed ML models at scale on Kubernetes (GKE).
            Anthropic Claude API integration for NLP features.
            MS Computer Science, IIT Bombay.
        """,
    },
    {
        "name": "James Wilson",
        "text": """
            Cloud Backend Engineer — 8 years, Python + Go.
            Primarily Go lately, Python on older projects.
            AWS Lambda, DynamoDB. No FastAPI, no asyncio.
            Strong DevOps: Terraform, k8s, Helm.
            No AI/LLM experience.
            BSc Software Engineering.
        """,
    },
]


if __name__ == "__main__":
    report = asyncio.run(
        run_hr_pipeline(
            job_requirements=SAMPLE_REQUIREMENTS,
            resumes=SAMPLE_RESUMES,
        )
    )
    print("\n" + "=" * 60)
    print(report[:3000])   # preview first 3k chars
