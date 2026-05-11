# HarnessAgent Tutorials

Four production-ready tutorials — one per major agent framework — showing how to
wrap real-world multi-agent workflows with `agent-haas` for tracing, safety,
cost tracking, evaluation, and self-improvement.

---

## Tutorials

| # | Framework | Use Case | Key HarnessAgent Features |
|---|-----------|----------|--------------------------|
| 01 | **LangGraph** | HR Recruitment Pipeline | TraceRecorder, span tree, event streaming |
| 02 | **CrewAI** | Best Open-Source Project Scout | EvalDataset, custom scorer, cost tracking |
| 03 | **AutoGen** | Multi-Agent Customer Support Team | HITL, FailureTracker, audit log |
| 04 | **Agno** | AI Research Team + Eval Flywheel | Full Hermes loop, OnlineLearningMonitor |

---

## 01 — LangGraph: HR Recruitment Pipeline

**File:** `01_langgraph_hr_pipeline.py`

```
requirements → JD Creator → Resume Filter → Resume Enhancer
                                                    ↓
                                         Quiz Generator → Follow-Up Generator → Report
```

**What it does:**
- `JDCreatorAgent` turns raw hiring requirements into a structured, ATS-optimised Job Description
- `ResumeFilterAgent` scores every resume against the JD (0–1 match score) and shortlists candidates ≥ 65%
- `ResumeEnhancerAgent` gives each shortlisted candidate 5 specific resume improvement suggestions
- `QuizGeneratorAgent` creates a 10-question technical quiz (mixed difficulty) from the JD
- `FollowUpAgent` generates 5 personalised interview questions per candidate based on resume gaps

**Run:**
```bash
pip install agent-haas[observe] langgraph
python tutorials/01_langgraph_hr_pipeline.py
```

**Output:** `output/hr/final_report.md` — full hiring report with rankings, quiz, and follow-up questions

---

## 02 — CrewAI: Best Open-Source Project Scout

**File:** `02_crewai_project_scout.py`

```
TechAnalyst → ProjectResearcher → QualityEvaluator → ReportCompiler
```

**What it does:**
- Takes a tech stack, business goal, and constraints
- Four sequential agents research, evaluate, and rank 6+ open-source projects
- Produces a Markdown report with comparison matrix, winner recommendation, adoption roadmap, and risk assessment

**Evaluation harness:**
```python
dataset = build_eval_dataset()   # 3 known-answer cases
runner  = EvalRunner(agent_runner)
report  = await runner.run(dataset, scorer=custom_scorer)
```

**Run:**
```bash
pip install agent-haas[observe] crewai crewai-tools
# Optional: export SERPER_API_KEY=... for live web search
python tutorials/02_crewai_project_scout.py
```

**Output:** `output/project_scout/recommendation_report.md`

---

## 03 — AutoGen: Multi-Agent Customer Support Team

**File:** `03_autogen_support_team.py`

```
Customer → TriageAgent → TechSupport / Billing / Escalation → QAReviewer → Customer
```

**What it does:**
- `TriageAgent` categorises ticket as TECHNICAL / BILLING / GENERAL / ESCALATION and routes it
- Specialist agents handle each category with deep domain knowledge
- `QAReviewer` checks tone, accuracy, completeness, and policy compliance before sending
- Custom `speaker_selection_func` routes conversation dynamically based on ticket state
- HITL gate: refunds > $100 or enterprise escalations pause for human approval

**Ticket examples included:**
1. API 503 intermittent errors (TECHNICAL — Pro tier)
2. Double billing charge (BILLING — Free tier, refund request)
3. Enterprise production outage — $15k/hour impact (ESCALATION — critical)

**Run:**
```bash
pip install agent-haas[observe] pyautogen
python tutorials/03_autogen_support_team.py
```

**Output:** `output/support/{ticket_id}_resolution.txt` per ticket

---

## 04 — Agno: Research Team + Full Eval Flywheel

**File:** `04_agno_research_team.py`

```
ResearchDirector
  ├── WebResearcher    (DuckDuckGo + article tools)
  ├── DataAnalyst      (pattern recognition, synthesis)
  ├── ContentWriter    (polished, cited content)
  └── FactChecker      (verify claims, approval gate)
```

**What it does:**
- 5-agent coordinated team produces fact-checked, cited research reports
- Covers any topic: AI frameworks, vector databases, safety techniques, cost optimisation
- FactChecker must APPROVE (>90% verified) before the team is done

**Full Hermes improvement loop:**
```
Eval cases → EvalRunner → FailureTracker → HermesLoop
    ↓                                           ↓
Score report              Patches bad prompts from failures
    ↓                                           ↓
Pass rate < 80%?    →    Apply if score > 70%
                                                ↓
                    OnlineLearningMonitor → Auto-rollback if regression > 30%
```

**Evaluation dataset (5 ground-truth cases):**
- LLM landscape 2024
- RAG best practices
- Agent framework comparison
- AI safety techniques
- Vector DB production comparison

**Custom scorer** measures: keyword presence (40%) + word count (40%) + structure (20%)

**Run:**
```bash
pip install agent-haas[vector,observe] agno
python tutorials/04_agno_research_team.py
```

**Output:** `output/research/{topic}_{timestamp}.md`

---

## Setup for all tutorials

```bash
# 1. Install agent-haas
pip install agent-haas[vector,observe,mcp]

# 2. Start Redis (required for tracing + state)
brew install redis && brew services start redis
# or: docker run -d -p 6379:6379 redis:7-alpine

# 3. Create .env in project root
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://localhost:6379
ENVIRONMENT=dev
EOF

# 4. Start API server (optional — for trace UI)
uvicorn harness.api.main:create_app --factory --port 8000

# 5. Run any tutorial
python tutorials/01_langgraph_hr_pipeline.py
```

---

## Viewing traces

After any tutorial run, inspect the span waterfall:

```bash
# Via API
curl http://localhost:8000/runs/{run_id}/trace | python3 -m json.tool

# Via dashboard — Traces tab
open http://localhost:8000
```

Or read the JSONL trace directly:
```bash
cat logs/runs/{run_id}/trace.jsonl | python3 -m json.tool
```

---

## HarnessAgent features demonstrated

| Feature | 01 LangGraph | 02 CrewAI | 03 AutoGen | 04 Agno |
|---------|-------------|-----------|------------|---------|
| TraceRecorder (span tree) | ✅ | ✅ | ✅ | ✅ |
| Cost tracking | ✅ | ✅ | ✅ | ✅ |
| Event streaming (SSE) | ✅ | — | ✅ | — |
| HITL gate | — | — | ✅ | — |
| FailureTracker | — | ✅ | ✅ | ✅ |
| EvalDataset + EvalRunner | — | ✅ | ✅ | ✅ |
| Hermes improvement loop | — | — | — | ✅ |
| OnlineLearningMonitor | — | — | — | ✅ |
| Checkpoint recovery | ✅ | — | — | — |
