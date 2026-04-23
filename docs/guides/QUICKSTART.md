# Quickstart — Codex Harness

> **Goal:** Have a running agent answering questions in under 15 minutes.
> No prior experience with multi-agent systems required.

---

## Prerequisites Checklist

Before you begin, confirm you have the following installed and available.

| Requirement | Version | Check | Get it |
|---|---|---|---|
| 🐍 Python | 3.11+ | `python --version` | [python.org](https://www.python.org/downloads/) |
| 📦 Poetry | 1.8+ | `poetry --version` | `pip install poetry` |
| 🐳 Docker Desktop | Latest | `docker info` | [docker.com](https://www.docker.com/products/docker-desktop/) |
| 🔑 API key **OR** local model | — | see paths below | Anthropic / OpenAI / Ollama |

> **Note:** You need at least one LLM source — an Anthropic key, an OpenAI key, or a
> locally running model. You do **not** need all three.

---

## Choose Your Setup Path

```
Do you have an Anthropic or OpenAI API key?
│
├── YES → Path A: Cloud Setup (5 min)   ──────────────┐
│                                                      │
└── NO                                                 │
    │                                                  │
    ├── I want to run fully offline                    │
    │       → Path B: Local LLM (15 min)   ────────────┤
    │                                                  │
    └── I need production-grade deployment             │
            → Path C: Production ──────────────────────┘
                       See DEPLOYMENT.md
```

---

## Path A — Cloud Setup (5 minutes)

You have an Anthropic or OpenAI API key. This is the fastest route.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/codex-harness.git
cd codex-harness
```

### Step 2 — Install Python dependencies

```bash
poetry install
```

This installs all Python packages defined in `pyproject.toml` into an isolated
virtual environment managed by Poetry.

### Step 3 — Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` in your editor and set at least one of these:

```bash
# For Claude (recommended — best tool-use reliability)
ANTHROPIC_API_KEY=sk-ant-...

# OR for OpenAI
OPENAI_API_KEY=sk-...
```

Everything else in `.env` has sensible defaults for local development.
You do not need to change anything else right now.

### Step 4 — Start infrastructure services

You only need three services to get started (Redis, Qdrant, MLflow).
You do **not** need to start the full stack yet.

```bash
docker compose up -d redis qdrant mlflow
```

Verify they are healthy:

```bash
docker compose ps
```

All three should show `healthy` within about 30 seconds.

### Step 5 — Start the API server

Open a terminal and run:

```bash
PYTHONPATH=src uvicorn harness.api.main:create_app \
  --factory \
  --port 8000 \
  --reload
```

You should see output similar to:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Step 6 — Start a background worker

Open a **second terminal** and run:

```bash
cd src && rq worker harness-default
```

The worker picks up agent jobs from the Redis queue and executes them
asynchronously. Without it, run requests will queue but never execute.

### Step 7 — Verify the setup

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "services": {
    "redis": true,
    "vector_db": true,
    "graph_db": true,
    "llm": true
  },
  "version": "0.1.0"
}
```

If any service shows `false`, refer to the [Common Errors](#common-errors-and-fixes)
table at the bottom of this guide.

### Step 8 — Open observability dashboards

| Dashboard | URL | Credentials |
|---|---|---|
| MLflow (traces & experiments) | http://localhost:5000 | none required |
| Prometheus (raw metrics) | http://localhost:9090 | none required |

---

## Path B — Fully Local Setup (15 minutes)

No API key required. Runs entirely on your machine using llama.cpp.

### Step 1 — Download a GGUF model

We recommend **Llama-3-8B-Instruct Q4_K_M** as a good balance of speed and quality.

```bash
mkdir -p ./models

# Download via huggingface-hub CLI (install with: pip install huggingface-hub)
huggingface-cli download \
  bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  --local-dir ./models
```

Direct link if you prefer a browser download:
`https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF`

### Step 2 — Start the local CPU stack

```bash
LLAMACPP_MODEL=Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  docker compose --profile local-cpu up -d
```

This starts llama.cpp on port 8081 in addition to Redis, Qdrant, and MLflow.

### Step 3 — Configure your `.env`

```bash
cp .env.example .env
```

Add this line to `.env`:

```bash
LLAMACPP_BASE_URL=http://localhost:8081
```

Leave `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` empty or unset.
The router will automatically detect and use the local model.

### Step 4 — Continue from Path A, Step 5

From here, follow [Step 5](#step-5--start-the-api-server) through
[Step 8](#step-8--open-observability-dashboards) above.

> **Note on tool calling:** llama.cpp uses ReAct-style text injection for tool
> calling instead of native structured tool use. This means the agent works
> correctly but may occasionally produce slightly different formatting.
> For critical production workloads, a provider with native tool support
> (Claude, GPT-4o) is recommended.

---

## Path C — Production Deployment

See the full **[DEPLOYMENT.md](DEPLOYMENT.md)** guide for:
- Docker Compose with external managed databases
- Kubernetes / Helm deployment
- Backup and recovery procedures
- Production security checklist

---

## Your First SQL Agent

Run a SQL query against a sample SQLite database using the REST API.

```bash
# Step 1: Create a run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "sql",
    "task": "Show me the top 5 customers by total order value",
    "metadata": {
      "connection_string": "sqlite:///./data/sample.db"
    }
  }'
```

The response contains a `run_id`:

```json
{
  "run_id": "run_abc123",
  "status": "queued",
  "agent_type": "sql",
  "task": "Show me the top 5 customers by total order value"
}
```

```bash
# Step 2: Poll for the result
curl http://localhost:8000/runs/run_abc123
```

```bash
# Step 3: Stream live steps via Server-Sent Events (optional)
# Each line is a JSON StepEvent as the agent works
curl -N http://localhost:8000/runs/run_abc123/steps
```

A successful result looks like:

```json
{
  "run_id": "run_abc123",
  "status": "completed",
  "result": {
    "output": "Here are the top 5 customers by total order value:\n\n1. Acme Corp — $142,500\n2. ...",
    "steps": 4,
    "tokens_used": 1240
  }
}
```

---

## Your First Code Agent

Ask the agent to write and validate Python code for you.

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "code",
    "task": "Write a Python function to calculate fibonacci numbers and test it"
  }'
```

The code agent will:
1. Write the function
2. Run `ruff` to lint it
3. Execute the code to verify it works
4. Return the final, tested code

---

## Using the Python SDK Directly

You can bypass the REST API and call the harness directly from Python.
This is useful for notebooks, scripts, and testing.

```python
import asyncio
import uuid
from harness.core.config import get_config
from harness.core.context import AgentContext
from harness.llm.factory import build_router
from harness.memory.manager import MemoryManager
from harness.agents.sql_agent import SQLAgent

# All agent dependencies — in production these are injected via the API startup
# For direct use you can pass lightweight stubs for components you don't need.

async def main():
    config = get_config()

    # Build the LLM router (handles provider selection, fallback, circuit breaking)
    router = build_router(config)

    # Build the memory manager (vector + graph + short-term Redis)
    memory = await MemoryManager.create(config)

    # Build the agent — pass None for components you're not using in a script
    agent = SQLAgent(
        llm_router=router,
        memory_manager=memory,
        tool_registry=None,       # tool registry optional for testing
        safety_pipeline=None,
        step_tracer=None,
        mlflow_tracer=None,
        failure_tracker=None,
        audit_logger=None,
        event_bus=None,
        cost_tracker=None,
        checkpoint_manager=None,
    )

    # Build the context that controls this specific run
    ctx = AgentContext(
        run_id=str(uuid.uuid4()),
        tenant_id="local-dev",
        task="List all tables in the database",
        agent_type="sql",
        max_steps=20,
        metadata={"connection_string": "sqlite:///./data/sample.db"},
        memory_manager=memory,
    )

    result = await agent.run(ctx)
    print(result.output)

asyncio.run(main())
```

---

## Using with LangGraph

The harness includes a `LangGraphAdapter` that wraps any LangGraph `StateGraph`
and provides full observability — step tracing, budget enforcement, and event
streaming — without changing your graph logic.

```python
import asyncio
import uuid
from langgraph.graph import StateGraph, END
from harness.adapters.langgraph import LangGraphAdapter
from harness.core.context import AgentContext

# --- Define your LangGraph workflow as normal ---

class State(dict):
    pass

def step_one(state: State) -> State:
    return {"messages": [*state.get("messages", []), "Step one complete"]}

def step_two(state: State) -> State:
    return {"output": "Final answer: " + state["messages"][-1]}

builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)
builder.set_entry_point("step_one")
graph = builder.compile()

# --- Wrap with the harness adapter ---

async def run_with_harness():
    ctx = AgentContext(
        run_id=str(uuid.uuid4()),
        tenant_id="local-dev",
        task="Run my LangGraph workflow",
        agent_type="orchestrator",
        max_steps=50,
        metadata={},
        memory_manager=None,
    )

    adapter = LangGraphAdapter(graph)

    # Stream StepEvents as each graph node executes
    async for step_event in adapter.run(ctx, input={"messages": []}):
        print(f"Node executed: {step_event.payload.get('node')} "
              f"(step {step_event.step})")

    result = await adapter.get_result()
    print(f"Output: {result.output}")
    print(f"Total steps: {result.steps}")

asyncio.run(run_with_harness())
```

---

## Viewing Your First Trace in MLflow

After running any agent, open **http://localhost:5000**.

```
MLflow UI
└── Experiments
    └── codex-harness                    ← default experiment name
        └── Run: run_abc123
            ├── Parameters
            │   ├── agent_type: sql
            │   ├── model: claude-sonnet-4-6
            │   └── tenant_id: local-dev
            ├── Metrics
            │   ├── tokens_input: 842
            │   ├── tokens_output: 398
            │   ├── cost_usd: 0.0031
            │   └── steps: 4
            └── Artifacts
                └── trace.json           ← full span tree
```

**What to look for:**

| Field | What it tells you |
|---|---|
| **Spans** | One span per LLM call and tool execution. Click to see exact prompts. |
| **Token counts** | Input vs output tokens per step — spot expensive calls. |
| **Tool calls** | Which tools were invoked, in what order, with what arguments. |
| **cost_usd** | Estimated cost for this run based on provider pricing. |
| **steps** | How many reasoning loops the agent took — high = complex task or regression. |

---

## Common Errors and Fixes

| Error message | Cause | Fix |
|---|---|---|
| `No LLM providers configured` | No API key or local URL set | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env`, or set `LLAMACPP_BASE_URL` |
| `Connection refused: redis://localhost:6379` | Redis container not running | `docker compose up -d redis` |
| `Circuit breaker OPEN for provider anthropic` | API provider is failing or key is invalid | Wait 60 s for auto-recovery; check your API key is valid and has credits |
| `BudgetExceeded: max_steps=20 reached` | Task too complex or agent looping | Increase `max_steps` in your request metadata, or simplify the task |
| `ModuleNotFoundError: harness` | PYTHONPATH not set | Run with `PYTHONPATH=src` prefix, e.g. `PYTHONPATH=src uvicorn ...` |
| `rq.exceptions.NoRedisConnectionException` | Worker cannot reach Redis | Ensure Redis is running and `REDIS_URL` in `.env` points to the correct host |
| `422 Unprocessable Entity` from API | Invalid request body | Check `agent_type` is one of: `sql`, `code`, `research`, `orchestrator` |
| `healthcheck: vector_db: false` | Qdrant not started | `docker compose up -d qdrant` |

---

## Next Steps

| Goal | Resource |
|---|---|
| Configure providers, memory, and safety limits | [CONFIGURATION.md](CONFIGURATION.md) |
| Deploy to production | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Understand the architecture | [../architecture/](../architecture/) |
| Add a custom tool or agent | [../reference/](../reference/) |
