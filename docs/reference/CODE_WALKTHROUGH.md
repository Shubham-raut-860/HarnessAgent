# HarnessAgent — Code Walkthrough

> **Style**: Azure Architecture Center  
> **Audience**: Software engineers new to the codebase  
> **Purpose**: A guided tour from first HTTP request to final agent answer, tracing every hop through the actual source code.

---

## Overview

This document follows a single request — "List all tables in the database" submitted to the SQL agent — from the moment it arrives at the API layer through worker execution, the eight-stage agent loop, and final result delivery. Every code block references the actual source file so you can open it alongside this walkthrough.

---

## Step 0: The entry point

You issue:

```bash
curl -X POST http://localhost:8000/runs \
  -H "X-Tenant-ID: acme" \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "sql",
    "task": "List all tables in the database",
    "metadata": {}
  }'
```

What happens next is a chain of eight components, each handing off to the next.

---

## Step 1: API receives the request

**File**: `src/harness/api/routes/runs.py`

FastAPI routes the `POST /runs` request to `create_run()`:

```python
@router.post("", status_code=status.HTTP_201_CREATED, response_model=RunRecordResponse)
async def create_run(
    body: CreateRunRequest,
    tenant_id: str = Depends(get_current_tenant),  # extracted from JWT
    redis: Any = Depends(get_redis),
) -> RunRecordResponse:
    if body.agent_type not in _ALLOWED_AGENT_TYPES:
        raise HTTPException(400, detail="Unknown agent_type...")

    runner = _get_runner(redis)
    record = await runner.create_run(
        tenant_id=tenant_id,
        agent_type=body.agent_type,
        task=body.task,
        metadata=body.metadata,
    )

    # Enqueue the run ID in Redis for the worker
    queue_key = f"harness:queue:{body.agent_type}"
    await redis.rpush(queue_key, record.run_id)

    return RunRecordResponse.from_record(record)
```

**What just happened:**
1. `CreateRunRequest` is validated by Pydantic — `agent_type` must be one of `sql`, `code`, `research`, `orchestrator`.
2. `get_current_tenant` decodes the JWT and extracts the tenant ID.
3. `AgentRunner.create_run()` persists a `RunRecord` to Redis with `status="pending"` and a generated `run_id`.
4. The `run_id` is pushed to the `harness:queue:sql` Redis list for a worker to pick up.
5. The handler returns **immediately** with HTTP 201 — the actual work happens asynchronously.

The caller receives a `RunRecord` with `status: "pending"`. They can poll `GET /runs/{run_id}` to watch for `status: "completed"`.

---

## Step 2: The worker picks it up

**File**: `src/harness/workers/agent_worker.py`

An RQ worker is running in a separate process, listening on the `harness:queue:sql` Redis list. When our `run_id` arrives, it calls `process_run_job()`:

```python
def process_run_job(run_id: str, config_dict: Optional[dict] = None) -> None:
    """Synchronous wrapper called by RQ. Bridges to async implementation."""
    asyncio.run(process_run_job_async(run_id, config_dict or {}))
```

Inside `process_run_job_async()`:

```python
async def process_run_job_async(run_id: str, config_dict=None) -> None:
    # 1. Connect to Redis
    redis_client = aioredis.from_url(cfg.redis_url, ...)
    await redis_client.ping()

    # 2. Build the agent factory
    def agent_factory(agent_type: str):
        if agent_type == "sql":
            from harness.agents.sql_agent import SQLAgent
            return SQLAgent(config=config_dict)
        elif agent_type == "code":
            from harness.agents.code_agent import CodeAgent
            return CodeAgent(config=config_dict)

    # 3. Build the runner and execute
    runner = AgentRunner(
        redis=redis_client,
        agent_factory=agent_factory,
        workspace_base=cfg.workspace_base_path,
        error_collector=error_collector,
    )
    record = await runner.execute_run(run_id)
```

The worker pattern achieves three things:
- **Isolation**: each run gets a fresh asyncio event loop (via `asyncio.run()`).
- **Fault tolerance**: if the worker process crashes, RQ marks the job as failed and retries.
- **Scaling**: you can run multiple worker processes in parallel.

---

## Step 3: AgentRunner builds the context

**File**: `src/harness/orchestrator/runner.py`

`AgentRunner.execute_run()` is responsible for translating a `RunRecord` into an `AgentContext` and handing it to the agent:

```python
async def execute_run(self, run_id: str) -> RunRecord:
    record = await self.get_run(run_id)

    # Mark as running
    record.status = "running"
    record.started_at = datetime.now(timezone.utc)
    await self.update_run(record)

    # Create isolated workspace directory
    workspace = self._workspace_base / record.tenant_id / record.run_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Build the agent and run it
    agent = self._agent_factory(record.agent_type)
    agent_result = await agent.run(
        tenant_id=record.tenant_id,
        task=record.task,
        run_id=record.run_id,
        workspace_path=workspace,
        metadata=record.metadata,
    )

    record.status = "completed" if agent_result.success else "failed"
    record.result = _serialise_result(agent_result)
    record.completed_at = datetime.now(timezone.utc)
    await self.update_run(record)
    return record
```

Key points:
- A per-run workspace directory (`/workspaces/acme/{run_id}/`) is created for file outputs.
- `RunRecord.status` is set to `"running"` before execution and updated to `"completed"` or `"failed"` afterward. Any polling client will see these transitions.
- Unhandled exceptions are caught, the status is set to `"failed"`, and the error is recorded to the `ErrorCollector` for Hermes to analyse later.

---

## Step 4: SQLAgent populates the schema graph

**File**: `src/harness/agents/sql_agent.py`

`SQLAgent` is a subclass of `BaseAgent`. Before the main agent loop begins, it introspects the database and populates the knowledge graph:

```python
class SQLAgent(BaseAgent):
    agent_type = "sql"

    async def _populate_schema(self, ctx: AgentContext) -> None:
        """Query database metadata and load it into the knowledge graph."""
        tables_info = await self._get_tables_info()
        # tables_info = [{"name": "orders", "columns": [...], "foreign_keys": [...]}]

        await ctx.memory._graph_rag.populate_schema(
            tables_info=tables_info,
            ctx=ctx,
        )
```

`populate_schema()` (in `memory/graph_rag.py`) creates graph nodes:
- `Table` nodes — one per table, keyed by table name.
- `Column` nodes — one per column, keyed by `table_name.column_name`, with `col_type` and `nullable` properties.
- `has_column` edges from Table → Column.
- `joins` edges between tables with `on` clauses from foreign key definitions.

**Why this matters for Graph RAG:** when the agent later asks "List all tables", the Graph RAG engine extracts the entity "tables" from the query, anchors on table nodes, and returns a compact `[SCHEMA]` block — instead of sending thousands of tokens of raw schema documentation with every call.

---

## Step 5: BaseAgent.run() — the eight-stage loop

**File**: `src/harness/agents/base.py`

This is the heart of the harness. Here is every stage in detail.

### Pre-loop setup

```python
async def run(self, ctx: AgentContext) -> AgentResult:
    run_start = time.monotonic()

    # Emit "started" event to event bus and message bus
    await self._emit_event(StepEvent.started(ctx))

    # Increment active_runs gauge
    metrics.active_runs.labels(agent_type=self.agent_type).inc()

    async with self._mlflow_context(ctx) as mlflow_run_id:
        # Attempt checkpoint resume
        await self._maybe_resume_checkpoint(ctx)
        history: list[dict[str, Any]] = []
```

### Stage 1 — Checkpoint resume

```python
async def _maybe_resume_checkpoint(self, ctx: AgentContext) -> None:
    checkpoint = await self._checkpoint_manager.load(ctx.run_id)
    if checkpoint is not None:
        ctx.step_count = checkpoint.get("step_count", 0)
        ctx.token_count = checkpoint.get("token_count", 0)
        # Resume from where the run was interrupted
```

If a previous run for this `run_id` was interrupted mid-way (e.g., worker process killed), the checkpoint restores the `step_count` and `token_count` so billing and budget enforcement remain accurate.

### Stage 2 — Context preparation

```python
while ctx.is_budget_ok():
    # 2a. Fit history to context window
    history = await self._fit_history(ctx, history)
    # Keeps the last 40 messages in the in-memory list.
    # ContextWindowManager handles the token-level sliding window.

    # 2b. Graph RAG retrieval
    retrieval_context = await self._smart_retrieve(ctx)
    # Returns a compact [SCHEMA] string from the knowledge graph,
    # supplemented by vector similarity hits.

    # 2c. Assemble messages for LLM
    messages = self.build_messages(ctx, history, retrieval_context)
    system_prompt = self.build_system_prompt(ctx)
```

`_smart_retrieve()` calls `ctx.memory.smart_retrieve(query=ctx.task, ...)`, which calls `GraphRAGEngine.retrieve()`. For our SQL task, this returns something like:

```
[SCHEMA]
orders: Table | cols: id(INTEGER), user_id(INTEGER), amount(DECIMAL)
users: Table | cols: id(INTEGER), email(VARCHAR)
orders --joins--> users ON user_id=id
```

This 3-line block replaces what might be dozens of pages of schema documentation.

### Stage 3 — LLM call

```python
async with self._llm_span(ctx):
    response = await self._call_llm(ctx, messages, system_prompt)

# response is an LLMResponse with:
#   .content      — text reply or empty string if only tool calls
#   .tool_calls   — list of ToolCall objects
#   .input_tokens / .output_tokens
#   .model        — e.g. "claude-sonnet-4-6"
#   .cached       — True if prompt cache hit

total_tokens = response.input_tokens + response.output_tokens
ctx.tick(tokens=total_tokens)
# tick() raises BudgetExceeded if step count, token count, or
# elapsed time exceeds ctx.max_steps / ctx.max_tokens / ctx.timeout_seconds
```

Inside `_call_llm()`:

```python
async def _call_llm(self, ctx, messages, system) -> LLMResponse:
    tools = self._tool_registry.to_anthropic_format()
    remaining_tokens = ctx.max_tokens - ctx.token_count
    max_tokens = min(4096, max(256, remaining_tokens // 2))

    return await self._llm_router.complete(
        messages=messages,
        system=system,
        tools=tools if tools else None,
        max_tokens=max_tokens,
    )
```

The router tries Claude first (priority 0), then GPT-4o-mini (priority 20). If Claude's circuit breaker is OPEN, GPT-4o-mini is used automatically.

### Stage 4 — Cost tracking

```python
run_cost = await self._cost_tracker.record(
    run_id=ctx.run_id,
    tenant_id=ctx.tenant_id,
    model=response.model,
    input_tokens=response.input_tokens,
    output_tokens=response.output_tokens,
)
total_cost_usd += run_cost.cost_usd
# Redis pipeline: writes cost ledger, increments monthly spend,
# increments daily spend — all in one round-trip.
```

### Stage 5 — Safety check on LLM output

```python
guard = await _safe_call(
    self._safety_pipeline.check_output,
    {"content": response.content},
)
if guard is not None and guard.blocked:
    raise SafetyViolation(
        f"Output blocked by safety pipeline: {guard.reason}",
        guard_source="output_guard",
        failure_class=FailureClass.SAFETY_OUTPUT,
    )
```

The PIIRedactor scans the response for SSN, phone, email, and credit card patterns. If the LLM accidentally included PII in its output, the guard blocks the response before it reaches the caller.

### Stage 5b — Emit MLflow span

```python
await self._log_llm_span(ctx, response)
await self._emit_event(StepEvent.llm_called(ctx, response))
# StepEvent carries model, provider, input_tokens, output_tokens, cached flag.
```

The MLflow span records the model, provider, and token counts. The StepEvent is published to the event bus, making it available to SSE subscribers at `GET /runs/{id}/steps`.

### Stage 6 — Tool execution

```python
if not response.tool_calls:
    # No tools requested — agent is done
    output = self.extract_final_answer(history)
    break

for call in response.tool_calls:
    # 6a. HITL check
    await self._check_hitl(ctx, call)
    # If policy.requires_hitl(call.name) is True:
    #   - Creates ApprovalRequest in Redis
    #   - Awaits human decision (up to 1 hour)
    #   - Raises HITLRejected if rejected or expired

    # 6b. Execute the tool
    try:
        result = await self._tool_registry.execute(ctx, call)
    except ToolError as exc:
        result = ToolResult(data=None, error=str(exc), ...)
    except SafetyViolation as exc:
        result = ToolResult(data=None, error=f"Blocked: {exc}")

    # 6c. Push result to short-term memory
    await ctx.memory.push_message(
        run_id=ctx.run_id,
        role="tool",
        content=result.to_text(),
    )

    # 6d. Emit StepEvent and audit log
    await self._emit_event(StepEvent.tool_called(ctx, call, result))
    await self._audit(ctx, call, result)
```

Inside `ToolRegistry.execute()` for our `list_tables` call:
1. Tool lookup — finds the `list_tables` `ToolExecutor`.
2. Schema validation — validates the empty args `{}` against `input_schema`.
3. Safety check — `check_step({"tool_name": "list_tables", "args": {}})` passes (list_tables is in the SQL allowlist).
4. Execute — runs `tool.execute(ctx, {})` with a 30-second timeout.
5. Audit log — records the call in the audit trail.
6. Prometheus — increments `harness_tool_calls_total{tool_name="list_tables", success="true"}`.

### Stage 7 — Memory update and checkpoint

After all tool calls in this step are processed, the assistant message and tool results are appended to `history`:

```python
history.append({"role": "assistant", "content": response.content})
history.extend(tool_results_for_history)
# tool_results_for_history = [
#   {"role": "tool", "tool_use_id": call.id, "content": result.to_text()}
# ]

# Checkpoint every 10 steps
if ctx.step_count % 10 == 0:
    await self._save_checkpoint(ctx, history)
    # Saves: {step_count, token_count, history_len}
```

The checkpoint is written atomically (`.tmp` → `rename`) so a crash mid-write never corrupts the checkpoint.

Agents can also write graph edges from tool results. For example, after running a SQL query, `SQLAgent` calls:

```python
await ctx.memory.add_fact(
    subject=f"query:{ctx.run_id}",
    predicate="uses",
    object_=f"table:{table_name}",
)
# Creates a directed edge: this run's query used this table.
```

### Stage 8 — Loop termination

```python
if not response.tool_calls:
    output = self.extract_final_answer(history)
    break
```

`extract_final_answer()` walks `history` in reverse to find the last `role=="assistant"` message. It handles both plain string content and Anthropic's structured content blocks (filtering out `tool_use` blocks to extract only text).

When the LLM decides it has enough information and responds without any tool calls — "Here are the tables in your database: orders, users, products…" — the loop exits and the output is returned.

---

## Step 6: Result collection and observability

**Files**: `src/harness/orchestrator/runner.py`, `src/harness/observability/`

After `BaseAgent.run()` returns:

```python
# In BaseAgent.run() finally block:
metrics.active_runs.labels(agent_type=self.agent_type).dec()
metrics.agent_runs_total.labels(
    agent_type=self.agent_type,
    success=str(not ctx.failed).lower(),
).inc()

# Emit completion event
await self._emit_event(StepEvent.completed(ctx, output))

# Return AgentResult
return AgentResult(
    run_id=ctx.run_id,
    output=output,
    steps=ctx.step_count,
    tokens=ctx.token_count,
    success=not ctx.failed,
    failure_class=ctx.failure_class,
    elapsed_seconds=elapsed,
    cost_usd=total_cost_usd,
)
```

Back in `AgentRunner.execute_run()`:

```python
record.status = "completed"
record.result = _serialise_result(agent_result)
# _serialise_result converts AgentResult to a plain dict:
# {"run_id": "...", "output": "Here are the tables...", "steps": 2,
#  "tokens": 1847, "success": true, "cost_usd": 0.0062, ...}

record.completed_at = datetime.now(timezone.utc)
await self.update_run(record)
# Writes back to Redis: harness:run:{run_id}
```

MLflow closes the parent span and logs final metrics: `step_count=2`, `token_count=1847`, `elapsed_seconds=4.3`, `failed=0`.

---

## Step 7: Client receives the result

The client has two options:

### Option A: Polling

```bash
# Poll until status is not "pending" or "running"
curl http://localhost:8000/runs/{run_id} | jq '.status, .result.output'
```

`GET /runs/{run_id}` reads the `RunRecord` from `harness:run:{run_id}` in Redis and returns it. Once `status` is `"completed"`, `.result.output` contains the final answer text.

### Option B: SSE streaming

```bash
curl http://localhost:8000/runs/{run_id}/steps
# Streams StepEvent objects as they are emitted by the event bus:
# data: {"run_id": "...", "step": 1, "event_type": "llm_call", "payload": {...}}
# data: {"run_id": "...", "step": 1, "event_type": "tool_call", "payload": {...}}
# data: {"run_id": "...", "step": 2, "event_type": "llm_call", "payload": {...}}
# data: {"run_id": "...", "step": 2, "event_type": "completed", "payload": {...}}
```

`GET /runs/{run_id}/steps` subscribes to the event bus for the given `run_id`. Each `StepEvent` emitted by the agent's `_emit_event()` call is serialised and pushed to the SSE stream. The stream closes when a `completed` or `failed` event is received.

---

## Step 8: Hermes runs in the background

**File**: `src/harness/improvement/hermes.py`

While the above run succeeded, some fraction of SQL agent runs fail — perhaps because the system prompt doesn't handle edge cases well. Every hour, the Hermes worker fires:

```python
# In hermes_worker.py, scheduled by APScheduler:
outcomes = await hermes_loop.run_all_agents(["sql", "code", "base"])
```

For the `sql` agent type:

```python
async def run_cycle(self, agent_type: str) -> PatchOutcome | None:
    # 1. Check if enough errors have accumulated
    error_count = await self._collector.count("sql")
    if error_count < 5:
        return None  # Not enough signal yet

    # 2. Sample recent errors from the vector store
    errors = await self._collector.get_recent("sql", limit=20)
    # errors = [ErrorRecord(task="...", error_message="...", failure_class="..."), ...]

    # 3. Generate a patch proposal
    patch = await self._generator.generate(
        agent_type="sql",
        errors=errors,
        max_errors_in_prompt=10,
    )
    # patch.op = "append"
    # patch.value = "When listing tables, always use SHOW TABLES FROM information_schema."

    # 4. Evaluate the patch against the most recent errors
    eval_result = await self._evaluator.score(
        patch=patch,
        test_cases=errors[:10],
        agent_type="sql",
    )
    # eval_result.score = 0.83

    # 5. Apply or queue
    if eval_result.score >= 0.7 and self.auto_apply:
        await self._apply_patch(patch, "sql")   # Updates prompt store
        patch.status = "applied"
    elif eval_result.score >= 0.7:
        patch.status = "approved"               # Queued for human review
        await self._store_patch(patch)
    else:
        patch.status = "rejected"
```

The failures from our runs feed directly into this cycle. If the agent consistently fails on a certain class of task, Hermes notices, proposes a prompt improvement, evaluates it, and queues it for review. After a human approves via `POST /improvement/patches/{id}/apply`, the next generation of SQL agents benefits from the improvement.

---

## Data flow summary

```
POST /runs
    |
    v
create_run() ──────────────────────────────────── RunRecord{status: pending}
    |                                                       |
    | rpush harness:queue:sql                               | GET /runs/{id}
    |                                                       |
    v                                                       v
RQ Worker picks up run_id                         Client polls RunRecord
    |
    v
process_run_job_async(run_id)
    |
    v
AgentRunner.execute_run()
    ├── workspace mkdir
    ├── agent_factory("sql") -> SQLAgent
    └── agent.run(ctx)
            |
            ├── _maybe_resume_checkpoint()
            |
            ├─ [LOOP] ──────────────────────────────────────────────────────┐
            │   ├── _fit_history()            [sliding window]              |
            │   ├── _smart_retrieve()         [Graph RAG → compact schema]  |
            │   ├── build_messages()          [history + retrieval context] |
            │   ├── LLMRouter.complete()      [Claude/GPT-4o fallback]      |
            │   ├── ctx.tick()                [budget enforcement]          |
            │   ├── cost_tracker.record()     [Redis cost ledger]           |
            │   ├── safety_pipeline.check_output()  [PII redaction]         |
            │   ├── StepEvent.llm_called()    [event bus + MLflow]          |
            │   ├── [per tool call]                                         |
            │   │     ├── _check_hitl()       [pause for human review]      |
            │   │     ├── tool_registry.execute()  [validate → safety → run]|
            │   │     ├── memory.push_message()    [tool result → Redis]    |
            │   │     └── StepEvent.tool_called()  [event bus]              |
            │   ├── checkpoint save (every 10 steps)                        |
            │   └── if no tool_calls: break ─────────────────────────────── ┘
            │
            └── AgentResult{output, steps, tokens, cost_usd, elapsed}
    |
    v
RunRecord{status: completed, result: {...}}  ←── persisted to Redis
    |
    v
GET /runs/{id} returns completed RunRecord   ←── client receives answer
GET /runs/{id}/steps SSE stream closes       ←── stream subscriber notified

[Background, hourly]
HermesLoop.run_all_agents()
    ├── sample recent errors from vector store
    ├── generate patch proposal via LLM
    ├── evaluate patch by replaying failures
    └── queue approved patches for human review
```
