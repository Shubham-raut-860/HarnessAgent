# HarnessAgent — Component Reference

> **Style**: Azure Architecture Center  
> **Audience**: Engineers, architects, and non-technical stakeholders  
> **Purpose**: Authoritative reference for every major component in the HarnessAgent multi-agent AI platform.

---

## Table of Contents

1. [LLM Router](#1-llm-router)
2. [Memory Manager](#2-memory-manager)
3. [Graph RAG Engine](#3-graph-rag-engine)
4. [Context Window Manager](#4-context-window-manager)
5. [BaseAgent](#5-baseagent)
6. [Safety Pipeline](#6-safety-pipeline)
7. [Tool Registry](#7-tool-registry)
8. [MCP Client](#8-mcp-client)
9. [Circuit Breaker](#9-circuit-breaker)
10. [Cost Tracker](#10-cost-tracker)
11. [Rate Limiter](#11-rate-limiter)
12. [HITL Manager](#12-hitl-manager)
13. [Hermes Loop](#13-hermes-loop)
14. [Planner](#14-planner)
15. [Framework Adapters](#15-framework-adapters)
16. [MLflow Tracer](#16-mlflow-tracer)
17. [Failure Tracker](#17-failure-tracker)

---

## 1. LLM Router

**What it does** Routes LLM completion requests across multiple AI providers, automatically falling back to the next available provider when one is unhealthy or circuit-broken.

**File**: `src/harness/llm/router.py`  
**Key class**: `LLMRouter`  
**Used by**: `BaseAgent`, `Planner`, `ContextWindowManager` (summarizer)

---

### How it works

Every call to `complete()` iterates providers in ascending priority order (lower number = tried first). Before attempting a provider, the router checks its context window is large enough for the request and performs a live `health_check()`. Providers that fail either test are silently skipped.

Each provider is protected by a dedicated `CircuitBreaker` instance, retrieved from a shared `CircuitBreakerRegistry`. If the breaker for that provider is OPEN, the router logs a warning and moves on. Only `LLM_RATE_LIMIT`, `LLM_TIMEOUT`, and `LLM_ERROR` failures trigger fallback; other errors (such as authentication failures) are re-raised immediately so operators see them quickly.

When all providers are exhausted, the router raises `LLMError("All providers exhausted")` with the last underlying error attached. The default provider priority chain, as configured by `build_router()` in `llm/factory.py`, is:

| Priority | Provider | Context window |
|---|---|---|
| 0 | Claude (claude-sonnet-4-6 by default) | 200 000 tokens |
| 10 | GPT-4o / GPT-5 | 128 000 tokens |
| 20 | GPT-4o-mini (cost-optimised fallback) | 128 000 tokens |
| 100 | vLLM (self-hosted) | 32 768 tokens |
| 110 | SGLang | 8 192 tokens |
| 120 | llama.cpp | 4 096 tokens |

The `stream()` method applies the same provider-selection logic, yielding tokens as they arrive via `AsyncIterator[str]`.

---

### Key methods

| Method | What it does |
|---|---|
| `register(provider, priority, context_window)` | Adds a provider to the router; re-sorts by priority after insertion |
| `complete(messages, *, max_tokens, required_context, system, tools, **kwargs)` | Routes a chat completion request; returns `LLMResponse` |
| `stream(messages, **kwargs)` | Streams tokens from the first healthy provider |
| `health_check_all()` | Runs `provider.health_check()` for all providers concurrently; returns `dict[str, bool]` |

---

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables Anthropic Claude provider |
| `OPENAI_API_KEY` | — | Enables OpenAI provider |
| `OPENAI_MODELS` | `gpt-4o-mini` | Comma-separated list of OpenAI models to register |
| `VLLM_BASE_URL` | — | Enables self-hosted vLLM provider |
| `SGLANG_BASE_URL` | — | Enables SGLang provider |
| `LLAMACPP_BASE_URL` | — | Enables llama.cpp provider |
| `DEFAULT_MODEL` | `claude-sonnet-4-6` | Primary Claude model |

---

### Example

```python
from harness.llm.factory import build_router
from harness.core.config import get_config

router = build_router(get_config())

response = await router.complete(
    messages=[{"role": "user", "content": "Summarise Q3 sales."}],
    system="You are a financial analyst.",
    max_tokens=512,
)
print(response.content)          # The text reply
print(response.model)            # e.g. "claude-sonnet-4-6"
print(response.input_tokens)     # Tokens charged for input
```

---

## 2. Memory Manager

**What it does** Provides a single unified interface to all three memory tiers — Redis short-term, vector long-term, and graph structured — with PII redaction applied before every write.

**File**: `src/harness/memory/manager.py`  
**Key class**: `MemoryManager`  
**Used by**: `BaseAgent`, `GraphRAGEngine`

---

### How it works

`MemoryManager` composes four sub-systems that each handle a specific type of recall:

1. **Short-term (hot)** — `ShortTermMemory` backed by Redis. Every message in an agent conversation is stored here, indexed by `run_id`. Messages expire when the run's Redis TTL lapses.
2. **Long-term (warm)** — A vector store (Chroma, Qdrant, or Weaviate) that receives semantic embeddings of facts, documents, and tool results. PII is redacted by five regex patterns (SSN, phone, email, Visa, Mastercard) before any text is embedded or stored.
3. **Knowledge graph (structured)** — A NetworkX or Neo4j graph where agents write subject–predicate–object triples via `add_fact()`. For example, `add_fact("query:abc123", "uses", "table:orders")` records which tables a query touched.
4. **GraphRAG engine** — `GraphRAGEngine` orchestrates graph traversal plus vector similarity in a single `smart_retrieve()` call, choosing the best strategy automatically.

The `create()` classmethod is the standard factory. It reads `config.redis_url`, `config.vector_store`, and `config.graph_type` to wire everything up without caller boilerplate.

---

### Key methods

| Method | What it does |
|---|---|
| `push_message(run_id, role, content, tokens)` | Appends a conversation turn to Redis short-term history |
| `get_history(run_id, last_n)` | Returns the most recent `last_n` messages in chronological order |
| `fit_history(run_id, max_tokens)` | Retrieves history and trims it to the given token budget via `ContextWindowManager` |
| `remember(text, metadata, tenant_id)` | PII-redacts text, embeds it, and upserts into the vector store; returns the doc ID |
| `recall(query, k, filter)` | Returns the top-k semantically similar memories as `MemoryEntry` objects |
| `add_fact(subject, predicate, object_, weight, metadata)` | Adds a directed triple to the knowledge graph |
| `smart_retrieve(query, ctx)` | Graph-first retrieval with vector fallback; delegates to `GraphRAGEngine` |
| `clear_session(run_id)` | Removes all short-term memory for a run (called after completion) |
| `MemoryManager.create(config)` | Async class factory — preferred way to instantiate |

---

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Short-term memory store |
| `VECTOR_STORE` | `chroma` | `chroma`, `qdrant`, or `weaviate` |
| `GRAPH_TYPE` | `networkx` | `networkx` (in-process) or `neo4j` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |

---

### Example

```python
manager = await MemoryManager.create(config)

# Store a fact with PII redaction
doc_id = await manager.remember(
    text="User john@example.com asked about orders",  # email will be redacted to [EMAIL]
    metadata={"agent_type": "sql", "run_id": ctx.run_id},
    tenant_id=ctx.tenant_id,
)

# Graph-first retrieval
result = await manager.smart_retrieve(
    query="List orders for user",
    ctx=ctx,
)
print(result.strategy)        # "graph_primary", "hybrid", or "vector_fallback"
print(result.graph_context)   # Compact schema text injected into the system prompt
```

---

## 3. Graph RAG Engine

**What it does** Retrieves context from a knowledge graph via multi-hop traversal, supplemented by vector similarity search, producing compact structured context that fits in far fewer tokens than naive retrieval.

**File**: `src/harness/memory/graph_rag.py`  
**Key class**: `GraphRAGEngine`  
**Used by**: `MemoryManager.smart_retrieve()`

---

### How it works

`retrieve()` runs a four-step pipeline:

1. **Entity extraction** — Three regex patterns scan the query for candidate entity names: quoted strings (highest confidence), CamelCase identifiers, and `snake_case`/generic identifiers. Common stop-words (`select`, `query`, `show`, etc.) are filtered out.
2. **Graph anchor + traversal** — Extracted entities are fuzzy-matched against graph nodes. Matched nodes become BFS anchor points. From there, `traverse(anchor_ids, max_hops=2)` follows edges up to two hops, collecting `GraphPath` objects containing typed nodes and edges.
3. **Vector supplement** — The raw query is always sent to the vector store regardless of graph results. This ensures a fallback when the graph is empty or the query targets unknown entities.
4. **Strategy selection and rendering** — The engine classifies the result into one of four strategies and calls `_render_paths()` to produce a compact `[SCHEMA]` block.

```
Strategy selection:
  ≥ 3 graph paths, no vector hits  →  "graph_primary"
  graph paths + vector hits         →  "hybrid"
  entities found but no graph paths →  "vector_fallback"
  no entities extracted             →  "vector_primary"
```

`_render_paths()` deduplicates nodes and edges, emitting one line per table and one line per foreign-key join. This design is why GraphRAG achieves roughly 83% token reduction versus naive "embed all documents" approaches: instead of repeating every column for every query, the agent receives only the schema subgraph relevant to the entities it mentioned.

`populate_schema()` is called by `SQLAgent` at startup to load database schema into the graph, creating `Table` nodes, `Column` nodes with `has_column` edges, and `joins` edges for foreign keys.

---

### Key methods

| Method | What it does |
|---|---|
| `retrieve(query, ctx, max_hops, vector_k)` | Full pipeline: entity extraction → graph traversal → vector supplement → `RetrievalResult` |
| `populate_schema(tables_info, ctx)` | Loads SQL schema (tables, columns, FKs) into the graph as typed nodes and edges |
| `_extract_entities(query)` | Extracts candidate entity names using three regex patterns |
| `_render_paths(paths)` | Renders graph paths as a compact `[SCHEMA]` text block |

---

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_hops` (retrieve arg) | `2` | BFS depth from anchor nodes |
| `vector_k` (retrieve arg) | `5` | Number of vector store results to retrieve |

---

### Example

```python
engine = GraphRAGEngine(graph=graph, vector_store=vs, embedder=embedder)

# Pre-populate with database schema
await engine.populate_schema(tables_info=[
    {
        "name": "orders",
        "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
        "foreign_keys": [{"col": "user_id", "ref_table": "users", "ref_col": "id"}],
    }
], ctx=ctx)

result = await engine.retrieve(
    query="Show me orders joined to users",
    ctx=ctx,
)
print(result.graph_context)
# [SCHEMA]
# orders: Table | cols: id(INTEGER)
# users: Table | cols: id(INTEGER)
# orders --joins--> users ON user_id=id
```

---

## 4. Context Window Manager

**What it does** Prevents context overflow by fitting conversation history into the available token budget, keeping the most recent messages and optionally compressing dropped history into a summary.

**File**: `src/harness/memory/context_manager.py`  
**Key class**: `ContextWindowManager`  
**Used by**: `MemoryManager.fit_history()`

---

### How it works

`fit()` receives a list of `ConversationMessage` objects and a budget broken down as:

```
available_tokens = max_tokens - system_tokens - retrieved_tokens - reserve_output
```

The `reserve_output` (default 2000) guarantees the LLM always has room to generate a response even after context is filled.

Messages are split into two groups: `system` messages (always kept) and conversational messages. The conversational messages are iterated in **reverse chronological order** — most recent first — and accumulated until the budget is exhausted. This is the "sliding window" behaviour: recent messages are always preserved; older messages are dropped when the window fills.

If any messages are dropped and a `summarizer_provider` was configured, the dropped messages are condensed into a 3-5 sentence summary using a secondary LLM call (max 400 output tokens). The summary is prepended as a `system` message so the agent retains key facts without the full token cost. If the summarizer is unavailable or fails, a rule-based fallback extracts one representative user/assistant snippet every 5 messages.

The returned `ContextWindow` carries `truncated=True` and the `summary` text when compression occurred, allowing callers to log or display this information.

---

### Key methods

| Method | What it does |
|---|---|
| `fit(messages, system_tokens, retrieved_tokens, reserve_output)` | Applies sliding window; returns `ContextWindow` with kept messages and optional summary |
| `summarize(messages)` | Compresses a message list into a short summary string using LLM or rule-based fallback |
| `count_tokens(messages)` | Returns total estimated token count for a list of messages |

---

### Configuration

| Constructor argument | Default | Description |
|---|---|---|
| `max_tokens` | `100_000` | Hard token budget for the full context window |
| `summarizer_provider` | `None` | Optional LLM provider used to compress dropped messages |

---

### Example

```python
manager = ContextWindowManager(max_tokens=100_000, summarizer_provider=router)

window = await manager.fit(
    messages=history,
    system_tokens=800,
    retrieved_tokens=400,
    reserve_output=2000,
)
print(f"Kept {len(window.messages)} messages, {window.total_tokens} tokens")
print(f"Truncated: {window.truncated}")
if window.summary:
    print(f"Summary: {window.summary[:100]}...")
```

---

## 5. BaseAgent

**What it does** Provides the full production-grade agent lifecycle — an eight-stage loop covering checkpoint recovery, memory retrieval, LLM calls, safety checks, tool execution, cost tracking, MLflow tracing, and event emission.

**File**: `src/harness/agents/base.py`  
**Key class**: `BaseAgent`  
**Used by**: `AgentRunner`, `SQLAgent`, `CodeAgent`

---

### How it works

`BaseAgent.run()` executes one continuous `while ctx.is_budget_ok()` loop. Each iteration of the loop is one "step" — one round-trip to the LLM. Here is the annotated lifecycle:

**Pre-loop**
- Emits a `StepEvent(event_type="started")` to the event bus and message bus.
- Increments the `harness_active_runs` Prometheus gauge.
- Opens an MLflow run via `_mlflow_context()`.
- Attempts to resume from a checkpoint via `_maybe_resume_checkpoint()`.

**Per-step (the main loop)**

| Stage | Code | What happens |
|---|---|---|
| 1. History fit | `_fit_history(ctx, history)` | Trims in-memory history list to `_MAX_HISTORY_MESSAGES` (40). Full sliding-window logic is applied when memory manager calls `fit_history()`. |
| 2. Memory retrieval | `_smart_retrieve(ctx)` | Calls `MemoryManager.smart_retrieve()` to get a compact graph/vector context string injected into the first user message. |
| 3. LLM call | `_call_llm(ctx, messages, system)` | Routes through `LLMRouter`. Token budget is computed as `min(4096, remaining // 2)` to avoid overshooting. |
| 4. Token + cost accounting | `ctx.tick()`, `cost_tracker.record()` | `tick()` raises `BudgetExceeded` if step, token, or time limit is crossed. `CostTracker` records per-run and per-tenant spend in Redis. |
| 5. Safety check | `safety_pipeline.check_output()` | Runs PIIRedactor and any configured output guards on the LLM response. Raises `SafetyViolation` if blocked. |
| 6. Tool execution | `_tool_registry.execute(ctx, call)` | For each tool call: (a) HITL approval check, (b) schema validation, (c) safety check, (d) execute with timeout, (e) write result to short-term memory. |
| 7. Checkpoint save | `_save_checkpoint()` | Every `_CHECKPOINT_INTERVAL` (10) steps, persists `step_count`, `token_count`, and `history_len` to the checkpoint manager. |
| 8. Loop termination | `if not response.tool_calls` | When the LLM responds with no tool calls, `extract_final_answer()` returns the last assistant message text and the loop exits. |

**Post-loop**
- Decrements `harness_active_runs` gauge.
- Increments `harness_agent_runs_total` counter with success label.
- Emits `StepEvent(event_type="completed")` or `StepEvent(event_type="failed")`.
- Returns an `AgentResult` with run_id, output, steps, tokens, cost, and elapsed time.

Subclasses override `build_system_prompt()` to inject agent-specific instructions and `extract_final_answer()` for specialised output parsing. `SQLAgent` and `CodeAgent` both inherit this structure.

---

### Key methods

| Method | What it does |
|---|---|
| `run(ctx)` | Full eight-stage lifecycle; returns `AgentResult` |
| `build_system_prompt(ctx)` | Returns the system prompt string (override in subclasses) |
| `build_messages(ctx, history, retrieval_context)` | Assembles the messages list for the LLM call |
| `extract_final_answer(history)` | Extracts final text from the last assistant message |
| `_check_hitl(ctx, call)` | Pauses execution if the tool requires human approval; raises `HITLRejected` on rejection |

---

### Configuration (via `AgentContext`)

| Field | Default | Description |
|---|---|---|
| `max_steps` | `50` | Maximum tool-call iterations |
| `max_tokens` | `100_000` | Maximum cumulative tokens across all LLM calls |
| `timeout_seconds` | `300.0` | Wall-clock time limit in seconds |
| `_CHECKPOINT_INTERVAL` | `10` | Steps between checkpoint saves |
| `_MAX_HISTORY_MESSAGES` | `40` | Maximum messages retained in in-memory history |

---

### Example

```python
from harness.core.context import AgentContext

ctx = AgentContext.create(
    tenant_id="acme",
    agent_type="sql",
    task="Find the top 10 customers by revenue this quarter",
    memory=memory_manager,
    workspace_path=Path("/workspaces/acme/run123"),
    max_steps=30,
    max_tokens=50_000,
)

result = await agent.run(ctx)
print(result.output)           # Final answer text
print(result.steps)            # Number of LLM steps taken
print(f"${result.cost_usd:.4f}")  # Total LLM cost
```

---

## 6. Safety Pipeline

**What it does** Guards every agent run with three independently configured check points — input, intermediate (per step), and output — using a composable pipeline of guardrail objects.

**File**: `src/harness/safety/pipeline_factory.py`  
**Key class**: `SafetyConfig`, `build_pipeline()`, `get_default_config()`  
**Used by**: `BaseAgent`, `ToolRegistry`

---

### How it works

`build_pipeline()` assembles a `guardrail.Pipeline` from three stage groups:

| Stage | Guards | Trigger |
|---|---|---|
| Input | `InjectionDetector` | Called once per run with the user's task string |
| Intermediate | `Budget` + `LoopDetector` + `ToolPolicy` | Called before every tool execution |
| Output | `PIIRedactor` | Called on every LLM response text |

**Guard details:**
- `InjectionDetector` — pattern-matches against known prompt injection signatures.
- `Budget` — enforces the `max_steps`, `max_tokens`, and `max_wall_seconds` from `SafetyConfig`. This guard is separate from `ctx.tick()` so that the guardrail layer independently validates.
- `LoopDetector` — scans the last `loop_window` (default 10) tool calls for repeated identical invocations, signalling an agent stuck in a loop.
- `ToolPolicy` — allows only tools in `allowed_tools` and blocks any tool in `blocked_tools`.
- `PIIRedactor` — strips SSN, phone numbers, emails, and credit card numbers from LLM output before it is stored or returned to the caller.

`get_default_config()` returns sensible per-agent-type defaults:
- **sql** — allows only `execute_sql`, `list_tables`, `describe_table`, `sample_rows`; PII redaction and injection detection on.
- **code** — allows `run_python`, `lint_code`, `read_file`, `write_file`, `apply_patch`, `list_workspace`; destructive commands off.
- **research** — read-only file tools only.

When the `guardrail` package is not installed, `build_pipeline()` returns a `_NullPipeline` that passes all checks silently, so the harness can run in environments without the safety library.

---

### Key methods

| Method | What it does |
|---|---|
| `build_pipeline(agent_type, config, budget)` | Assembles and returns the configured `Pipeline` (or `_NullPipeline` if guardrail absent) |
| `get_default_config(agent_type)` | Returns a sensible `SafetyConfig` for `sql`, `code`, `research`, or generic agents |
| `pipeline.check_input(payload)` | Checks input payload; returns guard result with `blocked` and `reason` |
| `pipeline.check_step(payload)` | Checks intermediate step (tool call); returns guard result |
| `pipeline.check_output(payload)` | Checks LLM output; returns guard result with redacted content |

---

### Configuration

```python
SafetyConfig(
    max_steps=50,                        # Steps before Budget guard triggers
    max_tokens=100_000,                  # Tokens before Budget guard triggers
    max_wall_seconds=300.0,              # Seconds before Budget guard triggers
    allowed_tools=["execute_sql", ...],  # Whitelist; None = all tools allowed
    blocked_tools=[],                    # Explicit blacklist
    allow_destructive_commands=False,    # Blocks DROP TABLE, rm -rf etc.
    pii_redact_output=True,              # Enable PIIRedactor on output
    injection_detect_input=True,         # Enable InjectionDetector on input
    loop_detection=True,                 # Enable LoopDetector
    loop_window=10,                      # Scan last N steps for loops
)
```

---

### Example

```python
from harness.safety.pipeline_factory import build_pipeline, get_default_config

config = get_default_config("sql")
pipeline = build_pipeline(agent_type="sql", config=config)

# Check user input before starting the agent
result = await pipeline.check_input({"content": user_task})
if result.blocked:
    return {"error": f"Input blocked: {result.reason}"}

# Check output before returning to user
output_result = await pipeline.check_output({"content": llm_response})
```

---

## 7. Tool Registry

**What it does** Manages all tools available to agents — validating inputs against JSON Schema, running safety checks, executing with a per-tool timeout, and emitting audit logs and Prometheus metrics for every call.

**File**: `src/harness/tools/registry.py`  
**Key class**: `ToolRegistry`  
**Used by**: `BaseAgent`

---

### How it works

`execute()` runs a strict seven-step pipeline for every tool call:

1. **Lookup** — `self._tools.get(call.name)`. Raises `ToolError(TOOL_NOT_FOUND)` if missing, including the list of registered tools in the context.
2. **Input schema validation** — `jsonschema.validate(call.args, tool.input_schema)`. Raises `ToolError(TOOL_SCHEMA_ERROR)` with the validation message on failure.
3. **Safety check** — Calls `safety_pipeline.check_step({"tool_name": ..., "args": ...})`. Raises `SafetyViolation(SAFETY_STEP)` if blocked.
4. **Execute with timeout** — `asyncio.timeout(tool.timeout_seconds)` wraps the actual `tool.execute(ctx, call.args)` call. Times out as `ToolError(TOOL_TIMEOUT)`.
5. **Output schema validation** — If the tool declares `output_schema`, the result is validated. Failures are non-blocking: they attach an error message to the result but do not raise.
6. **Audit log** — `_audit_logger.log(event_type="tool_executed", ...)` records the tool name, ID, args, and whether it errored.
7. **Prometheus counter** — Increments `harness_tool_calls_total{tool_name, success}`.

Registration is done via `register(tool: ToolExecutor)` where `ToolExecutor` is a protocol requiring `name`, `description`, `input_schema`, `timeout_seconds`, and an `execute(ctx, args)` coroutine.

`to_anthropic_format()` and `to_openai_format()` convert the registry's tools to the JSON formats expected by each provider's API, so the same registry works with any LLM.

---

### Key methods

| Method | What it does |
|---|---|
| `register(tool)` | Adds a `ToolExecutor` to the registry; warns if overwriting |
| `execute(ctx, call)` | Seven-step pipeline: lookup → schema → safety → execute → output schema → audit → metrics |
| `get(name)` | Returns the registered `ToolExecutor` or `None` |
| `list_tools(agent_type)` | Returns all registered tools |
| `to_anthropic_format()` | Returns tool list in Anthropic API format (`name`, `description`, `input_schema`) |
| `to_openai_format()` | Returns tool list in OpenAI function-calling format |

---

### Configuration

| Field | Default | Description |
|---|---|---|
| `tool.timeout_seconds` | `30.0` | Per-tool execution timeout |
| `safety_pipeline` | `None` | Optional safety pipeline for `check_step()` |
| `audit_logger` | `None` | Optional audit logger |
| `metrics` | `None` | Optional `HarnessMetrics` for counters |

---

### Example

```python
from harness.tools.registry import ToolRegistry

registry = ToolRegistry(
    safety_pipeline=pipeline,
    audit_logger=audit_logger,
)
registry.register(my_sql_tool)

# Used internally by BaseAgent:
result = await registry.execute(ctx, ToolCall(id="tc1", name="execute_sql", args={"query": "SELECT 1"}))
if result.is_error:
    print(f"Error: {result.error}")
else:
    print(result.to_text())  # JSON-serialised output
```

---

## 8. MCP Client

**What it does** Connects to any MCP (Model Context Protocol) server over `stdio` or SSE transports, auto-discovers its tools, and wraps each tool as a `ToolExecutor` that can be registered directly in `ToolRegistry`.

**File**: `src/harness/tools/mcp_client.py`  
**Key class**: `MCPToolAdapter`, `MCPToolWrapper`  
**Used by**: Agent startup code, `ToolRegistry`

---

### How it works

`MCPToolAdapter` manages a single MCP server connection. On `connect()`, it:

1. Selects the transport: for `stdio`, it spawns the configured subprocess command via `StdioServerParameters`; for `sse`, it opens an HTTP SSE connection to the configured URL.
2. Opens a `ClientSession`, calls `session.initialize()`, then calls `session.list_tools()` to discover all tools the server exposes.
3. Wraps each discovered tool as an `MCPToolWrapper`, which stores the session reference and implements the `ToolExecutor` protocol.

`MCPToolWrapper.execute()` calls `session.call_tool(name, arguments=args)`, iterates the MCP content blocks in the response, and returns a `ToolResult`. Error blocks (`block.type == "error"`) are surfaced as `ToolResult.error`. Timeouts raise `ToolError(TOOL_TIMEOUT)` and other exceptions raise `ToolError(MCP_TOOL_ERROR)`.

Environment variables in server configurations are interpolated using `${ENV_VAR}` syntax, parsed from `configs/mcp_servers.yaml` via `load_mcp_servers_from_config()`.

---

### Key methods

| Method | What it does |
|---|---|
| `connect()` | Establishes transport, initializes MCP session, discovers tools; returns `list[MCPToolWrapper]` |
| `disconnect()` | Closes session and underlying transport streams |
| `list_resources()` | Lists MCP resources (documents, schemas) exposed by the server |
| `MCPToolWrapper.execute(ctx, args)` | Calls the MCP tool via `session.call_tool()`; returns `ToolResult` |
| `load_mcp_servers_from_config(config_path)` | Parses `mcp_servers.yaml` and returns `list[MCPServerConfig]` |

---

### Configuration (`configs/mcp_servers.yaml`)

```yaml
servers:
  - name: my_db_server
    transport: stdio
    command: ["python", "-m", "my_mcp_server"]
    env:
      DB_PASSWORD: "${DB_PASSWORD}"
    timeout: 30.0

  - name: web_search
    transport: sse
    url: "http://localhost:8080/sse"
    timeout: 15.0
```

---

### Example

```python
from harness.tools.mcp_client import MCPToolAdapter, MCPServerConfig

adapter = MCPToolAdapter(MCPServerConfig(
    name="analytics_server",
    transport="stdio",
    command=["python", "-m", "analytics_mcp"],
))

tools = await adapter.connect()
for tool in tools:
    registry.register(tool)   # Plug directly into ToolRegistry

# Later:
await adapter.disconnect()
```

---

## 9. Circuit Breaker

**What it does** Protects every LLM provider from cascading failures by tracking consecutive errors and refusing calls when a provider is unhealthy, then automatically probing recovery after a configurable timeout.

**File**: `src/harness/core/circuit_breaker.py`  
**Key class**: `CircuitBreaker`, `CircuitBreakerRegistry`  
**Used by**: `LLMRouter`

---

### How it works

Each `CircuitBreaker` is a state machine with three states:

```
CLOSED ──(5 failures)──► OPEN ──(60 s timeout)──► HALF_OPEN ──(2 successes)──► CLOSED
                                                         │
                                              (any failure)──► OPEN
```

All state transitions are protected by an `asyncio.Lock` so concurrent calls don't race.

The primary API is the `call()` context manager, used as `async with breaker.call():`. The `_CircuitBreakerCall` context manager calls `can_proceed()` on entry — which raises `CircuitOpenError` immediately if the breaker is OPEN — and calls `record_success()` or `record_failure()` on exit depending on whether an exception was raised.

The `protect` property is a decorator equivalent: `@cb.protect` wraps an async function so every call goes through the breaker.

`CircuitBreakerRegistry` is a central store of named `CircuitBreaker` instances. `LLMRouter` uses `get_or_create(name=f"{provider_name}:{model}")` to ensure each provider has exactly one breaker. `all_states()` returns a snapshot of all breaker states, which is surfaced in the `GET /metrics` endpoint as the `harness_circuit_breaker_state` gauge.

---

### Key methods

| Method | What it does |
|---|---|
| `call(fn, *args, **kwargs)` | Context manager or direct call guard; raises `CircuitOpenError` if OPEN |
| `protect` | Decorator that wraps an async function with this circuit breaker |
| `record_success()` | Increments success counter; closes circuit when `success_threshold` reached |
| `record_failure()` | Increments failure counter; opens circuit when `failure_threshold` reached |
| `can_proceed()` | Returns `True` if calls may proceed; transitions OPEN→HALF_OPEN if timeout elapsed |
| `CircuitBreakerRegistry.get_or_create(name, ...)` | Returns or creates a named breaker with given thresholds |
| `CircuitBreakerRegistry.all_states()` | Snapshot of all breaker states |
| `CircuitBreakerRegistry.reset(name)` | Forcibly closes a named breaker |

---

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `failure_threshold` | `5` | Consecutive failures before opening |
| `recovery_timeout` | `60.0` s | Seconds to wait before probing recovery |
| `success_threshold` | `2` | Successes in HALF_OPEN before closing |

---

### Example

```python
from harness.core.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(name="claude", failure_threshold=5, recovery_timeout=60.0)

# Context manager usage
async with breaker.call():
    response = await anthropic_client.complete(...)

# Decorator usage
@breaker.protect
async def call_claude(messages):
    return await anthropic_client.complete(messages)
```

---

## 10. Cost Tracker

**What it does** Records the USD cost of every LLM call per run and per tenant in Redis, and enforces configurable monthly budget caps that raise an error before spend overruns.

**File**: `src/harness/core/cost_tracker.py`  
**Key class**: `CostTracker`, `MODEL_COSTS`  
**Used by**: `BaseAgent`

---

### How it works

`record()` uses `MODEL_COSTS` — a dict of per-million-token prices — to compute the USD cost for a given model and token counts:

```python
cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
```

Prefix matching handles versioned model names: `claude-sonnet-4-6-20250101` matches the `claude-sonnet-4-6` entry.

After computing the cost, a Redis `pipeline` (non-transactional for performance) executes three operations atomically:
1. `HSET harness:cost_ledger:{run_id}` — stores the full `RunCost` record with a 90-day TTL.
2. `INCRBYFLOAT harness:tenant_spend:{tenant_id}:{YYYY-MM}` — accumulates the monthly spend float with a 32-day TTL.
3. `INCRBYFLOAT harness:tenant_spend:{tenant_id}:{YYYY-MM-DD}` — accumulates the daily spend float with a 2-day TTL.

`check_budget()` fetches the current monthly spend and raises `RateLimitError(LLM_RATE_LIMIT)` when it meets or exceeds the configured limit. `BaseAgent` calls this before each LLM call.

The `MODEL_COSTS` dict covers Anthropic Claude models (`claude-sonnet-4-6` at $3/$15 per M tokens), OpenAI GPT-4o, GPT-5, o-series reasoning models, and all local providers at $0.

---

### Key methods

| Method | What it does |
|---|---|
| `record(run_id, tenant_id, model, input_tokens, output_tokens)` | Computes cost, writes ledger record, increments monthly and daily spend; returns `RunCost` |
| `check_budget(tenant_id)` | Returns `True` if under budget; raises `RateLimitError` if over |
| `get_tenant_spend(tenant_id, window)` | Returns total USD spend for `"month"` or `"day"` |
| `get_run_cost(run_id)` | Retrieves the `RunCost` record for a specific run |

---

### Configuration

| Environment variable / constructor | Default | Description |
|---|---|---|
| `budget_usd_per_tenant` | `$100.00` | Monthly USD budget per tenant |
| `REDIS_URL` | `redis://localhost:6379` | Redis for spend counters |

---

### Example

```python
tracker = CostTracker(redis_client=redis, budget_usd_per_tenant=50.0)

await tracker.check_budget("acme")          # Raises if over $50/month

run_cost = await tracker.record(
    run_id="abc123",
    tenant_id="acme",
    model="claude-sonnet-4-6",
    input_tokens=1200,
    output_tokens=300,
)
print(f"This call: ${run_cost.cost_usd:.6f}")

spend = await tracker.get_tenant_spend("acme", window="month")
print(f"Monthly spend: ${spend:.4f}")
```

---

## 11. Rate Limiter

**What it does** Enforces per-tenant, per-resource request-per-minute limits using a Redis sorted-set sliding window that expires old entries automatically.

**File**: `src/harness/core/rate_limiter.py`  
**Key class**: `RateLimiter`, `RateLimitResult`  
**Used by**: `RateLimitMiddleware` (FastAPI), any component that calls `require()`

---

### How it works

`check()` implements a Redis sliding-window counter using a sorted set:

```
key = "harness:rate_limit:{tenant_id}:{resource}"

ZREMRANGEBYSCORE key -inf (now - window_seconds)   # expire old entries
ZRANGE key 0 -1 WITHSCORES                         # count current window
ZADD key {now} "{now}:{unique_id}"                 # record this request
EXPIRE key {window_seconds * 2}                    # ensure cleanup
```

The unique member value (`{now}:{id(object())}`) ensures concurrent requests from the same tenant at the same millisecond don't collide in the sorted set. The current count is compared against the effective limit; if the limit is met or exceeded, the new entry is immediately removed with `ZREMRANGEBYSCORE` and the function returns `allowed=False` with `retry_after` computed from the oldest entry in the window.

`require()` is a thin wrapper that calls `check()` and raises `RateLimitError` if not allowed, carrying `retry_after` so callers can surface the value in HTTP 429 responses.

`RateLimitMiddleware` integrates with FastAPI. It reads `X-Tenant-ID` from the request headers (falling back to `"anonymous"`) and calls `require(tenant_id, resource="api")` before forwarding to the next handler. On success it appends `X-RateLimit-Remaining` and `X-RateLimit-Reset` headers to the response.

---

### Key methods

| Method | What it does |
|---|---|
| `check(tenant_id, resource, cost, limit)` | Checks and records the request; returns `RateLimitResult` |
| `require(tenant_id, resource, cost, limit)` | Same as `check()` but raises `RateLimitError` when denied |
| `RateLimitMiddleware.dispatch(request, call_next)` | FastAPI middleware that enforces limits on every incoming HTTP request |

---

### Configuration

| Constructor argument | Default | Description |
|---|---|---|
| `default_rpm` | `60` | Requests per minute per tenant per resource |
| `window_seconds` | `60` | Sliding window duration |
| `tenant_header` | `X-Tenant-ID` | HTTP header used to identify the tenant |

---

### Example

```python
limiter = RateLimiter(redis_client=redis, default_rpm=60)

result = await limiter.check("acme", resource="llm_calls", limit=10)
if not result.allowed:
    print(f"Rate limited. Retry in {result.retry_after:.1f}s")
else:
    print(f"Allowed. {result.remaining} requests remaining this minute.")

# Strict version (raises on deny):
await limiter.require("acme", resource="api")
```

---

## 12. HITL Manager

**What it does** Pauses agent execution when a sensitive tool call requires human review, persists the approval request in Redis with a 1-hour TTL, and lets the agent resume or abort based on the reviewer's decision.

**File**: `src/harness/orchestrator/hitl.py`  
**Key class**: `HITLManager`, `ApprovalRequest`  
**Used by**: `BaseAgent._check_hitl()`

---

### How it works

When `BaseAgent._check_hitl()` detects that a tool name matches the HITL policy (`policy.requires_hitl(call.name)`), it calls `hitl_manager.request_approval()` to create an `ApprovalRequest` and persist it in Redis under `harness:hitl:{request_id}` with a double-TTL expiry (so the record is preserved for audit even after resolution).

The agent then polls `await_decision(request_id, timeout=3600.0)` — blocking the asyncio loop cheaply while periodically checking the Redis key for status changes. When a human calls `POST /hitl/{request_id}/approve` or `/reject`, the `approve()` or `reject()` method updates the `status` field in Redis and removes the request from the pending set.

If the reviewer approves, `await_decision()` returns `"approved"` and execution continues normally. If rejected or expired, it raises `HITLRejected`, which `BaseAgent` catches and records as a failure of class `INTER_AGENT_REJECT`.

`list_pending(tenant_id)` scans all `harness:hitl:*` keys, filtering for `status == "pending"` and not expired, and returns them sorted by creation time. This is what `GET /hitl/pending` calls.

---

### Key methods

| Method | What it does |
|---|---|
| `request_approval(run_id, tenant_id, tool_name, tool_args, reason)` | Creates and persists an `ApprovalRequest`; returns the request object |
| `approve(request_id, resolved_by)` | Sets status to `approved`, timestamps, removes from pending; raises if already resolved/expired |
| `reject(request_id, resolved_by)` | Sets status to `rejected`; raises if already resolved/expired |
| `get(request_id)` | Retrieves a single `ApprovalRequest` from Redis |
| `list_pending(tenant_id)` | Returns all non-expired pending requests, optionally filtered by tenant |

---

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `ttl_seconds` | `3600` (1 hour) | Auto-expiry of approval requests |
| `event_bus` | `None` | Optional event bus for real-time HITL notifications |

---

### Example

```python
# Approve a pending request (e.g., from a webhook or UI action)
request = await hitl_manager.approve(
    request_id="abc123def456",
    resolved_by="jane.operator",
)
print(f"Approved: {request.tool_name} for run {request.run_id}")

# List all pending requests for a tenant
pending = await hitl_manager.list_pending(tenant_id="acme")
for req in pending:
    print(f"[{req.request_id[:8]}] {req.tool_name}: {req.tool_args}")
```

---

## 13. Hermes Loop

**What it does** Runs a continuous self-improvement cycle: it samples recent agent failures, asks an LLM to propose a prompt patch, evaluates the patch by replaying failing tasks, and applies it automatically (or queues it for human review).

**File**: `src/harness/improvement/hermes.py`  
**Key class**: `HermesLoop`, `PatchOutcome`  
**Used by**: `HermesWorker`, background scheduler

---

### How it works

`run_cycle(agent_type)` implements a six-step improvement loop for one agent type:

1. **Error threshold check** — `collector.count(agent_type)`. If fewer than `min_errors` (default 5) have occurred recently, the cycle is skipped.
2. **Error sampling** — `collector.get_recent(agent_type, limit=max(10, min_errors * 2))`. Returns recent `ErrorRecord` objects from the vector store.
3. **Current config retrieval** — Fetches the agent's current system prompt from `prompt_store`.
4. **Patch proposal** — `generator.generate(agent_type, errors)`. An LLM analyses the error patterns and proposes a `Patch` with an `op` field (`append`, `prepend`, `replace`, `set`, or `remove`) and a `value`.
5. **Patch evaluation** — `evaluator.score(patch, test_cases, agent_type)`. The most-recent half of the sampled errors are replayed with the proposed patch applied. The evaluator returns an `EvalResult` with a `score` from 0.0 to 1.0.
6. **Apply or queue** — Decision logic based on `score` and `auto_apply`:
   - `score >= threshold` AND `auto_apply=True` → applied immediately via `_apply_patch()`
   - `score >= threshold` AND `auto_apply=False` → status set to `"approved"`, stored for human application
   - `score < threshold` → status set to `"rejected"`, stored for review

`run_all_agents()` runs `run_cycle()` concurrently for all agent types using `asyncio.gather()`, so the entire fleet is evaluated in parallel.

`start_background()` schedules `run_all_agents()` via APScheduler's `AsyncIOScheduler` at the configured interval (default every 3600 seconds). Falls back to a plain `asyncio.sleep` loop if APScheduler is not installed.

**Auto-apply is `False` by default.** All patches land in a review queue unless `HERMES_AUTO_APPLY=true` is explicitly set.

---

### Key methods

| Method | What it does |
|---|---|
| `run_cycle(agent_type)` | One full improvement cycle for one agent type; returns `PatchOutcome` or `None` if skipped |
| `run_all_agents(agent_types)` | Concurrent cycles for all agent types; returns `list[PatchOutcome]` |
| `start_background(agent_types)` | Starts APScheduler loop at `hermes_interval_seconds` intervals |

---

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `HERMES_PATCH_SCORE_THRESHOLD` | `0.7` | Minimum score for a patch to be approved |
| `HERMES_AUTO_APPLY` | `false` | If `true`, approved patches are applied without human review |
| `HERMES_MIN_ERRORS_TO_TRIGGER` | `5` | Minimum recent errors before a cycle runs |
| `HERMES_INTERVAL_SECONDS` | `3600` | Seconds between background cycles |

---

### Example

```python
# Manual trigger from the API
outcome = await hermes_loop.run_cycle(agent_type="sql")
if outcome and outcome.applied:
    print(f"Patch applied! Score: {outcome.eval_result.score:.3f}")
elif outcome and not outcome.applied:
    print(f"Patch queued for review. Reason: {outcome.reason}")

# Run for all agents concurrently
outcomes = await hermes_loop.run_all_agents(["sql", "code"])
```

---

## 14. Planner

**What it does** Decomposes a complex multi-step task into a validated directed acyclic graph (DAG) of sub-tasks, each assigned to a specialised agent type, ready for parallel execution by the Scheduler.

**File**: `src/harness/orchestrator/planner.py`  
**Key class**: `Planner`, `TaskPlan`, `SubTask`  
**Used by**: `AgentRunner` (orchestrator mode)

---

### How it works

`plan()` sends the user's task to the LLM with a structured prompt that instructs it to output a JSON array of sub-task objects. Each object must include:
- `"id"` — a short unique identifier (e.g., `"t1"`)
- `"agent_type"` — which agent type to run
- `"task"` — a self-contained task description
- `"depends_on"` — list of prerequisite sub-task IDs (empty for independent tasks)

The response is parsed in `_parse_plan()`, which strips Markdown code fences, parses JSON, normalises agent types against the `available_agents` allowlist, and constructs `SubTask` dataclass instances.

After parsing, the plan is validated:
1. All `depends_on` IDs reference known sub-task IDs.
2. Kahn's topological sort verifies the DAG has no cycles. If a cycle is detected, `topological_order()` raises `ValueError`, and `plan()` raises accordingly.

`TaskPlan.get_ready_tasks(completed_ids)` returns the sub-tasks whose prerequisites are all in `completed_ids`. The Scheduler calls this after each task completes to determine the next parallel batch to launch.

---

### Key methods

| Method | What it does |
|---|---|
| `plan(task, available_agents, ctx)` | Calls LLM, parses response, validates DAG; returns `TaskPlan` |
| `TaskPlan.get_ready_tasks(completed_ids)` | Returns sub-tasks with all prerequisites met |
| `TaskPlan.topological_order()` | Returns sub-tasks in valid execution order via Kahn's algorithm |
| `TaskPlan.validate()` | Returns list of validation errors (unknown deps, cycles) |

---

### Configuration

The planner uses the same `LLMRouter` as agents. The prompt template is embedded in the module (`_PLAN_PROMPT_TEMPLATE`) and enforces a maximum of 10 sub-tasks.

---

### Example

```python
planner = Planner(llm_provider=router)

plan = await planner.plan(
    task="Analyse Q3 sales data and produce a board-ready summary",
    available_agents=["sql", "code"],
    ctx=ctx,
)

print(f"Plan {plan.plan_id}: {len(plan.subtasks)} sub-tasks")
completed: set[str] = set()

while True:
    ready = plan.get_ready_tasks(completed)
    if not ready:
        break
    # Launch all ready tasks in parallel
    results = await asyncio.gather(*[run_agent(st) for st in ready])
    completed.update(st.id for st in ready)
```

---

## 15. Framework Adapters

**What it does** Wraps third-party multi-agent frameworks (LangGraph, AutoGen, CrewAI) inside the HarnessAgent lifecycle, emitting standard `StepEvent` objects for every framework-specific action so full observability and budget enforcement apply regardless of the underlying framework.

**File**: `src/harness/adapters/langgraph.py`, `autogen.py`, `crewai.py`  
**Key classes**: `LangGraphAdapter`, `AutoGenAdapter`, `CrewAIAdapter`  
**Used by**: Applications integrating existing framework-based agents

---

### How it works

All three adapters extend `FrameworkAdapter` and implement two methods: `run(ctx, input)` (async generator) and `get_result()`.

**LangGraph** (`LangGraphAdapter`): Calls `compiled_graph.astream(input)`, iterating over `{node_name: node_output}` chunks. Each chunk triggers `ctx.tick()` for budget enforcement and yields a `StepEvent(event_type="tool_call")` with the node name and output keys. Token usage is extracted from LangChain `AIMessage.usage_metadata` if present. The final state is captured for `get_result()`.

**AutoGen** (`AutoGenAdapter`): AutoGen is synchronous. The adapter monkey-patches `recipient.receive` before the call to capture each message as `{"role": sender_name, "content": ...}`. `initiate_chat()` runs in `asyncio.get_running_loop().run_in_executor(None, ...)` so the event loop stays unblocked. After the chat completes, the original `receive` method is restored in a `finally` block. Each captured message is yielded as a `StepEvent(event_type="message")`.

**CrewAI** (`CrewAIAdapter`): Also synchronous. The adapter wraps `crew.step_callback` to capture individual agent step outputs (forwarding to any existing callback). `crew.kickoff(inputs=input)` runs in the thread pool. After completion, one `StepEvent(event_type="tool_call")` is yielded per captured step.

All adapters check `ctx.is_budget_ok()` before yielding each event and emit a `budget_exceeded` event to stop the stream if the budget is crossed.

---

### Key methods

| Adapter | Method | What it does |
|---|---|---|
| `LangGraphAdapter` | `run(ctx, input)` | Streams graph; emits one `StepEvent` per node |
| `LangGraphAdapter` | `get_result()` | Returns `FrameworkResult` from the final graph state |
| `AutoGenAdapter` | `run(ctx, input)` | Runs chat in thread pool; emits one `StepEvent` per message |
| `AutoGenAdapter` | `get_result()` | Returns `FrameworkResult` from the last captured message |
| `CrewAIAdapter` | `run(ctx, input)` | Runs crew in thread pool; emits one `StepEvent` per step |
| `CrewAIAdapter` | `get_result()` | Returns `FrameworkResult` from `crew.kickoff()` return value |

---

### Configuration

All adapters accept an optional `event_bus` for real-time `StepEvent` publishing. `LangGraphAdapter` additionally accepts a `step_tracer` for OTel spans.

---

### Example

```python
from harness.adapters.langgraph import LangGraphAdapter

adapter = LangGraphAdapter(graph=my_state_graph, event_bus=event_bus)

async for event in adapter.run(ctx, input={"messages": [HumanMessage(content=task)]}):
    print(f"Step {event.step}: {event.payload.get('node')}")

result = await adapter.get_result()
print(result.output)
```

---

## 16. MLflow Tracer

**What it does** Records every agent run as an MLflow experiment with nested spans for LLM calls, tool executions, and inter-agent messages, logging token counts, cost, latency, and a success flag for every run.

**File**: `src/harness/observability/mlflow_tracer.py`  
**Key class**: `MLflowAgentTracer`  
**Used by**: `BaseAgent`

---

### How it works

One MLflow **run** is opened per agent invocation via the `agent_run()` async context manager. On entry, it calls `mlflow.start_run()` with a run name of `{agent_type}_{run_id[:8]}` and logs agent parameters: `run_id`, `tenant_id`, `agent_type`, `max_steps`, `max_tokens`, and the first 256 characters of `task`. Inside the MLflow run, a parent span of type `"AGENT"` is opened.

On context exit (success or failure), final metrics are logged: `step_count`, `token_count`, `elapsed_seconds`, and `failed` (1.0 or 0.0).

**Nested spans** are opened via:
- `llm_span(model, messages, provider)` — span type `"LLM"`, inputs set to messages and model metadata.
- `tool_span(tool_name, args)` — span type `"TOOL"`, inputs set to the tool arguments.
- `inter_agent_span(sender, recipient, msg_type)` — span type `"AGENT"`, attributes set to sender, recipient, and message type.

When `mlflow` is not installed, all methods yield a `_NoOpSpan` that ignores all calls, so the harness works without MLflow in development.

`evaluate_run()` logs evaluation metrics including `eval_response_length`, `eval_task_length`, and optionally `eval_word_overlap` when an expected output is provided.

---

### Key methods

| Method | What it does |
|---|---|
| `agent_run(ctx)` | Opens MLflow run + parent AGENT span; logs params on entry, metrics on exit |
| `llm_span(model, messages, provider)` | Nested LLM span with input messages |
| `tool_span(tool_name, args)` | Nested TOOL span with input arguments |
| `inter_agent_span(sender, recipient, msg_type)` | Nested AGENT span for inter-agent communication |
| `evaluate_run(run_id, task, output, expected)` | Logs evaluation metrics to the active or a new MLflow run |
| `get_run_trace(run_id)` | Returns MLflow run data for a given harness `run_id` |

---

### Configuration

| Constructor argument | Description |
|---|---|
| `tracking_uri` | MLflow tracking server URI (e.g., `http://localhost:5000`) |
| `experiment_name` | MLflow experiment name (e.g., `"harness-agent"`) |

---

### Example

```python
tracer = MLflowAgentTracer(
    tracking_uri="http://mlflow:5000",
    experiment_name="production-agents",
)

async with tracer.agent_run(ctx) as span:
    async with tracer.llm_span("claude-sonnet-4-6", messages, "anthropic") as llm_span:
        response = await router.complete(messages)
        llm_span.set_outputs({"content": response.content[:100]})

    async with tracer.tool_span("execute_sql", {"query": "SELECT ..."}):
        result = await registry.execute(ctx, call)
```

---

## 17. Failure Tracker

**What it does** Records every agent step failure across four sinks — structured logs, Prometheus counters, a vector store (for semantic sampling by Hermes), and a Redis stream — and provides `get_heatmap()` to visualise failure distribution by agent type and failure class.

**File**: `src/harness/observability/failures.py`  
**Key class**: `FailureTracker`, `StepFailure`, `FailureClass`  
**Used by**: `BaseAgent`, `HermesLoop`, `ErrorCollector`

---

### How it works

`record(failure: StepFailure)` writes to four destinations in sequence:

1. **Structured JSON log** — `logger.error("Step failure recorded", extra={"failure": failure_dict})`. This feeds into any centralised log aggregation (Loki, CloudWatch, Datadog).
2. **Prometheus counter** — `harness_failure_total{failure_class, agent_type}.inc()`. Exposed on `GET /metrics` for alerting.
3. **Vector store** — The failure class, message, and first 500 chars of the stack trace are embedded and upserted. This enables `HermesLoop` to sample semantically similar failures using `sample_batch()`.
4. **Redis stream** — `XADD harness:failures {data: json.dumps(failure_dict)} MAXLEN 10000 ~`. The stream is capped at 10 000 entries with approximate trimming.

`get_heatmap()` reads the last 5 000 stream entries and builds a 2D count matrix `{agent_type: {failure_class: count}}`. This is the foundation of the failure heatmap UI.

`get_summary(agent_type, window_hours)` reads the last 1 000 stream entries, filters by agent type and time window, and returns top failure classes, failures by tool, and total count.

The `FailureClass` enum (imported from `core/errors.py`) defines 27 canonical failure categories across LLM, tool, MCP, safety, budget, memory, inter-agent, and orchestration domains.

---

### Key methods

| Method | What it does |
|---|---|
| `record(failure)` | Writes to log, Prometheus, vector store, and Redis stream |
| `sample_batch(agent_type, failure_class, k)` | Returns up to k `StepFailure` objects via semantic vector search |
| `get_summary(agent_type, window_hours)` | Returns aggregated stats from the Redis stream |
| `get_heatmap()` | Returns `{agent_type: {failure_class: count}}` from the Redis stream |
| `StepFailure.from_exception(exc, run_id, ...)` | Convenience constructor that captures the current traceback |

---

### The FailureClass taxonomy

| Category | Classes |
|---|---|
| LLM | `LLM_ERROR`, `LLM_TIMEOUT`, `LLM_RATE_LIMIT`, `LLM_CONTEXT_LIMIT`, `LLM_PARSE_ERROR` |
| Tool | `TOOL_NOT_FOUND`, `TOOL_SCHEMA_ERROR`, `TOOL_EXEC_ERROR`, `TOOL_TIMEOUT`, `TOOL_OUTPUT_ERROR` |
| MCP | `MCP_CONNECT_ERROR`, `MCP_TOOL_ERROR` |
| Safety | `SAFETY_INPUT`, `SAFETY_STEP`, `SAFETY_OUTPUT` |
| Budget | `BUDGET_STEPS`, `BUDGET_TOKENS`, `BUDGET_TIME` |
| Memory | `MEMORY_REDIS`, `MEMORY_VECTOR`, `MEMORY_GRAPH` |
| Inter-agent | `INTER_AGENT_TIMEOUT`, `INTER_AGENT_REJECT` |
| Orchestration | `PLAN_ERROR`, `SKILL_MISSING` |
| Catch-all | `UNKNOWN` |

---

### Example

```python
from harness.observability.failures import FailureTracker, StepFailure
from harness.core.errors import FailureClass

tracker = FailureTracker(
    vector_store=vs,
    embedder=embedder,
    redis_client=redis,
)

failure = StepFailure.from_exception(
    exc=exc,
    run_id=ctx.run_id,
    step_number=ctx.step_count,
    agent_type="sql",
    failure_class=FailureClass.TOOL_EXEC_ERROR,
    tool_name="execute_sql",
)
await tracker.record(failure)

heatmap = await tracker.get_heatmap()
# {"sql": {"TOOL_EXEC_ERROR": 12, "LLM_TIMEOUT": 3}, "code": {...}}
```
