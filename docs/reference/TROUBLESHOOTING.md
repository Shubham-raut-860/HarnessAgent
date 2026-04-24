# HarnessAgent — Troubleshooting Guide

> **Style**: Azure Architecture Center  
> **Audience**: Platform engineers, SREs, and on-call operators  
> **Purpose**: Diagnose and resolve the most common production issues in the HarnessAgent, with concrete commands, root causes, and fixes.

---

## Table of Contents

1. [Quick health check commands](#1-quick-health-check-commands)
2. [Common issues and fixes](#2-common-issues-and-fixes)
3. [How to approve a HITL request](#3-how-to-approve-a-hitl-request)
4. [How to rollback a Hermes patch](#4-how-to-rollback-a-hermes-patch)
5. [How to replay a failed run from the DLQ](#5-how-to-replay-a-failed-run-from-the-dlq)
6. [Performance tuning guide](#6-performance-tuning-guide)
7. [Understanding the failure taxonomy](#7-understanding-the-failure-taxonomy)
8. [Reading MLflow traces](#8-reading-mlflow-traces)
9. [Emergency procedures](#9-emergency-procedures)

---

## 1. Quick health check commands

Run these first whenever something looks wrong. They give a complete picture of system state in under 30 seconds.

### Check overall service health

```bash
curl -s http://localhost:8000/health | jq
```

Expected response:

```json
{
  "status": "ok",
  "redis": "connected",
  "vector_store": "connected",
  "graph": "connected",
  "llm_providers": {
    "anthropic:claude-sonnet-4-6": true,
    "openai:gpt-4o-mini": true
  }
}
```

If any field is `false` or `"disconnected"`, see the relevant section below.

### Check circuit breaker states

```bash
curl -s http://localhost:8000/metrics | grep circuit_breaker_state
```

Expected output:

```
harness_circuit_breaker_state{service="anthropic:claude-sonnet-4-6"} 0.0
harness_circuit_breaker_state{service="openai:gpt-4o-mini"} 0.0
```

State encoding: `0 = CLOSED (healthy)`, `1 = OPEN (blocking)`, `2 = HALF_OPEN (recovering)`.

Any breaker at `1.0` means that provider is currently refusing all calls.

### Check DLQ depth (dead-letter queue of failed runs)

```bash
curl -s http://localhost:8000/improvement/errors?limit=5 | jq
```

A growing DLQ indicates a systematic failure pattern that Hermes has not yet patched. See [Section 5](#5-how-to-replay-a-failed-run-from-the-dlq).

### Check pending Hermes patches awaiting review

```bash
curl -s "http://localhost:8000/improvement/patches?status=pending" | jq
```

Patches with `status: "approved"` are ready to apply. Patches with `status: "pending"` need evaluation. See [Section 4](#4-how-to-rollback-a-hermes-patch).

### Check pending HITL approval requests

```bash
curl -s http://localhost:8000/hitl/pending | jq
```

If runs are stuck with `hitl_pending: true` in their `RunRecord`, it means the agent is waiting for human approval before executing a sensitive tool. See [Section 3](#3-how-to-approve-a-hitl-request).

### Check active runs and queue depth

```bash
curl -s http://localhost:8000/metrics | grep -E "active_runs|dlq_depth|agent_runs_total"
```

### Check token and cost totals

```bash
curl -s http://localhost:8000/metrics | grep -E "agent_tokens_total|cost_usd_total"
```

---

## 2. Common issues and fixes

| Symptom | Likely cause | Diagnostic command | Fix |
|---|---|---|---|
| Agent returns empty output | LLM responded with no text and no tool calls | Check MLflow trace for the run | Review system prompt; check for empty `content` in LLM response |
| `"All providers exhausted"` error | All LLM providers failing simultaneously | `curl /health \| jq .llm_providers` | Check API keys; check internet connectivity; add a local provider as a last-resort fallback |
| Agent loops forever, never finishes | LoopDetector not triggering; same tool called repeatedly with same args | Check `agent_steps_total` counter; inspect MLflow trace | Reduce `loop_window` in `SafetyConfig`; improve system prompt to break loops |
| `"Budget exceeded"` at step 3 | `max_tokens` set too low relative to schema/context size | Check `agent_tokens_total` for the run; inspect `retrieved_tokens` in the MLflow trace | Increase `MAX_TOKENS` env var; enable Graph RAG to reduce context size |
| Graph RAG returns empty `[SCHEMA]` block | No schema nodes in the graph for this tenant | Query the graph node count directly | Run `SQLAgent` first to trigger `populate_schema()`; verify `GRAPH_TYPE` env var |
| Circuit breaker stuck in OPEN for Anthropic | Too many consecutive LLM failures (5+ in a row) | `grep circuit_breaker_state /metrics` | Wait 60 seconds for automatic HALF_OPEN probe; fix underlying API key issue; call `CircuitBreakerRegistry.reset("anthropic:claude-sonnet-4-6")` to force-close |
| HITL request expired before approval | No reviewer acted within 1 hour | `curl /hitl/pending \| jq` | Re-run the task; increase `HITL_TTL_SECONDS` env var; set up alerting on `harness:hitl:*` key creation events |
| Hermes applied a bad patch that worsened performance | `HERMES_AUTO_APPLY=true` with too-low threshold | `curl /improvement/patches?status=applied \| jq` | Immediately rollback (see Section 4); set `HERMES_AUTO_APPLY=false`; raise `HERMES_PATCH_SCORE_THRESHOLD` to 0.85 |
| Embeddings very slow (>5s per call) | Large batch size or slow embedding model | Check `tool_latency_seconds` histogram for embedding calls | Switch to a smaller embedding model; reduce vector store batch size |
| Worker process OOM (out of memory) killed | Too many tokens in context, too many steps | Check `harness_agent_tokens_total` and `active_runs` gauges | Reduce `MAX_TOKENS`; reduce `MAX_STEPS`; enable context window compression |
| MCP server not connecting | Wrong transport type or missing command in YAML | `cat configs/mcp_servers.yaml` | Test command manually in terminal; check `${ENV_VAR}` interpolation; verify `transport: stdio` vs `transport: sse` |
| `TOOL_NOT_FOUND` errors | Tool not registered before agent starts | Check logs for `Registered tool:` lines on startup | Add `registry.register(tool)` to agent startup; verify MCP server connected |
| `TOOL_SCHEMA_ERROR` | LLM passing wrong argument types | Inspect the tool call args in MLflow trace | Improve tool description and example in system prompt; add type coercion in the tool's `execute()` |
| PII appearing in stored memories | `pii_redact_output=False` or custom vector store not going through `remember()` | Grep vector store for known PII patterns | Set `pii_redact_output=True` in SafetyConfig; ensure all writes go through `MemoryManager.remember()` |
| `LLM_CONTEXT_LIMIT` errors | Messages + schema too large for the model's context window | Check `required_context` vs provider `context_window` in router | Enable Graph RAG schema compression; reduce `_MAX_HISTORY_MESSAGES`; use a provider with a larger context window |
| Hermes cycle skips every time | Fewer than `min_errors` failures recorded | Check `harness_failure_total` counter | Lower `HERMES_MIN_ERRORS_TO_TRIGGER`; verify `FailureTracker` is initialised and writing to vector store |
| `SAFETY_INPUT` blocks all user requests | Injection detector too sensitive | Check `harness_safety_blocks_total{stage="input"}` | Review `InjectionDetector` patterns; whitelist known-safe task patterns |
| `INTER_AGENT_REJECT` failures | HITL reviewer rejected a critical tool call | Look up the `ApprovalRequest` in Redis by `request_id` | Re-run the task; adjust HITL policy to not require approval for that tool type |
| Slow `GET /runs` list endpoint | Scanning all Redis keys on every call | Check `active_runs` count and `list_runs()` call time | Add a Redis sorted set index for runs per tenant; paginate aggressively |

---

### Detailed diagnostics for the top 3 issues

#### "All providers exhausted"

```bash
# 1. Check which providers are failing
curl -s http://localhost:8000/health | jq '.llm_providers'

# 2. Check circuit breaker states
curl -s http://localhost:8000/metrics | grep circuit_breaker_state

# 3. Test Anthropic API key directly
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5","max_tokens":10,"messages":[{"role":"user","content":"ping"}]}'

# 4. If circuit breaker is OPEN, wait 60s for it to probe
# OR force-reset via the internal API (development only)
curl -X POST http://localhost:8000/debug/circuit-breakers/reset \
  -d '{"name": "anthropic:claude-sonnet-4-6"}'
```

#### Circuit breaker OPEN on Anthropic

The circuit breaker opens after 5 consecutive failures. It waits 60 seconds, then moves to HALF_OPEN and allows one probe call. If the probe succeeds twice, the circuit closes. If the probe fails, it reopens.

```bash
# Check when it opened (look at LLM errors in logs)
docker logs harness-worker --since 5m | grep "Circuit breaker"

# Force-close if you've verified the underlying issue is fixed
# (do this only if you know the API key is valid and Anthropic is up)
python3 -c "
import asyncio
from harness.core.circuit_breaker import default_registry
asyncio.run(default_registry.reset('anthropic:claude-sonnet-4-6'))
print('Circuit closed')
"
```

#### Agent loops forever

```bash
# 1. Check the run's step count
curl http://localhost:8000/runs/{run_id} | jq '.result.steps'

# 2. Check MLflow for repeated tool calls
# Open MLflow UI: http://localhost:5000
# Find the run → look at tool:execute_sql spans → check if query is identical across steps

# 3. Check loop detector config
grep LOOP_WINDOW .env
# Default is 10 — the detector scans the last 10 steps for repeated (tool_name, args) pairs

# 4. Cancel the stuck run
curl -X DELETE http://localhost:8000/runs/{run_id}
```

---

## 3. How to approve a HITL request

When an agent encounters a tool that requires human approval (e.g., a `DELETE` query or a write to a production database), it pauses and creates a `ApprovalRequest` in Redis. The run's `RunRecord` has `hitl_pending: true`.

### Find pending requests

```bash
curl -s http://localhost:8000/hitl/pending | jq '.[].request_id, .[].tool_name, .[].tool_args'
```

### Review a specific request

```bash
curl http://localhost:8000/hitl/{request_id} | jq
# Shows: run_id, tenant_id, tool_name, tool_args, reason, expires_at
```

### Approve a request

```bash
curl -X POST http://localhost:8000/hitl/{request_id}/approve \
  -H "Content-Type: application/json" \
  -d '{"resolved_by": "jane.operator"}'
```

After approval, the HITLManager sets `status: "approved"` in Redis. The agent's `await_decision()` call detects this change and resumes execution.

### Reject a request

```bash
curl -X POST http://localhost:8000/hitl/{request_id}/reject \
  -H "Content-Type: application/json" \
  -d '{"resolved_by": "jane.operator"}'
```

After rejection, the agent receives a `HITLRejected` exception, which terminates the run with `failure_class: "INTER_AGENT_REJECT"`.

### Expired requests

Requests expire after 1 hour (configurable via `HITL_TTL_SECONDS`). Expired requests automatically transition to `status: "expired"` when `list_pending()` encounters them. The agent that was waiting receives an `HITLRejected` exception (expired).

```bash
# Identify expired requests
curl http://localhost:8000/hitl/pending | jq '.[] | select(.status == "expired")'

# Increase TTL for future requests
echo "HITL_TTL_SECONDS=7200" >> .env
# Then restart the API and workers
```

---

## 4. How to rollback a Hermes patch

Hermes stores all applied patches in the prompt store with version history. If an applied patch degrades performance, you can roll back to the previous version.

### List prompt versions for an agent type

```bash
curl http://localhost:8000/prompts/sql/versions | jq
# Returns: [{version_id, created_at, score, status, preview}, ...]
```

### Roll back one version

```bash
curl -X POST http://localhost:8000/prompts/sql/rollback | jq
# Rolls back to the previous version immediately
# The response includes the version that was activated
```

### Roll back to a specific version

```bash
# 1. List versions to find the target version_id
curl http://localhost:8000/prompts/sql/versions | jq '.[].version_id'

# 2. Promote that specific version
curl -X POST http://localhost:8000/prompts/sql/promote/{version_id} | jq
```

### Prevent future auto-applies

```bash
# Disable auto-apply entirely (safest setting for production)
echo "HERMES_AUTO_APPLY=false" >> .env

# Raise the score threshold
echo "HERMES_PATCH_SCORE_THRESHOLD=0.85" >> .env

# Restart workers
docker restart harness-worker harness-hermes-worker
```

### Review pending patches before they are applied

```bash
# List patches awaiting human review
curl "http://localhost:8000/improvement/patches?status=approved" | jq

# Inspect a specific patch
curl http://localhost:8000/improvement/patches/{patch_id} | jq '.op, .path, .value, .score'

# Apply a specific approved patch manually
curl -X POST http://localhost:8000/improvement/patches/{patch_id}/apply | jq

# Reject a patch
curl -X DELETE http://localhost:8000/improvement/patches/{patch_id} | jq
```

---

## 5. How to replay a failed run from the DLQ

The dead-letter queue collects error records from runs that failed. Hermes reads from this queue, but you can also inspect it and trigger manual cycles.

### Inspect the DLQ

```bash
# Get recent error records
curl "http://localhost:8000/improvement/errors?limit=10" | jq

# Get errors for a specific agent type
curl "http://localhost:8000/improvement/errors?agent_type=sql&limit=10" | jq

# Get failure heatmap (agent_type × failure_class count matrix)
curl http://localhost:8000/improvement/heatmap | jq
```

### Replay a failed run

There is no automatic replay of a specific run (replay would re-run the same failed task, which may fail again for the same reason). Instead, the recommended approach is:

1. Diagnose the root cause using the error record and MLflow trace.
2. Fix the underlying issue (update system prompt, fix a tool, adjust config).
3. Create a new run with the same task:

```bash
curl -X POST http://localhost:8000/runs \
  -H "X-Tenant-ID: acme" \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "sql",
    "task": "<same task as failed run>",
    "metadata": {"replay_of": "<original_run_id>"}
  }'
```

### Trigger a Hermes cycle manually

If the DLQ is deep and you want Hermes to analyse it now rather than waiting for the hourly trigger:

```bash
# Trigger a cycle for a specific agent type
curl -X POST "http://localhost:8000/improvement/cycle?agent_type=sql" | jq

# Trigger cycles for all agent types
curl -X POST "http://localhost:8000/improvement/cycle" | jq
```

The response shows whether a patch was proposed, its score, and whether it was applied or queued.

---

## 6. Performance tuning guide

Use this table when a specific metric is outside acceptable bounds.

| Issue | Metric to check | Knob to turn | How |
|---|---|---|---|
| Slow end-to-end response time | `harness_agent_latency_seconds` p95 | Reduce `MAX_STEPS`; use faster model | Lower `max_steps` in `SafetyConfig`; switch to `claude-haiku-4-5` or `gpt-4o-mini` |
| High token cost per run | `harness_agent_tokens_total` | Enable Graph RAG; tune context window | Ensure `SQLAgent._populate_schema()` is called; increase `HERMES_INTERVAL_SECONDS` to improve prompts over time |
| Too many LLM steps (agent takes many rounds to finish) | `harness_agent_steps_total` | Improve system prompt (via Hermes) | Lower `HERMES_MIN_ERRORS_TO_TRIGGER` to make Hermes more aggressive; manually tune the system prompt |
| Circuit breakers opening frequently | `harness_circuit_breaker_state` | Add more fallback providers | Register a vLLM or SGLang provider at priority 100 as the last resort; reduce `circuit_failure_threshold` to detect issues earlier |
| High DLQ depth | `harness_dlq_depth` | Run Hermes cycles; improve prompts | Trigger manual Hermes cycle; review failure heatmap to identify the dominant failure class |
| Tool calls timing out | `harness_tool_latency_seconds` p99 for the slow tool | Increase `tool.timeout_seconds` | Update the tool's `timeout_seconds` attribute; optimise the underlying operation |
| High safety block rate | `harness_safety_blocks_total` | Tune safety config | Reduce injection detection sensitivity for known-safe task patterns; adjust `allowed_tools` list |
| Slow memory retrieval | `harness_graph_rag_hops` histogram | Reduce `max_hops`; optimise graph | Reduce `max_hops` from 2 to 1; ensure Neo4j has appropriate indexes; reduce `vector_k` |
| Memory growing without bound | Redis `DBSIZE` command | Set TTLs on Redis keys | Verify `push_message()` keys have TTL; call `memory.clear_session(run_id)` after completion |
| Hermes patch quality poor | `eval_result.score` in patch records | Improve evaluator test cases | Add more diverse test cases to the eval dataset; increase `max_errors_in_prompt` for richer context |
| Worker OOM (Out of Memory) | Process memory via `ps aux` | Limit context size | Reduce `MAX_TOKENS`; reduce `_MAX_HISTORY_MESSAGES` in `base.py`; run fewer concurrent workers |

### Finding the bottleneck quickly

```bash
# Latency breakdown by component
curl -s http://localhost:8000/metrics | grep -E "latency_seconds" | sort -t'{' -k2

# Top failure classes in the last hour
curl http://localhost:8000/improvement/errors/summary?window_hours=1 | jq '.top_classes'

# Cost per tenant this month
curl http://localhost:8000/costs/monthly | jq
```

---

## 7. Understanding the failure taxonomy

The harness classifies every failure into a `FailureClass` (defined in `src/harness/core/errors.py`). This table maps failure classes to their most common causes and remediations.

| FailureClass | Meaning | Common causes | Fix |
|---|---|---|---|
| `LLM_ERROR` | Provider returned an error | API key invalid; model deprecated; provider outage | Check API key; check provider status page |
| `LLM_TIMEOUT` | LLM call exceeded timeout | Heavy load on provider; very large prompt | Retry with backoff; reduce prompt size |
| `LLM_RATE_LIMIT` | Provider rate limit exceeded | Too many concurrent runs; token budget exceeded | Implement request queuing; increase per-tenant budget; add a local fallback |
| `LLM_CONTEXT_LIMIT` | Prompt too large for model | Schema too large; history too long | Enable Graph RAG; reduce `_MAX_HISTORY_MESSAGES` |
| `LLM_PARSE_ERROR` | Could not parse LLM response | Model returned malformed JSON or tool calls | Add output format instructions to system prompt; enable structured output mode |
| `TOOL_NOT_FOUND` | Tool name not in registry | Typo in LLM tool call; tool not registered | Register the tool; improve tool name description |
| `TOOL_SCHEMA_ERROR` | Args failed JSON Schema validation | LLM passed wrong arg types | Improve tool description with examples; add type hints |
| `TOOL_EXEC_ERROR` | Tool raised an unhandled exception | Bug in tool; external service down | Fix the tool; add error handling |
| `TOOL_TIMEOUT` | Tool exceeded `timeout_seconds` | Slow database; slow external API | Increase `timeout_seconds`; optimise the operation |
| `MCP_CONNECT_ERROR` | Cannot connect to MCP server | Server not running; wrong URL/command | Start the MCP server; check YAML config |
| `MCP_TOOL_ERROR` | MCP tool returned an error block | Bug in MCP server | Check MCP server logs |
| `SAFETY_INPUT` | Input guard blocked user task | Prompt injection detected | Review and whitelist legitimate patterns |
| `SAFETY_STEP` | Intermediate guard blocked tool call | Tool not in allowed list; ToolPolicy violation | Add tool to `allowed_tools` if legitimate |
| `SAFETY_OUTPUT` | Output guard blocked LLM response | PII in output; toxicity detected | Check model behaviour; add post-processing |
| `BUDGET_STEPS` | Exceeded `max_steps` | Looping agent; complex task needs more steps | Increase `max_steps`; fix loop via Hermes |
| `BUDGET_TOKENS` | Exceeded `max_tokens` | Large context; many LLM calls | Increase `max_tokens`; enable Graph RAG |
| `BUDGET_TIME` | Exceeded `timeout_seconds` | Slow tools; slow LLM | Increase `timeout_seconds`; optimise tools |
| `MEMORY_REDIS` | Redis unavailable | Redis down; network issue | Restart Redis; check connectivity |
| `MEMORY_VECTOR` | Vector store unavailable | Chroma/Qdrant/Weaviate down | Restart vector store; check disk space |
| `MEMORY_GRAPH` | Graph store unavailable | NetworkX serialisation error; Neo4j down | Restart graph store; check Neo4j logs |
| `INTER_AGENT_TIMEOUT` | Inter-agent call timed out | Sub-agent taking too long | Increase `timeout_seconds`; optimise sub-agent |
| `INTER_AGENT_REJECT` | HITL reviewer rejected the action | Deliberate rejection; expiry | Re-run with different parameters |
| `PLAN_ERROR` | Planner could not parse LLM plan | Bad JSON from planner LLM | Check planner system prompt; use structured output |
| `UNKNOWN` | Unclassified error | Bug in harness code | Check logs; file a bug report |

---

## 8. Reading MLflow traces

Every agent run produces a nested MLflow trace. Here is how to navigate it:

### Finding a run in MLflow

```bash
# Get the MLflow run ID for a harness run
curl http://localhost:8000/runs/{run_id} | jq '.result.mlflow_run_id'

# Open MLflow UI
open http://localhost:5000
# Navigate to: Experiments → harness-agent → find the run by name (sql_{run_id[:8]})
```

### What each span type means

| Span type | Name pattern | What it shows |
|---|---|---|
| `AGENT` | `agent:sql` | Full run: params (task, max_steps), metrics (step_count, token_count, elapsed, failed) |
| `LLM` | `llm:anthropic/claude-sonnet-4-6` | One LLM call: input messages, model, tokens used |
| `TOOL` | `tool:execute_sql` | One tool execution: arguments, output |
| `AGENT` | `inter_agent:sql→code` | Inter-agent message: sender, recipient, message type |

### Diagnosing from the trace

1. **High step count** — Look at the sequence of `tool:` spans. Repeated identical calls signal a loop.
2. **High token count on a single LLM span** — The input messages were very large. Check if Graph RAG was active (small `[SCHEMA]` block) or if the full raw schema was injected.
3. **`failed=1.0`** — Look at the final span. The `failure_class` attribute shows the classification.
4. **Slow overall run** — Compare wall time across spans. A slow `tool:execute_sql` span points to a database issue, not an LLM issue.
5. **`cached=true` on LLM spans** — Prompt caching is working. High cache hit rates reduce cost significantly.

### Evaluation metrics

```bash
# Trigger evaluation of a completed run
curl -X POST http://localhost:8000/runs/{run_id}/evaluate \
  -H "Content-Type: application/json" \
  -d '{"expected_output": "The database contains 5 tables: orders, users..."}'
```

This calls `MLflowAgentTracer.evaluate_run()`, which logs `eval_word_overlap` to the MLflow run.

---

## 9. Emergency procedures

### Scenario: All agents failing simultaneously

1. Check if Redis is up: `redis-cli ping`
2. Check circuit breakers: `curl /metrics | grep circuit_breaker_state`
3. Check API keys: `curl /health | jq .llm_providers`
4. Scale down workers to stop new runs from failing: `docker scale harness-worker=0`
5. Fix the root cause.
6. Scale workers back up: `docker scale harness-worker=2`

### Scenario: Budget runaway (tenant spending too much)

```bash
# Check current spend
curl http://localhost:8000/costs/tenant/acme | jq

# Emergency: reduce the budget cap immediately
curl -X PATCH http://localhost:8000/tenants/acme \
  -H "Content-Type: application/json" \
  -d '{"budget_usd_per_month": 10.0}'

# Cancel all pending runs for the tenant
curl http://localhost:8000/runs?status=pending | jq '.[].run_id' | \
  xargs -I{} curl -X DELETE http://localhost:8000/runs/{}
```

### Scenario: Hermes applied a bad patch and agents are degraded

```bash
# 1. Disable Hermes immediately
echo "HERMES_AUTO_APPLY=false" >> .env
docker restart harness-hermes-worker

# 2. Rollback the problematic prompt
curl -X POST http://localhost:8000/prompts/sql/rollback | jq

# 3. Verify agents are recovering
curl http://localhost:8000/metrics | grep agent_runs_total

# 4. Investigate the bad patch
curl "http://localhost:8000/improvement/patches?status=applied" | jq '.[0]'
```

### Scenario: Redis running out of memory

```bash
# Check Redis memory
redis-cli INFO memory | grep used_memory_human

# Check for large keys
redis-cli --bigkeys

# Flush only the rate limit keys (safe)
redis-cli --scan --pattern "harness:rate_limit:*" | xargs redis-cli DEL

# Flush failure stream (Hermes will start fresh)
redis-cli DEL harness:failures

# If disk space allows, increase Redis maxmemory
# redis.conf: maxmemory 2gb
```

### Scenario: Worker processes stuck and not processing runs

```bash
# Check RQ queue depth
redis-cli llen harness:queue:sql

# Check if workers are running
ps aux | grep agent_worker

# Restart workers
docker restart harness-worker

# If jobs are stuck in "started" state in RQ (worker died mid-job)
# Use RQ Dashboard or rq-cli to requeue
rq requeue --all --queue sql
```
