# HarnessAgent API Reference

Production-grade multi-agent orchestration harness

## GET /health
**Health**

Aggregate health check for all service dependencies.

Returns:
    200 with status="ok" if all services healthy.
    200 with status="degraded" if any service is unhealthy (but process is alive).

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |

## GET /health/ready
**Readiness**

Kubernetes readiness probe.

The service is ready when Redis (our primary state store) is reachable.

Returns:
    200 if ready, 503 if not ready.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |

## GET /health/live
**Liveness**

Kubernetes liveness probe.

Always returns 200 as long as the process is alive.

Returns:
    200 always.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |

## GET /metrics
**Metrics**

Prometheus metrics endpoint in text exposition format.

Returns:
    text/plain with Prometheus metrics.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |

## POST /runs
**Create Run**

Create a new agent run and enqueue it for a worker.

Args:
    body:      Run creation payload.
    tenant_id: Extracted from JWT.
    redis:     Redis client from app state.

Returns:
    201 with the created RunRecord.

Raises:
    400 if agent_type is not recognised.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `X-API-Key` | header | No | string |  |

### Request Body

Requires `CreateRunRequest` JSON object.

### Responses

| Code | Description |
|---|---|
| 201 | Successful Response |
| 422 | Validation Error |

## GET /runs
**List Runs**

List runs for the authenticated tenant.

Args:
    limit:     Maximum number of runs to return (1-100).
    offset:    Pagination offset.
    tenant_id: Extracted from JWT.
    redis:     Redis client.

Returns:
    200 with list of RunRecords, newest first.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `limit` | query | No | integer |  |
| `offset` | query | No | integer |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /runs/{run_id}
**Get Run**

Retrieve a run by ID.

Args:
    run_id:    The run identifier.
    tenant_id: Extracted from JWT.
    redis:     Redis client.

Returns:
    200 with RunRecord.

Raises:
    404 if run not found.
    403 if run belongs to a different tenant.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `run_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## DELETE /runs/{run_id}
**Cancel Run**

Cancel a run that is pending or running.

Args:
    run_id:    The run to cancel.
    tenant_id: Extracted from JWT.
    redis:     Redis client.

Returns:
    204 on success.

Raises:
    404 if run not found.
    403 if run belongs to another tenant.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `run_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 204 | Successful Response |
| 422 | Validation Error |

## GET /runs/{run_id}/stream
**Stream Run Events**

Stream run events as Server-Sent Events (SSE).

Subscribes to the Redis event stream for *run_id* and forwards events
as SSE until the run completes or the timeout expires.

Each event is a JSON-encoded StepEvent payload::

    data: {"event_type": "token_delta", "payload": {"delta": "Hello"}, ...}

Args:
    run_id:  The run to stream events from.
    timeout: Maximum seconds to keep the connection open (default 300).

Returns:
    text/event-stream response.

Raises:
    404 if run not found.
    403 if run belongs to another tenant.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `run_id` | path | Yes | string |  |
| `timeout` | query | No | number |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /runs/{run_id}/steps
**Stream Run Steps**

Stream real-time step events for a run via Server-Sent Events.

The stream uses a Redis pub/sub channel keyed by run_id.  Workers
publish StepEvent JSON objects to this channel as the agent executes.
A ``[DONE]`` sentinel is sent when the run completes or the connection
times out.

Args:
    run_id:    The run to stream.
    request:   FastAPI request object (for disconnect detection).
    tenant_id: Extracted from JWT.
    redis:     Redis client.

Returns:
    EventSourceResponse with text/event-stream content type.

Raises:
    404 if the run is not found.
    503 if the SSE library is not available.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `run_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /memory/remember
**Remember**

Store a text entry in long-term memory.

Args:
    body:      The text and metadata to store.
    tenant_id: From JWT.
    memory:    MemoryManager.

Returns:
    201 with the ID of the stored entry.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `X-API-Key` | header | No | string |  |

### Request Body

Requires `RememberRequest` JSON object.

### Responses

| Code | Description |
|---|---|
| 201 | Successful Response |
| 422 | Validation Error |

## POST /memory/recall
**Recall**

Retrieve relevant memories via semantic search.

Args:
    body:      Query, k, and optional filter.
    tenant_id: From JWT.
    memory:    MemoryManager.

Returns:
    List of MemoryEntry objects ordered by relevance.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `X-API-Key` | header | No | string |  |

### Request Body

Requires `RecallRequest` JSON object.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /memory/graph
**Query Graph**

Retrieve knowledge graph context for a query.

Performs entity extraction on the query, traverses the graph, and
returns the discovered paths and rendered context string.

Args:
    query:     Natural language query.
    max_hops:  Maximum graph traversal depth.
    tenant_id: From JWT.
    memory:    MemoryManager.

Returns:
    Graph paths and rendered context.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `query` | query | Yes | string | Search query for graph traversal |
| `max_hops` | query | No | integer | Maximum graph hops |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## DELETE /memory/session/{run_id}
**Clear Session**

Clear short-term (session) memory for a specific run.

Args:
    run_id:    The run whose session memory should be cleared.
    tenant_id: From JWT.
    memory:    MemoryManager.

Returns:
    204 on success.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `run_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 204 | Successful Response |
| 422 | Validation Error |

## GET /evals/suites
**List Eval Suites**

List built-in eval suites available from the operator console.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |

## POST /evals/smoke/run
**Run Smoke Eval**

Run the built-in single-agent smoke evaluation.

### Request Body

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /evals/multi/run
**Run Multi Eval**

Run the built-in multi-agent handoff evaluation.

### Request Body

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /evals/compare
**Compare Prompt Eval**

Compare baseline and patched prompt labels over the smoke suite.

### Request Body

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /improvement/patches
**List Patches**

List improvement patches, optionally filtered by agent_type and status.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `agent_type` | query | No | string |  |
| `status` | query | No | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /improvement/patches/{patch_id}/approve
**Approve Patch**

Approve a patch and apply it to the active prompt.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `patch_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /improvement/patches/{patch_id}/reject
**Reject Patch**

Reject a pending patch.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `patch_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /improvement/cycle
**Trigger Improvement Cycle**

Manually trigger a Hermes improvement cycle for an agent type.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `agent_type` | query | No | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /improvement/errors
**List Errors**

List recent error records.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `agent_type` | query | No | string |  |
| `limit` | query | No | integer |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /improvement/failures/heatmap
**Failure Heatmap**

Return a heatmap of failure_class counts per agent_type.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /hitl/pending
**List Hitl Pending**

List all pending HITL approval requests for the tenant.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /hitl/{request_id}/approve
**Approve Hitl**

Approve a HITL request.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `request_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Request Body

Requires `ResolveRequest` JSON object.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /hitl/{request_id}/reject
**Reject Hitl**

Reject a HITL request.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `request_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Request Body

Requires `ResolveRequest` JSON object.

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## GET /prompts/{agent_type}/versions
**List Prompt Versions**

List all prompt versions for an agent type.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `agent_type` | path | Yes | string |  |
| `limit` | query | No | integer |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /prompts/{agent_type}/promote/{version_id}
**Promote Prompt**

Promote a specific prompt version to active.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `agent_type` | path | Yes | string |  |
| `version_id` | path | Yes | string |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## POST /prompts/{agent_type}/rollback
**Rollback Prompt**

Roll back the active prompt N steps.

### Parameters

| Name | In | Required | Type | Description |
|---|---|---|---|---|
| `agent_type` | path | Yes | string |  |
| `steps` | query | No | integer |  |
| `X-API-Key` | header | No | string |  |

### Responses

| Code | Description |
|---|---|
| 200 | Successful Response |
| 422 | Validation Error |

## Schemas

### ApprovalRequestResponse
| Property | Type | Description |
|---|---|---|
| `request_id` | string |  |
| `run_id` | string |  |
| `tenant_id` | string |  |
| `tool_name` | string |  |
| `tool_args` | object |  |
| `reason` | string |  |
| `status` | string |  |
| `created_at` | string |  |
| `expires_at` | string |  |
| `resolved_at` | union |  |
| `resolved_by` | union |  |

### CreateRunRequest
Request body for creating a new agent run.

| Property | Type | Description |
|---|---|---|
| `agent_type` | string | Runnable native agent type: sql or code |
| `task` | string | The task for the agent to execute |
| `metadata` | object | Optional metadata |

### ErrorRecordResponse
| Property | Type | Description |
|---|---|---|
| `record_id` | string |  |
| `agent_type` | string |  |
| `task` | string |  |
| `failure_class` | string |  |
| `error_message` | string |  |
| `stack_trace` | string |  |
| `created_at` | string |  |

### EvalCompareRequest
Optional controls for a prompt comparison.

| Property | Type | Description |
|---|---|---|
| `baseline_prompt_version` | string |  |
| `patched_prompt_version` | string |  |

### EvalRunRequest
Optional controls for an eval run.

| Property | Type | Description |
|---|---|---|
| `prompt_version` | string | Prompt version label |

### GraphQueryResponse
Response for graph traversal queries.

| Property | Type | Description |
|---|---|---|
| `paths` | array |  |
| `context` | string |  |

### HTTPValidationError
| Property | Type | Description |
|---|---|---|
| `detail` | array |  |

### MemoryEntryResponse
A single retrieved memory entry.

| Property | Type | Description |
|---|---|---|
| `id` | string |  |
| `text` | string |  |
| `metadata` | object |  |
| `score` | number |  |
| `source` | string |  |

### PatchOutcomeResponse
| Property | Type | Description |
|---|---|---|
| `patch_id` | string |  |
| `baseline_score` | number |  |
| `patched_score` | number |  |
| `improvement` | number |  |
| `accepted` | boolean |  |
| `eval_summary` | string |  |

### PatchResponse
| Property | Type | Description |
|---|---|---|
| `patch_id` | string |  |
| `agent_type` | string |  |
| `target` | string |  |
| `op` | string |  |
| `path` | string |  |
| `value` | string |  |
| `rationale` | string |  |
| `proposed_by` | string |  |
| `proposed_at` | string |  |
| `score` | union |  |
| `status` | string |  |
| `based_on_errors` | array |  |

### PromptVersionResponse
| Property | Type | Description |
|---|---|---|
| `version_id` | string |  |
| `agent_type` | string |  |
| `content` | string |  |
| `version_number` | integer |  |
| `active` | boolean |  |
| `score` | union |  |
| `patch_id` | union |  |
| `created_by` | string |  |
| `created_at` | string |  |
| `tags` | array |  |
| `metadata` | object |  |

### RecallRequest
Request body for retrieving memory entries.

| Property | Type | Description |
|---|---|---|
| `query` | string | Search query |
| `k` | integer | Number of results |
| `filter` | object | Metadata filter |

### RememberRequest
Request body for storing a memory entry.

| Property | Type | Description |
|---|---|---|
| `text` | string | Text to store |
| `metadata` | object | Optional metadata |

### RememberResponse
Response for a remember operation.

| Property | Type | Description |
|---|---|---|
| `id` | string |  |

### ResolveRequest
| Property | Type | Description |
|---|---|---|
| `resolved_by` | string | Username or system ID of resolver |

### RunRecordResponse
API response model for a RunRecord.

| Property | Type | Description |
|---|---|---|
| `run_id` | string |  |
| `tenant_id` | string |  |
| `agent_type` | string |  |
| `task` | string |  |
| `status` | string |  |
| `result` | union |  |
| `created_at` | string |  |
| `started_at` | union |  |
| `completed_at` | union |  |
| `hitl_pending` | boolean |  |
| `metadata` | object |  |

### ValidationError
| Property | Type | Description |
|---|---|---|
| `loc` | array |  |
| `msg` | string |  |
| `type` | string |  |
| `input` | string |  |
| `ctx` | object |  |
