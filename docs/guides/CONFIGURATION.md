# Configuration Reference — Codex Harness

> All runtime behaviour is controlled by environment variables.
> No code changes are required to switch providers, backends, or feature flags.

---

## How Configuration Works

The harness uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
to load configuration. Values are resolved in this priority order (highest wins):

```
1. Environment variable     (e.g., export ANTHROPIC_API_KEY=sk-ant-...)
        ↓
2. .env file                (PROJECT_ROOT/.env)
        ↓
3. Default value in code    (src/harness/core/config.py)
```

**Start here:**

```bash
cp .env.example .env
# Edit .env with your values
```

The `.env` file is never committed to version control. The `.env.example` file
shows every available variable with its default.

---

## LLM Providers

The `LLMRouter` tries providers in priority order until one succeeds.
Setting an API key automatically registers that provider.

### Anthropic (Claude)

```bash
ANTHROPIC_API_KEY=sk-ant-...
# Type:    string
# Default: "" (disabled)
# Get it:  https://console.anthropic.com

DEFAULT_MODEL=claude-sonnet-4-6
# Type:    string
# Default: "claude-sonnet-4-6"
# Options: claude-sonnet-4-6   — balanced quality and speed (recommended)
#          claude-haiku-4-5    — fastest and cheapest, good for simple tasks
#          claude-opus-4-7     — most capable, use for complex reasoning
```

> **Tip:** Claude is the recommended primary provider. It has native tool-calling
> support, which makes SQL and code agents more reliable.

### OpenAI

```bash
OPENAI_API_KEY=sk-...
# Type:    string
# Default: "" (disabled)
# Get it:  https://platform.openai.com

OPENAI_MODELS=gpt-4o-mini
# Type:    comma-separated string
# Default: "gpt-4o-mini"
# Example: "gpt-4o-mini,gpt-4o,o4-mini"
# Note:    Each model is registered as a separate provider entry.
#          The router will use gpt-4o-mini as the first OpenAI option.

OPENAI_BASE_URL=
# Type:    string (URL)
# Default: "" (uses api.openai.com)
# Example: "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT"
# Use for: Azure OpenAI Service, LiteLLM proxy, or any OpenAI-compatible endpoint
```

### vLLM (local GPU)

```bash
VLLM_BASE_URL=http://localhost:8000
# Type:    string (URL)
# Default: "" (disabled)
# Start:   docker compose --profile local-gpu up -d vllm

VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
# Type:    string (HuggingFace model ID)
# Default: "mistralai/Mistral-7B-Instruct-v0.3"
# Note:    Must match the model loaded in the vLLM container
```

### SGLang (local GPU, alternative runtime)

```bash
SGLANG_BASE_URL=http://localhost:8000
# Type:    string (URL)
# Default: "" (disabled)

SGLANG_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
# Type:    string (HuggingFace model ID)
# Default: "meta-llama/Meta-Llama-3-8B-Instruct"
```

### llama.cpp (local CPU, no GPU required)

```bash
LLAMACPP_BASE_URL=http://localhost:8081
# Type:    string (URL)
# Default: "" (disabled)
# Start:   docker compose --profile local-cpu up -d llamacpp
# Note:    Uses OpenAI-compatible REST API via llama.cpp server.
#          Tool calling is handled via ReAct text injection.
```

---

## Memory

The harness has three memory tiers. Each can be configured independently.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Memory Architecture                          │
│                                                                     │
│  Short-term  ──► Redis         Fast, ephemeral conversation history │
│  Long-term   ──► Vector Store  Semantic search over past sessions   │
│  Structured  ──► Graph Store   Entity relationships and knowledge   │
└─────────────────────────────────────────────────────────────────────┘
```

### Short-Term Memory (Redis)

```bash
REDIS_URL=redis://localhost:6379
# Type:    string (Redis URL)
# Default: "redis://localhost:6379"
# Use:     Conversation history, run queues, short-term checkpoints
```

### Vector Store (Long-Term Semantic Memory)

```bash
VECTOR_BACKEND=chroma
# Type:    "chroma" | "qdrant" | "weaviate"
# Default: "chroma"
# chroma  — easiest setup, stored on local disk. Good for development.
# qdrant  — production-grade, Docker or managed cloud. Recommended for prod.
# weaviate — alternative with strong query language support.
```

**Chroma (development default):**

```bash
CHROMA_PATH=/data/chroma
# Type:    string (filesystem path)
# Default: "/data/chroma"
# Note:    Chroma persists data here. Map this path to a Docker volume
#          if you want data to survive container restarts.
```

**Qdrant (recommended for production):**

```bash
QDRANT_URL=http://localhost:6333
# Type:    string (URL)
# Default: "http://localhost:6333"
# Docker:  docker compose up -d qdrant
# Cloud:   Use your Qdrant Cloud cluster URL
```

**Weaviate:**

```bash
WEAVIATE_URL=http://localhost:8080
# Type:    string (URL)
# Default: "http://localhost:8080"
```

**Embedding model (all vector backends):**

```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
# Type:    string (sentence-transformers model name)
# Default: "all-MiniLM-L6-v2"
# Note:    This model runs locally — no API key required.
#          Larger models (e.g., all-mpnet-base-v2) give better retrieval
#          at the cost of more memory and slower indexing.
```

### Graph Store (Structured Knowledge Memory)

```bash
GRAPH_BACKEND=networkx
# Type:    "networkx" | "neo4j"
# Default: "networkx"
# networkx — in-memory, zero setup, does not persist across restarts.
#            Perfect for development and testing.
# neo4j    — persistent, supports complex graph queries. Recommended for prod.
```

**Neo4j (production):**

```bash
NEO4J_URL=bolt://localhost:7687
# Type:    string (Bolt URL)
# Default: "bolt://localhost:7687"

NEO4J_USER=neo4j
# Type:    string
# Default: "neo4j"

NEO4J_PASSWORD=harnesspassword
# Type:    string
# Default: "harnesspassword"
# IMPORTANT: Change this in production!
```

---

## Observability

### MLflow (Experiment Tracking and Traces)

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
# Type:    string (URL)
# Default: "http://localhost:5000"
# UI:      Open this URL in a browser to see all experiments and traces

MLFLOW_EXPERIMENT_NAME=codex-harness
# Type:    string
# Default: "codex-harness"
# Note:    All runs are grouped under this experiment name in the MLflow UI.
#          Use different names to separate environments or projects.
```

### OpenTelemetry (Distributed Tracing)

```bash
OTEL_EXPORTER_ENDPOINT=http://localhost:4317
# Type:    string (OTLP gRPC endpoint)
# Default: "http://localhost:4317"
# Use:     Any OTLP-compatible backend: Jaeger, Tempo, Honeycomb, Datadog, etc.

OTEL_SERVICE_NAME=codex-harness
# Type:    string
# Default: "codex-harness"
# Note:    Appears as the service name in your tracing backend.
#          Use a unique name per deployment environment.
```

### Logging

```bash
LOG_LEVEL=INFO
# Type:    "DEBUG" | "INFO" | "WARNING" | "ERROR"
# Default: "INFO"
# Note:    "DEBUG" logs every LLM call, tool argument, and step transition.
#          Use carefully in production — very verbose.
```

---

## Safety and Budget Controls

These settings protect against runaway costs and abuse.

```bash
COST_BUDGET_USD_PER_TENANT=100.0
# Type:    float
# Default: 100.0
# Unit:    US dollars per tenant per calendar month
# Note:    When a tenant's cumulative spend reaches this threshold, all new
#          runs are rejected with a 429 response until the budget resets.
#          Set to 0.0 to disable budget enforcement.

RATE_LIMIT_RPM=60
# Type:    int
# Default: 60
# Unit:    Requests per minute per tenant
# Note:    Enforced at the API level via a Redis-backed sliding window.
#          Exceeding this returns HTTP 429 Too Many Requests.

WORKSPACE_BASE_PATH=/workspaces
# Type:    string (filesystem path)
# Default: "/workspaces"
# Note:    Code agents write files inside a per-run subdirectory of this path.
#          Keep this path inside a Docker volume for isolation.
#          The agent cannot write outside this directory (sandboxed).
```

---

## Hermes (Self-Improvement Engine)

Hermes is the autonomous component that periodically analyses agent errors,
generates patches, and (optionally) applies them automatically.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Hermes Improvement Loop                        │
│                                                                   │
│  Collect errors  ──► Evaluate patterns  ──► Generate patch       │
│       ▲                                         │                 │
│       │                 Score ≥ threshold?       │                 │
│       │                 YES ──────────────► Apply patch           │
│       │                 NO  ──────────────► Flag for human review │
│       │                                         │                 │
│       └─────────────── Wait interval ───────────┘                │
└──────────────────────────────────────────────────────────────────┘
```

```bash
HERMES_AUTO_APPLY=false
# Type:    "true" | "false"
# Default: "false"
# IMPORTANT: Leave this "false" in production unless you have reviewed
#            the patch pipeline and trust the eval scores completely.
#            Auto-apply can modify agent prompts and tool configurations.

HERMES_INTERVAL_SECONDS=3600.0
# Type:    float
# Default: 3600.0 (1 hour)
# Note:    How often Hermes wakes up to check for new errors.
#          Minimum recommended: 300 (5 minutes). Lower values increase
#          LLM costs since Hermes itself calls the LLM to analyse errors.

HERMES_MIN_ERRORS_TO_TRIGGER=5
# Type:    int
# Default: 5
# Note:    Hermes only runs a patch cycle if at least this many errors
#          have been recorded since the last cycle. This prevents
#          unnecessary analysis of one-off or expected errors.

HERMES_PATCH_SCORE_THRESHOLD=0.7
# Type:    float (0.0–1.0)
# Default: 0.7
# Note:    Patches must score at least this high on the internal eval
#          before they are considered for auto-apply. A score of 1.0
#          means the patch must pass all eval tests — very strict.
```

---

## Authentication

```bash
JWT_SECRET_KEY=change-me-in-production
# Type:    string (secret)
# Default: "change-me-in-production" (INSECURE — change before deploying)
# Generate a strong key:
#   python -c "import secrets; print(secrets.token_hex(32))"
# IMPORTANT: This value is used to sign all JWT tokens. If it leaks,
#            all tokens can be forged. Treat it like a database password.

ENVIRONMENT=dev
# Type:    "dev" | "staging" | "prod"
# Default: "dev"
# Note:    In "prod" mode, the API enforces stricter security checks,
#          disables debug endpoints, and requires valid JWT tokens for
#          all requests.
```

---

## Runtime

```bash
WORKER_CONCURRENCY=4
# Type:    int
# Default: 4
# Note:    Number of agent jobs each worker process handles in parallel.
#          Increase this for higher throughput on large machines.
#          Each concurrent agent opens LLM connections and uses memory,
#          so do not set this higher than your machine can support.
#          A safe formula: (available_memory_GB / 2) workers.
```

---

## MCP Server Configuration

MCP (Model Context Protocol) servers extend the agent with external tools —
databases, filesystems, APIs — without modifying the harness code.

Configuration file: `configs/mcp_servers.yaml`

### Example: Filesystem Access

```yaml
mcp_servers:
  - name: filesystem
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - "${WORKSPACE_BASE_PATH:-/workspaces}"
    env: {}
    enabled: true
    description: "Provides file read/write/list tools scoped to the agent workspace."
```

### Example: PostgreSQL Database

```yaml
  - name: postgres
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-postgres"
    env:
      DATABASE_URL: "${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/codex}"
    enabled: true
    description: "Exposes PostgreSQL query tools; DATABASE_URL must be set."
```

### Example: Custom REST API via SSE

Use this pattern to wrap any internal API as an MCP server.

```yaml
  - name: custom_api
    transport: sse
    url: "${CUSTOM_MCP_URL:-http://localhost:9000/mcp/sse}"
    headers:
      Authorization: "Bearer ${CUSTOM_MCP_TOKEN:-}"
      Content-Type: "application/json"
    env: {}
    enabled: false
    description: "Connects to a custom HTTP/SSE MCP server."
```

### Global Connection Settings

```yaml
connection:
  connect_timeout_seconds: 10.0    # Wait up to 10s for each server to start
  max_reconnect_attempts: 3        # Retry 3 times on connection loss
  reconnect_backoff_seconds: 2.0   # Wait 2s between retries (doubles each attempt)
```

**To add a new MCP server:**
1. Add an entry to `configs/mcp_servers.yaml`
2. Set `enabled: true`
3. Export any required environment variables (e.g., `DATABASE_URL`)
4. Restart the API and worker — MCP servers are loaded at startup

---

## Configuration Profiles

Three reference `.env` configurations for common scenarios.

### Profile 1 — Minimal Development

Fastest to get running. In-memory graph, local Chroma, single Claude provider.
Suitable for a single developer on a laptop.

```bash
# LLM
ANTHROPIC_API_KEY=sk-ant-your-key-here
DEFAULT_MODEL=claude-haiku-4-5

# Memory
VECTOR_BACKEND=chroma
CHROMA_PATH=/data/chroma
GRAPH_BACKEND=networkx

# Infrastructure
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=codex-harness-dev

# Safety
COST_BUDGET_USD_PER_TENANT=10.0
RATE_LIMIT_RPM=30

# Runtime
ENVIRONMENT=dev
LOG_LEVEL=DEBUG
WORKER_CONCURRENCY=2
JWT_SECRET_KEY=dev-not-a-secret
```

Start infrastructure:

```bash
docker compose up -d redis mlflow
```

### Profile 2 — Fully Local (No API Keys)

Runs entirely on your machine. No cloud costs. Good for offline development
or sensitive environments where data cannot leave the machine.

```bash
# LLM — local only
LLAMACPP_BASE_URL=http://localhost:8081
# Leave ANTHROPIC_API_KEY and OPENAI_API_KEY empty

# Memory
VECTOR_BACKEND=chroma
CHROMA_PATH=/data/chroma
GRAPH_BACKEND=networkx

# Infrastructure
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000

# Safety
COST_BUDGET_USD_PER_TENANT=0.0
RATE_LIMIT_RPM=60

# Runtime
ENVIRONMENT=dev
LOG_LEVEL=INFO
WORKER_CONCURRENCY=2
JWT_SECRET_KEY=local-dev-key
```

Start infrastructure:

```bash
LLAMACPP_MODEL=Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
  docker compose --profile local-cpu up -d
```

### Profile 3 — Production

Full stack: Claude primary + OpenAI fallback, Qdrant for vector memory,
Neo4j for the knowledge graph, full observability, strict security.

```bash
# LLM
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENAI_MODELS=gpt-4o-mini,gpt-4o
DEFAULT_MODEL=claude-sonnet-4-6

# Memory
VECTOR_BACKEND=qdrant
QDRANT_URL=http://qdrant:6333
GRAPH_BACKEND=neo4j
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=a-very-strong-password-here
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Infrastructure
REDIS_URL=redis://redis:6379
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=codex-harness-prod
OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
OTEL_SERVICE_NAME=codex-harness

# Safety
COST_BUDGET_USD_PER_TENANT=100.0
RATE_LIMIT_RPM=60
WORKSPACE_BASE_PATH=/workspaces

# Hermes — keep manual review in prod
HERMES_AUTO_APPLY=false
HERMES_INTERVAL_SECONDS=3600.0
HERMES_MIN_ERRORS_TO_TRIGGER=5
HERMES_PATCH_SCORE_THRESHOLD=0.7

# Auth & Runtime
JWT_SECRET_KEY=your-256-bit-random-key-here
ENVIRONMENT=prod
LOG_LEVEL=INFO
WORKER_CONCURRENCY=4
```

Start infrastructure:

```bash
docker compose up -d
```

---

## All Configuration Variables — Quick Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | string | `""` | Claude API key |
| `DEFAULT_MODEL` | string | `claude-sonnet-4-6` | Default LLM model |
| `OPENAI_API_KEY` | string | `""` | OpenAI API key |
| `OPENAI_MODELS` | string | `gpt-4o-mini` | Comma-separated model list |
| `OPENAI_BASE_URL` | string | `""` | Override for Azure or proxy |
| `VLLM_BASE_URL` | string | `""` | vLLM server URL |
| `VLLM_MODEL` | string | `mistralai/Mistral-7B-Instruct-v0.3` | vLLM model ID |
| `SGLANG_BASE_URL` | string | `""` | SGLang server URL |
| `SGLANG_MODEL` | string | `meta-llama/Meta-Llama-3-8B-Instruct` | SGLang model ID |
| `LLAMACPP_BASE_URL` | string | `""` | llama.cpp server URL |
| `VECTOR_BACKEND` | enum | `chroma` | `chroma` / `qdrant` / `weaviate` |
| `CHROMA_PATH` | string | `/data/chroma` | Chroma data directory |
| `QDRANT_URL` | string | `http://localhost:6333` | Qdrant server URL |
| `WEAVIATE_URL` | string | `http://localhost:8080` | Weaviate server URL |
| `GRAPH_BACKEND` | enum | `networkx` | `networkx` / `neo4j` |
| `NEO4J_URL` | string | `bolt://localhost:7687` | Neo4j Bolt URL |
| `NEO4J_USER` | string | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | string | `harnesspassword` | Neo4j password |
| `EMBEDDING_MODEL` | string | `all-MiniLM-L6-v2` | sentence-transformers model |
| `REDIS_URL` | string | `redis://localhost:6379` | Redis connection URL |
| `MLFLOW_TRACKING_URI` | string | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | string | `codex-harness` | MLflow experiment name |
| `OTEL_EXPORTER_ENDPOINT` | string | `http://localhost:4317` | OTLP gRPC endpoint |
| `OTEL_SERVICE_NAME` | string | `codex-harness` | Service name in traces |
| `LOG_LEVEL` | enum | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `COST_BUDGET_USD_PER_TENANT` | float | `100.0` | Monthly spend cap per tenant |
| `RATE_LIMIT_RPM` | int | `60` | Requests per minute per tenant |
| `WORKSPACE_BASE_PATH` | string | `/workspaces` | Agent sandbox root directory |
| `HERMES_AUTO_APPLY` | bool | `false` | Auto-apply Hermes patches |
| `HERMES_INTERVAL_SECONDS` | float | `3600.0` | Hermes run interval |
| `HERMES_MIN_ERRORS_TO_TRIGGER` | int | `5` | Errors required before Hermes runs |
| `HERMES_PATCH_SCORE_THRESHOLD` | float | `0.7` | Minimum eval score to apply patch |
| `JWT_SECRET_KEY` | string | `change-me-in-production` | JWT signing secret |
| `ENVIRONMENT` | enum | `dev` | `dev` / `staging` / `prod` |
| `WORKER_CONCURRENCY` | int | `4` | Parallel agent workers per process |
