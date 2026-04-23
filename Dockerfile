# syntax=docker/dockerfile:1.7
# =============================================================================
# Codex Harness — Multi-stage Dockerfile
# =============================================================================
# Stages:
#   builder  — Install all Python dependencies in a virtualenv
#   api      — FastAPI / uvicorn server
#   worker   — RQ agent worker
#   hermes   — Hermes self-improvement scheduler
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="Codex Harness <noreply@example.com>"

# System dependencies needed at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so we can copy it into the next stage cleanly
ENV VENV_PATH=/opt/venv
RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip and wheel first
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy only the dependency manifests first (improves layer caching)
WORKDIR /build
COPY pyproject.toml ./
COPY configs/ ./configs/

# Install the package with all production extras
# (src/ will be copied in later stages)
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || \
    pip install --no-cache-dir \
        fastapi[all]>=0.110 \
        uvicorn[standard]>=0.29 \
        pydantic>=2.6 \
        pydantic-settings>=2.2 \
        redis>=5.0 \
        hiredis>=2.3 \
        rq>=1.16 \
        apscheduler>=3.10 \
        anthropic>=0.25 \
        openai>=1.30 \
        httpx>=0.27 \
        sse-starlette>=2.0 \
        python-jose[cryptography]>=3.3 \
        prometheus-client>=0.20 \
        opentelemetry-sdk>=1.24 \
        opentelemetry-exporter-otlp>=1.24 \
        opentelemetry-instrumentation-fastapi>=0.45 \
        chromadb>=0.5 \
        qdrant-client>=1.9 \
        sentence-transformers>=3.0 \
        networkx>=3.3 \
        sqlparse>=0.5 \
        jsonschema>=4.21 \
        fakeredis>=2.23 \
        pymupdf>=1.24 \
        trafilatura>=1.9 \
        python-docx>=1.1 \
        mlflow>=2.13 \
        boto3>=1.34 \
        structlog>=24.1 \
        freezegun>=1.4

# ---------------------------------------------------------------------------
# Stage 2: api — FastAPI server
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS api

# Runtime system packages only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy virtualenv from builder
COPY --from=builder "$VENV_PATH" "$VENV_PATH"

# Create non-root user
RUN groupadd --gid 1001 harness && \
    useradd --uid 1001 --gid harness --shell /bin/bash --create-home harness

# Copy application source
WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/

RUN chown -R harness:harness /app

# Create writable workspace directory
RUN mkdir -p /workspaces && chown harness:harness /workspaces

USER harness

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "harness.api.main:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]

# ---------------------------------------------------------------------------
# Stage 3: worker — RQ agent worker
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS worker

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

COPY --from=builder "$VENV_PATH" "$VENV_PATH"

RUN groupadd --gid 1001 harness && \
    useradd --uid 1001 --gid harness --shell /bin/bash --create-home harness

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p /workspaces && chown -R harness:harness /app /workspaces

USER harness

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "harness.workers.agent_worker"]

# ---------------------------------------------------------------------------
# Stage 4: hermes — Hermes self-improvement scheduler
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS hermes

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

COPY --from=builder "$VENV_PATH" "$VENV_PATH"

RUN groupadd --gid 1001 harness && \
    useradd --uid 1001 --gid harness --shell /bin/bash --create-home harness

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/

RUN chown -R harness:harness /app

USER harness

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "harness.workers.hermes_worker"]
