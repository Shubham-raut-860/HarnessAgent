# Deployment Guide — HarnessAgent

> This guide covers running HarnessAgent in environments beyond a developer laptop.
> It covers single-node Docker Compose deployments, scaling strategies, production
> hardening, Kubernetes concepts, and backup/recovery procedures.

---

## Deployment Options Overview

Choose the option that matches your scale and operational maturity.

| Option | Complexity | Scale | Cost | Best for |
|---|---|---|---|---|
| Docker Compose (all-in-one) | ⭐ Low | Single node | Low | Development, small teams, proof-of-concept |
| Docker Compose + managed DBs | ⭐⭐ Medium | Single node | Medium | Startups, internal tools |
| Kubernetes (Helm) | ⭐⭐⭐ High | Multi-node, auto-scale | Higher | Enterprise, high-traffic production |

> **Recommendation for most teams:** Start with Docker Compose + managed databases.
> Migrate to Kubernetes when you need horizontal scaling or multi-region.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │              Docker Network              │
                    │                                         │
  Users/API calls   │  ┌─────────┐    ┌─────────────────┐    │
  ──────────────►   │  │   API   │───►│  Redis (queue)  │    │
                    │  │ :8000   │    │   + short-term  │    │
                    │  └────┬────┘    └─────────────────┘    │
                    │       │                 ▲               │
                    │       │         ┌───────┘               │
                    │       ▼         ▼                       │
                    │  ┌─────────────────────────────────┐    │
                    │  │     Worker(s) — Agent Executor  │    │
                    │  └──────────────┬──────────────────┘    │
                    │                 │                        │
                    │    ┌────────────┼────────────┐          │
                    │    ▼            ▼             ▼          │
                    │  ┌──────┐  ┌───────┐  ┌──────────┐     │
                    │  │Qdrant│  │ Neo4j │  │  MLflow  │     │
                    │  │(vec) │  │(graph)│  │ (traces) │     │
                    │  └──────┘  └───────┘  └──────────┘     │
                    │                                         │
                    │  ┌──────────┐   ┌──────────────────┐   │
                    │  │  Hermes  │   │ Prometheus+Grafana│   │
                    │  │(improver)│   │   (monitoring)   │   │
                    │  └──────────┘   └──────────────────┘   │
                    └─────────────────────────────────────────┘
```

---

## Docker Compose Production Deployment

### Step 1 — Prepare the server

A Linux VM (Ubuntu 22.04+) with:
- 4+ CPU cores
- 8 GB+ RAM (16 GB recommended)
- 50 GB+ SSD storage
- Docker Engine 24+ and Docker Compose v2

```bash
# Install Docker on Ubuntu
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker info
docker compose version
```

### Step 2 — Clone and configure

```bash
git clone https://github.com/your-org/harness-agent.git
cd harness-agent
cp .env.example .env
```

### Step 3 — Generate a strong JWT secret

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
# Output: e.g. a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1
```

Add this to your `.env`:

```bash
JWT_SECRET_KEY=a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1
```

### Step 4 — Configure production `.env`

```bash
# LLM (at minimum, set one of these)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Auth
JWT_SECRET_KEY=<your-generated-key>
ENVIRONMENT=prod

# Memory (production backends)
VECTOR_BACKEND=qdrant
GRAPH_BACKEND=neo4j
NEO4J_PASSWORD=<strong-random-password>

# Safety
HERMES_AUTO_APPLY=false
COST_BUDGET_USD_PER_TENANT=100.0
RATE_LIMIT_RPM=60

# Observability
LOG_LEVEL=INFO
```

### Step 5 — Enable Redis persistence

By default, the `docker-compose.yml` runs Redis without persistence
(`--save ""` `--appendonly no`) for performance. In production, enable
at least one persistence mode.

Edit `docker-compose.yml` under the `redis` service:

```yaml
services:
  redis:
    command: >
      redis-server
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfsync everysec
```

The `appendonly yes` setting ensures no more than one second of data
can be lost if the server crashes.

### Step 6 — Set resource limits

Add resource limits to the `api` and `worker` services in `docker-compose.yml`
to prevent runaway agents from consuming all host resources:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M

  worker:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M
      replicas: 2     # start with 2 workers; scale up as needed
```

### Step 7 — Start the full production stack

```bash
docker compose up -d
```

Verify all services are healthy:

```bash
docker compose ps
```

All services should show `healthy` within 60 seconds. If any stay in
`starting`, check their logs:

```bash
docker compose logs <service-name> --tail 50
```

---

## Scaling Workers

Workers are stateless — you can run as many as your host machine supports.

```bash
# Scale to 4 worker replicas
docker compose up -d --scale worker=4

# Alternatively, set concurrency within each worker process
WORKER_CONCURRENCY=8 docker compose up -d worker
```

**When to scale:**

| Signal | Action |
|---|---|
| Queue depth > 10 jobs consistently | Add more worker replicas |
| Worker CPU consistently > 80% | Increase `WORKER_CONCURRENCY` |
| Memory usage > 80% of host | Add more host memory or reduce `WORKER_CONCURRENCY` |
| Queue depth is 0, workers idle | Scale down to reduce cost |

```bash
# Monitor queue depth via Redis CLI
docker exec harness-redis redis-cli llen harness-default
```

---

## Health Checks and Monitoring

### Service health endpoint

```bash
curl http://localhost:8000/health
```

Expected response when all services are healthy:

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

When any service is unhealthy, `status` changes to `"degraded"` and the
affected service shows `false`. The API continues to serve requests for
healthy components.

### Liveness and readiness probes

```bash
# Liveness — is the process alive?
curl http://localhost:8000/health/live

# Readiness — is the service ready to receive traffic?
curl http://localhost:8000/health/ready
```

These are the same endpoints used by Docker healthchecks and Kubernetes probes.

### Prometheus metrics

```bash
# Raw Prometheus metrics (Prometheus scrapes this automatically)
curl http://localhost:8000/metrics
```

Key metrics to monitor:

| Metric | What it means |
|---|---|
| `harness_agent_runs_total` | Total runs completed, by status and agent type |
| `harness_llm_request_duration_seconds` | LLM call latency histogram |
| `harness_tokens_total` | Token consumption by provider |
| `harness_circuit_breaker_state` | 0=closed (healthy), 1=open (failing) |
| `harness_cost_usd_total` | Cumulative spend by tenant |
| `harness_queue_depth` | Pending jobs in the worker queue |

### Grafana dashboards

```bash
open http://localhost:3000
# Username: admin
# Password: harnesspassword   (change this in GF_SECURITY_ADMIN_PASSWORD)
```

The `infra/` directory includes a pre-built Grafana dashboard JSON
(`harness.json`) that is auto-loaded at startup. It shows:
- Request rate and error rate
- LLM cost and token burn rate
- Queue depth and worker utilisation
- Circuit breaker status per provider

---

## Graceful Shutdown

Stop the API and workers cleanly — in-flight requests complete before the
process exits. Jobs that were being processed are re-queued automatically
by Redis.

```bash
# Stop application services, leave infrastructure running
docker compose stop api worker hermes

# Restart application services
docker compose start api worker hermes

# Full teardown (infrastructure kept)
docker compose stop

# Full teardown including infrastructure (data in volumes is preserved)
docker compose down
```

> **Do not use `docker compose kill`** in production. It sends SIGKILL immediately,
> which can leave agent runs in an inconsistent state.

---

## Production Security Checklist

Complete this checklist before exposing the harness to the internet or
connecting it to production data.

- [ ] **JWT_SECRET_KEY** is a 64-character random hex string (not the default)
- [ ] **ENVIRONMENT=prod** is set
- [ ] **NEO4J_PASSWORD** is changed from `harnesspassword`
- [ ] **HERMES_AUTO_APPLY=false** — Hermes patches require human review in prod
- [ ] Redis persistence is enabled (`appendonly yes`)
- [ ] Qdrant data volume is mapped to persistent storage
- [ ] Neo4j data volume is mapped to persistent storage
- [ ] MLflow artifact store is on a persistent volume
- [ ] Resource limits are set on API and worker services
- [ ] Rate limits are configured per tenant (`RATE_LIMIT_RPM`)
- [ ] Cost budget is set per tenant (`COST_BUDGET_USD_PER_TENANT`)
- [ ] API is behind a reverse proxy (nginx/Caddy/Traefik) with TLS
- [ ] Grafana admin password changed from `harnesspassword`
- [ ] Grafana alerts configured for circuit breaker and error rate spikes
- [ ] Secrets (`.env`) are not committed to version control
- [ ] Docker socket is not mounted into agent containers

---

## Kubernetes Deployment

For teams that need horizontal scaling, high availability, or multi-region
deployments, Kubernetes is the recommended path. A Helm chart is the most
maintainable approach.

### Key Kubernetes concepts for this stack

```
┌────────────────────────────────────────────────────────────────┐
│                   Kubernetes Cluster                           │
│                                                                │
│  Namespace: harness-agent                                      │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Deployments                                             │  │
│  │  • harness-api       (Deployment, 2+ replicas)          │  │
│  │  • harness-worker    (Deployment, HPA managed)          │  │
│  │  • harness-hermes    (Deployment, 1 replica)            │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Auto-scaling                                            │  │
│  │  • HorizontalPodAutoscaler on harness-worker            │  │
│  │    scales on: CPU utilization, queue depth custom metric │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Configuration                                           │  │
│  │  • Secret: harness-secrets (API keys, JWT key)          │  │
│  │  • ConfigMap: harness-config (non-secret env vars)      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ External Services (managed or self-hosted)              │  │
│  │  • Redis     → AWS ElastiCache or Upstash               │  │
│  │  • Qdrant    → Qdrant Cloud or self-hosted StatefulSet  │  │
│  │  • Neo4j     → Neo4j AuraDB or self-hosted StatefulSet  │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Secrets

Never store API keys in ConfigMaps. Use Kubernetes Secrets:

```bash
kubectl create secret generic harness-secrets \
  --namespace harness-agent \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
  --from-literal=NEO4J_PASSWORD=strong-password
```

For production, use an external secrets manager:
- **AWS:** AWS Secrets Manager + External Secrets Operator
- **GCP:** Secret Manager + Workload Identity
- **Azure:** Azure Key Vault + Azure Key Vault Provider for Secrets Store CSI Driver
- **HashiCorp Vault:** Vault Agent Injector

### Readiness and liveness probes

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 15
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3
```

### Horizontal Pod Autoscaler for workers

Scale workers automatically based on CPU utilisation:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: harness-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: harness-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

> For queue-depth-based scaling (more accurate for async workloads), use
> KEDA (Kubernetes Event-Driven Autoscaling) with the Redis list scaler.

---

## Backup and Recovery

Back up these data stores regularly. Agent runs are ephemeral but
memory, knowledge graphs, and audit logs are valuable long-term.

| Data | Location | Recommended backup method | Recovery |
|---|---|---|---|
| Vector memories | Qdrant volume (`qdrant_data`) | `qdrant snapshot create` API | Restore snapshot via Qdrant REST API |
| Knowledge graph | Neo4j volume (`neo4j_data`) | `neo4j-admin database dump` | `neo4j-admin database load` |
| Run records + queues | Redis volume (`redis_data`) | Redis RDB snapshot or AOF replay | Copy `dump.rdb` back, restart Redis |
| Prompt versions | Redis | Redis backup (same as above) | Same as Redis |
| Audit logs | Host filesystem (`/var/log/harness/`) | `rsync` or `cp` to object storage | Copy files back |
| MLflow runs and artifacts | MLflow volume (`mlflow_data`) | `cp -r` the volume to object storage | Copy volume back, restart MLflow |
| Agent workspace files | Workspace volume (`workspaces`) | `rsync` to backup location | Copy files back |

### Qdrant snapshot backup

```bash
# Create a snapshot (runs inside the container)
curl -X POST http://localhost:6333/collections/harness_memory/snapshots

# List snapshots to get the snapshot name
curl http://localhost:6333/collections/harness_memory/snapshots

# Download the snapshot file from Qdrant's storage directory
docker exec harness-qdrant \
  tar czf /tmp/qdrant-backup.tar.gz /qdrant/storage/snapshots

docker cp harness-qdrant:/tmp/qdrant-backup.tar.gz ./backups/
```

### Neo4j backup

```bash
# Dump the Neo4j database (requires stopping or using online backup)
docker exec harness-neo4j \
  neo4j-admin database dump neo4j --to-path=/var/lib/neo4j/export/

docker cp harness-neo4j:/var/lib/neo4j/export/neo4j.dump ./backups/
```

### Redis backup

```bash
# Trigger an immediate RDB snapshot
docker exec harness-redis redis-cli BGSAVE

# Wait for the snapshot to complete
docker exec harness-redis redis-cli LASTSAVE

# Copy the dump file
docker cp harness-redis:/data/dump.rdb ./backups/redis-dump.rdb
```

### Automated backup script

Create a cron job (or a scheduled container) that runs this daily:

```bash
#!/usr/bin/env bash
# save as: scripts/backup.sh
set -euo pipefail

BACKUP_DIR="./backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

echo "Backing up Redis..."
docker exec harness-redis redis-cli BGSAVE
docker cp harness-redis:/data/dump.rdb "$BACKUP_DIR/redis-dump.rdb"

echo "Backing up Qdrant..."
curl -s -X POST http://localhost:6333/collections/harness_memory/snapshots > /dev/null
docker exec harness-qdrant \
  tar czf /tmp/qdrant-snapshot.tar.gz /qdrant/storage
docker cp harness-qdrant:/tmp/qdrant-snapshot.tar.gz "$BACKUP_DIR/qdrant.tar.gz"

echo "Backing up Neo4j..."
docker exec harness-neo4j \
  neo4j-admin database dump neo4j --to-path=/var/lib/neo4j/export/
docker cp harness-neo4j:/var/lib/neo4j/export/neo4j.dump "$BACKUP_DIR/neo4j.dump"

echo "Backing up MLflow artifacts..."
docker run --rm \
  -v mlflow_data:/source:ro \
  -v "$(pwd)/$BACKUP_DIR":/dest \
  alpine tar czf /dest/mlflow.tar.gz -C /source .

echo "Backup complete: $BACKUP_DIR"

# Optional: upload to S3
# aws s3 sync "$BACKUP_DIR" "s3://your-bucket/harness-backups/$(date +%Y-%m-%d)/"
```

Make it executable and add to crontab:

```bash
chmod +x scripts/backup.sh

# Run daily at 2:00 AM
crontab -e
# Add: 0 2 * * * /path/to/harness-agent/scripts/backup.sh >> /var/log/harness-backup.log 2>&1
```

---

## Upgrading the Harness

Follow this procedure for zero-downtime upgrades using Docker Compose.

```bash
# 1. Pull the latest code
git pull origin main

# 2. Build new images without stopping anything
docker compose build api worker hermes

# 3. Restart services one at a time
docker compose up -d --no-deps api
docker compose up -d --no-deps worker
docker compose up -d --no-deps hermes

# 4. Verify health
curl http://localhost:8000/health

# 5. If something is wrong, roll back
git checkout <previous-tag>
docker compose build api worker hermes
docker compose up -d --no-deps api worker hermes
```

> Database schema migrations (if any) are always backwards-compatible.
> Run `docker compose run --rm api python -m harness.migrations` before
> upgrading if the release notes mention schema changes.

---

## Troubleshooting Production Issues

### API is slow or timing out

1. Check worker queue depth: `docker exec harness-redis redis-cli llen harness-default`
2. Scale workers if queue is deep: `docker compose up -d --scale worker=8`
3. Check circuit breaker status in Prometheus: query `harness_circuit_breaker_state`
4. Check LLM provider status pages (Anthropic, OpenAI)

### Out of memory

1. Check which container is using memory: `docker stats`
2. Reduce `WORKER_CONCURRENCY` in `.env`
3. Add memory limits to `docker-compose.yml` `deploy.resources.limits.memory`
4. Consider moving to a larger instance or splitting API and workers to separate hosts

### Qdrant queries are slow

1. Check disk I/O on the host: `iostat -x 1`
2. Ensure the Qdrant volume is on SSD storage
3. Consider switching to Qdrant Cloud for managed performance

### Neo4j out of heap

1. Increase heap in `docker-compose.yml`:

```yaml
environment:
  NEO4J_dbms_memory_heap_max__size: 1G
  NEO4J_dbms_memory_pagecache_size: 512m
```

---

## Next Steps

| Goal | Resource |
|---|---|
| Configure LLM providers and memory backends | [CONFIGURATION.md](CONFIGURATION.md) |
| Get started quickly | [QUICKSTART.md](QUICKSTART.md) |
| Understand the system architecture | [../architecture/](../architecture/) |
| Extend with custom tools and agents | [../reference/](../reference/) |
