# Tutorial 02 — Project Scout: Best Observability Tool

Below is an **executive‑ready recommendation report** aligned with your stack, constraints, and evaluation framework.

---

## 1. Executive Summary

For a Python‑based, real‑time ML inference platform requiring distributed tracing and metrics, **SigNoz** is the strongest open‑source, self‑hosted option. It provides end‑to‑end OpenTelemetry support with the lowest operational overhead for a 6‑engineer team, while delivering an ML‑friendly UI out of the box. Grafana Tempo is a strong alternative but requires more ecosystem assembly to reach parity.

---

## 2. Winner: **SigNoz**

### Rationale

**SigNoz** offers the best balance of:
- **Native OpenTelemetry (Python‑first)** support for tracing and metrics
- **Unified observability** (traces + metrics + basic logs) without stitching multiple tools
- **Low operational complexity** via Docker‑Compose/Kubernetes presets
- **ML‑appropriate debugging UX**, including latency breakdowns, high‑cardinality attributes, and real‑time views

For a small team and an ML inference workload, SigNoz minimizes integration and maintenance cost while remaining fully open‑source and self‑hosted.

---

## 3. Comparison Table

| Criterion | Jaeger | Grafana Tempo | **SigNoz (Winner)** |
|--------|--------|---------------|---------------------|
| OTel Python SDK Compatibility | ✅ Strong | ✅ Excellent | ✅✅ Excellent |
| Distributed Tracing (Real‑Time) | ✅ Mature | ✅✅ Scalable | ✅✅ Real‑time + ML‑friendly |
| Metrics Support | ❌ External only | ✅ Prometheus‑native | ✅ Built‑in (OTel + Prom) |
| Operational Complexity | ✅ Simple | ⚠️ Medium (Grafana stack) | ✅✅ Low (all‑in‑one) |
| UI & Debugging for ML Inference | ⚠️ Basic | ✅ Flexible | ✅✅ Purpose‑built |
| Community & Maturity | ✅ Very mature | ✅ Growing fast | ✅ Active, focused |
| **Overall Fit** | Good tracer | Infra‑heavy | **Best end‑to‑end fit** |

---

## 4. 3‑Step Adoption Roadmap

### Step 1 – Foundation (Week 1)
- Deploy SigNoz via **Docker Compose**
- Instrument FastAPI inference services with **OTel Python SDK**
- Export traces + metrics via OTLP

### Step 2 – ML‑Specific Instrumentation (Weeks 2–3)
- Add custom spans for:
  - Model loading
  - Feature extraction
  - Inference execution
- Attach high‑cardinality attributes (model version, batch size, GPU/CPU)

### Step 3 – Production Hardening (Weeks 4–6)
- Tune retention and sampling policies
- Add SLOs (p95/p99 inference latency)
- Integrate alerting (Grafana/Alertmanager)

---

## 5. Top 3 Risks & Mitigations

| Risk | Impact | Mitigation |
|----|-------|-----------|
| **High cardinality from ML metadata** | Storage/query cost | Use attribute whitelisting + tail sampling |
| **SigNoz maturity vs Grafana stack** | Long‑term scalability concerns | Validate with load tests; Tempo remains drop‑in fallback |
| **Team unfamiliarity with OTel** | Slower rollout | Start with auto‑instrumentation; incrementally add custom spans |

---

## 6. Alternatives Considered

1. **Grafana Tempo + Prometheus + Grafana**
   - Best for large infra teams
   - More flexible, but higher setup and cognitive load

2. **Jaeger + Prometheus**
   - Excellent pure tracing
   - Not sufficient alone for ML observability without significant add‑ons

3. **Elastic APM (OSS components)**
   - Powerful, but heavier operational footprint and licensing caveats over time

---

### Final Recommendation
Adopt **SigNoz** as the primary observability platform, with Grafana Tempo as a contingency if long‑term scale or ecosystem integration becomes the dominant concern.