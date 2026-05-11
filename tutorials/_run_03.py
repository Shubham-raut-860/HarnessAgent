"""Run Tutorial 03 — AutoGen Multi-Agent Support Team with HarnessAgent."""
import asyncio, json, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")
sys.path.insert(0, "tutorials")
from _azure_client import chat as azure_chat, MODEL_NAME

from harness.core.config import get_config
from harness.core.context import AgentContext
from harness.observability.trace_recorder import TraceRecorder
from harness.observability.trace_schema import SpanKind, SpanStatus

cfg = get_config()
recorder = TraceRecorder.create(redis_url=cfg.redis_url, log_dir="logs")
OUT = Path("tutorials/output/03_support_team")
OUT.mkdir(parents=True, exist_ok=True)

TICKETS = [
    {
        "id": "TKT-001", "tier": "pro", "customer": "Sarah Johnson",
        "subject": "API returning 503 errors intermittently",
        "message": (
            "Getting 503 errors on POST /api/v1/analyze since 2 hours. "
            "Happens 30% of the time. Error: upstream_timeout retry_after 30. "
            "This is impacting our production pipeline."
        ),
    },
    {
        "id": "TKT-002", "tier": "free", "customer": "Mark Thompson",
        "subject": "Charged twice for the same month",
        "message": (
            "Charged 29 USD twice in March 2024: March 1st and March 3rd. "
            "Both show YourApp Pro Monthly. One account only. "
            "Please refund the duplicate. TXN-89234 and TXN-89456."
        ),
    },
    {
        "id": "TKT-003", "tier": "enterprise", "customer": "CTO - Nexus Corp",
        "subject": "CRITICAL: Complete outage - losing $15k/hour",
        "message": (
            "100% API calls failing with 500 errors. Started 45 mins ago. "
            "50k active users affected. We are $2M ARR customer losing $15k/hour. "
            "Status page shows green but is WRONG. Need engineering call in 10 min."
        ),
    },
]


async def handle_ticket(ticket: dict) -> dict:
    ctx = AgentContext.create(
        tenant_id=ticket["tier"],
        agent_type="autogen_support",
        task=f"Support: {ticket['subject']}",
        memory=None,
        workspace_path=OUT / ticket["id"],
        max_steps=10,
        max_tokens=30_000,
    )
    (OUT / ticket["id"]).mkdir(exist_ok=True)

    print(f"\n{'─'*54}")
    print(f"  {ticket['id']} | {ticket['tier'].upper()} | {ticket['customer']}")
    print(f"  Subject: {ticket['subject']}")

    # ── HarnessAgent: open RUN span ──────────────────────────────────────
    run_sid = await recorder.start_span(
        ctx.run_id, SpanKind.RUN, f"support:{ticket['id']}", ctx,
        input_preview=ticket["subject"],
    )

    # ── Agent 1: TriageAgent ─────────────────────────────────────────────
    triage_sid = await recorder.start_span(
        ctx.run_id, SpanKind.TOOL, "agent:TriageAgent", ctx,
        input_preview=ticket["message"][:200],
    )
    raw = await azure_chat(
        "You are the support triage specialist. Return JSON only.",
        (
            f"Ticket from {ticket['customer']} (Tier: {ticket['tier']}).\n"
            f"Subject: {ticket['subject']}\n"
            f"Message: {ticket['message']}\n\n"
            "Return JSON: {\"category\": \"TECHNICAL|BILLING|GENERAL|ESCALATION\", "
            "\"urgency\": \"LOW|MEDIUM|HIGH|CRITICAL\", "
            "\"summary\": str, "
            "\"route_to\": \"TechSupportAgent|BillingAgent|EscalationAgent\"}"
        ),
    )
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        triage = json.loads(raw)
    except Exception:
        m_cat = "ESCALATION" if "critical" in ticket["message"].lower() else "TECHNICAL"
        triage = {"category": m_cat, "urgency": "HIGH",
                  "summary": ticket["subject"],
                  "route_to": "EscalationAgent" if m_cat == "ESCALATION" else "TechSupportAgent"}

    await recorder.end_span(ctx.run_id, triage_sid, SpanStatus.OK,
        output_preview=f"Category:{triage['category']} → {triage['route_to']}")
    print(f"  [TriageAgent]   → {triage['category']} | {triage['urgency']} | {triage['route_to']}")

    # ── Agent 2: Specialist ──────────────────────────────────────────────
    route = triage.get("route_to", "TechSupportAgent")
    spec_sid = await recorder.start_span(
        ctx.run_id, SpanKind.TOOL, f"agent:{route}", ctx,
        input_preview=ticket["message"][:200],
    )

    system_map = {
        "TechSupportAgent": (
            "You are a Senior Technical Support Engineer. "
            "Diagnose and resolve API, infrastructure, and integration issues. "
            "Give step-by-step solutions. End with: RESOLUTION: [one-line summary]"
        ),
        "BillingAgent": (
            "You are a Billing Specialist. Handle refunds and billing disputes professionally. "
            "Policy: auto-approve refunds within 30 days. "
            "End with: BILLING_ACTION: [what was done]"
        ),
        "EscalationAgent": (
            "You are a Senior Escalation Manager. Handle critical enterprise issues. "
            "Take personal ownership. Offer concrete SLA compensation. "
            "End with: ESCALATION_STATUS: RESOLVED | IN_PROGRESS | ESCALATED_FURTHER"
        ),
    }
    specialist_resp = await azure_chat(
        system_map.get(route, system_map["TechSupportAgent"]),
        (
            f"Customer: {ticket['customer']} | Tier: {ticket['tier']}\n"
            f"Issue: {ticket['subject']}\n"
            f"Details: {ticket['message']}\n"
            f"Triage summary: {triage['summary']}\n\n"
            "Provide a complete, professional response that resolves the customer issue."
        ),
    )
    await recorder.end_span(ctx.run_id, spec_sid, SpanStatus.OK,
        output_preview=specialist_resp[:150])
    print(f"  [{route}]  Response ready ({len(specialist_resp)} chars)")

    # ── Agent 3: QAReviewer (GUARDRAIL span) ─────────────────────────────
    qa_sid = await recorder.start_span(
        ctx.run_id, SpanKind.GUARDRAIL, "agent:QAReviewer", ctx,
        input_preview=specialist_resp[:200],
    )
    qa_resp = await azure_chat(
        (
            "You are the QA Reviewer for customer support responses. "
            "Check: accuracy, tone, completeness, policy compliance. "
            "Output APPROVED: [final customer message] or REVISION_NEEDED: [feedback]"
        ),
        (
            f"Review this response for {ticket['customer']}:\n\n"
            f"{specialist_resp}\n\n"
            "If good, output APPROVED: followed by the polished final message."
        ),
    )
    approved = "APPROVED" in qa_resp.upper()
    final = qa_resp.split("APPROVED:")[-1].strip() if approved else specialist_resp
    await recorder.end_span(ctx.run_id, qa_sid,
        SpanStatus.OK if approved else SpanStatus.ERROR,
        output_preview="APPROVED" if approved else "REVISION_NEEDED")
    print(f"  [QAReviewer]    {'✅ APPROVED' if approved else '⚠️  NEEDS REVISION'}")

    # ── Close RUN span ───────────────────────────────────────────────────
    await recorder.end_span(ctx.run_id, run_sid, SpanStatus.OK,
        output_preview=f"{triage['category']}/{triage['urgency']} resolved")

    trace = await recorder.get_trace(ctx.run_id)
    trace_info = f"{trace.span_count} spans | ${trace.total_cost_usd:.4f}" if trace else "n/a"
    print(f"  HarnessAgent    → {trace_info}")

    return {
        "ticket_id": ticket["id"],
        "customer":  ticket["customer"],
        "tier":      ticket["tier"],
        "triage":    triage,
        "approved":  approved,
        "final_response": final,
        "trace": trace_info,
        "run_id": ctx.run_id,
    }


async def main():
    print("=" * 54)
    print(" TUTORIAL 03 — AutoGen Support Team")
    print(f" Model: {MODEL_NAME}")
    print(" HarnessAgent: TraceRecorder + GUARDRAIL spans")
    print("=" * 54)

    results = []
    for t in TICKETS:
        r = await handle_ticket(t)
        results.append(r)

    # ── Build report ─────────────────────────────────────────────────────
    lines = [
        "# Tutorial 03 — AutoGen Multi-Agent Support Team\n",
        f"*Model: {MODEL_NAME} | HarnessAgent: RUN → TOOL × 2 → GUARDRAIL spans*\n",
        "## Ticket Summary\n",
        "| Ticket | Customer | Tier | Category | Urgency | QA | HarnessAgent Trace |",
        "|--------|----------|------|----------|---------|----|--------------------|",
    ]
    for r in results:
        t = r["triage"]
        lines.append(
            f"| {r['ticket_id']} | {r['customer']} | {r['tier']} "
            f"| {t.get('category','?')} | {t.get('urgency','?')} "
            f"| {'✅' if r['approved'] else '⚠️'} | {r['trace']} |"
        )

    lines.append("\n## Full Resolutions\n")
    for r, ticket in zip(results, TICKETS):
        t = r["triage"]
        lines += [
            f"### {r['ticket_id']} — {ticket['subject']}",
            f"**Customer:** {r['customer']} ({r['tier']})",
            f"**Triage:** {t.get('category')} | {t.get('urgency')} | Routed → {t.get('route_to')}",
            f"\n**HarnessAgent spans:**",
            f"- `run:support:{r['ticket_id']}` (RUN)",
            f"- `agent:TriageAgent` (TOOL)",
            f"- `agent:{t.get('route_to','TechSupportAgent')}` (TOOL)",
            f"- `agent:QAReviewer` (GUARDRAIL) — {'passed' if r['approved'] else 'flagged'}",
            f"- Trace: `GET /runs/{r['run_id']}/trace`\n",
            f"**Final Response to Customer:**\n\n{r['final_response']}\n",
        ]

    full = "\n".join(lines)
    (OUT / "final_report.md").write_text(full)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: tutorials/output/03_support_team/final_report.md")
    return results, full


if __name__ == "__main__":
    results, report = asyncio.run(main())
    print("\n" + report)
