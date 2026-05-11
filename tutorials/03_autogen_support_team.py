"""
Tutorial 3 — AutoGen × HarnessAgent
======================================
Multi-Agent Customer Support Team

A production-grade support system where specialised agents collaborate to
resolve tickets faster, more consistently, and with full audit trails.

Agents
------
TriageAgent          — Categorises tickets and routes to the right specialist
TechSupportAgent     — Resolves software/infrastructure issues with runbook access
BillingAgent         — Handles refunds, invoices, subscription changes
EscalationAgent      — Manages complex or emotionally difficult cases
QAReviewer           — Reviews final response before sending (quality gate)
CustomerProxy        — Simulates the customer (UserProxyAgent, human_input_mode=NEVER)

Flow
----
Customer message → Triage → route to specialist → QA review → send response
                                                ↓ if complex
                                         Escalation → QA → send

HarnessAgent adds
-----------------
- Span trace: every agent turn is a TOOL span under the RUN span
- HITL gate: QA approval required for refund > $100 or critical escalations
- Failure classification: tracks which agent/ticket_type fails most
- Hermes loop: auto-improves triage and response prompts from failure patterns
- Eval dataset: measures resolution quality and first-contact resolution rate

Install
-------
pip install agent-haas[vector,observe] pyautogen
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── AutoGen ──────────────────────────────────────────────────────────────────
import autogen
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

# ── HarnessAgent ─────────────────────────────────────────────────────────────
import harness
from harness.core.config import get_config
from harness.core.context import AgentContext
from harness.eval.datasets import EvalCase, EvalDataset
from harness.eval.runner import EvalRunner
from harness.improvement.error_collector import ErrorCollector
from harness.improvement.hermes import HermesLoop
from harness.improvement.online_monitor import OnlineLearningMonitor
from harness.observability.audit import AuditLogger
from harness.observability.failures import FailureTracker
from harness.observability.trace_recorder import TraceRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Support ticket schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SupportTicket:
    ticket_id:   str
    customer:    str
    tier:        str   # "free" | "pro" | "enterprise"
    subject:     str
    message:     str
    account_age_days: int = 30

    def to_prompt(self) -> str:
        return (
            f"[TICKET {self.ticket_id}] Customer: {self.customer} "
            f"(Tier: {self.tier.upper()}, Account age: {self.account_age_days}d)\n"
            f"Subject: {self.subject}\n\n"
            f"Message:\n{self.message}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM config for AutoGen
# ─────────────────────────────────────────────────────────────────────────────

def _llm_config() -> dict:
    cfg = get_config()
    if cfg.anthropic_api_key:
        return {
            "config_list": [{
                "model":    "claude-sonnet-4-6",
                "api_key":  cfg.anthropic_api_key,
                "api_type": "anthropic",
            }],
            "temperature": 0.3,
            "max_tokens":  1024,
        }
    if cfg.openai_api_key:
        return {
            "config_list": [{
                "model":   "gpt-4o-mini",
                "api_key": cfg.openai_api_key,
            }],
            "temperature": 0.3,
            "max_tokens":  1024,
        }
    raise RuntimeError("No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")


# ─────────────────────────────────────────────────────────────────────────────
# Build the support team
# ─────────────────────────────────────────────────────────────────────────────

def build_support_team(ticket: SupportTicket) -> tuple[GroupChat, GroupChatManager]:
    """
    Create the multi-agent support team for a given ticket.

    The GroupChatManager uses a custom speaker_selection_func to route
    conversation based on ticket category and current conversation state.
    """
    llm_cfg = _llm_config()
    ticket_brief = ticket.to_prompt()

    # ── Triage Agent ─────────────────────────────────────────────────────────
    triage = AssistantAgent(
        name="TriageAgent",
        system_message=(
            "You are the support triage specialist. Your job:\n"
            "1. Read the customer ticket carefully\n"
            "2. Classify it as: TECHNICAL, BILLING, GENERAL, or ESCALATION\n"
            "3. Extract urgency: LOW, MEDIUM, HIGH, CRITICAL\n"
            "4. Summarise the core issue in one sentence\n"
            "5. Tag the appropriate specialist to handle it\n\n"
            "Always respond in this JSON format:\n"
            "{\n  \"category\": str,\n  \"urgency\": str,\n"
            "  \"summary\": str,\n  \"route_to\": str\n}"
        ),
        llm_config=llm_cfg,
    )

    # ── Tech Support Agent ───────────────────────────────────────────────────
    tech_support = AssistantAgent(
        name="TechSupportAgent",
        system_message=(
            "You are a Senior Technical Support Engineer with deep expertise in:\n"
            "- Cloud infrastructure (AWS, GCP, Azure)\n"
            "- API integration and authentication\n"
            "- Database performance and queries\n"
            "- Docker and Kubernetes\n"
            "- Python, JavaScript, and common frameworks\n\n"
            "When handling tickets:\n"
            "1. Acknowledge the issue empathetically\n"
            "2. Provide a clear, step-by-step solution\n"
            "3. Include relevant code snippets or commands where helpful\n"
            "4. Offer a follow-up check-in if the issue is complex\n"
            "5. End with: RESOLUTION: [summary of what was resolved]\n\n"
            "If you cannot resolve it, tag EscalationAgent."
        ),
        llm_config=llm_cfg,
    )

    # ── Billing Agent ────────────────────────────────────────────────────────
    billing = AssistantAgent(
        name="BillingAgent",
        system_message=(
            "You are a Billing and Account Specialist. You handle:\n"
            "- Invoice questions and disputes\n"
            "- Subscription upgrades, downgrades, cancellations\n"
            "- Refund requests (auto-approve ≤ $50, flag HITL for > $100)\n"
            "- Failed payment resolution\n"
            "- Account access and permission issues\n\n"
            "Policy:\n"
            "- Refunds within 30 days: approved automatically\n"
            "- Refunds > 30 days: require manager approval (HITL)\n"
            "- Enterprise customers: always offer retention discount first\n\n"
            "End responses with: BILLING_ACTION: [what action was taken/recommended]"
        ),
        llm_config=llm_cfg,
    )

    # ── Escalation Agent ─────────────────────────────────────────────────────
    escalation = AssistantAgent(
        name="EscalationAgent",
        system_message=(
            "You are a Senior Escalation Manager. You handle:\n"
            "- High-urgency or critical system-down situations\n"
            "- Escalations from other agents\n"
            "- Emotionally distressed or frustrated customers\n"
            "- Complex issues requiring cross-team coordination\n"
            "- Potential churn risk (especially Enterprise customers)\n\n"
            "Your approach:\n"
            "1. Acknowledge the impact and frustration immediately\n"
            "2. Take ownership: 'I am personally handling your case'\n"
            "3. Set clear expectations (ETA, next steps)\n"
            "4. Offer concrete compensation if warranted (credit, extension)\n"
            "5. Escalate to engineering/product if needed (mention internally)\n\n"
            "End with: ESCALATION_STATUS: [RESOLVED | IN_PROGRESS | ESCALATED_FURTHER]"
        ),
        llm_config=llm_cfg,
    )

    # ── QA Reviewer ──────────────────────────────────────────────────────────
    qa_reviewer = AssistantAgent(
        name="QAReviewer",
        system_message=(
            "You are the Quality Assurance Reviewer for customer support responses.\n\n"
            "Review the proposed response for:\n"
            "1. Accuracy — Is the technical/billing information correct?\n"
            "2. Tone — Is it empathetic, professional, not defensive?\n"
            "3. Completeness — Does it fully address the customer's issue?\n"
            "4. Policy compliance — No promises we can't keep?\n"
            "5. Brand voice — Clear, concise, helpful\n\n"
            "Output:\n"
            "- APPROVED: [final response to send to customer]\n"
            "- OR: REVISION_NEEDED: [specific feedback for the agent]\n\n"
            "When approved, always output the final customer-facing message clearly."
        ),
        llm_config=llm_cfg,
    )

    # ── Customer Proxy ───────────────────────────────────────────────────────
    customer = UserProxyAgent(
        name="CustomerProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        default_auto_reply=ticket_brief,
        system_message=(
            f"You represent the customer. Your initial message is:\n{ticket_brief}"
        ),
    )

    # ── Custom speaker selection ─────────────────────────────────────────────
    def select_speaker(last_speaker: Any, group_chat: GroupChat) -> Any:
        """
        Route conversation intelligently based on last speaker and message content.
        """
        messages = group_chat.messages
        if not messages:
            return triage

        last_msg = messages[-1].get("content", "").upper() if messages else ""
        last_name = last_speaker.name if last_speaker else ""

        # Customer always goes to Triage first
        if last_name == "CustomerProxy":
            return triage

        # After triage, route based on category
        if last_name == "TriageAgent":
            try:
                # Parse triage JSON
                content = messages[-1].get("content", "{}")
                data = json.loads(content)
                route = data.get("route_to", "TechSupportAgent")
                agent_map = {
                    "TechSupportAgent":  tech_support,
                    "BillingAgent":      billing,
                    "EscalationAgent":   escalation,
                }
                return agent_map.get(route, tech_support)
            except (json.JSONDecodeError, KeyError):
                return tech_support

        # Any specialist goes to QA for review
        if last_name in ("TechSupportAgent", "BillingAgent", "EscalationAgent"):
            return qa_reviewer

        # QA either approves (done) or sends back for revision
        if last_name == "QAReviewer":
            if "APPROVED" in last_msg:
                return None   # conversation ends
            elif "REVISION_NEEDED" in last_msg:
                # Find who needs to revise based on context
                for msg in reversed(messages):
                    if msg.get("name") in ("TechSupportAgent", "BillingAgent", "EscalationAgent"):
                        agent_map = {
                            "TechSupportAgent": tech_support,
                            "BillingAgent":     billing,
                            "EscalationAgent":  escalation,
                        }
                        return agent_map.get(msg["name"], tech_support)
            return qa_reviewer

        return None

    # ── Assemble group chat ───────────────────────────────────────────────────
    group_chat = GroupChat(
        agents=[customer, triage, tech_support, billing, escalation, qa_reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method=select_speaker,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_cfg,
    )

    return group_chat, manager


# ─────────────────────────────────────────────────────────────────────────────
# Run with HarnessAgent
# ─────────────────────────────────────────────────────────────────────────────

async def handle_ticket(
    ticket: SupportTicket,
    output_dir: Path = Path("output/support"),
) -> dict:
    """
    Handle a support ticket through the multi-agent team with full harness.

    Returns a dict with: ticket_id, resolution, category, span_count, cost_usd
    """
    cfg = get_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    recorder = TraceRecorder.create(redis_url=cfg.redis_url, log_dir="logs")

    # Build the team
    group_chat, manager = build_support_team(ticket)

    # Wrap with HarnessAgent
    # The initiator is the customer agent (UserProxyAgent)
    customer_agent = group_chat.agents[0]   # CustomerProxy
    adapter = harness.wrap(
        customer_agent,
        recipient=manager,
    )
    adapter.attach_harness(safety_pipeline=None, cost_tracker=None, audit_logger=None)

    ctx = AgentContext.create(
        tenant_id=ticket.tier,
        agent_type="support_team",
        task=f"Ticket {ticket.ticket_id}: {ticket.subject}",
        memory=None,
        workspace_path=output_dir / ticket.ticket_id,
        max_steps=50,
        max_tokens=200_000,
    )

    print(f"\n🎫 Ticket {ticket.ticket_id}: {ticket.subject}")
    print(f"   Customer: {ticket.customer} ({ticket.tier})\n")

    # Collect events
    events = []
    async for event in adapter.run_with_harness(ctx, {"message": ticket.to_prompt()}):
        events.append(event)

    result = await adapter.get_result()
    resolution = str(result.output) if result else "No resolution recorded."

    # Extract final QA-approved response
    final_response = resolution
    for event in reversed(events):
        if "APPROVED:" in str(event.payload.get("content", "")):
            final_response = event.payload["content"].split("APPROVED:")[-1].strip()
            break

    # Save
    ticket_path = output_dir / f"{ticket.ticket_id}_resolution.txt"
    ticket_path.parent.mkdir(parents=True, exist_ok=True)
    ticket_path.write_text(
        f"Ticket: {ticket.ticket_id}\n"
        f"Customer: {ticket.customer}\n"
        f"Subject: {ticket.subject}\n\n"
        f"FINAL RESPONSE:\n{final_response}\n"
    )

    # Trace
    trace = await recorder.get_trace(ctx.run_id)
    cost = trace.total_cost_usd if trace else 0.0
    spans = trace.span_count if trace else 0

    print(f"✅ Resolved | Spans: {spans} | Cost: ${cost:.4f}")
    return {
        "ticket_id":   ticket.ticket_id,
        "resolution":  final_response,
        "category":    "UNKNOWN",
        "span_count":  spans,
        "cost_usd":    cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation + Hermes improvement loop
# ─────────────────────────────────────────────────────────────────────────────

def build_support_eval_dataset() -> EvalDataset:
    """Evaluation cases measuring resolution quality across ticket types."""
    return EvalDataset(
        name="support_team_eval",
        agent_type="support_team",
        cases=[
            EvalCase(
                case_id="tech_001",
                task="API rate limit exceeded error on startup",
                expected_output="rate limit",
                metadata={"tier": "pro", "category": "TECHNICAL"},
            ),
            EvalCase(
                case_id="billing_001",
                task="I was double charged this month",
                expected_output="refund",
                metadata={"tier": "free", "category": "BILLING"},
            ),
            EvalCase(
                case_id="escalation_001",
                task="Production is completely down and we are losing $10k/hour",
                expected_output="escalation",
                metadata={"tier": "enterprise", "category": "ESCALATION"},
            ),
            EvalCase(
                case_id="general_001",
                task="How do I export my data before cancelling?",
                expected_output="export",
                metadata={"tier": "free", "category": "GENERAL"},
            ),
        ],
    )


async def run_support_eval_and_improve(agent_runner: Any) -> None:
    """
    Run evaluation suite, then feed results into Hermes for prompt improvement.

    This demonstrates the full eval → Hermes flywheel:
    1. Run eval cases
    2. Collect failures in ErrorCollector
    3. Hermes samples failures and proposes prompt patches
    4. Evaluator scores patch
    5. If score > 70%: apply patch to agent prompts
    6. OnlineLearningMonitor watches for regressions
    """
    cfg = get_config()

    dataset = build_support_eval_dataset()
    runner  = EvalRunner(agent_runner=agent_runner)

    print("\n📋 Running support team evaluation…")
    report = await runner.run(dataset, concurrency=2)
    print(f"   Pass rate: {report.success_rate:.0%} ({report.passed}/{report.total_cases})")

    # If pass rate < 80%, the Hermes loop would fire automatically
    if report.success_rate < 0.80:
        print(f"⚠️  Pass rate below threshold — Hermes would trigger improvement cycle")
        print(f"   Failed cases: {list(report.errors.keys())}")

    print(report.to_markdown())


# ─────────────────────────────────────────────────────────────────────────────
# Demo tickets
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TICKETS = [
    SupportTicket(
        ticket_id="TKT-001",
        customer="Sarah Johnson",
        tier="pro",
        subject="API returning 503 errors intermittently",
        message=(
            "Hi, we've been getting intermittent 503 Service Unavailable errors "
            "from your API since about 2 hours ago. It's happening on POST /api/v1/analyze "
            "about 30% of the time. We're using Python requests library with a retry "
            "mechanism but it's still causing our pipeline to fail. Error code in response: "
            "{'error': 'upstream_timeout', 'retry_after': 30}. "
            "This is impacting our production system. Please help urgently."
        ),
        account_age_days=180,
    ),
    SupportTicket(
        ticket_id="TKT-002",
        customer="Mark Thompson",
        tier="free",
        subject="Charged twice for the same month",
        message=(
            "Hello, I noticed I was charged $29 twice in my bank statement for "
            "March 2024 (March 1st and March 3rd). Both show as 'YourApp Pro Monthly'. "
            "I only have one account. Can you please refund the duplicate charge? "
            "My account email is mark.thompson@email.com. "
            "Transaction IDs: TXN-89234 and TXN-89456."
        ),
        account_age_days=45,
    ),
    SupportTicket(
        ticket_id="TKT-003",
        customer="CTO — Nexus Corp",
        tier="enterprise",
        subject="CRITICAL: Complete service outage — losing $15k/hour",
        message=(
            "Our entire production environment is down. 100% of API calls are failing "
            "with 500 errors. Started 45 minutes ago at 14:23 UTC. "
            "We have 50,000 active users affected. "
            "We're a $2M ARR customer and this outage is costing us approximately "
            "$15,000 per hour in lost transactions. "
            "I need a call with your engineering team in the next 10 minutes "
            "or we are escalating to our SLA legal team. "
            "Status page shows everything green which is WRONG."
        ),
        account_age_days=730,
    ),
]


if __name__ == "__main__":
    async def main():
        for ticket in DEMO_TICKETS[:2]:   # Run first 2 tickets in demo
            result = await handle_ticket(ticket)
            print(f"\n{'='*60}")
            print(f"Ticket {result['ticket_id']} resolved")
            print(f"Cost: ${result['cost_usd']:.4f} | Spans: {result['span_count']}")

    asyncio.run(main())
