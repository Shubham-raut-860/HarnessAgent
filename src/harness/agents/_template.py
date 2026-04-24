"""Agent creation template for HarnessAgent.

Copy this file to create a new agent type:

    cp src/harness/agents/_template.py src/harness/agents/my_agent.py

Then:
  1. Rename MyAgent / MyAgentConfig to your agent name
  2. Set AGENT_TYPE to a unique string
  3. Write your SYSTEM_PROMPT
  4. Register in src/harness/orchestrator/factory.py
  5. Call POST /runs with {"agent_type": "my_agent", "task": "..."}

Everything else — ReAct loop, memory, tools, safety, tracing,
checkpointing, cost tracking — is inherited from BaseAgent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from harness.agents.base import BaseAgent
from harness.core.context import AgentContext, AgentResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Agent Config — domain-specific settings for this agent type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MyAgentConfig:
    """Configuration specific to MyAgent.

    Add fields for any settings that control agent behaviour.
    All fields should have defaults so the agent can be started
    with zero config and be customised per-run via metadata.
    """

    # How many results to return in the final answer
    max_results: int = 10

    # Domain label used in the system prompt
    domain: str = "general"

    # Whether to apply strict validation on tool outputs
    strict_mode: bool = False

    # Any extra metadata the agent should know about
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent Implementation
# ─────────────────────────────────────────────────────────────────────────────

class MyAgent(BaseAgent):
    """Domain-specialized agent built on HarnessAgent's BaseAgent.

    Inherited for free (no extra code needed):
    - ReAct reasoning loop (think → tool call → observe → repeat)
    - 3-tier memory (Redis short-term → vector DB long-term → knowledge graph)
    - LLM routing with circuit breaker and provider fallback
    - Budget enforcement (hard stop when cost cap is hit)
    - Safety guardrail pipeline (input / step / output)
    - Human-in-the-loop approval gates
    - MLflow traces + Prometheus metrics per step
    - Checkpoint + restore every 10 steps
    - Structured audit log (JSONL)

    To customise, override:
    - system_prompt  : the agent's persona and instructions
    - build_initial_messages : how to frame the first user message
    - post_process   : optional result transformation before returning
    """

    # ── Registration ──────────────────────────────────────────────────────────
    # This string maps to the agent in AgentFactory and in POST /runs payloads.
    AGENT_TYPE = "my_agent"

    # ── System Prompt Template ────────────────────────────────────────────────
    # Use {domain} and {tool_list} — they are filled in at runtime.
    # Keep instructions concise; verbose prompts cost tokens every step.
    _SYSTEM = """\
You are a specialized {domain} assistant.

Available tools:
{tool_list}

Guidelines:
1. Think step-by-step before calling any tool.
2. Call only one tool per reasoning step.
3. Verify results before presenting them to the user.
4. If a tool fails, explain why and try a different approach.
5. When you have enough information to answer, respond directly.
6. Never reveal internal reasoning or tool names to the user.
"""

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, config: MyAgentConfig | None = None, **kwargs: Any) -> None:
        """Initialise the agent.

        Args:
            config: Domain-specific settings. Defaults to MyAgentConfig().
            **kwargs: Forwarded to BaseAgent (llm_router, memory_manager, etc.).
        """
        super().__init__(**kwargs)
        self.config = config or MyAgentConfig()

    # ── Override: System Prompt ───────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        """Return the system prompt with runtime values filled in."""
        tool_names = ", ".join(self.tool_registry.list_tool_names())
        return self._SYSTEM.format(
            domain=self.config.domain,
            tool_list=tool_names or "(no tools registered)",
        )

    # ── Override: Initial Messages ────────────────────────────────────────────

    def build_initial_messages(
        self, task: str, context: AgentContext
    ) -> list[dict[str, str]]:
        """Frame the task as the first user message.

        Args:
            task:    The raw task string from the run request.
            context: AgentContext with tenant_id, run_id, metadata, etc.

        Returns:
            A list of chat messages (role + content dicts).
            Typically just one user message, but you can add system
            context or few-shot examples here.
        """
        lines = [f"Task: {task}"]

        if self.config.max_results != 10:
            lines.append(f"Return at most {self.config.max_results} results.")

        if self.config.strict_mode:
            lines.append("Apply strict validation to all tool outputs.")

        if context.metadata:
            # Pass any run-level metadata the agent should know about
            relevant = {
                k: v for k, v in context.metadata.items()
                if k not in ("tenant_id", "run_id")
            }
            if relevant:
                lines.append(f"Additional context: {relevant}")

        return [{"role": "user", "content": "\n".join(lines)}]

    # ── Override: Post-Processing (optional) ──────────────────────────────────

    async def post_process(self, result: AgentResult) -> AgentResult:
        """Optional: transform or validate the result before returning.

        Common uses:
        - Strip internal reasoning from the final answer
        - Parse structured output (JSON / table) from the text
        - Redact additional sensitive fields
        - Validate against a schema and set result.success = False on failure

        Args:
            result: The raw AgentResult from the ReAct loop.

        Returns:
            The (optionally modified) AgentResult.
        """
        # Example: truncate to max_results if the answer is a list
        # if isinstance(result.output, list):
        #     result.output = result.output[:self.config.max_results]

        return result

    # ── Optional: Custom Tool Hooks ───────────────────────────────────────────
    # If you need to intercept tool calls (e.g., inject auth headers, log
    # domain-specific events), override _before_tool and _after_tool from
    # BaseAgent. These are called for every tool execution in the ReAct loop.
    #
    # async def _before_tool(self, tool_name: str, args: dict) -> dict:
    #     logger.info("About to call %s with %s", tool_name, args)
    #     return args  # return (possibly modified) args
    #
    # async def _after_tool(self, tool_name: str, result: Any) -> Any:
    #     logger.info("Tool %s returned %s", tool_name, result)
    #     return result  # return (possibly modified) result
