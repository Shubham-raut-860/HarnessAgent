"""Safety pipeline factory for Codex Harness.

Constructs Guardrail Pipeline instances with appropriate stages
based on agent type and configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for a safety guardrail pipeline."""

    max_steps: int = 50
    max_tokens: int = 100_000
    max_wall_seconds: float = 300.0
    allowed_tools: list[str] | None = None  # None means all tools are allowed
    blocked_tools: list[str] = field(default_factory=list)
    allow_destructive_commands: bool = False
    pii_redact_output: bool = True
    injection_detect_input: bool = True
    loop_detection: bool = True
    loop_window: int = 10


def build_pipeline(
    agent_type: str,
    config: SafetyConfig,
    budget: Any | None = None,  # guardrail.intermediate.budget.Budget
) -> Any:
    """Build a Guardrail Pipeline configured for the given agent type.

    Pipeline composition:
    - Input stage:  InjectionDetector (if config.injection_detect_input)
    - Intermediate: Budget (steps / tokens / time), LoopDetector (if loop_detection),
                    ToolPolicy (allowed_tools / blocked_tools)
    - Output stage: PIIRedactor (if pii_redact_output)

    Returns a configured Pipeline instance.
    """
    try:
        from guardrail.pipeline import Pipeline, Stage
    except ImportError:
        logger.warning(
            "guardrail package not installed — returning NullPipeline for agent_type=%s",
            agent_type,
        )
        return _NullPipeline()

    input_stages: list[Any] = []
    intermediate_stages: list[Any] = []
    output_stages: list[Any] = []

    # ------------------------------------------------------------------
    # Input guards
    # ------------------------------------------------------------------
    if config.injection_detect_input:
        try:
            from guardrail.input.injection_detector import InjectionDetector
            input_stages.append(InjectionDetector())
        except ImportError:
            logger.debug("InjectionDetector not available — skipping")

    # ------------------------------------------------------------------
    # Intermediate guards
    # ------------------------------------------------------------------

    # Budget guard
    if budget is None:
        try:
            from guardrail.intermediate.budget import Budget
            budget = Budget(
                max_steps=config.max_steps,
                max_tokens=config.max_tokens,
                max_wall_seconds=config.max_wall_seconds,
            )
        except ImportError:
            logger.debug("Budget guard not available — skipping")

    if budget is not None:
        intermediate_stages.append(budget)

    # Loop detector
    if config.loop_detection:
        try:
            from guardrail.intermediate.loop_detector import LoopDetector
            intermediate_stages.append(LoopDetector(window=config.loop_window))
        except ImportError:
            logger.debug("LoopDetector not available — skipping")

    # Tool policy
    if config.allowed_tools is not None or config.blocked_tools:
        try:
            from guardrail.intermediate.tool_policy import ToolPolicy
            intermediate_stages.append(
                ToolPolicy(
                    allowed=config.allowed_tools,
                    blocked=config.blocked_tools,
                )
            )
        except ImportError:
            logger.debug("ToolPolicy not available — skipping")

    # ------------------------------------------------------------------
    # Output guards
    # ------------------------------------------------------------------
    if config.pii_redact_output:
        try:
            from guardrail.output.pii_redactor import PIIRedactor
            output_stages.append(PIIRedactor())
        except ImportError:
            logger.debug("PIIRedactor not available — skipping")

    # ------------------------------------------------------------------
    # Assemble pipeline
    # ------------------------------------------------------------------
    stages: list[Any] = []
    if input_stages:
        try:
            stages.append(Stage(name="input", guards=input_stages))
        except Exception:
            stages.extend(input_stages)
    if intermediate_stages:
        try:
            stages.append(Stage(name="intermediate", guards=intermediate_stages))
        except Exception:
            stages.extend(intermediate_stages)
    if output_stages:
        try:
            stages.append(Stage(name="output", guards=output_stages))
        except Exception:
            stages.extend(output_stages)

    try:
        pipeline = Pipeline(stages=stages, name=f"harness_{agent_type}")
    except Exception:
        # Some versions of guardrail use positional args or different API
        pipeline = Pipeline(stages)  # type: ignore[call-arg]

    logger.info(
        "Built safety pipeline for agent_type=%s: %d stages, %d guards",
        agent_type,
        len(stages),
        len(input_stages) + len(intermediate_stages) + len(output_stages),
    )
    return pipeline


def get_default_config(agent_type: str) -> SafetyConfig:
    """Return a sensible default SafetyConfig for the given agent type."""
    match agent_type:
        case "sql":
            return SafetyConfig(
                allowed_tools=[
                    "execute_sql",
                    "list_tables",
                    "describe_table",
                    "sample_rows",
                ],
                allow_destructive_commands=False,
                pii_redact_output=True,
                injection_detect_input=True,
                loop_detection=True,
            )
        case "code":
            return SafetyConfig(
                allowed_tools=[
                    "run_python",
                    "lint_code",
                    "read_file",
                    "write_file",
                    "apply_patch",
                    "list_workspace",
                ],
                allow_destructive_commands=False,
                pii_redact_output=True,
                injection_detect_input=True,
                loop_detection=True,
            )
        case "research":
            return SafetyConfig(
                allowed_tools=["read_file", "write_file", "list_workspace"],
                pii_redact_output=True,
                injection_detect_input=True,
            )
        case _:
            return SafetyConfig()


# ---------------------------------------------------------------------------
# Null implementation for when guardrail is unavailable
# ---------------------------------------------------------------------------

class _NullGuardResult:
    """Stub GuardResult that always allows the call through."""

    blocked: bool = False
    reason: str = ""
    decision: str = "allow"


class _NullPipeline:
    """Fallback pipeline that passes all checks when guardrail is not installed."""

    async def check_input(self, payload: Any) -> _NullGuardResult:
        return _NullGuardResult()

    async def check_step(self, payload: Any) -> _NullGuardResult:
        return _NullGuardResult()

    async def check_output(self, payload: Any) -> _NullGuardResult:
        return _NullGuardResult()

    def redact(self, text: str) -> str:
        return text
