"""Patch generation — creates prompt improvement patches from error analysis."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from harness.improvement.error_collector import ErrorRecord

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    """A proposed prompt improvement patch.

    Attributes:
        patch_id:       Unique identifier.
        agent_type:     Which agent's prompt to patch.
        target:         Target resource (e.g. "prompt" or "tool_config").
        op:             Operation: append | prepend | replace | remove | set.
        path:           For replace/remove: the text to find in the current prompt.
        value:          The new text to insert / replace with.
        rationale:      LLM-generated explanation of why this patch helps.
        proposed_by:    Who proposed this patch ("hermes" or user ID).
        proposed_at:    UTC timestamp.
        score:          Eval score after testing (None until evaluated).
        status:         pending | approved | rejected | applied.
        based_on_errors: List of ErrorRecord IDs this patch addresses.
    """

    patch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_type: str = ""
    target: str = "prompt"
    op: str = "append"
    path: str = ""
    value: str = ""
    rationale: str = ""
    proposed_by: str = "hermes"
    proposed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: Optional[float] = None
    status: str = "pending"  # pending | approved | rejected | applied
    based_on_errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "agent_type": self.agent_type,
            "target": self.target,
            "op": self.op,
            "path": self.path,
            "value": self.value,
            "rationale": self.rationale,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at.isoformat(),
            "score": self.score,
            "status": self.status,
            "based_on_errors": self.based_on_errors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Patch":
        proposed_at = data.get("proposed_at")
        if isinstance(proposed_at, str):
            try:
                proposed_at = datetime.fromisoformat(proposed_at)
            except ValueError:
                proposed_at = datetime.now(timezone.utc)
        elif not isinstance(proposed_at, datetime):
            proposed_at = datetime.now(timezone.utc)

        return cls(
            patch_id=data.get("patch_id", uuid.uuid4().hex),
            agent_type=data.get("agent_type", ""),
            target=data.get("target", "prompt"),
            op=data.get("op", "append"),
            path=data.get("path", ""),
            value=data.get("value", ""),
            rationale=data.get("rationale", ""),
            proposed_by=data.get("proposed_by", "hermes"),
            proposed_at=proposed_at,
            score=data.get("score"),
            status=data.get("status", "pending"),
            based_on_errors=list(data.get("based_on_errors", [])),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Patch":
        return cls.from_dict(json.loads(raw))


@dataclass
class PatchOutcome:
    """Result of applying a patch and re-evaluating.

    Attributes:
        patch:          The Patch that was applied.
        baseline_score: Score before applying the patch.
        patched_score:  Score after applying the patch.
        improvement:    patched_score - baseline_score.
        accepted:       Whether the patch was accepted (improvement >= threshold).
        eval_summary:   Human-readable evaluation summary.
    """

    patch: Patch
    baseline_score: float = 0.0
    patched_score: float = 0.0
    improvement: float = 0.0
    accepted: bool = False
    eval_summary: str = ""

    def __post_init__(self) -> None:
        self.improvement = self.patched_score - self.baseline_score


# ---------------------------------------------------------------------------
# Patch generation prompt
# ---------------------------------------------------------------------------

_GEN_SYSTEM = (
    "You are Hermes, an AI self-improvement engine for multi-agent systems. "
    "You analyze error patterns and generate targeted improvements to agent prompts. "
    "Always output valid JSON."
)

_GEN_PROMPT_TEMPLATE = """\
You are analyzing failures in an AI agent system to improve its prompt.

## Agent Type
{agent_type}

## Current Prompt
```
{current_prompt}
```

## Recent Errors ({error_count} failures)
{error_summary}

## Most Common Failure Classes
{failure_classes}

## Task
Generate a single targeted patch to improve the agent's prompt based on these errors.

The patch should:
1. Address the root cause of the most common failures
2. Be specific and actionable (not generic advice)
3. Add guidance that was missing or clarify confusing instructions

Respond with a JSON object in exactly this format:
{{
  "op": "<append|prepend|replace|remove|set>",
  "path": "<exact text to find in prompt for replace/remove, empty for append/prepend/set>",
  "value": "<the new text to insert or replace with>",
  "rationale": "<2-3 sentence explanation of why this patch addresses the failures>"
}}

JSON:"""


class PatchGenerator:
    """Generates prompt patches by analyzing error records with an LLM.

    Args:
        llm_provider:   LLMProvider for generating patches.
        prompt_manager: PromptManager for reading current active prompts.
        patch_store:    Optional Redis-backed store for persisting patches.
    """

    def __init__(
        self,
        llm_provider: Any,
        prompt_manager: Any,
        patch_store: Optional[Any] = None,
    ) -> None:
        self._llm = llm_provider
        self._prompt_manager = prompt_manager
        self._patch_store = patch_store

    async def generate(
        self,
        agent_type: str,
        errors: list[ErrorRecord],
        max_errors_in_prompt: int = 10,
    ) -> Optional[Patch]:
        """Generate a patch based on recent error records.

        Args:
            agent_type:            The agent type to patch.
            errors:                List of recent ErrorRecord objects.
            max_errors_in_prompt:  Max error entries to include in the LLM prompt.

        Returns:
            A Patch object if generation succeeded, else None.
        """
        if not errors:
            logger.info("No errors to generate patch from for agent_type=%s", agent_type)
            return None

        # Get current prompt
        current_prompt = await self._prompt_manager.get_prompt(agent_type)

        # Summarise errors
        sample_errors = errors[:max_errors_in_prompt]
        error_lines = []
        for i, err in enumerate(sample_errors, 1):
            error_lines.append(
                f"{i}. [{err.failure_class}] {err.error_message[:200]}"
                + (f"\n   Task: {err.task[:100]}" if err.task else "")
            )
        error_summary = "\n".join(error_lines)

        # Count failure classes
        fc_counts: dict[str, int] = {}
        for err in errors:
            fc_counts[err.failure_class] = fc_counts.get(err.failure_class, 0) + 1
        failure_classes = ", ".join(
            f"{fc}={count}"
            for fc, count in sorted(fc_counts.items(), key=lambda x: -x[1])[:5]
        )

        prompt = _GEN_PROMPT_TEMPLATE.format(
            agent_type=agent_type,
            current_prompt=current_prompt[:3000],
            error_count=len(errors),
            error_summary=error_summary,
            failure_classes=failure_classes,
        )

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                system=_GEN_SYSTEM,
            )
            raw = response.content.strip()

            # Strip markdown
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()

            data = json.loads(raw)

            patch = Patch(
                agent_type=agent_type,
                target="prompt",
                op=str(data.get("op", "append")).lower(),
                path=str(data.get("path", "")),
                value=str(data.get("value", "")),
                rationale=str(data.get("rationale", "")),
                proposed_by="hermes",
                based_on_errors=[err.record_id for err in sample_errors],
            )

            if self._patch_store is not None:
                await self._patch_store.save(patch)

            logger.info(
                "Generated patch %s for agent_type=%s (op=%s)",
                patch.patch_id[:8],
                agent_type,
                patch.op,
            )
            return patch

        except json.JSONDecodeError as exc:
            logger.warning("Patch generation JSON parse error: %s", exc)
            return None
        except Exception as exc:
            logger.error("Patch generation failed for agent_type=%s: %s", agent_type, exc)
            return None
