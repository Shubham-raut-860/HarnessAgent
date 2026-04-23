"""HermesLoop: orchestrates the full self-improvement cycle for Codex Harness agents."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from harness.improvement.error_collector import ErrorCollector, ErrorRecord
from harness.improvement.evaluator import EvalResult, Evaluator
from harness.improvement.patch_generator import Patch, PatchGenerator

logger = logging.getLogger(__name__)

# Default agent types for the Hermes loop
_DEFAULT_AGENT_TYPES = ["sql", "code", "base"]

# Redis key prefix for storing pending patches
_PATCH_KEY_PREFIX = "harness:hermes:patch:"
_PATCH_INDEX_KEY = "harness:hermes:patch_index"


@dataclass
class PatchOutcome:
    """Result of one Hermes improvement cycle for a single agent type.

    Attributes:
        patch:       The proposed Patch (may be None if no proposal was generated).
        eval_result: The evaluation result (may be None if not evaluated).
        applied:     Whether the patch was applied to the prompt store.
        reason:      Human-readable explanation of why the patch was applied or skipped.
    """

    patch: Patch | None
    eval_result: EvalResult | None
    applied: bool
    reason: str


class HermesLoop:
    """Orchestrates the Hermes self-improvement loop.

    One cycle per agent type:
    1. Check error count in rolling window — skip if fewer than min_errors.
    2. Sample a batch of recent errors.
    3. Generate a patch proposal from the LLM.
    4. Evaluate the patch by replaying failing tasks.
    5. If score > threshold AND auto_apply: apply via prompt_store.
       Otherwise: store with status="pending" for human review.
    6. Record metrics.

    Can be run on a schedule via APScheduler using start_background().
    """

    def __init__(
        self,
        collector: ErrorCollector,
        generator: PatchGenerator,
        evaluator: Evaluator,
        prompt_store: Any,  # PromptStore
        metrics: Any,
        config: Any,  # Settings
    ) -> None:
        """
        Args:
            collector:    ErrorCollector for sampling agent failures.
            generator:    PatchGenerator for proposing patches.
            evaluator:    Evaluator for scoring patches.
            prompt_store: PromptStore (or PromptManager) for applying patches.
            metrics:      HarnessMetrics instance for recording patch counts.
            config:       Settings with hermes_* configuration keys.
        """
        self._collector = collector
        self._generator = generator
        self._evaluator = evaluator
        self._prompt_store = prompt_store
        self._metrics = metrics
        self._config = config

        self.threshold: float = getattr(config, "hermes_patch_score_threshold", 0.7)
        self.auto_apply: bool = getattr(config, "hermes_auto_apply", False)
        self.min_errors: int = getattr(config, "hermes_min_errors_to_trigger", 5)
        self._interval_seconds: float = getattr(config, "hermes_interval_seconds", 3600.0)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self, agent_type: str) -> PatchOutcome | None:
        """Run one improvement cycle for the given agent type.

        Args:
            agent_type: The agent type to improve.

        Returns:
            PatchOutcome if a patch was proposed/evaluated, None if skipped.
        """
        logger.info("Hermes cycle starting for agent_type=%s", agent_type)

        # 1. Check error count in window
        try:
            error_count = await self._collector.count(agent_type)
        except Exception as exc:
            logger.warning("Hermes: error count check failed for %s: %s", agent_type, exc)
            return None

        if error_count < self.min_errors:
            logger.info(
                "Hermes: insufficient errors for %s (%d < %d) — skipping cycle",
                agent_type,
                error_count,
                self.min_errors,
            )
            return None

        # 2. Sample error batch
        try:
            errors: list[ErrorRecord] = await self._collector.get_recent(
                agent_type, limit=max(10, self.min_errors * 2)
            )
        except Exception as exc:
            logger.error("Hermes: error sampling failed for %s: %s", agent_type, exc)
            return None

        if not errors:
            logger.info("Hermes: no errors sampled for %s", agent_type)
            return None

        logger.info(
            "Hermes: sampled %d errors for agent_type=%s", len(errors), agent_type
        )

        # 3. Get current agent config
        current_config: dict[str, Any] = {}
        try:
            if hasattr(self._prompt_store, "get_prompt"):
                current_prompt = await self._prompt_store.get_prompt(agent_type)
                current_config = {"system_prompt": current_prompt}
            elif hasattr(self._prompt_store, "get"):
                current_prompt = await self._prompt_store.get(agent_type)
                current_config = {"system_prompt": str(current_prompt)}
        except Exception as exc:
            logger.warning("Hermes: could not load current config for %s: %s", agent_type, exc)

        # 4. Generate patch proposal
        patch: Patch | None = None
        try:
            patch = await self._generator.generate(
                agent_type=agent_type,
                errors=errors,
                max_errors_in_prompt=10,
            )
        except Exception as exc:
            logger.error("Hermes: patch generation failed for %s: %s", agent_type, exc)

        if patch is None:
            logger.info("Hermes: no patch generated for agent_type=%s", agent_type)
            return PatchOutcome(
                patch=None,
                eval_result=None,
                applied=False,
                reason="Patch generator returned no proposal.",
            )

        logger.info(
            "Hermes: generated patch %s for %s (op=%s)",
            patch.patch_id[:8],
            agent_type,
            patch.op,
        )

        # 5. Evaluate patch
        eval_result: EvalResult | None = None
        try:
            # Use a subset of errors as test cases (most recent half)
            test_cases = errors[: max(3, len(errors) // 2)]
            eval_result = await self._evaluator.score(
                patch=patch,
                test_cases=test_cases,
                agent_type=agent_type,
            )
            patch.score = eval_result.score
        except Exception as exc:
            logger.error(
                "Hermes: patch evaluation failed for %s (patch=%s): %s",
                agent_type,
                patch.patch_id[:8],
                exc,
            )
            eval_result = None

        # 6. Apply or queue for human review
        applied = False
        reason: str = ""

        score = eval_result.score if eval_result is not None else 0.0

        if eval_result is None:
            patch.status = "pending"
            reason = "Evaluation failed — patch queued for manual review."
            await self._store_patch(patch)

        elif score >= self.threshold and self.auto_apply:
            # Apply the patch automatically
            try:
                await self._apply_patch(patch, agent_type)
                patch.status = "applied"
                applied = True
                reason = (
                    f"Score {score:.3f} >= threshold {self.threshold:.3f} "
                    f"and auto_apply=True — patch applied."
                )
                logger.info(
                    "Hermes: auto-applied patch %s for %s (score=%.3f)",
                    patch.patch_id[:8],
                    agent_type,
                    score,
                )
            except Exception as exc:
                patch.status = "pending"
                reason = f"Application failed: {exc} — patch queued for manual review."
                logger.error(
                    "Hermes: patch application failed for %s: %s", agent_type, exc
                )
                await self._store_patch(patch)

        elif score >= self.threshold and not self.auto_apply:
            patch.status = "approved"
            reason = (
                f"Score {score:.3f} >= threshold {self.threshold:.3f} "
                f"but auto_apply=False — patch approved, awaiting manual application."
            )
            await self._store_patch(patch)

        else:
            patch.status = "rejected"
            reason = (
                f"Score {score:.3f} < threshold {self.threshold:.3f} — patch rejected."
            )
            await self._store_patch(patch)

        # 7. Record metrics
        self._record_metric(agent_type, patch.status)

        logger.info(
            "Hermes cycle complete for %s: patch=%s status=%s score=%.3f applied=%s",
            agent_type,
            patch.patch_id[:8],
            patch.status,
            score,
            applied,
        )

        return PatchOutcome(
            patch=patch,
            eval_result=eval_result,
            applied=applied,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Multi-agent cycle
    # ------------------------------------------------------------------

    async def run_all_agents(
        self, agent_types: list[str] | None = None
    ) -> list[PatchOutcome]:
        """Run improvement cycles for multiple agent types concurrently.

        Args:
            agent_types: Agent types to run. Defaults to ["sql", "code", "base"].

        Returns:
            List of PatchOutcome objects (one per agent type, None entries excluded).
        """
        if agent_types is None:
            agent_types = _DEFAULT_AGENT_TYPES

        results = await asyncio.gather(
            *[self.run_cycle(at) for at in agent_types],
            return_exceptions=True,
        )

        outcomes: list[PatchOutcome] = []
        for agent_type, result in zip(agent_types, results):
            if isinstance(result, Exception):
                logger.error(
                    "Hermes: run_cycle raised for %s: %s", agent_type, result
                )
            elif result is not None:
                outcomes.append(result)

        return outcomes

    # ------------------------------------------------------------------
    # Background scheduler
    # ------------------------------------------------------------------

    async def start_background(self, agent_types: list[str] | None = None) -> None:
        """Start the Hermes loop as a background APScheduler job.

        Runs run_all_agents every hermes_interval_seconds.

        Args:
            agent_types: Agent types to run on each interval.
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self.run_all_agents,
                "interval",
                seconds=self._interval_seconds,
                kwargs={"agent_types": agent_types},
                id="hermes_loop",
                max_instances=1,
                coalesce=True,
            )
            scheduler.start()

            logger.info(
                "Hermes background loop started: interval=%.0fs, agents=%s",
                self._interval_seconds,
                agent_types or _DEFAULT_AGENT_TYPES,
            )

            # Keep running until cancelled
            while True:
                await asyncio.sleep(60)

        except ImportError:
            logger.warning(
                "apscheduler not installed — running Hermes loop as a simple asyncio task. "
                "Install with: pip install apscheduler"
            )
            # Fallback: manual asyncio loop
            if agent_types is None:
                agent_types = _DEFAULT_AGENT_TYPES
            while True:
                try:
                    await self.run_all_agents(agent_types)
                except Exception as exc:
                    logger.error("Hermes background cycle failed: %s", exc)
                await asyncio.sleep(self._interval_seconds)

        except asyncio.CancelledError:
            logger.info("Hermes background loop cancelled")
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _apply_patch(self, patch: Patch, agent_type: str) -> None:
        """Apply a patch to the prompt store."""
        try:
            if hasattr(self._prompt_store, "apply_patch"):
                await _maybe_await(self._prompt_store.apply_patch(patch))
            elif hasattr(self._prompt_store, "update_prompt"):
                # Build the new prompt by fetching current and applying op
                current = await _maybe_await(self._prompt_store.get_prompt(agent_type))
                new_prompt = _apply_op(current, patch.op, patch.path, patch.value)
                await _maybe_await(
                    self._prompt_store.update_prompt(agent_type, new_prompt)
                )
            else:
                raise AttributeError(
                    f"prompt_store has no apply_patch or update_prompt method"
                )
        except Exception as exc:
            logger.error("Patch application failed: %s", exc)
            raise

    async def _store_patch(self, patch: Patch) -> None:
        """Persist a patch for later review."""
        try:
            if hasattr(self._generator, "_patch_store") and self._generator._patch_store:
                await _maybe_await(self._generator._patch_store.save(patch))
        except Exception as exc:
            logger.debug("Could not persist patch %s: %s", patch.patch_id[:8], exc)

    def _record_metric(self, agent_type: str, status: str) -> None:
        """Increment the hermes_patches_total Prometheus counter."""
        if self._metrics is None:
            return
        try:
            counter = getattr(self._metrics, "hermes_patches_total", None)
            if counter is not None:
                counter.labels(agent_type=agent_type, status=status).inc()
        except Exception as exc:
            logger.debug("Hermes metric recording failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _maybe_await(obj: Any) -> Any:
    """Await if coroutine, else return directly."""
    if asyncio.iscoroutine(obj):
        return await obj
    return obj


def _apply_op(current: str, op: str, path: str, value: Any) -> str:
    """Apply a patch operation to a string prompt."""
    value_str = str(value)
    op = op.lower()
    if op == "append":
        return (current + "\n" + value_str).strip()
    elif op == "prepend":
        return (value_str + "\n" + current).strip()
    elif op == "replace":
        return current.replace(path, value_str)
    elif op == "remove":
        return current.replace(path, "").strip()
    elif op in ("set",):
        return value_str
    else:
        return (current + "\n" + value_str).strip()
