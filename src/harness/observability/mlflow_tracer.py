"""MLflow agent-level tracing and experiment tracking."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore[import]
    import mlflow.tracking  # type: ignore[import]

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning(
        "mlflow not installed — experiment tracking is a no-op. "
        "Install with: pip install mlflow"
    )

if TYPE_CHECKING:
    from harness.core.context import AgentContext


class _NoOpSpan:
    """Returned when MLflow is unavailable."""

    def set_inputs(self, inputs: Any) -> None:
        pass

    def set_outputs(self, outputs: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class MLflowAgentTracer:
    """
    Provides MLflow run / span lifecycle management for agent execution.

    One MLflow run per agent invocation, with nested spans for LLM calls,
    tool calls, and inter-agent messages.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._active_run_ids: dict[str, str] = {}  # harness run_id → mlflow run_id

        if _MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Agent-level run
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def agent_run(
        self, ctx: "AgentContext"
    ) -> AsyncIterator[Any]:
        """
        Open an MLflow run for the duration of an agent execution.

        Logs agent parameters on entry and metrics on exit.
        """
        if not _MLFLOW_AVAILABLE:
            yield _NoOpSpan()
            return

        try:
            with mlflow.start_run(run_name=f"{ctx.agent_type}_{ctx.run_id[:8]}") as run:
                self._active_run_ids[ctx.run_id] = run.info.run_id

                # Log parameters
                mlflow.log_params(
                    {
                        "run_id": ctx.run_id,
                        "tenant_id": ctx.tenant_id,
                        "agent_type": ctx.agent_type,
                        "max_steps": ctx.max_steps,
                        "max_tokens": ctx.max_tokens,
                        "task": ctx.task[:256],  # truncate for MLflow limit
                    }
                )

                try:
                    with mlflow.start_span(
                        name=f"agent:{ctx.agent_type}",
                        span_type="AGENT",
                        attributes={
                            "run_id": ctx.run_id,
                            "tenant_id": ctx.tenant_id,
                        },
                    ) as span:
                        yield span
                finally:
                    # Log final metrics
                    mlflow.log_metrics(
                        {
                            "step_count": float(ctx.step_count),
                            "token_count": float(ctx.token_count),
                            "elapsed_seconds": float(ctx.elapsed_seconds),
                            "failed": float(int(ctx.failed)),
                        }
                    )
                    self._active_run_ids.pop(ctx.run_id, None)
        except Exception as exc:
            logger.warning("MLflow agent_run failed: %s", exc)
            yield _NoOpSpan()

    # ------------------------------------------------------------------
    # LLM span
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def llm_span(
        self,
        model: str,
        messages: list[dict[str, Any]],
        provider: str,
    ) -> AsyncIterator[Any]:
        """Nested span for a single LLM completion call."""
        if not _MLFLOW_AVAILABLE:
            yield _NoOpSpan()
            return

        try:
            with mlflow.start_span(
                name=f"llm:{provider}/{model}",
                span_type="LLM",
                attributes={"model": model, "provider": provider},
            ) as span:
                span.set_inputs({"messages": messages, "model": model, "provider": provider})
                yield span
        except Exception as exc:
            logger.debug("MLflow llm_span failed: %s", exc)
            yield _NoOpSpan()

    # ------------------------------------------------------------------
    # Tool span
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def tool_span(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Nested span for a tool execution."""
        if not _MLFLOW_AVAILABLE:
            yield _NoOpSpan()
            return

        try:
            with mlflow.start_span(
                name=f"tool:{tool_name}",
                span_type="TOOL",
                attributes={"tool_name": tool_name},
            ) as span:
                span.set_inputs(args)
                yield span
        except Exception as exc:
            logger.debug("MLflow tool_span failed: %s", exc)
            yield _NoOpSpan()

    # ------------------------------------------------------------------
    # Inter-agent span
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def inter_agent_span(
        self,
        sender: str,
        recipient: str,
        msg_type: str,
    ) -> AsyncIterator[Any]:
        """Nested span for inter-agent communication."""
        if not _MLFLOW_AVAILABLE:
            yield _NoOpSpan()
            return

        try:
            with mlflow.start_span(
                name=f"inter_agent:{sender}→{recipient}",
                span_type="AGENT",
                attributes={
                    "sender": sender,
                    "recipient": recipient,
                    "msg_type": msg_type,
                },
            ) as span:
                yield span
        except Exception as exc:
            logger.debug("MLflow inter_agent_span failed: %s", exc)
            yield _NoOpSpan()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate_run(
        self,
        run_id: str,
        task: str,
        output: str,
        expected: str | None = None,
    ) -> dict[str, Any]:
        """
        Log evaluation metrics for a completed agent run.

        Always computes:
        - response_length: character count of the output.
        - step_count: taken from active run metrics if available.

        If ``expected`` is provided, additional quality metrics are logged.
        """
        if not _MLFLOW_AVAILABLE:
            return {}

        mlflow_run_id = self._active_run_ids.get(run_id)

        metrics: dict[str, float] = {
            "eval_response_length": float(len(output)),
            "eval_task_length": float(len(task)),
        }

        if expected:
            # Simple character-level overlap as a proxy for quality
            overlap = len(set(output.lower().split()) & set(expected.lower().split()))
            total = len(set(expected.lower().split())) or 1
            metrics["eval_word_overlap"] = float(overlap / total)

        try:
            if mlflow_run_id:
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_metrics(metrics)
            else:
                # No active run — log to a new evaluation run
                with mlflow.start_run(run_name=f"eval_{run_id[:8]}"):
                    mlflow.log_metrics(metrics)
                    mlflow.log_params({"task": task[:256], "run_id": run_id})
        except Exception as exc:
            logger.warning("MLflow evaluate_run failed: %s", exc)

        return metrics

    # ------------------------------------------------------------------
    # Trace retrieval
    # ------------------------------------------------------------------

    def get_run_trace(self, run_id: str) -> dict[str, Any]:
        """
        Retrieve MLflow run data for the given harness run_id.

        Returns an empty dict if the run is not found or MLflow unavailable.
        """
        if not _MLFLOW_AVAILABLE:
            return {}

        mlflow_run_id = self._active_run_ids.get(run_id)
        if not mlflow_run_id:
            return {}

        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(mlflow_run_id)
            return {
                "mlflow_run_id": mlflow_run_id,
                "status": run.info.status,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }
        except Exception as exc:
            logger.warning("get_run_trace failed: %s", exc)
            return {}
