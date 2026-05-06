"""Unit tests for MLflowAgentTracer — new monitoring methods."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.observability.mlflow_tracer import MLflowAgentTracer

# Use a real file-backed MLflow store so __init__ doesn't fail.
# All actual MLflow calls in individual tests are mocked out.
_TMP_DIR = tempfile.mkdtemp(prefix="harness_mlflow_test_")


def _tracer() -> MLflowAgentTracer:
    return MLflowAgentTracer(
        tracking_uri=f"file://{_TMP_DIR}",
        experiment_name="test_experiment",
    )


# ---------------------------------------------------------------------------
# log_llm_call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_llm_call_noop_when_no_active_run():
    """log_llm_call silently does nothing when no run is active for run_id."""
    tracer = _tracer()
    # No active run — should not raise
    await tracer.log_llm_call(
        run_id="unknown_run",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=200,
    )


@pytest.mark.asyncio
async def test_log_llm_call_calls_mlflow_client_when_run_active():
    tracer = _tracer()
    tracer._active_run_ids["run1"] = "mlflow_run_abc"

    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", True), \
         patch("harness.observability.mlflow_tracer.mlflow") as mock_mlflow:
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        await tracer.log_llm_call(
            run_id="run1",
            model="claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=1000,
        )

        assert mock_client.log_metric.called
        calls = [c[0][1] for c in mock_client.log_metric.call_args_list]
        assert "llm_input_tokens" in calls
        assert "llm_output_tokens" in calls
        assert "llm_cost_usd" in calls


@pytest.mark.asyncio
async def test_log_llm_call_noop_when_mlflow_unavailable():
    tracer = _tracer()
    tracer._active_run_ids["run1"] = "mlflow_run_abc"

    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", False):
        # Must not raise
        await tracer.log_llm_call(
            run_id="run1",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )


# ---------------------------------------------------------------------------
# log_hermes_cycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_hermes_cycle_noop_when_mlflow_unavailable():
    tracer = _tracer()
    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", False):
        await tracer.log_hermes_cycle(
            agent_type="sql",
            patch_id="patch123",
            score=0.85,
            applied=True,
            errors_sampled=10,
            reason="Test reason",
        )


@pytest.mark.asyncio
async def test_log_hermes_cycle_logs_to_experiment():
    tracer = _tracer()

    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", True), \
         patch("harness.observability.mlflow_tracer.mlflow") as mock_mlflow:
        mock_run_ctx = MagicMock()
        mock_run_ctx.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_run_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run_ctx

        await tracer.log_hermes_cycle(
            agent_type="sql",
            patch_id="abc123",
            score=0.9,
            applied=True,
            errors_sampled=15,
            reason="Score above threshold",
            eval_successes=8,
            eval_total=10,
            rolled_back=False,
        )

        mock_mlflow.set_experiment.assert_called_with("harness_hermes_sql")
        assert mock_mlflow.log_params.called
        assert mock_mlflow.log_metrics.called

        metric_call = mock_mlflow.log_metrics.call_args[0][0]
        assert abs(metric_call["patch_score"] - 0.9) < 1e-6
        assert abs(metric_call["eval_success_rate"] - 0.8) < 1e-6


@pytest.mark.asyncio
async def test_log_hermes_cycle_with_rollback_flag():
    tracer = _tracer()

    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", True), \
         patch("harness.observability.mlflow_tracer.mlflow") as mock_mlflow:
        mock_run_ctx = MagicMock()
        mock_run_ctx.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_run_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run_ctx

        await tracer.log_hermes_cycle(
            agent_type="code",
            patch_id="def456",
            score=0.2,
            applied=False,
            errors_sampled=5,
            reason="Regression detected",
            rolled_back=True,
        )

        param_call = mock_mlflow.log_params.call_args[0][0]
        assert param_call["rolled_back"] == "True"
        assert param_call["applied"] == "False"


# ---------------------------------------------------------------------------
# log_online_metrics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_online_metrics_noop_when_mlflow_unavailable():
    tracer = _tracer()
    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", False):
        await tracer.log_online_metrics(
            agent_type="sql",
            version_id="v123",
            version_number=3,
            success_rate=0.9,
            avg_cost=0.005,
            avg_steps=4.2,
            sample_count=50,
        )


@pytest.mark.asyncio
async def test_log_online_metrics_logs_to_correct_experiment():
    tracer = _tracer()

    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", True), \
         patch("harness.observability.mlflow_tracer.mlflow") as mock_mlflow:
        mock_run_ctx = MagicMock()
        mock_run_ctx.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_run_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run_ctx

        await tracer.log_online_metrics(
            agent_type="code",
            version_id="ver_abc",
            version_number=5,
            success_rate=0.88,
            avg_cost=0.012,
            avg_steps=7.1,
            sample_count=100,
        )

        mock_mlflow.set_experiment.assert_called_with("harness_online_code")
        metric_call = mock_mlflow.log_metrics.call_args[0][0]
        assert abs(metric_call["success_rate"] - 0.88) < 1e-6
        assert metric_call["sample_count"] == 100.0


# ---------------------------------------------------------------------------
# log_prompt_version
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_log_prompt_version_noop_when_mlflow_unavailable():
    tracer = _tracer()
    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", False):
        await tracer.log_prompt_version(
            agent_type="sql",
            version_id="v1",
            version_number=1,
            created_by="human",
        )


@pytest.mark.asyncio
async def test_log_prompt_version_logs_to_correct_experiment():
    tracer = _tracer()

    with patch("harness.observability.mlflow_tracer._MLFLOW_AVAILABLE", True), \
         patch("harness.observability.mlflow_tracer.mlflow") as mock_mlflow:
        mock_run_ctx = MagicMock()
        mock_run_ctx.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_run_ctx.__exit__ = MagicMock(return_value=False)
        mock_mlflow.start_run.return_value = mock_run_ctx

        await tracer.log_prompt_version(
            agent_type="research",
            version_id="v_xyz",
            version_number=2,
            created_by="hermes",
            patch_id="patch_abc",
        )

        mock_mlflow.set_experiment.assert_called_with("harness_prompts_research")
        param_call = mock_mlflow.log_params.call_args[0][0]
        assert param_call["created_by"] == "hermes"
        assert param_call["patch_id"] == "patch_abc"
        assert param_call["version_number"] == "2"
