"""Unit tests for OnlineLearningMonitor — rolling metrics and regression detection."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from harness.improvement.online_monitor import (
    OnlineLearningMonitor,
    PendingRollbackCheck,
    VersionMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monitor(redis_client, mlflow_tracer=None, window_size=10,
             regression_threshold=0.30, min_samples=3, snapshot_interval=5):
    return OnlineLearningMonitor(
        redis=redis_client,
        mlflow_tracer=mlflow_tracer,
        window_size=window_size,
        regression_threshold=regression_threshold,
        min_samples=min_samples,
        snapshot_interval=snapshot_interval,
    )


# ---------------------------------------------------------------------------
# record_run + windowed metrics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_run_stores_entry(redis_client):
    mon = _monitor(redis_client)
    vid = uuid.uuid4().hex
    await mon.record_run("sql", vid, 1, success=True, cost_usd=0.01, steps=5)

    vm = await mon.get_windowed_metrics("sql", vid)
    assert vm.sample_count == 1
    assert vm.success_rate == 1.0
    assert abs(vm.avg_cost_usd - 0.01) < 1e-6
    assert vm.avg_steps == 5.0


@pytest.mark.asyncio
async def test_windowed_metrics_empty(redis_client):
    mon = _monitor(redis_client)
    vm = await mon.get_windowed_metrics("sql", "nonexistent")
    assert vm.sample_count == 0
    assert vm.success_rate == 0.0
    assert vm.is_reliable is False


@pytest.mark.asyncio
async def test_success_rate_mixed_outcomes(redis_client):
    mon = _monitor(redis_client)
    vid = uuid.uuid4().hex
    for i in range(10):
        await mon.record_run("sql", vid, 1, success=(i % 2 == 0))

    vm = await mon.get_windowed_metrics("sql", vid)
    assert vm.sample_count == 10
    assert abs(vm.success_rate - 0.5) < 0.01


@pytest.mark.asyncio
async def test_window_caps_at_window_size(redis_client):
    mon = _monitor(redis_client, window_size=5)
    vid = uuid.uuid4().hex
    for _ in range(15):
        await mon.record_run("sql", vid, 1, success=True)

    vm = await mon.get_windowed_metrics("sql", vid)
    assert vm.sample_count == 5  # capped at window_size


@pytest.mark.asyncio
async def test_is_reliable_true_when_enough_samples(redis_client):
    mon = _monitor(redis_client, min_samples=3)
    vid = uuid.uuid4().hex
    for _ in range(3):
        await mon.record_run("sql", vid, 1, success=True)

    vm = await mon.get_windowed_metrics("sql", vid)
    assert vm.is_reliable is True


@pytest.mark.asyncio
async def test_is_reliable_false_when_too_few_samples(redis_client):
    mon = _monitor(redis_client, min_samples=10)
    vid = uuid.uuid4().hex
    await mon.record_run("sql", vid, 1, success=True)

    vm = await mon.get_windowed_metrics("sql", vid)
    assert vm.is_reliable is False


# ---------------------------------------------------------------------------
# Pending rollback check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schedule_and_pop_rollback_check(redis_client):
    mon = _monitor(redis_client)
    agent_type = "sql"
    patch_id = uuid.uuid4().hex

    await mon.schedule_rollback_check(
        agent_type=agent_type,
        patch_id=patch_id,
        baseline_version_id="v_old",
        new_version_id="v_new",
        baseline_error_count=5,
    )

    check = await mon.pop_pending_check(agent_type)
    assert check is not None
    assert check.patch_id == patch_id
    assert check.baseline_version_id == "v_old"
    assert check.baseline_error_count == 5

    # Second pop returns None (deleted after first)
    check2 = await mon.pop_pending_check(agent_type)
    assert check2 is None


@pytest.mark.asyncio
async def test_pop_returns_none_when_no_check(redis_client):
    mon = _monitor(redis_client)
    result = await mon.pop_pending_check("code")
    assert result is None


# ---------------------------------------------------------------------------
# PendingRollbackCheck serialisation
# ---------------------------------------------------------------------------

def test_pending_check_serialise_roundtrip():
    check = PendingRollbackCheck(
        agent_type="sql",
        patch_id="p123",
        baseline_version_id="v_old",
        new_version_id="v_new",
        baseline_error_count=8,
        applied_at=1_700_000_000.0,
    )
    raw = check.to_json()
    restored = PendingRollbackCheck.from_json(raw)
    assert restored.agent_type == "sql"
    assert restored.patch_id == "p123"
    assert restored.baseline_error_count == 8
    assert abs(restored.applied_at - 1_700_000_000.0) < 1.0


# ---------------------------------------------------------------------------
# Regression detection and rollback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_rollback_when_no_pending_check(redis_client):
    mon = _monitor(redis_client)
    collector = AsyncMock()
    prompt_manager = AsyncMock()

    rolled = await mon.check_and_maybe_rollback("sql", collector, prompt_manager)
    assert rolled is False
    prompt_manager.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_no_rollback_when_errors_stable(redis_client):
    mon = _monitor(redis_client, regression_threshold=0.30)
    await mon.schedule_rollback_check(
        agent_type="sql",
        patch_id="p1",
        baseline_version_id="v_old",
        new_version_id="v_new",
        baseline_error_count=10,
    )

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=11)  # only +10%, below 30% threshold
    prompt_manager = AsyncMock()

    rolled = await mon.check_and_maybe_rollback("sql", collector, prompt_manager)
    assert rolled is False
    prompt_manager.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_rollback_triggered_when_errors_spike(redis_client):
    mon = _monitor(redis_client, regression_threshold=0.30)
    await mon.schedule_rollback_check(
        agent_type="sql",
        patch_id="p2",
        baseline_version_id="v_old",
        new_version_id="v_new",
        baseline_error_count=10,
    )

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=20)  # +100%, above 30% threshold
    prompt_manager = AsyncMock()
    prompt_manager.rollback = AsyncMock()

    rolled = await mon.check_and_maybe_rollback("sql", collector, prompt_manager)
    assert rolled is True
    prompt_manager.rollback.assert_called_once_with("sql", steps=1)


@pytest.mark.asyncio
async def test_rollback_triggered_when_baseline_zero_and_errors_appear(redis_client):
    mon = _monitor(redis_client)
    await mon.schedule_rollback_check(
        agent_type="code",
        patch_id="p3",
        baseline_version_id="v_old",
        new_version_id="v_new",
        baseline_error_count=0,   # no errors before patch
    )

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=5)  # errors appeared after patch
    prompt_manager = AsyncMock()
    prompt_manager.rollback = AsyncMock()

    rolled = await mon.check_and_maybe_rollback("code", collector, prompt_manager)
    assert rolled is True
    prompt_manager.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_rollback_check_consumed_after_no_regression(redis_client):
    """After a clean check, the pending record is consumed — not checked again."""
    mon = _monitor(redis_client, regression_threshold=0.30)
    await mon.schedule_rollback_check(
        agent_type="sql",
        patch_id="p4",
        baseline_version_id="v_old",
        new_version_id="v_new",
        baseline_error_count=10,
    )

    collector = AsyncMock()
    collector.count = AsyncMock(return_value=10)
    prompt_manager = AsyncMock()

    # First call — check performed, no rollback
    rolled = await mon.check_and_maybe_rollback("sql", collector, prompt_manager)
    assert rolled is False

    # Second call — no pending check, so nothing happens
    collector.count = AsyncMock(return_value=50)  # errors spiked but check was already consumed
    rolled2 = await mon.check_and_maybe_rollback("sql", collector, prompt_manager)
    assert rolled2 is False


# ---------------------------------------------------------------------------
# MLflow snapshot on interval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mlflow_snapshot_called_on_interval(redis_client):
    mlflow_tracer = AsyncMock()
    mlflow_tracer.log_online_metrics = AsyncMock()

    mon = _monitor(redis_client, mlflow_tracer=mlflow_tracer, snapshot_interval=3)
    vid = uuid.uuid4().hex

    # Record 3 runs — should trigger snapshot on 3rd
    for i in range(3):
        await mon.record_run("sql", vid, 1, success=True)

    # MLflow should have been called at least once
    assert mlflow_tracer.log_online_metrics.called


@pytest.mark.asyncio
async def test_no_mlflow_call_when_tracer_is_none(redis_client):
    """Should not raise even without an MLflow tracer."""
    mon = _monitor(redis_client, mlflow_tracer=None, snapshot_interval=1)
    vid = uuid.uuid4().hex
    # Should not raise
    await mon.record_run("sql", vid, 1, success=True)


# ---------------------------------------------------------------------------
# Version metrics isolation between agent types
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_isolated_per_agent_type(redis_client):
    mon = _monitor(redis_client)
    vid = uuid.uuid4().hex

    for _ in range(5):
        await mon.record_run("sql", vid, 1, success=True)
    for _ in range(5):
        await mon.record_run("code", vid, 1, success=False)

    sql_vm = await mon.get_windowed_metrics("sql", vid)
    code_vm = await mon.get_windowed_metrics("code", vid)

    assert sql_vm.success_rate == 1.0
    assert code_vm.success_rate == 0.0
