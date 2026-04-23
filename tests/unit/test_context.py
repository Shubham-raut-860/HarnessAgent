"""Unit tests for AgentContext."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.core.context import AgentContext
from harness.core.errors import BudgetExceeded, FailureClass


@pytest.fixture
def ctx(tmp_path):
    """Create a fresh AgentContext for testing."""
    return AgentContext(
        run_id="test-run-001",
        tenant_id="test-tenant",
        agent_type="sql",
        task="SELECT * FROM users",
        memory=MagicMock(),
        workspace_path=tmp_path / "ws",
        max_steps=5,
        max_tokens=1000,
        timeout_seconds=10.0,
    )


def test_budget_ok_on_fresh_context(ctx):
    """A newly created context should report budget as OK."""
    assert ctx.is_budget_ok() is True


def test_budget_exceeded_steps(ctx):
    """Calling tick() beyond max_steps should raise BudgetExceeded."""
    # Tick up to the limit
    for _ in range(5):
        ctx.tick(tokens=10)

    with pytest.raises(BudgetExceeded) as exc_info:
        ctx.tick(tokens=10)

    assert exc_info.value.failure_class == FailureClass.BUDGET_STEPS
    assert ctx.failed is True
    assert ctx.failure_class == FailureClass.BUDGET_STEPS.value


def test_budget_exceeded_tokens(ctx):
    """Accumulating tokens beyond max_tokens should raise BudgetExceeded."""
    with pytest.raises(BudgetExceeded) as exc_info:
        ctx.tick(tokens=1001)

    assert exc_info.value.failure_class == FailureClass.BUDGET_TOKENS
    assert ctx.failed is True


def test_budget_exceeded_time(ctx, tmp_path):
    """Elapsed time beyond timeout_seconds should raise BudgetExceeded."""
    past = datetime.now(timezone.utc) - timedelta(seconds=20)
    ctx.started_at = past

    # Step count is fine, token count is fine, but time exceeded
    assert ctx.elapsed_seconds > ctx.timeout_seconds

    with pytest.raises(BudgetExceeded) as exc_info:
        ctx.tick(tokens=0)

    assert exc_info.value.failure_class == FailureClass.BUDGET_TIME


def test_tick_increments_counts(ctx):
    """tick() should increment step_count and accumulate token_count."""
    assert ctx.step_count == 0
    assert ctx.token_count == 0

    ctx.tick(tokens=100)
    assert ctx.step_count == 1
    assert ctx.token_count == 100

    ctx.tick(tokens=50)
    assert ctx.step_count == 2
    assert ctx.token_count == 150


def test_elapsed_seconds(ctx):
    """elapsed_seconds should return a non-negative float."""
    elapsed = ctx.elapsed_seconds
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_is_budget_ok_step_limit(ctx):
    """is_budget_ok() should return False when step limit is reached."""
    ctx.step_count = ctx.max_steps
    assert ctx.is_budget_ok() is False


def test_is_budget_ok_token_limit(ctx):
    """is_budget_ok() should return False when token limit is reached."""
    ctx.token_count = ctx.max_tokens
    assert ctx.is_budget_ok() is False


def test_is_budget_ok_time_limit(ctx):
    """is_budget_ok() should return False when time limit is exceeded."""
    ctx.started_at = datetime.now(timezone.utc) - timedelta(seconds=100)
    assert ctx.is_budget_ok() is False


def test_context_create_factory(tmp_path):
    """AgentContext.create() should auto-generate run_id and trace_id."""
    ctx = AgentContext.create(
        tenant_id="tenant",
        agent_type="code",
        task="Write a sort function",
        memory=MagicMock(),
        workspace_path=tmp_path / "ws",
    )
    assert len(ctx.run_id) == 32  # uuid hex
    assert len(ctx.trace_id) == 32
    assert ctx.failed is False
    assert ctx.failure_class is None
