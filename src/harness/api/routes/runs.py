"""Agent run management API routes."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from harness.api.deps import get_current_tenant, get_redis
from harness.orchestrator.runner import AgentRunner, RunRecord

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    """Request body for creating a new agent run."""

    agent_type: str = Field(..., description="Agent type: sql, code, research, orchestrator")
    task: str = Field(..., description="The task for the agent to execute", min_length=1)
    metadata: dict = Field(default_factory=dict, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_type": "sql",
                "task": "List all tables in the database",
                "metadata": {"priority": "high"},
            }
        }


class RunRecordResponse(BaseModel):
    """API response model for a RunRecord."""

    run_id: str
    tenant_id: str
    agent_type: str
    task: str
    status: str
    result: Optional[dict] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    hitl_pending: bool = False
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_record(cls, record: RunRecord) -> "RunRecordResponse":
        return cls(
            run_id=record.run_id,
            tenant_id=record.tenant_id,
            agent_type=record.agent_type,
            task=record.task,
            status=record.status,
            result=record.result,
            created_at=record.created_at.isoformat(),
            started_at=record.started_at.isoformat() if record.started_at else None,
            completed_at=record.completed_at.isoformat() if record.completed_at else None,
            hitl_pending=record.hitl_pending,
            metadata=record.metadata,
        )


_ALLOWED_AGENT_TYPES = {"sql", "code", "research", "orchestrator"}


def _get_runner(redis: Any) -> AgentRunner:
    """Build a minimal AgentRunner from the Redis client.

    In production this would be injected from app.state.  For now it is
    reconstructed per-request using the Redis connection.
    """
    # Lazy import to avoid circular deps
    from harness.orchestrator.runner import AgentRunner

    def _noop_factory(agent_type: str):
        raise NotImplementedError(
            f"Agent factory not configured for type: {agent_type}"
        )

    return AgentRunner(redis=redis, agent_factory=_noop_factory)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED, response_model=RunRecordResponse)
async def create_run(
    body: CreateRunRequest,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> RunRecordResponse:
    """Create a new agent run and enqueue it for a worker.

    Args:
        body:      Run creation payload.
        tenant_id: Extracted from JWT.
        redis:     Redis client from app state.

    Returns:
        201 with the created RunRecord.

    Raises:
        400 if agent_type is not recognised.
    """
    if body.agent_type not in _ALLOWED_AGENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent_type '{body.agent_type}'. Allowed: {sorted(_ALLOWED_AGENT_TYPES)}",
        )

    runner = _get_runner(redis)
    record = await runner.create_run(
        tenant_id=tenant_id,
        agent_type=body.agent_type,
        task=body.task,
        metadata=body.metadata,
    )

    # Enqueue in Redis for the worker to pick up
    try:
        queue_key = f"harness:queue:{body.agent_type}"
        await redis.rpush(queue_key, record.run_id)
        logger.info("Enqueued run %s on queue %s", record.run_id, queue_key)
    except Exception as exc:
        logger.warning("Failed to enqueue run %s: %s", record.run_id, exc)

    return RunRecordResponse.from_record(record)


@router.get("/{run_id}", response_model=RunRecordResponse)
async def get_run(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> RunRecordResponse:
    """Retrieve a run by ID.

    Args:
        run_id:    The run identifier.
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        200 with RunRecord.

    Raises:
        404 if run not found.
        403 if run belongs to a different tenant.
    """
    runner = _get_runner(redis)
    record = await runner.get_run(run_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )
    if record.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return RunRecordResponse.from_record(record)


@router.get("", response_model=list[RunRecordResponse])
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> list[RunRecordResponse]:
    """List runs for the authenticated tenant.

    Args:
        limit:     Maximum number of runs to return (1-100).
        offset:    Pagination offset.
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        200 with list of RunRecords, newest first.
    """
    runner = _get_runner(redis)
    records = await runner.list_runs(
        tenant_id=tenant_id, limit=limit, offset=offset
    )
    return [RunRecordResponse.from_record(r) for r in records]


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis: Any = Depends(get_redis),
) -> None:
    """Cancel a run that is pending or running.

    Args:
        run_id:    The run to cancel.
        tenant_id: Extracted from JWT.
        redis:     Redis client.

    Returns:
        204 on success.

    Raises:
        404 if run not found.
        403 if run belongs to another tenant.
    """
    runner = _get_runner(redis)
    record = await runner.get_run(run_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )
    if record.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    await runner.cancel_run(run_id)
