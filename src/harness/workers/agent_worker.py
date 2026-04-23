"""RQ-based agent worker for processing run jobs from the queue."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async job handler
# ---------------------------------------------------------------------------


async def process_run_job_async(run_id: str, config_dict: Optional[dict] = None) -> None:
    """Process a single agent run job asynchronously.

    Initialises all required dependencies (Redis, memory manager, LLM router),
    delegates to AgentRunner.execute_run(), and updates the run record on
    completion or failure.

    Args:
        run_id:      The run identifier to execute.
        config_dict: Optional configuration overrides for this run.
    """
    config_dict = config_dict or {}

    # --- Import harness dependencies ---
    import redis.asyncio as aioredis
    from harness.core.config import get_config

    cfg = get_config()

    # Initialise Redis
    redis_client = aioredis.from_url(
        config_dict.get("redis_url", cfg.redis_url),
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=10,
    )

    try:
        await redis_client.ping()
        logger.info("Worker: Redis connected for run %s", run_id)
    except Exception as exc:
        logger.error("Worker: Redis connection failed: %s", exc)
        raise

    # Build a minimal agent factory
    def agent_factory(agent_type: str) -> Any:
        """Create the appropriate agent for the given type."""
        # Lazy imports to keep startup fast
        try:
            if agent_type == "sql":
                from harness.agents.sql_agent import SQLAgent
                return SQLAgent(config=config_dict)
            elif agent_type == "code":
                from harness.agents.code_agent import CodeAgent
                return CodeAgent(config=config_dict)
            else:
                raise ValueError(f"Unknown agent_type: {agent_type!r}")
        except ImportError as exc:
            logger.warning("Agent module not found for %s: %s", agent_type, exc)
            raise RuntimeError(f"Agent '{agent_type}' is not available: {exc}") from exc

    # Initialise error collector for failure recording
    from harness.improvement.error_collector import ErrorCollector
    error_collector = ErrorCollector(redis=redis_client)

    # Build the AgentRunner
    from harness.orchestrator.runner import AgentRunner
    runner = AgentRunner(
        redis=redis_client,
        agent_factory=agent_factory,
        workspace_base=config_dict.get("workspace_base_path", cfg.workspace_base_path),
        error_collector=error_collector,
    )

    # Execute the run
    try:
        record = await runner.execute_run(run_id)
        logger.info(
            "Worker: run %s completed with status=%s",
            run_id,
            record.status,
        )
    except KeyError as exc:
        logger.error("Worker: run %s not found: %s", run_id, exc)
        raise
    except Exception as exc:
        logger.exception("Worker: run %s raised unhandled exception: %s", run_id, exc)

        # Attempt to mark as failed in Redis
        try:
            from harness.orchestrator.runner import RunRecord, _run_key
            import json
            from datetime import datetime, timezone

            raw = await redis_client.get(_run_key(run_id))
            if raw:
                rec = RunRecord.from_json(raw if isinstance(raw, str) else raw.decode())
                rec.status = "failed"
                rec.completed_at = datetime.now(timezone.utc)
                rec.result = {
                    "run_id": run_id,
                    "output": "",
                    "steps": 0,
                    "tokens": 0,
                    "success": False,
                    "error_message": str(exc),
                }
                await redis_client.set(_run_key(run_id), rec.to_json())
        except Exception as update_exc:
            logger.warning("Worker: failed to update run status: %s", update_exc)

        raise
    finally:
        await redis_client.aclose()


# ---------------------------------------------------------------------------
# Synchronous job entry point (called by RQ)
# ---------------------------------------------------------------------------


def process_run_job(run_id: str, config_dict: Optional[dict] = None) -> None:
    """Synchronous wrapper called by RQ workers.

    RQ enqueues calls to this function. It bridges RQ's synchronous
    interface to the async implementation.

    Args:
        run_id:      The run identifier to execute.
        config_dict: Optional configuration overrides.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("RQ job: processing run %s", run_id)
    asyncio.run(process_run_job_async(run_id, config_dict or {}))


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def start_worker(
    queues: Optional[list[str]] = None,
    redis_url: Optional[str] = None,
) -> None:
    """Start an RQ worker listening on the specified queues.

    Args:
        queues:    Queue names to listen on.  Defaults to ["default", "agent"].
        redis_url: Redis URL.  Falls back to config.
    """
    import redis as sync_redis  # type: ignore
    from rq import Worker, Queue  # type: ignore
    from harness.core.config import get_config

    cfg = get_config()
    effective_redis_url = redis_url or cfg.redis_url
    effective_queues = queues or ["default", "agent", "sql", "code", "research"]

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    conn = sync_redis.from_url(effective_redis_url, decode_responses=False)
    queue_objects = [Queue(q, connection=conn) for q in effective_queues]

    logger.info(
        "Starting RQ worker on queues: %s (redis=%s)",
        effective_queues,
        effective_redis_url,
    )

    worker = Worker(queue_objects, connection=conn)
    worker.work(with_scheduler=True)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    start_worker()
