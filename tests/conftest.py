"""Comprehensive pytest fixtures for HarnessAgent test suite."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Event loop (session-scoped for async tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default asyncio event loop policy (pytest-asyncio 0.24+ style)."""
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Redis (fakeredis)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def redis_client():
    """Provide a fakeredis async client (function scope — clean state per test)."""
    try:
        import fakeredis.aioredis as fake_aioredis  # type: ignore

        client = fake_aioredis.FakeRedis(decode_responses=True)
        yield client
        await client.aclose()
    except ImportError:
        pytest.skip("fakeredis not installed")


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider():
    """Async mock implementing the LLMProvider protocol.

    complete() returns a standard LLMResponse.
    health_check() returns True.
    stream() yields a single delta.
    """
    from harness.core.context import LLMResponse

    provider = AsyncMock()
    provider.provider_name = "mock"
    provider.model = "mock-model"

    default_response = LLMResponse(
        content="Test response",
        tool_calls=[],
        input_tokens=10,
        output_tokens=20,
        model="mock-model",
        provider="mock",
        cached=False,
    )

    provider.complete = AsyncMock(return_value=default_response)
    provider.health_check = AsyncMock(return_value=True)

    async def _stream(*args, **kwargs):
        yield "Test "
        yield "response"

    provider.stream = _stream
    return provider


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    """Async mock returning fixed-dimension embeddings."""
    DIMS = 384  # matches all-MiniLM-L6-v2

    embedder = AsyncMock()
    embedder.model = "mock-embedder"
    embedder.dimensions = DIMS

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * DIMS for _ in texts]

    embedder.embed = _embed
    return embedder


# ---------------------------------------------------------------------------
# AgentContext factory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_context(tmp_workspace):
    """Factory fixture that creates AgentContext instances."""

    def _factory(
        run_id: str = None,
        tenant_id: str = "test-tenant",
        agent_type: str = "sql",
        task: str = "Test task",
        memory=None,
    ):
        from harness.core.context import AgentContext

        return AgentContext(
            run_id=run_id or uuid.uuid4().hex,
            tenant_id=tenant_id,
            agent_type=agent_type,
            task=task,
            memory=memory or MagicMock(),
            workspace_path=tmp_workspace,
            max_steps=10,
            max_tokens=10_000,
            timeout_seconds=60.0,
        )

    return _factory


# ---------------------------------------------------------------------------
# Temp workspace
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory with the standard structure."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "artifacts").mkdir(exist_ok=True)
    (ws / "logs").mkdir(exist_ok=True)
    return ws


# ---------------------------------------------------------------------------
# Mock vector store
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store():
    """AsyncMock implementing the VectorStore protocol."""
    from harness.core.protocols import VectorHit

    store = AsyncMock()

    async def _query(text: str, k: int = 5, filter=None, hybrid_alpha=None):
        return [
            VectorHit(id=f"hit-{i}", text=f"Result {i}", score=0.9 - i * 0.1)
            for i in range(min(k, 3))
        ]

    store.upsert = AsyncMock(return_value=None)
    store.query = _query
    store.delete = AsyncMock(return_value=None)
    store.count = AsyncMock(return_value=0)
    return store


# ---------------------------------------------------------------------------
# Mock graph store
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph_store():
    """AsyncMock implementing the GraphStore protocol."""
    from harness.core.protocols import GraphNode, GraphPath

    store = AsyncMock()

    async def _traverse(start_ids: list[str], max_hops: int = 2):
        return []

    async def _find_nodes(names: list[str], fuzzy: bool = True):
        return [
            GraphNode(id=name.lower().replace(" ", "_"), type="entity", props={"name": name})
            for name in names[:2]
        ]

    store.add_node = AsyncMock(return_value=None)
    store.add_edge = AsyncMock(return_value=None)
    store.traverse = _traverse
    store.find_nodes = _find_nodes
    return store


# ---------------------------------------------------------------------------
# Mock memory manager
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_manager(mock_vector_store, mock_graph_store):
    """Mock MemoryManager with mocked backends."""
    from harness.memory.schemas import MemoryEntry

    mm = AsyncMock()
    mm.vector_store = mock_vector_store
    mm.graph_store = mock_graph_store

    async def _remember(text: str, metadata: dict = None, tenant_id: str = ""):
        return uuid.uuid4().hex

    async def _recall(query: str, k: int = 5, filter: dict = None, tenant_id: str = ""):
        return [
            MemoryEntry(
                id=f"mem-{i}",
                text=f"Memory result {i} for: {query[:30]}",
                metadata={},
                score=0.9 - i * 0.1,
            )
            for i in range(min(k, 3))
        ]

    async def _add_fact(subject: str, predicate: str, object_: str, tenant_id: str = ""):
        pass

    mm.remember = _remember
    mm.recall = _recall
    mm.add_fact = _add_fact
    return mm


# ---------------------------------------------------------------------------
# Mock tool registry
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tool_registry():
    """ToolRegistry with no-op tools registered for standard tool names."""
    from harness.core.context import ToolResult
    from harness.tools.registry import ToolRegistry

    registry = ToolRegistry()

    class NoOpTool:
        name = "noop"
        description = "No-operation test tool"
        input_schema = {"type": "object", "properties": {}, "required": []}
        timeout_seconds = 5.0

        async def execute(self, ctx, args):
            return ToolResult(data={"status": "ok"})

    class EchoTool:
        name = "echo"
        description = "Echoes the input message"
        input_schema = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }
        timeout_seconds = 5.0

        async def execute(self, ctx, args):
            return ToolResult(data={"echo": args.get("message", "")})

    registry.register(NoOpTool())
    registry.register(EchoTool())
    return registry
