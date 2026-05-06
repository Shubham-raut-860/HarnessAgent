"""Memory management API routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field

from harness.api.deps import get_current_tenant, get_memory_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RememberRequest(BaseModel):
    """Request body for storing a memory entry."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Users table has columns: id, email, created_at",
                "metadata": {"source": "schema_docs"},
            }
        }
    )

    text: str = Field(..., description="Text to store", min_length=1)
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class RememberResponse(BaseModel):
    """Response for a remember operation."""

    id: str


class RecallRequest(BaseModel):
    """Request body for retrieving memory entries."""

    query: str = Field(..., description="Search query", min_length=1)
    k: int = Field(default=5, ge=1, le=50, description="Number of results")
    filter: dict = Field(default_factory=dict, description="Metadata filter")


class MemoryEntryResponse(BaseModel):
    """A single retrieved memory entry."""

    id: str
    text: str
    metadata: dict
    score: float
    source: str = "long"


class GraphQueryResponse(BaseModel):
    """Response for graph traversal queries."""

    paths: list[dict]
    context: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/remember", response_model=RememberResponse, status_code=status.HTTP_201_CREATED)
async def remember(
    body: RememberRequest,
    tenant_id: str = Depends(get_current_tenant),
    memory: Any = Depends(get_memory_manager),
) -> RememberResponse:
    """Store a text entry in long-term memory.

    Args:
        body:      The text and metadata to store.
        tenant_id: From JWT.
        memory:    MemoryManager.

    Returns:
        201 with the ID of the stored entry.
    """
    try:
        result = await memory.remember(
            text=body.text,
            metadata={**body.metadata, "tenant_id": tenant_id},
            tenant_id=tenant_id,
        )
        # result may be an ID string or an object with .id
        if isinstance(result, str):
            entry_id = result
        elif hasattr(result, "id"):
            entry_id = result.id
        else:
            import uuid
            entry_id = uuid.uuid4().hex
        return RememberResponse(id=entry_id)
    except Exception as exc:
        logger.error("remember failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {exc}",
        ) from exc


@router.post("/recall", response_model=list[MemoryEntryResponse])
async def recall(
    body: RecallRequest,
    tenant_id: str = Depends(get_current_tenant),
    memory: Any = Depends(get_memory_manager),
) -> list[MemoryEntryResponse]:
    """Retrieve relevant memories via semantic search.

    Args:
        body:      Query, k, and optional filter.
        tenant_id: From JWT.
        memory:    MemoryManager.

    Returns:
        List of MemoryEntry objects ordered by relevance.
    """
    try:
        filter_with_tenant = {**body.filter, "tenant_id": tenant_id}
        entries = await memory.recall(
            query=body.query,
            k=body.k,
            filter=filter_with_tenant,
            tenant_id=tenant_id,
        )
        return [
            MemoryEntryResponse(
                id=getattr(e, "id", ""),
                text=getattr(e, "text", ""),
                metadata=getattr(e, "metadata", {}),
                score=getattr(e, "score", 0.0),
                source=getattr(e, "source", "long"),
            )
            for e in (entries or [])
        ]
    except Exception as exc:
        logger.error("recall failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recall memories: {exc}",
        ) from exc


@router.get("/graph", response_model=GraphQueryResponse)
async def query_graph(
    query: str = Query(..., description="Search query for graph traversal"),
    max_hops: int = Query(default=2, ge=1, le=5, description="Maximum graph hops"),
    tenant_id: str = Depends(get_current_tenant),
    memory: Any = Depends(get_memory_manager),
) -> GraphQueryResponse:
    """Retrieve knowledge graph context for a query.

    Performs entity extraction on the query, traverses the graph, and
    returns the discovered paths and rendered context string.

    Args:
        query:     Natural language query.
        max_hops:  Maximum graph traversal depth.
        tenant_id: From JWT.
        memory:    MemoryManager.

    Returns:
        Graph paths and rendered context.
    """
    try:
        # Try graph_rag retrieval if available
        if hasattr(memory, "retrieve_graph"):
            result = await memory.retrieve_graph(
                query=query,
                max_hops=max_hops,
                tenant_id=tenant_id,
            )
            paths = []
            for path in getattr(result, "graph_paths", []):
                paths.append({
                    "nodes": [
                        {"id": n.id, "type": n.type, "props": n.props}
                        for n in getattr(path, "nodes", [])
                    ],
                    "edges": [
                        {"source": e.source_id, "target": e.target_id, "type": e.type}
                        for e in getattr(path, "edges", [])
                    ],
                })
            return GraphQueryResponse(
                paths=paths,
                context=getattr(result, "graph_context", ""),
            )

        # Fallback: use graph_rag module directly
        return GraphQueryResponse(paths=[], context="Graph RAG not configured")

    except Exception as exc:
        logger.error("graph query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph query failed: {exc}",
        ) from exc


@router.delete(
    "/session/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
async def clear_session(
    run_id: str,
    tenant_id: str = Depends(get_current_tenant),
    memory: Any = Depends(get_memory_manager),
) -> None:
    """Clear short-term (session) memory for a specific run.

    Args:
        run_id:    The run whose session memory should be cleared.
        tenant_id: From JWT.
        memory:    MemoryManager.

    Returns:
        204 on success.
    """
    try:
        if hasattr(memory, "clear_session"):
            result = memory.clear_session(run_id=run_id, tenant_id=tenant_id)
            if hasattr(result, "__await__"):
                await result
        elif hasattr(memory, "short_term") and hasattr(memory.short_term, "clear"):
            result = memory.short_term.clear(run_id)
            if hasattr(result, "__await__"):
                await result
        else:
            logger.debug("Memory manager has no clear_session method for run %s", run_id)
    except Exception as exc:
        logger.error("clear_session failed for run %s: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {exc}",
        ) from exc
