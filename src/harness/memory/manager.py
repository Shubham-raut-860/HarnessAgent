"""MemoryManager: unified interface to all memory tiers."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from harness.core.protocols import EmbeddingProvider, GraphPath, VectorStore
from harness.memory.context_manager import ContextWindowManager
from harness.memory.graph import get_graph_memory
from harness.memory.graph_rag import GraphRAGEngine
from harness.memory.schemas import (
    ConversationMessage,
    ContextWindow,
    MemoryEntry,
    RetrievalResult,
)
from harness.memory.short_term import ShortTermMemory
from harness.memory.vector_factory import build_embedding_provider, build_vector_store

if TYPE_CHECKING:
    from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

# PII redaction patterns
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),                    # SSN
    (re.compile(r"\b\d{3}[.\-\s]?\d{3}[.\-\s]?\d{4}\b"), "[PHONE]"),   # US phone
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),  # email
    (re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"), "[CC]"),               # Visa card
    (re.compile(r"\b5[1-5][0-9]{14}\b"), "[CC]"),                       # Mastercard
]


def _redact_pii(text: str) -> str:
    """Apply basic PII masking to text before long-term storage."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class MemoryManager:
    """
    Unified memory interface providing access to:
    - Short-term conversation history (Redis)
    - Long-term vector memory (Chroma / Qdrant / Weaviate)
    - Knowledge graph (NetworkX / Neo4j)
    - GraphRAG smart retrieval
    - Context window management
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        vector_store: VectorStore,
        graph: Any,
        embedder: EmbeddingProvider,
        context_manager: ContextWindowManager,
    ) -> None:
        self._short_term = short_term
        self._vector_store = vector_store
        self._graph = graph
        self._embedder = embedder
        self._context_manager = context_manager
        self._graph_rag = GraphRAGEngine(graph, vector_store, embedder)

    # ------------------------------------------------------------------
    # Conversation (short-term)
    # ------------------------------------------------------------------

    async def push_message(
        self,
        run_id: str,
        role: str,
        content: str,
        tokens: int = 0,
    ) -> None:
        """Append a new message to the conversation history for ``run_id``."""
        await self._short_term.push_message(run_id, role, content, tokens)

    async def get_history(
        self,
        run_id: str,
        last_n: int = 20,
    ) -> list[ConversationMessage]:
        """Return the most recent ``last_n`` messages (chronological order)."""
        return await self._short_term.get_history(run_id, last_n=last_n)

    async def fit_history(
        self,
        run_id: str,
        max_tokens: int,
        **kwargs: Any,
    ) -> ContextWindow:
        """Retrieve history and fit it into the given token budget."""
        messages = await self._short_term.get_history(run_id)
        return await self._context_manager.fit(
            messages=messages,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Long-term (vector)
    # ------------------------------------------------------------------

    async def remember(
        self,
        text: str,
        metadata: dict[str, Any],
        tenant_id: str | None = None,
    ) -> str:
        """
        Store text in the long-term vector store after PII redaction.

        Returns the generated document ID.
        """
        clean_text = _redact_pii(text)
        doc_id = uuid.uuid4().hex

        if tenant_id:
            metadata = {**metadata, "tenant_id": tenant_id}

        embeddings = await self._embedder.embed([clean_text])
        await self._vector_store.upsert(
            id=doc_id,
            text=clean_text,
            metadata=metadata,
            embedding=embeddings[0],
        )
        return doc_id

    async def recall(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve the top-k most semantically similar memories."""
        hits = await self._vector_store.query(text=query, k=k, filter=filter)
        return [
            MemoryEntry(
                id=h.id,
                text=h.text,
                metadata=h.metadata,
                score=h.score,
                created_at=datetime.now(timezone.utc),
                source="long",
            )
            for h in hits
        ]

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a subject-predicate-object triple to the knowledge graph.

        Creates nodes for subject and object if they don't exist, then adds
        a directed edge with the predicate as the edge type.
        """
        props = metadata or {}
        await self._graph.add_node(id=subject, type="Entity", props={"name": subject})
        await self._graph.add_node(id=object_, type="Entity", props={"name": object_})
        await self._graph.add_edge(
            src=subject,
            tgt=object_,
            type=predicate.replace(" ", "_").upper(),
            props={**props, "weight": weight},
        )

    async def graph_query(
        self,
        start_ids: list[str],
        max_hops: int = 2,
    ) -> list[GraphPath]:
        """Traverse the knowledge graph from the given start node IDs."""
        return await self._graph.traverse(start_ids=start_ids, max_hops=max_hops)

    # ------------------------------------------------------------------
    # Smart retrieval
    # ------------------------------------------------------------------

    async def smart_retrieve(
        self,
        query: str,
        ctx: "AgentContext",
    ) -> RetrievalResult:
        """
        Graph-first retrieval with vector fallback.

        Delegates to GraphRAGEngine which handles entity extraction,
        graph traversal, vector supplementation, and strategy annotation.
        """
        return await self._graph_rag.retrieve(query=query, ctx=ctx)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def clear_session(self, run_id: str) -> None:
        """Clear all short-term memory for the given run_id."""
        await self._short_term.clear(run_id)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls, config: Any) -> "MemoryManager":
        """
        Build a fully configured MemoryManager from harness config.

        Constructs:
        - ShortTermMemory (Redis)
        - EmbeddingProvider (SentenceTransformer)
        - VectorStore (Chroma / Qdrant / Weaviate based on config)
        - GraphMemory (NetworkX / Neo4j based on config)
        - ContextWindowManager (100k token budget by default)
        """
        embedder = build_embedding_provider(config)
        vector_store = build_vector_store(config, embedder)
        short_term = ShortTermMemory(redis_url=config.redis_url)
        graph = get_graph_memory(config)
        context_manager = ContextWindowManager(max_tokens=100_000)

        return cls(
            short_term=short_term,
            vector_store=vector_store,
            graph=graph,
            embedder=embedder,
            context_manager=context_manager,
        )
