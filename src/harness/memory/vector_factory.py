"""Factory functions for vector store and embedding provider construction."""

from __future__ import annotations

from typing import Any

from harness.core.protocols import EmbeddingProvider, VectorStore
from harness.memory.embedder import SentenceTransformerEmbedder


def build_embedding_provider(config: Any) -> EmbeddingProvider:
    """Construct a SentenceTransformerEmbedder from harness config."""
    return SentenceTransformerEmbedder(model_name=config.embedding_model)  # type: ignore[return-value]


def build_vector_store(config: Any, embedder: EmbeddingProvider) -> VectorStore:
    """
    Construct and return the configured vector store backend.

    Supported backends (config.vector_backend):
      - "chroma"   → ChromaVectorStore
      - "qdrant"   → QdrantVectorStore
      - "weaviate" → WeaviateVectorStore
    """
    backend = config.vector_backend

    if backend == "chroma":
        from harness.memory.backends.chroma import ChromaVectorStore

        return ChromaVectorStore(  # type: ignore[return-value]
            path=config.chroma_path,
            embedder=embedder,
        )

    if backend == "qdrant":
        from harness.memory.backends.qdrant import QdrantVectorStore

        return QdrantVectorStore(  # type: ignore[return-value]
            url=config.qdrant_url,
            embedder=embedder,
        )

    if backend == "weaviate":
        from harness.memory.backends.weaviate import WeaviateVectorStore

        return WeaviateVectorStore(  # type: ignore[return-value]
            url=config.weaviate_url,
            embedder=embedder,
        )

    raise ValueError(
        f"Unknown vector_backend '{backend}'. "
        "Valid choices are: 'chroma', 'qdrant', 'weaviate'."
    )


class VectorStoreFactory:
    """
    Class-based factory for vector store instantiation.

    Provides the same functionality as the module-level ``build_vector_store``
    function but in a class form for dependency-injection patterns.
    """

    @staticmethod
    def build(config: Any, embedder: EmbeddingProvider) -> VectorStore:
        """Build a VectorStore from config and embedder."""
        return build_vector_store(config, embedder)

    @staticmethod
    def build_embedder(config: Any) -> EmbeddingProvider:
        """Build an EmbeddingProvider from config."""
        return build_embedding_provider(config)
