"""IngestionPipeline — orchestrates the full document ingestion flow."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from harness.ingestion.chunker import chunk_documents
from harness.ingestion.extractor import extract_facts as _extract_facts
from harness.ingestion.loaders import detect_type, load

logger = logging.getLogger(__name__)

# Supported file extensions for directory ingestion
_SUPPORTED_EXTENSIONS = {
    ".pdf", ".html", ".htm", ".md", ".markdown",
    ".docx", ".doc", ".csv", ".tsv", ".txt", ".text", ".rst",
}


@dataclass
class IngestionResult:
    """Result of a single source ingestion.

    Attributes:
        source:              The source file path or URL.
        documents_loaded:    Number of Document objects loaded.
        chunks_created:      Number of Chunk objects created.
        embeddings_stored:   Number of chunks stored in the vector store.
        facts_extracted:     Number of knowledge triples extracted.
        errors:              List of error messages encountered.
        duration_seconds:    Wall-clock seconds to complete ingestion.
    """

    source: str
    documents_loaded: int = 0
    chunks_created: int = 0
    embeddings_stored: int = 0
    facts_extracted: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Return True if ingestion completed with no critical errors."""
        return self.documents_loaded > 0 and len(self.errors) == 0

    def __repr__(self) -> str:
        return (
            f"IngestionResult(source={self.source!r}, "
            f"docs={self.documents_loaded}, chunks={self.chunks_created}, "
            f"embeddings={self.embeddings_stored}, facts={self.facts_extracted}, "
            f"errors={len(self.errors)}, duration={self.duration_seconds:.2f}s)"
        )


class IngestionPipeline:
    """Full ingestion pipeline: load → chunk → embed → extract facts.

    Args:
        memory_manager: A MemoryManager instance that provides:
            - ``remember(text, metadata, tenant_id)`` — stores embedding
            - ``add_fact(subject, predicate, object_, tenant_id)`` — stores graph fact
        llm_provider:  Optional LLMProvider for entity extraction.
                       If None, fact extraction is skipped.
    """

    def __init__(
        self,
        memory_manager: Any,
        llm_provider: Optional[Any] = None,
    ) -> None:
        self._memory = memory_manager
        self._llm = llm_provider

    # ------------------------------------------------------------------
    # Single source ingestion
    # ------------------------------------------------------------------

    async def ingest(
        self,
        source: str,
        tenant_id: str,
        agent_type: str = "research",
        extract_facts: bool = True,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> IngestionResult:
        """Ingest a single file or URL.

        Pipeline:
            1. load(source)  → list[Document]
            2. chunk_documents(docs) → list[Chunk]
            3. For each chunk: memory_manager.remember(chunk.content, metadata, tenant_id)
            4. If extract_facts and llm_provider: extract triples and store via add_fact

        Args:
            source:         File path or URL to ingest.
            tenant_id:      Tenant to associate all memories with.
            agent_type:     Tag used for routing/filtering in the memory store.
            extract_facts:  Whether to run LLM-based entity extraction.
            chunk_size:     Target chunk size in tokens.
            chunk_overlap:  Overlap between consecutive chunks in tokens.

        Returns:
            IngestionResult summarising what was done.
        """
        result = IngestionResult(source=source)
        t_start = time.monotonic()

        # Step 1: Load documents
        try:
            docs = await load(source)
            result.documents_loaded = len(docs)
            if not docs:
                result.errors.append(f"No documents loaded from: {source}")
                result.duration_seconds = time.monotonic() - t_start
                return result
        except Exception as exc:
            msg = f"Failed to load {source}: {exc}"
            logger.error(msg)
            result.errors.append(msg)
            result.duration_seconds = time.monotonic() - t_start
            return result

        logger.info(
            "Loaded %d documents from %s (agent_type=%s, tenant=%s)",
            len(docs), source, agent_type, tenant_id,
        )

        # Step 2: Chunk documents
        try:
            chunks = chunk_documents(
                docs,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
            result.chunks_created = len(chunks)
        except Exception as exc:
            msg = f"Chunking failed for {source}: {exc}"
            logger.error(msg)
            result.errors.append(msg)
            result.duration_seconds = time.monotonic() - t_start
            return result

        logger.info("Created %d chunks from %s", len(chunks), source)

        # Step 3: Store embeddings
        embedded_count = 0
        for chunk in chunks:
            chunk_meta = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "agent_type": agent_type,
                "source": source,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            try:
                await self._remember(chunk.content, chunk_meta, tenant_id)
                embedded_count += 1
            except Exception as exc:
                msg = f"Failed to store chunk {chunk.chunk_id}: {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        result.embeddings_stored = embedded_count
        logger.info("Stored %d embeddings for %s", embedded_count, source)

        # Step 4: Extract and store facts
        if extract_facts and self._llm is not None:
            try:
                facts = await _extract_facts(
                    chunks=chunks,
                    llm_provider=self._llm,
                )
                facts_stored = 0
                for fact in facts:
                    try:
                        await self._add_fact(
                            subject=fact.subject,
                            predicate=fact.predicate,
                            object_=fact.object_,
                            tenant_id=tenant_id,
                        )
                        facts_stored += 1
                    except Exception as exc:
                        logger.warning("Failed to store fact %s: %s", fact, exc)
                result.facts_extracted = facts_stored
                logger.info("Stored %d facts from %s", facts_stored, source)
            except Exception as exc:
                msg = f"Fact extraction failed for {source}: {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        result.duration_seconds = time.monotonic() - t_start
        return result

    # ------------------------------------------------------------------
    # Directory ingestion
    # ------------------------------------------------------------------

    async def ingest_directory(
        self,
        directory: str,
        tenant_id: str,
        concurrency: int = 3,
        **kwargs,
    ) -> list[IngestionResult]:
        """Ingest all supported files in a directory concurrently.

        Args:
            directory:   Path to the directory to scan.
            tenant_id:   Tenant to associate memories with.
            concurrency: Max concurrent ingestion tasks.
            **kwargs:    Forwarded to self.ingest() per file.

        Returns:
            List of IngestionResult, one per discovered file.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.error("Directory not found: %s", directory)
            return [
                IngestionResult(
                    source=directory,
                    errors=[f"Directory not found: {directory}"],
                )
            ]

        # Find all supported files recursively
        files: list[Path] = []
        for p in dir_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                files.append(p)

        if not files:
            logger.warning("No supported files found in directory: %s", directory)
            return []

        logger.info(
            "Ingesting %d files from directory %s (concurrency=%d)",
            len(files), directory, concurrency,
        )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._ingest_with_semaphore(str(f), tenant_id, semaphore, **kwargs)
            for f in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ingestion_results: list[IngestionResult] = []
        for f, res in zip(files, results):
            if isinstance(res, Exception):
                ingestion_results.append(
                    IngestionResult(
                        source=str(f),
                        errors=[f"Unhandled exception: {res}"],
                    )
                )
            else:
                ingestion_results.append(res)

        total_docs = sum(r.documents_loaded for r in ingestion_results)
        total_chunks = sum(r.chunks_created for r in ingestion_results)
        logger.info(
            "Directory ingestion complete: %d files, %d docs, %d chunks",
            len(files), total_docs, total_chunks,
        )
        return ingestion_results

    async def _ingest_with_semaphore(
        self,
        source: str,
        tenant_id: str,
        semaphore: asyncio.Semaphore,
        **kwargs,
    ) -> IngestionResult:
        """Wrap ingest() with a semaphore for concurrency control."""
        async with semaphore:
            return await self.ingest(source=source, tenant_id=tenant_id, **kwargs)

    # ------------------------------------------------------------------
    # Memory interface helpers (normalise different memory manager APIs)
    # ------------------------------------------------------------------

    async def _remember(
        self, text: str, metadata: dict, tenant_id: str
    ) -> None:
        """Call the appropriate remember method on the memory manager."""
        mem = self._memory
        if hasattr(mem, "remember"):
            result = mem.remember(text=text, metadata=metadata, tenant_id=tenant_id)
            if asyncio.iscoroutine(result):
                await result
        elif hasattr(mem, "store"):
            result = mem.store(text=text, metadata=metadata, tenant_id=tenant_id)
            if asyncio.iscoroutine(result):
                await result
        else:
            logger.warning(
                "MemoryManager has no 'remember' or 'store' method; skipping embedding"
            )

    async def _add_fact(
        self, subject: str, predicate: str, object_: str, tenant_id: str
    ) -> None:
        """Call the appropriate add_fact method on the memory manager."""
        mem = self._memory
        if hasattr(mem, "add_fact"):
            result = mem.add_fact(
                subject=subject,
                predicate=predicate,
                object_=object_,
                tenant_id=tenant_id,
            )
            if asyncio.iscoroutine(result):
                await result
        elif hasattr(mem, "graph") and hasattr(mem.graph, "add_edge"):
            # Direct graph access fallback
            result = mem.graph.add_edge(subject, object_, type=predicate)
            if asyncio.iscoroutine(result):
                await result
        else:
            logger.debug(
                "MemoryManager has no 'add_fact' method; skipping fact storage"
            )
