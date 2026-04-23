"""Ingestion module — document loading, chunking, entity extraction, pipeline."""

from harness.ingestion.chunker import Chunk, chunk_documents, chunk_text
from harness.ingestion.extractor import ExtractedFact, extract_facts, extract_sql_schema_facts
from harness.ingestion.loaders import Document, detect_type, load
from harness.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "Chunk",
    "chunk_documents",
    "chunk_text",
    "Document",
    "detect_type",
    "ExtractedFact",
    "extract_facts",
    "extract_sql_schema_facts",
    "IngestionPipeline",
    "IngestionResult",
    "load",
]
