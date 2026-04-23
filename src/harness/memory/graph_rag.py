"""GraphRAG engine: multi-hop graph retrieval with vector fallback."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from harness.core.protocols import EmbeddingProvider, GraphPath, VectorStore
from harness.memory.schemas import RetrievalResult

if TYPE_CHECKING:
    from harness.core.context import AgentContext

logger = logging.getLogger(__name__)

# Regex patterns for entity extraction
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")
_QUOTED_RE = re.compile(r'["\']([^"\']{2,})["\']')
_CAMEL_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")


class GraphRAGEngine:
    """
    Multi-hop retrieval combining graph traversal with vector similarity search.

    Strategy selection:
    - ≥3 graph paths found → ``graph_primary``
    - graph + vector results  → ``hybrid``
    - No graph results         → ``vector_fallback``
    - No entities extracted   → ``vector_primary``
    """

    def __init__(
        self,
        graph: Any,  # GraphMemory (NetworkX or Neo4j)
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._vector_store = vector_store
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        ctx: "AgentContext",
        max_hops: int = 2,
        vector_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve context for ``query`` using graph + vector combination.

        Steps:
        1. Extract entity names from the query.
        2. Anchor: find graph nodes matching those entities.
        3. Traverse: BFS/DFS up to ``max_hops`` from anchor nodes.
        4. Render paths to compact text.
        5. Supplement with vector store if <3 graph paths found.
        6. Return annotated RetrievalResult.
        """
        entities = await self._extract_entities(query)
        graph_paths: list[GraphPath] = []
        graph_context = ""

        if entities:
            try:
                anchor_nodes = await self._graph.find_nodes(entities, fuzzy=True)
                anchor_ids = [n.id for n in anchor_nodes]
                if anchor_ids:
                    graph_paths = await self._graph.traverse(anchor_ids, max_hops=max_hops)
                    graph_context = self._render_paths(graph_paths)
            except Exception as exc:
                logger.warning("Graph retrieval failed (continuing with vector): %s", exc)

        # Always query vector store; use result to supplement graph
        vector_hits = []
        vector_context: list[str] = []
        try:
            tenant_filter = {"tenant_id": ctx.tenant_id} if ctx.tenant_id else None
            vector_hits = await self._vector_store.query(
                text=query,
                k=vector_k,
                filter=tenant_filter,
            )
            vector_context = [h.text for h in vector_hits]
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)

        # Determine strategy
        if len(graph_paths) >= 3 and not vector_hits:
            strategy = "graph_primary"
        elif graph_paths and vector_hits:
            strategy = "hybrid"
        elif not graph_paths and entities:
            strategy = "vector_fallback"
        else:
            strategy = "vector_primary"

        total_tokens = self._estimate_tokens(
            graph_context, " ".join(vector_context)
        )

        return RetrievalResult(
            graph_paths=graph_paths,
            graph_context=graph_context,
            vector_hits=vector_hits,
            vector_context=vector_context,
            total_tokens_estimate=total_tokens,
            strategy=strategy,  # type: ignore[arg-type]
        )

    async def populate_schema(
        self,
        tables_info: list[dict[str, Any]],
        ctx: "AgentContext",
    ) -> None:
        """
        Populate the graph with SQL schema information.

        ``tables_info`` format:
        [{
            "name": "orders",
            "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
            "foreign_keys": [{"col": "user_id", "ref_table": "users", "ref_col": "id"}]
        }]
        """
        for table in tables_info:
            table_name = table["name"]
            await self._graph.add_node(
                id=table_name,
                type="Table",
                props={"name": table_name, "tenant_id": ctx.tenant_id},
            )

            for col in table.get("columns", []):
                col_id = f"{table_name}.{col['name']}"
                await self._graph.add_node(
                    id=col_id,
                    type="Column",
                    props={
                        "name": col["name"],
                        "col_type": col.get("type", "UNKNOWN"),
                        "nullable": col.get("nullable", True),
                        "table": table_name,
                    },
                )
                await self._graph.add_edge(
                    src=table_name,
                    tgt=col_id,
                    type="has_column",
                    props={},
                )

            for fk in table.get("foreign_keys", []):
                await self._graph.add_edge(
                    src=table_name,
                    tgt=fk["ref_table"],
                    type="joins",
                    props={
                        "on": f"{fk['col']}={fk['ref_col']}",
                        "local_col": fk["col"],
                        "ref_col": fk["ref_col"],
                    },
                )

        logger.info(
            "Populated schema graph with %d tables for tenant %s",
            len(tables_info),
            ctx.tenant_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_entities(self, query: str) -> list[str]:
        """
        Extract potential entity names from the query string.

        Combines:
        - Quoted strings (exact names)
        - snake_case / identifier-like tokens (≥3 chars)
        - CamelCase tokens
        """
        entities: list[str] = []

        # Quoted strings first (highest confidence)
        for match in _QUOTED_RE.finditer(query):
            entities.append(match.group(1))

        # CamelCase identifiers
        for match in _CAMEL_RE.finditer(query):
            entities.append(match.group(1))

        # Generic identifiers (skip very common English words)
        _STOP = {
            "the", "and", "for", "are", "not", "that", "this", "with",
            "from", "have", "all", "was", "been", "can", "will", "how",
            "what", "where", "which", "when", "who", "but", "list", "get",
            "show", "give", "find", "return", "select", "query",
        }
        for match in _IDENTIFIER_RE.finditer(query):
            word = match.group(1)
            if word.lower() not in _STOP:
                entities.append(word)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        return unique

    def _render_paths(self, paths: list[GraphPath]) -> str:
        """
        Render graph paths as a compact, deduplicated context string.

        Format:
        [SCHEMA]
        table_name: Table | cols: col1(type), col2(type)
        table1 --joins--> table2 ON col=refcol
        """
        lines: list[str] = ["[SCHEMA]"]
        seen: set[str] = set()

        table_cols: dict[str, list[str]] = {}
        join_lines: list[str] = []

        for path in paths:
            for node in path.nodes:
                if node.type == "Table":
                    if node.id not in table_cols:
                        table_cols[node.id] = []
                elif node.type == "Column":
                    table_name = node.props.get("table", "")
                    col_name = node.props.get("name", node.id)
                    col_type = node.props.get("col_type", "?")
                    if table_name and f"{table_name}.{col_name}" not in seen:
                        seen.add(f"{table_name}.{col_name}")
                        if table_name not in table_cols:
                            table_cols[table_name] = []
                        table_cols[table_name].append(f"{col_name}({col_type})")

            for edge in path.edges:
                if edge.type == "joins":
                    on_clause = edge.props.get("on", "")
                    join_line = f"{edge.source_id} --joins--> {edge.target_id}"
                    if on_clause:
                        join_line += f" ON {on_clause}"
                    if join_line not in seen:
                        seen.add(join_line)
                        join_lines.append(join_line)

        for table_name, cols in sorted(table_cols.items()):
            col_str = ", ".join(cols) if cols else "(no columns)"
            lines.append(f"{table_name}: Table | cols: {col_str}")

        lines.extend(join_lines)

        return "\n".join(lines)

    def _estimate_tokens(self, *texts: str) -> int:
        """Rough token estimate: total chars / 4."""
        total_chars = sum(len(t) for t in texts)
        return max(1, total_chars // 4)
