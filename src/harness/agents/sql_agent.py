"""SQLAgent: specialized agent for database query tasks."""

from __future__ import annotations

import logging
from typing import Any

from harness.agents.base import BaseAgent, _safe_call
from harness.core.context import AgentContext, AgentResult

logger = logging.getLogger(__name__)


class SQLAgent(BaseAgent):
    """Agent specialised for SQL and database exploration tasks.

    Before entering the main run loop it populates the graph memory
    with schema information (tables + columns) so downstream queries
    benefit from structured retrieval.
    """

    agent_type: str = "sql"

    def build_system_prompt(self, ctx: AgentContext) -> str:  # type: ignore[override]
        """Return the SQL agent system prompt."""
        return """You are a SQL expert assistant with access to a relational database.

Rules:
1. ALWAYS call list_tables first to understand what tables are available.
2. ALWAYS call describe_table for each relevant table before writing any query.
   Never assume column names — they may differ from common conventions.
3. Call sample_rows to understand data distributions when needed.
4. Only write SELECT queries (read-only by default).
   If INSERT/UPDATE/DELETE is explicitly authorised, confirm before executing.
5. Always add LIMIT to queries that don't use aggregations (e.g., COUNT, SUM).
6. When reporting results, provide:
   - The SQL you used
   - A clear interpretation of the results in plain language
   - Any caveats or data quality observations
7. If a query fails, carefully read the error message, check the schema, and fix it.
   Do not guess — look up the actual column names with describe_table.

Database context and schema information will be provided from memory."""

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Pre-populate schema in graph memory, then run the base agent loop."""
        await self._populate_schema(ctx)
        return await super().run(ctx)

    async def _populate_schema(self, ctx: AgentContext) -> None:
        """Introspect the database and store schema in graph memory.

        This is a one-time operation per session, gated by ctx.metadata["schema_loaded"].
        Subsequent calls within the same session are no-ops.
        """
        if ctx.metadata.get("schema_loaded"):
            return

        # Mark as loading to prevent concurrent calls
        ctx.metadata["schema_loaded"] = True

        logger.info(
            "Populating database schema into graph memory for run %s", ctx.run_id
        )

        # Try to get list_tables and describe_table from tool registry
        if self._tool_registry is None:
            logger.debug("No tool_registry — skipping schema population")
            return

        list_tables_tool = self._tool_registry.get("list_tables")
        describe_table_tool = self._tool_registry.get("describe_table")

        if list_tables_tool is None:
            logger.debug("list_tables tool not registered — skipping schema population")
            return

        # Create a lightweight ToolCall to invoke list_tables
        from harness.core.context import ToolCall
        import uuid

        list_call = ToolCall(
            id=uuid.uuid4().hex,
            name="list_tables",
            args={},
        )

        try:
            list_result = await list_tables_tool.execute(ctx, list_call.args)
        except Exception as exc:
            logger.warning("list_tables during schema population failed: %s", exc)
            return

        if list_result.is_error or not list_result.data:
            logger.debug("list_tables returned no data: %s", list_result.error)
            return

        tables: list[dict] = list_result.data if isinstance(list_result.data, list) else []
        table_names = [
            t.get("table_name") or t.get("name")
            for t in tables
            if t.get("table_name") or t.get("name")
        ]

        logger.info("Discovered %d tables for schema population", len(table_names))

        # Populate graph memory with table and column nodes
        graph_rag_available = False
        if ctx.memory is not None and hasattr(ctx.memory, "graph_rag"):
            graph_rag_available = True

        for table_name in table_names[:50]:  # cap at 50 tables
            # Add table node to graph memory
            if ctx.memory is not None and hasattr(ctx.memory, "graph") and ctx.memory.graph is not None:
                try:
                    await ctx.memory.graph.add_node(
                        id=f"table:{table_name}",
                        type="Table",
                        props={"name": table_name, "run_id": ctx.run_id},
                    )
                except Exception as exc:
                    logger.debug("Failed to add table node '%s': %s", table_name, exc)

            # Describe table to get columns
            if describe_table_tool is None:
                continue

            describe_call = ToolCall(
                id=uuid.uuid4().hex,
                name="describe_table",
                args={"table_name": table_name},
            )

            try:
                desc_result = await describe_table_tool.execute(ctx, describe_call.args)
                if desc_result.is_error or not desc_result.data:
                    continue

                table_info = desc_result.data
                columns = table_info.get("columns", []) if isinstance(table_info, dict) else []

                for col in columns:
                    col_name = col.get("column_name") or col.get("name")
                    if not col_name:
                        continue
                    if ctx.memory is not None and hasattr(ctx.memory, "graph") and ctx.memory.graph is not None:
                        try:
                            await ctx.memory.graph.add_node(
                                id=f"column:{table_name}.{col_name}",
                                type="Column",
                                props={
                                    "name": col_name,
                                    "table": table_name,
                                    "data_type": col.get("data_type", "unknown"),
                                    "nullable": col.get("is_nullable", "YES"),
                                },
                            )
                            await ctx.memory.graph.add_edge(
                                src=f"table:{table_name}",
                                tgt=f"column:{table_name}.{col_name}",
                                type="HAS_COLUMN",
                            )
                        except Exception as exc:
                            logger.debug(
                                "Failed to add column node '%s.%s': %s",
                                table_name,
                                col_name,
                                exc,
                            )

                # Also store schema description in vector memory for RAG
                if ctx.memory is not None and hasattr(ctx.memory, "push_to_vector"):
                    col_summary = ", ".join(
                        f"{c.get('column_name', c.get('name', '?'))}({c.get('data_type', '?')})"
                        for c in columns[:20]
                    )
                    schema_text = f"Table: {table_name}\nColumns: {col_summary}"
                    try:
                        await ctx.memory.push_to_vector(
                            run_id=ctx.run_id,
                            text=schema_text,
                            metadata={"type": "schema", "table": table_name},
                        )
                    except Exception as exc:
                        logger.debug("Failed to push schema to vector: %s", exc)

            except Exception as exc:
                logger.warning(
                    "describe_table failed for '%s' during schema population: %s",
                    table_name,
                    exc,
                )
                continue

        # Store schema summary in ctx metadata
        ctx.metadata["schema_tables"] = table_names
        logger.info(
            "Schema population complete: %d tables stored in graph memory",
            len(table_names),
        )
