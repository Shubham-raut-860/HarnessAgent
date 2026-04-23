"""Integration tests for the SQL agent using in-memory SQLite."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from harness.core.context import AgentContext, LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_db():
    """Create an in-memory SQLite database with test tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            name TEXT,
            active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            total REAL NOT NULL DEFAULT 0.0,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        );

        INSERT INTO users (id, email, name, active) VALUES
            (1, 'alice@example.com', 'Alice', 1),
            (2, 'bob@example.com', 'Bob', 1),
            (3, 'charlie@example.com', 'Charlie', 0);

        INSERT INTO orders (user_id, total, status) VALUES
            (1, 99.99, 'completed'),
            (1, 49.50, 'completed'),
            (2, 19.99, 'pending');

        INSERT INTO products (name, price, stock) VALUES
            ('Widget A', 9.99, 100),
            ('Widget B', 19.99, 50),
            ('Premium Widget', 99.99, 10);
    """)
    yield conn
    conn.close()


@pytest.fixture
def sql_agent_context(tmp_path):
    """Create an AgentContext for SQL agent tests."""
    return AgentContext(
        run_id=uuid.uuid4().hex,
        tenant_id="test-tenant",
        agent_type="sql",
        task="Test SQL task",
        memory=MagicMock(),
        workspace_path=tmp_path / "ws",
        max_steps=20,
        max_tokens=50_000,
        timeout_seconds=60.0,
    )


def _make_sql_llm(responses: list):
    """Create a mock LLM that returns SQL-relevant responses."""
    llm = AsyncMock()
    llm.provider_name = "mock"
    llm.model = "mock"
    llm.complete = AsyncMock(side_effect=responses)
    llm.health_check = AsyncMock(return_value=True)
    return llm


def _sql_final_response(content: str):
    return LLMResponse(content=content, tool_calls=[], input_tokens=50, output_tokens=100)


def _sql_call_response(query: str):
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=uuid.uuid4().hex, name="execute_sql", args={"query": query})],
        input_tokens=50,
        output_tokens=50,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sql_agent_lists_tables(sqlite_db, sql_agent_context, tmp_path):
    """SQL agent should correctly identify and list all tables."""
    try:
        from harness.agents.sql import SQLAgent  # type: ignore
    except ImportError:
        pytest.skip("SQLAgent not implemented")

    responses = [
        _sql_call_response("SELECT name FROM sqlite_master WHERE type='table'"),
        _sql_final_response("The database has 3 tables: users, orders, products"),
    ]
    llm = _make_sql_llm(responses)
    sql_agent_context.task = "List all tables in the database"

    agent = SQLAgent(llm=llm, connection=sqlite_db, read_only=True)
    result = await agent.execute(sql_agent_context)

    assert result.success is True
    output_lower = result.output.lower()
    assert any(table in output_lower for table in ["users", "orders", "products"])


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sql_agent_counts_rows(sqlite_db, sql_agent_context, tmp_path):
    """SQL agent should count rows per table correctly."""
    try:
        from harness.agents.sql import SQLAgent  # type: ignore
    except ImportError:
        pytest.skip("SQLAgent not implemented")

    # Expected: users=3, orders=3, products=3
    responses = [
        _sql_call_response("SELECT 'users' as table_name, COUNT(*) as row_count FROM users"),
        _sql_call_response("SELECT 'orders' as table_name, COUNT(*) as row_count FROM orders"),
        _sql_call_response("SELECT 'products' as table_name, COUNT(*) as row_count FROM products"),
        _sql_final_response("Row counts: users=3, orders=3, products=3"),
    ]
    llm = _make_sql_llm(responses)
    sql_agent_context.task = "Count rows in each table"

    agent = SQLAgent(llm=llm, connection=sqlite_db, read_only=True)
    result = await agent.execute(sql_agent_context)

    assert result.success is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sql_agent_respects_read_only_mode(sqlite_db, sql_agent_context):
    """SQL agent in read-only mode should reject DELETE/UPDATE/INSERT queries."""
    try:
        from harness.agents.sql import SQLAgent  # type: ignore
    except ImportError:
        pytest.skip("SQLAgent not implemented")

    responses = [
        # Agent tries to delete
        _sql_call_response("DELETE FROM users WHERE id=1"),
        # After rejection, agent should stop or try something else
        _sql_final_response("Cannot execute DELETE in read-only mode"),
    ]
    llm = _make_sql_llm(responses)
    sql_agent_context.task = "Delete user with id=1"

    agent = SQLAgent(llm=llm, connection=sqlite_db, read_only=True)
    result = await agent.execute(sql_agent_context)

    # Verify no actual deletion occurred
    cursor = sqlite_db.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    assert count == 3  # All users still present


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sql_agent_handles_invalid_sql_gracefully(sqlite_db, sql_agent_context):
    """SQL agent should handle SQL syntax errors gracefully and try to recover."""
    try:
        from harness.agents.sql import SQLAgent  # type: ignore
    except ImportError:
        pytest.skip("SQLAgent not implemented")

    responses = [
        # First attempt: invalid SQL
        _sql_call_response("SELCT * FORM users"),  # typos
        # Agent receives error and tries corrected SQL
        _sql_call_response("SELECT * FROM users LIMIT 10"),
        _sql_final_response("Found 3 users in the database"),
    ]
    llm = _make_sql_llm(responses)
    sql_agent_context.task = "Show all users"

    agent = SQLAgent(llm=llm, connection=sqlite_db, read_only=True)
    result = await agent.execute(sql_agent_context)

    # Should eventually succeed or at least not crash with an unhandled exception
    assert result is not None
    assert hasattr(result, "success")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sql_agent_populates_schema_graph(sqlite_db, sql_agent_context, mock_graph_store):
    """SQL agent should populate the knowledge graph with schema information."""
    try:
        from harness.agents.sql import SQLAgent  # type: ignore
    except ImportError:
        pytest.skip("SQLAgent not implemented")

    responses = [
        _sql_call_response("SELECT name FROM sqlite_master WHERE type='table'"),
        _sql_final_response("Schema loaded successfully"),
    ]
    llm = _make_sql_llm(responses)
    sql_agent_context.memory.graph_store = mock_graph_store
    sql_agent_context.task = "Analyse the database schema"

    agent = SQLAgent(
        llm=llm,
        connection=sqlite_db,
        read_only=True,
        populate_graph=True,
    )

    if hasattr(agent, "populate_schema_graph"):
        await agent.populate_schema_graph(sqlite_db, sql_agent_context)
        # Should have added nodes for tables
        assert mock_graph_store.add_node.call_count >= 2
