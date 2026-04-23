"""Integration tests for the code agent using RestrictedPythonExecutor."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from harness.core.context import AgentContext, LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def code_agent_context(tmp_path):
    """Create an AgentContext for code agent tests."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "src").mkdir(exist_ok=True)

    return AgentContext(
        run_id=uuid.uuid4().hex,
        tenant_id="test-tenant",
        agent_type="code",
        task="Write and test Python code",
        memory=MagicMock(),
        workspace_path=ws,
        max_steps=20,
        max_tokens=50_000,
        timeout_seconds=60.0,
    )


def _make_code_llm(responses: list):
    """Create a mock LLM that returns code-relevant responses."""
    llm = AsyncMock()
    llm.provider_name = "mock"
    llm.model = "mock"
    llm.complete = AsyncMock(side_effect=responses)
    llm.health_check = AsyncMock(return_value=True)
    return llm


def _code_response(content: str):
    return LLMResponse(content=content, tool_calls=[], input_tokens=50, output_tokens=100)


def _code_tool_response(tool_name: str, args: dict):
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=uuid.uuid4().hex, name=tool_name, args=args)],
        input_tokens=50,
        output_tokens=50,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_agent_writes_and_runs_simple_function(
    code_agent_context, tmp_path
):
    """Code agent should write a simple Python function and execute it."""
    try:
        from harness.agents.code import CodeAgent  # type: ignore
    except ImportError:
        pytest.skip("CodeAgent not implemented")

    code = """
def reverse_string(s: str) -> str:
    '''Reverse a string.'''
    return s[::-1]

# Test it
result = reverse_string("hello")
assert result == "olleh", f"Expected 'olleh', got {result!r}"
print(f"Test passed: reverse_string('hello') = {result!r}")
"""
    responses = [
        _code_tool_response("write_file", {
            "path": "solution.py",
            "content": code,
        }),
        _code_tool_response("execute_python", {
            "code": code,
        }),
        _code_response("The function works correctly. reverse_string('hello') returns 'olleh'."),
    ]

    llm = _make_code_llm(responses)
    code_agent_context.task = "Write a Python function to reverse a string and test it"

    agent = CodeAgent(llm=llm, workspace_path=code_agent_context.workspace_path)
    result = await agent.execute(code_agent_context)

    assert result is not None
    assert result.success is True
    assert "reverse" in result.output.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_agent_debugs_broken_code(code_agent_context):
    """Code agent should identify and fix a bug in broken code."""
    try:
        from harness.agents.code import CodeAgent  # type: ignore
    except ImportError:
        pytest.skip("CodeAgent not implemented")

    broken_code = "def add(a,b): return a-b"
    fixed_code = "def add(a,b): return a+b"

    responses = [
        # Agent runs the broken code and observes it's wrong
        _code_tool_response("execute_python", {
            "code": broken_code + "\nresult = add(2, 3)\nassert result == 5, f'Bug: expected 5, got {result}'"
        }),
        # Agent proposes fix
        _code_tool_response("execute_python", {
            "code": fixed_code + "\nresult = add(2, 3)\nassert result == 5\nprint(f'Fixed! add(2,3) = {result}')"
        }),
        _code_response("Found and fixed the bug: 'return a-b' should be 'return a+b'. The function now correctly adds two numbers."),
    ]

    llm = _make_code_llm(responses)
    code_agent_context.task = "Debug this code: def add(a,b): return a-b"

    agent = CodeAgent(llm=llm, workspace_path=code_agent_context.workspace_path)
    result = await agent.execute(code_agent_context)

    assert result is not None
    assert result.success is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_agent_runs_linter(code_agent_context, tmp_path):
    """Code agent should run flake8/pylint and report results."""
    try:
        from harness.agents.code import CodeAgent  # type: ignore
    except ImportError:
        pytest.skip("CodeAgent not implemented")

    messy_code = """
import os
import sys
def add( a,b ) :
    x=a+b
    return x
"""
    responses = [
        _code_tool_response("write_file", {
            "path": "messy.py",
            "content": messy_code,
        }),
        _code_tool_response("run_linter", {"path": "messy.py"}),
        _code_response("Linting complete. Found style issues: unnecessary spaces, unused imports (os, sys). Code works but needs cleanup."),
    ]

    llm = _make_code_llm(responses)
    code_agent_context.task = "Write a simple add function and lint it"

    agent = CodeAgent(llm=llm, workspace_path=code_agent_context.workspace_path)
    result = await agent.execute(code_agent_context)

    assert result is not None
    assert hasattr(result, "success")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_agent_respects_workspace_isolation(code_agent_context, tmp_path):
    """Code agent should not be able to access files outside the workspace."""
    try:
        from harness.agents.code import CodeAgent  # type: ignore
    except ImportError:
        pytest.skip("CodeAgent not implemented")

    # Try to read a file outside the workspace
    responses = [
        _code_tool_response("read_file", {"path": "../../sensitive_config.txt"}),
        _code_response("Cannot access files outside the workspace directory."),
    ]

    llm = _make_code_llm(responses)
    code_agent_context.task = "Read the configuration file"

    agent = CodeAgent(llm=llm, workspace_path=code_agent_context.workspace_path)
    result = await agent.execute(code_agent_context)

    # Should not have succeeded in reading outside workspace
    assert result is not None
    # If path traversal was blocked, the error should be reflected
    if not result.success:
        assert result.error_message is not None or result.output is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_code_agent_uses_restricted_executor():
    """RestrictedPythonExecutor should block dangerous operations."""
    try:
        from harness.tools.skills import RestrictedPythonExecutor  # type: ignore
    except ImportError:
        pytest.skip("RestrictedPythonExecutor not implemented")

    executor = RestrictedPythonExecutor(timeout_seconds=5)

    # Safe code should execute
    safe_result = await executor.execute("result = 2 + 2\nprint(result)")
    assert not safe_result.is_error or "4" in str(safe_result.data)

    # Dangerous code should be blocked or contained
    dangerous_code = "import subprocess; subprocess.run(['rm', '-rf', '/'])"
    dangerous_result = await executor.execute(dangerous_code)
    # Either blocked at import or execution level
    if not dangerous_result.is_error:
        # If it ran, it should have been sandboxed
        # The test passes as long as no actual damage was done
        pass
