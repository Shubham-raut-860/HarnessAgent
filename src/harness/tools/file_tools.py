"""File I/O tools for Codex Harness agents.

All operations are workspace-scoped — path traversal outside the workspace
is rejected. Writes are atomic (temp file + rename).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from harness.core.context import AgentContext, ToolResult

logger = logging.getLogger(__name__)


def _resolve_safe(workspace: Path, rel_path: str) -> Path | None:
    """Resolve rel_path within workspace, returning None if it escapes."""
    workspace_abs = workspace.resolve()
    try:
        target = (workspace_abs / rel_path).resolve()
    except Exception:
        return None
    # Ensure target is inside workspace
    try:
        target.relative_to(workspace_abs)
    except ValueError:
        return None
    return target


class ReadFileTool:
    """Read a file from the agent workspace."""

    name = "read_file"
    description = "Read a file from the agent workspace and return its contents."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the agent workspace.",
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "File encoding (default: utf-8).",
            },
            "max_bytes": {
                "type": "integer",
                "default": 1_048_576,  # 1 MB
                "description": "Maximum bytes to read (default: 1 MB).",
            },
        },
        "required": ["path"],
    }
    timeout_seconds: float = 10.0

    def __init__(self, workspace_manager: Any | None = None) -> None:
        self._workspace_manager = workspace_manager

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Read and return file contents."""
        rel_path: str = args["path"]
        encoding: str = args.get("encoding", "utf-8")
        max_bytes: int = int(args.get("max_bytes", 1_048_576))

        # Resolve workspace path
        if self._workspace_manager is not None:
            try:
                resolved = await _maybe_await(
                    self._workspace_manager.resolve(
                        ctx.run_id, ctx.tenant_id, rel_path
                    )
                )
                target = Path(resolved)
            except Exception as exc:
                return ToolResult(data=None, error=f"Path resolution failed: {exc}")
        else:
            target = _resolve_safe(ctx.workspace_path, rel_path)
            if target is None:
                return ToolResult(
                    data=None,
                    error=f"Path '{rel_path}' escapes workspace boundary.",
                )

        if not target.exists():
            return ToolResult(
                data=None, error=f"File not found: {rel_path}"
            )
        if not target.is_file():
            return ToolResult(
                data=None, error=f"Path is not a file: {rel_path}"
            )

        try:
            stat = target.stat()
            file_size = stat.st_size
            truncated = False

            with target.open("rb") as fh:
                raw = fh.read(max_bytes)
            if file_size > max_bytes:
                truncated = True

            content = raw.decode(encoding, errors="replace")
            return ToolResult(
                data=content,
                metadata={
                    "path": rel_path,
                    "size_bytes": file_size,
                    "encoding": encoding,
                    "truncated": truncated,
                    "lines": content.count("\n"),
                },
            )
        except Exception as exc:
            logger.exception("read_file failed for '%s': %s", rel_path, exc)
            return ToolResult(data=None, error=f"Failed to read file: {exc}")


class WriteFileTool:
    """Write content to a file in the agent workspace (atomic write)."""

    name = "write_file"
    description = (
        "Write content to a file in the agent workspace. "
        "Creates intermediate directories if needed. "
        "Uses atomic temp-file + rename to prevent partial writes."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination file path relative to workspace.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "encoding": {
                "type": "string",
                "default": "utf-8",
                "description": "File encoding.",
            },
            "append": {
                "type": "boolean",
                "default": False,
                "description": "If true, append to existing file instead of overwriting.",
            },
        },
        "required": ["path", "content"],
    }
    timeout_seconds: float = 10.0

    def __init__(self, workspace_manager: Any | None = None) -> None:
        self._workspace_manager = workspace_manager

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """Atomically write content to the resolved workspace path."""
        rel_path: str = args["path"]
        content: str = args["content"]
        encoding: str = args.get("encoding", "utf-8")
        append: bool = bool(args.get("append", False))

        # Resolve workspace path
        if self._workspace_manager is not None:
            try:
                resolved = await _maybe_await(
                    self._workspace_manager.resolve(
                        ctx.run_id, ctx.tenant_id, rel_path
                    )
                )
                target = Path(resolved)
            except Exception as exc:
                return ToolResult(data=None, error=f"Path resolution failed: {exc}")
        else:
            target = _resolve_safe(ctx.workspace_path, rel_path)
            if target is None:
                return ToolResult(
                    data=None,
                    error=f"Path '{rel_path}' escapes workspace boundary.",
                )

        # Create parent directories
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return ToolResult(
                data=None, error=f"Failed to create directories: {exc}"
            )

        try:
            if append and target.exists():
                # Append mode: just open and append
                with target.open("a", encoding=encoding) as fh:
                    fh.write(content)
            else:
                # Atomic write: write to temp file in same directory, then rename
                dir_path = str(target.parent)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding=encoding,
                    dir=dir_path,
                    delete=False,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                ) as tmp_fh:
                    tmp_fh.write(content)
                    tmp_path = tmp_fh.name

                os.replace(tmp_path, str(target))

            size_bytes = target.stat().st_size
            return ToolResult(
                data={
                    "success": True,
                    "path": rel_path,
                    "size_bytes": size_bytes,
                    "appended": append,
                },
                metadata={"path": rel_path},
            )
        except Exception as exc:
            logger.exception("write_file failed for '%s': %s", rel_path, exc)
            # Clean up temp file if it exists
            try:
                if "tmp_path" in dir() and os.path.exists(tmp_path):  # type: ignore[name-defined]
                    os.unlink(tmp_path)  # type: ignore[name-defined]
            except Exception:
                pass
            return ToolResult(data=None, error=f"Failed to write file: {exc}")


class ListWorkspaceTool:
    """List files in the agent workspace directory."""

    name = "list_workspace"
    description = (
        "List files and directories in the agent workspace. "
        "Optionally filter by a subdirectory path."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "default": ".",
                "description": "Subdirectory to list (default: workspace root).",
            },
            "recursive": {
                "type": "boolean",
                "default": False,
                "description": "If true, list recursively.",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., '*.py').",
            },
        },
        "required": [],
    }
    timeout_seconds: float = 10.0

    async def execute(self, ctx: AgentContext, args: dict[str, Any]) -> ToolResult:
        """List files in the workspace or a subdirectory."""
        rel_path: str = args.get("path", ".")
        recursive: bool = bool(args.get("recursive", False))
        pattern: str | None = args.get("pattern")

        target = _resolve_safe(ctx.workspace_path, rel_path)
        if target is None:
            return ToolResult(
                data=None,
                error=f"Path '{rel_path}' escapes workspace boundary.",
            )

        if not target.exists():
            return ToolResult(data=None, error=f"Path not found: {rel_path}")
        if not target.is_dir():
            return ToolResult(data=None, error=f"Path is not a directory: {rel_path}")

        try:
            workspace_abs = ctx.workspace_path.resolve()
            entries: list[dict[str, Any]] = []

            if recursive:
                glob_pattern = f"**/{pattern}" if pattern else "**/*"
                paths = list(target.glob(glob_pattern))
            else:
                glob_pattern = pattern if pattern else "*"
                paths = list(target.glob(glob_pattern))

            # Sort and build entries
            for p in sorted(paths):
                if not recursive and p.parent != target:
                    continue
                try:
                    rel = str(p.relative_to(workspace_abs))
                    stat = p.stat()
                    entries.append(
                        {
                            "path": rel,
                            "name": p.name,
                            "type": "dir" if p.is_dir() else "file",
                            "size_bytes": stat.st_size if p.is_file() else None,
                            "modified": stat.st_mtime,
                        }
                    )
                except Exception:
                    continue

            return ToolResult(
                data=entries,
                metadata={
                    "directory": rel_path,
                    "count": len(entries),
                    "recursive": recursive,
                    "pattern": pattern,
                },
            )
        except Exception as exc:
            logger.exception("list_workspace failed: %s", exc)
            return ToolResult(data=None, error=f"Failed to list workspace: {exc}")


async def _maybe_await(obj: Any) -> Any:
    """Await if coroutine, otherwise return directly."""
    import asyncio
    if asyncio.iscoroutine(obj):
        return await obj
    return obj
