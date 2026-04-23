"""Workspace manager: isolated per-run filesystem sandboxes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TEMP_PATTERNS = ["*.tmp", "*.temp", "__pycache__", "*.pyc", ".pytest_cache"]


class WorkspaceManager:
    """
    Manages per-run filesystem workspaces under a shared base directory.

    Each run gets its own directory: ``{base_path}/{tenant_id}/{run_id}/``

    Path traversal is prevented: any attempt to resolve a path that escapes
    the workspace root raises ``PermissionError``.
    """

    def __init__(self, base_path: str | Path) -> None:
        self._base_path = Path(base_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create(self, run_id: str, tenant_id: str) -> Path:
        """
        Create and return the workspace directory for this run.

        Creates all intermediate directories as needed.
        """
        workspace = self._workspace_root(run_id, tenant_id)
        workspace.mkdir(parents=True, exist_ok=True)
        logger.debug("Workspace created: %s", workspace)
        return workspace

    def resolve(self, run_id: str, tenant_id: str, relative: str | Path) -> Path:
        """
        Resolve ``relative`` within the workspace, raising PermissionError
        if the resolved path escapes the workspace directory.
        """
        workspace = self._workspace_root(run_id, tenant_id)
        resolved = (workspace / relative).resolve()

        try:
            resolved.relative_to(workspace.resolve())
        except ValueError:
            raise PermissionError(
                f"Path traversal attempt detected: '{relative}' resolves to "
                f"'{resolved}' which is outside workspace '{workspace}'"
            )

        return resolved

    async def cleanup(
        self,
        run_id: str,
        tenant_id: str,
        keep_artifacts: bool = True,
    ) -> None:
        """
        Clean up workspace files.

        If ``keep_artifacts=True``, only temporary files are removed (*.tmp,
        __pycache__, etc.).  Otherwise the entire workspace directory is deleted.
        """
        workspace = self._workspace_root(run_id, tenant_id)

        if not workspace.exists():
            return

        if keep_artifacts:
            await self._remove_temp_files(workspace)
        else:
            shutil.rmtree(workspace, ignore_errors=True)
            logger.debug("Workspace removed: %s", workspace)

    async def list_files(
        self,
        run_id: str,
        tenant_id: str,
        pattern: str = "*",
    ) -> list[Path]:
        """Return all files matching ``pattern`` in the workspace (recursive)."""
        workspace = self._workspace_root(run_id, tenant_id)
        if not workspace.exists():
            return []
        return [p for p in workspace.rglob(pattern) if p.is_file()]

    async def get_size(self, run_id: str, tenant_id: str) -> int:
        """Return total bytes used by the workspace."""
        workspace = self._workspace_root(run_id, tenant_id)
        if not workspace.exists():
            return 0

        total = 0
        for path in workspace.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _workspace_root(self, run_id: str, tenant_id: str) -> Path:
        return self._base_path / tenant_id / run_id

    async def _remove_temp_files(self, workspace: Path) -> None:
        """Remove temp files and directories from workspace."""
        for pattern in _TEMP_PATTERNS:
            for path in workspace.rglob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    logger.debug("Removed temp artifact: %s", path)
                except OSError as exc:
                    logger.debug("Failed to remove %s: %s", path, exc)
