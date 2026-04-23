"""PromptManager — high-level interface with in-memory caching."""

from __future__ import annotations

import logging
from typing import Any, Optional

from harness.prompts.schemas import PromptVersion
from harness.prompts.store import PromptStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default hardcoded prompts (fallback when no version is stored)
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS: dict[str, str] = {
    "sql": """\
You are an expert SQL analyst. Your job is to help users query databases accurately and efficiently.

Guidelines:
- Always start by understanding the database schema using the list_tables and describe_table tools.
- Write clear, well-formatted SQL queries.
- Prefer SELECT statements; ask for explicit confirmation before executing DML (INSERT/UPDATE/DELETE).
- If a query might return many rows, add a LIMIT clause.
- Explain your reasoning before executing queries.
- Handle errors gracefully and try alternative approaches if a query fails.
- Always report query results clearly, with column names and row counts.
""",
    "code": """\
You are an expert software engineer. Your job is to write, debug, and test Python code.

Guidelines:
- Read the task carefully and plan your approach before writing code.
- Write clean, well-documented code following PEP 8.
- Always test the code you write using the execute_python tool.
- If code has errors, debug them systematically: read the error, identify the root cause, fix it.
- Run the linter (flake8) on completed code.
- Write meaningful docstrings and type annotations.
- Prefer simple, readable solutions over clever ones.
""",
    "research": """\
You are an expert research analyst. Your job is to research topics thoroughly and produce accurate reports.

Guidelines:
- Use available memory and knowledge graph tools to retrieve relevant context first.
- Cross-reference multiple sources when possible.
- Clearly distinguish between established facts and inferences.
- Structure your output with clear sections, bullet points, and citations.
- Flag uncertainty explicitly: "It appears that...", "This is unclear, but..."
- Keep summaries concise; provide details in appendices when needed.
""",
    "orchestrator": """\
You are an AI orchestrator that coordinates specialized sub-agents to solve complex tasks.

Guidelines:
- Break down complex tasks into clear sub-tasks, each assigned to the most appropriate agent.
- Monitor sub-agent progress and handle failures gracefully.
- Synthesize results from multiple agents into a coherent final response.
- Escalate to human-in-the-loop (HITL) when you encounter ambiguity or high-risk decisions.
- Track task progress and update the user on status.
""",
}


class PromptManager:
    """High-level prompt interface with in-memory caching.

    Wraps a PromptStore and adds:
    - In-process cache per agent_type (cleared on promote/apply_patch).
    - Fallback to hardcoded default prompts.
    - One-line patch application.
    - Default version initialization.

    Args:
        store: A PromptStore instance for persistent storage.
    """

    def __init__(self, store: PromptStore) -> None:
        self._store = store
        self._cache: dict[str, str] = {}  # agent_type -> prompt content

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    async def get_prompt(self, agent_type: str) -> str:
        """Return the active prompt content for *agent_type*.

        Checks the in-memory cache first, then Redis, then falls back to
        the hardcoded default.

        Args:
            agent_type: The agent type to fetch the prompt for.

        Returns:
            The prompt string (never empty; guaranteed to have a fallback).
        """
        if agent_type in self._cache:
            return self._cache[agent_type]

        version = await self._store.get_active(agent_type)
        if version is not None:
            self._cache[agent_type] = version.content
            return version.content

        # Fall back to hardcoded default
        default = _DEFAULT_PROMPTS.get(agent_type)
        if default:
            logger.debug(
                "No active prompt in store for agent_type=%s; using default",
                agent_type,
            )
            return default

        # Generic fallback
        generic = (
            f"You are a helpful AI assistant specialized in {agent_type} tasks. "
            "Complete the given task accurately and thoroughly."
        )
        logger.warning(
            "No prompt found for agent_type=%s; using generic fallback", agent_type
        )
        return generic

    async def get_version(self, agent_type: str) -> Optional[PromptVersion]:
        """Return the active PromptVersion for *agent_type*, or None.

        Args:
            agent_type: The agent type to look up.

        Returns:
            PromptVersion if an active version exists, else None.
        """
        return await self._store.get_active(agent_type)

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------

    async def apply_patch(self, patch: Any) -> PromptVersion:
        """Apply a Hermes patch to the current active prompt.

        Supported patch operations (patch.op):
            - ``append``:  Append patch.value to the end of the prompt.
            - ``prepend``: Prepend patch.value to the beginning.
            - ``replace``: Replace patch.path substring with patch.value.
            - ``remove``:  Remove patch.path substring from the prompt.
            - ``set``:     Replace the entire prompt with patch.value.

        Creates a new PromptVersion and promotes it.  Clears the cache.

        Args:
            patch: A Patch dataclass (harness.improvement.patch_generator).

        Returns:
            The newly created and promoted PromptVersion.

        Raises:
            ValueError: If the patch operation is unknown.
        """
        agent_type = getattr(patch, "agent_type", "")
        op = getattr(patch, "op", "").lower()
        path = getattr(patch, "path", "") or ""
        value = getattr(patch, "value", "") or ""
        patch_id = getattr(patch, "patch_id", None)

        # Get current content
        current_content = await self.get_prompt(agent_type)

        # Apply the operation
        if op == "append":
            new_content = current_content.rstrip() + "\n\n" + value.strip()
        elif op == "prepend":
            new_content = value.strip() + "\n\n" + current_content.lstrip()
        elif op == "replace":
            if not path:
                raise ValueError("Patch op='replace' requires a non-empty path (substring to replace)")
            new_content = current_content.replace(path, value, 1)
        elif op == "remove":
            if not path:
                raise ValueError("Patch op='remove' requires a non-empty path (substring to remove)")
            new_content = current_content.replace(path, "", 1)
        elif op == "set":
            new_content = value
        else:
            raise ValueError(
                f"Unknown patch operation: {op!r}. "
                "Supported: append, prepend, replace, remove, set"
            )

        # Create new version
        new_version = await self._store.create_version(
            agent_type=agent_type,
            content=new_content,
            patch_id=patch_id,
            created_by="hermes",
            tags=["auto-generated", f"patch:{patch_id}" if patch_id else "patch"],
            metadata={
                "patch_op": op,
                "patch_rationale": getattr(patch, "rationale", ""),
                "based_on_errors": getattr(patch, "based_on_errors", []),
            },
        )

        # Promote new version and clear cache
        await self._store.promote(new_version.version_id)
        self._clear_cache(agent_type)

        logger.info(
            "Applied patch %s to agent_type=%s (new version_id=%s)",
            patch_id,
            agent_type,
            new_version.version_id[:8],
        )
        return new_version

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize_defaults(self) -> None:
        """Create initial prompt versions for known agent types if none exist.

        Idempotent — only creates versions when the agent_type has no stored
        versions at all.
        """
        for agent_type, content in _DEFAULT_PROMPTS.items():
            existing = await self._store.list_versions(agent_type, limit=1)
            if existing:
                logger.debug(
                    "Default prompt already exists for agent_type=%s; skipping",
                    agent_type,
                )
                continue

            version = await self._store.create_version(
                agent_type=agent_type,
                content=content,
                created_by="system",
                tags=["default"],
                metadata={"source": "initialize_defaults"},
            )
            await self._store.promote(version.version_id)
            logger.info(
                "Initialized default prompt for agent_type=%s (version_id=%s)",
                agent_type,
                version.version_id[:8],
            )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _clear_cache(self, agent_type: Optional[str] = None) -> None:
        """Evict cache entries for *agent_type* (or all if None)."""
        if agent_type is not None:
            self._cache.pop(agent_type, None)
        else:
            self._cache.clear()

    async def promote(self, version_id: str) -> PromptVersion:
        """Promote a version and clear the cache for its agent_type.

        Args:
            version_id: The version to promote.

        Returns:
            The promoted PromptVersion.
        """
        version = await self._store.promote(version_id)
        self._clear_cache(version.agent_type)
        return version

    async def rollback(self, agent_type: str, steps: int = 1) -> PromptVersion:
        """Roll back to N-steps-prior version and clear cache.

        Args:
            agent_type: The agent type to roll back.
            steps:      Number of versions to step back.

        Returns:
            The newly active PromptVersion.
        """
        version = await self._store.rollback(agent_type, steps=steps)
        self._clear_cache(agent_type)
        return version

    async def list_versions(
        self, agent_type: str, limit: int = 20
    ) -> list[PromptVersion]:
        """List versions for an agent type, newest first.

        Args:
            agent_type: The agent type to list.
            limit:      Maximum number of versions to return.

        Returns:
            List of PromptVersion objects.
        """
        return await self._store.list_versions(agent_type, limit=limit)
