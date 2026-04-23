"""Skill system: named, versioned, composable capabilities for agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A named, versioned, composable agent capability.

    Skills bundle a system prompt, required tools, optional tools,
    sub-skills, few-shot examples, and searchable tags.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    system_prompt: str = ""
    required_tools: list[str] = field(default_factory=list)
    optional_tools: list[str] = field(default_factory=list)
    sub_skills: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def all_tools(self) -> list[str]:
        """Return the union of required and optional tools."""
        seen: set[str] = set()
        result: list[str] = []
        for t in self.required_tools + self.optional_tools:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result


class SkillRegistry:
    """Registry for Skill objects with version management and composition."""

    def __init__(self) -> None:
        # { name -> { version -> Skill } }
        self._store: dict[str, dict[str, Skill]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a skill under its name and version."""
        if skill.name not in self._store:
            self._store[skill.name] = {}
        self._store[skill.name][skill.version] = skill
        logger.debug("Registered skill '%s' v%s", skill.name, skill.version)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str, version: str = "latest") -> Skill | None:
        """Return a skill by name and version.

        If version is "latest", the highest semantic version is returned.
        Returns None if not found.
        """
        versions = self._store.get(name)
        if not versions:
            return None
        if version == "latest":
            # Sort by version string; works for simple semver like "1.0.0"
            latest_ver = sorted(versions.keys(), key=_version_key)[-1]
            return versions[latest_ver]
        return versions.get(version)

    def list_for_tags(self, tags: list[str]) -> list[Skill]:
        """Return all latest-version skills that have at least one of the given tags."""
        tag_set = set(tags)
        result: list[Skill] = []
        for name in self._store:
            skill = self.get(name)
            if skill and tag_set.intersection(skill.tags):
                result.append(skill)
        return result

    def all_skills(self) -> list[Skill]:
        """Return all latest-version skills."""
        return [self.get(name) for name in self._store]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, *names: str) -> Skill:
        """Merge multiple skills into a single composite skill.

        System prompts are concatenated; tool lists are unioned;
        examples and tags are merged with deduplication.
        """
        skills = [self.get(n) for n in names]
        missing = [n for n, s in zip(names, skills) if s is None]
        if missing:
            raise KeyError(f"Skills not found: {missing}")

        combined_prompt_parts: list[str] = []
        combined_required: list[str] = []
        combined_optional: list[str] = []
        combined_sub: list[str] = []
        combined_examples: list[dict[str, Any]] = []
        combined_tags: list[str] = []
        seen_required: set[str] = set()
        seen_optional: set[str] = set()
        seen_sub: set[str] = set()
        seen_tags: set[str] = set()

        for skill in skills:
            if skill.system_prompt:
                combined_prompt_parts.append(
                    f"# {skill.name}\n{skill.system_prompt}"
                )
            for t in skill.required_tools:
                if t not in seen_required:
                    seen_required.add(t)
                    combined_required.append(t)
            for t in skill.optional_tools:
                if t not in seen_optional:
                    seen_optional.add(t)
                    combined_optional.append(t)
            for s in skill.sub_skills:
                if s not in seen_sub:
                    seen_sub.add(s)
                    combined_sub.append(s)
            combined_examples.extend(skill.examples)
            for tag in skill.tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    combined_tags.append(tag)

        composed_name = "+".join(names)
        return Skill(
            name=composed_name,
            version="composed",
            description=f"Composition of: {', '.join(names)}",
            system_prompt="\n\n".join(combined_prompt_parts),
            required_tools=combined_required,
            optional_tools=combined_optional,
            sub_skills=combined_sub,
            examples=combined_examples,
            tags=combined_tags,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_tools_available(
        self, skill: Skill, registry: "ToolRegistry"
    ) -> list[str]:
        """Return a list of required tools that are missing from the registry."""
        missing: list[str] = []
        for tool_name in skill.required_tools:
            if registry.get(tool_name) is None:
                missing.append(tool_name)
        return missing


def _version_key(version_str: str) -> tuple[int, ...]:
    """Parse a semver string into a comparable tuple of ints."""
    try:
        return tuple(int(x) for x in version_str.split("."))
    except (ValueError, AttributeError):
        return (0,)


# ---------------------------------------------------------------------------
# Built-in Skills
# ---------------------------------------------------------------------------

SQL_SCHEMA_EXPLORER = Skill(
    name="sql_schema_explorer",
    version="1.0.0",
    description="Explore and understand database schema before writing queries.",
    required_tools=["list_tables", "describe_table", "sample_rows"],
    optional_tools=["execute_sql"],
    system_prompt=(
        "Before writing any query, always:\n"
        "1. Call list_tables to see all available tables.\n"
        "2. Call describe_table for each relevant table to understand column names and types.\n"
        "3. Call sample_rows to understand data shape and value distributions.\n"
        "Never assume column names. Always verify schema before writing SQL."
    ),
    tags=["sql", "database", "schema"],
    examples=[
        {
            "user": "How many orders do we have per customer?",
            "steps": [
                {"tool": "list_tables", "reason": "Discover available tables"},
                {"tool": "describe_table", "args": {"table_name": "orders"}, "reason": "Understand schema"},
                {"tool": "sample_rows", "args": {"table_name": "orders", "n": 3}, "reason": "Understand data"},
                {"tool": "execute_sql", "args": {"sql": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id"}, "reason": "Answer question"},
            ],
        }
    ],
)

CODE_DEBUGGER = Skill(
    name="code_debugger",
    version="1.0.0",
    description="Debug Python code by reading, running, and analyzing errors.",
    required_tools=["read_file", "run_python", "lint_code"],
    optional_tools=["write_file", "apply_patch"],
    system_prompt=(
        "When debugging code:\n"
        "1. Read the file(s) involved using read_file.\n"
        "2. Run the code with run_python and carefully analyze the traceback.\n"
        "3. Identify the root cause — do not just patch symptoms.\n"
        "4. Write the fixed code using write_file or apply_patch.\n"
        "5. Run lint_code to verify no new issues were introduced.\n"
        "6. Re-run the code to confirm the fix.\n"
        "Always explain your reasoning and the root cause."
    ),
    tags=["code", "python", "debugging"],
    examples=[],
)

RESEARCH_SYNTHESIZER = Skill(
    name="research_synthesizer",
    version="1.0.0",
    description=(
        "Synthesize information from multiple sources into structured reports."
    ),
    required_tools=["read_file", "write_file"],
    optional_tools=["list_workspace"],
    system_prompt=(
        "When synthesizing research:\n"
        "1. Read all relevant files from the workspace.\n"
        "2. Identify key themes, contradictions, and gaps.\n"
        "3. Organize findings into a structured format with clear sections.\n"
        "4. Cite specific sources for each claim.\n"
        "5. Write a concise executive summary followed by detailed findings.\n"
        "6. Save the synthesized report to the workspace."
    ),
    tags=["research", "synthesis", "writing"],
    examples=[],
)


def build_default_skill_registry() -> SkillRegistry:
    """Create and populate a SkillRegistry with all built-in skills."""
    registry = SkillRegistry()
    registry.register(SQL_SCHEMA_EXPLORER)
    registry.register(CODE_DEBUGGER)
    registry.register(RESEARCH_SYNTHESIZER)
    return registry
