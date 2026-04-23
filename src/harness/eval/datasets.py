"""EvalDataset and EvalCase definitions for Codex Harness evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalCase:
    """A single evaluation case with task, optional expected output, and tags.

    Attributes:
        case_id:         Unique identifier for the case.
        agent_type:      Which agent to run this case against.
        task:            The user task string to submit.
        expected_output: Ground-truth substring or None if success-only check.
        metadata:        Arbitrary extra data (e.g. difficulty, dataset source).
        tags:            List of classification tags for filtering.
    """

    case_id: str
    agent_type: str
    task: str
    expected_output: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSONL storage."""
        return {
            "case_id": self.case_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "expected_output": self.expected_output,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        """Deserialise from a dict (e.g. loaded from JSONL)."""
        return cls(
            case_id=data["case_id"],
            agent_type=data["agent_type"],
            task=data["task"],
            expected_output=data.get("expected_output"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


@dataclass
class EvalDataset:
    """A named collection of EvalCases for a specific agent type.

    Attributes:
        name:       Human-readable dataset name.
        agent_type: The agent type all cases belong to.
        cases:      Ordered list of EvalCase instances.
    """

    name: str
    agent_type: str
    cases: list[EvalCase]

    # ------------------------------------------------------------------
    # Filtering and sampling
    # ------------------------------------------------------------------

    def filter(
        self,
        tags: Optional[list[str]] = None,
        n: Optional[int] = None,
    ) -> "EvalDataset":
        """Return a new EvalDataset containing only cases matching the given tags.

        Args:
            tags: If provided, only include cases that have at least one
                  matching tag.  If None, all cases pass.
            n:    If provided, cap the result at n cases (preserves order).

        Returns:
            A new EvalDataset with the filtered subset of cases.
        """
        filtered = self.cases
        if tags:
            tag_set = set(tags)
            filtered = [c for c in filtered if tag_set.intersection(c.tags)]
        if n is not None:
            filtered = filtered[:n]
        return EvalDataset(name=self.name, agent_type=self.agent_type, cases=filtered)

    def sample(self, n: int) -> "EvalDataset":
        """Return a new EvalDataset with n cases randomly sampled without replacement.

        Args:
            n: Number of cases to sample; clamped to len(self.cases).

        Returns:
            A new EvalDataset with the sampled cases in random order.
        """
        k = min(n, len(self.cases))
        sampled = random.sample(self.cases, k)
        return EvalDataset(name=self.name, agent_type=self.agent_type, cases=sampled)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str) -> "EvalDataset":
        """Load a dataset from a JSONL file.

        Each line must be a JSON object matching the EvalCase schema.
        The first case's agent_type determines the dataset's agent_type.
        The file stem is used as the dataset name.

        Args:
            path: Filesystem path to the .jsonl file.

        Returns:
            An EvalDataset populated from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If any line is invalid JSON.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"EvalDataset file not found: {path}")

        cases: list[EvalCase] = []
        with p.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cases.append(EvalCase.from_dict(data))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_num}: {exc}"
                    ) from exc

        agent_type = cases[0].agent_type if cases else "unknown"
        return cls(name=p.stem, agent_type=agent_type, cases=cases)

    def to_jsonl(self, path: str) -> None:
        """Persist the dataset to a JSONL file (one EvalCase per line).

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __repr__(self) -> str:
        return (
            f"EvalDataset(name={self.name!r}, agent_type={self.agent_type!r}, "
            f"cases={len(self.cases)})"
        )


# ---------------------------------------------------------------------------
# Built-in minimal datasets
# ---------------------------------------------------------------------------

SQL_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="sql_01",
        agent_type="sql",
        task="List all tables in the database",
        tags=["basic"],
    ),
    EvalCase(
        case_id="sql_02",
        agent_type="sql",
        task="Count rows in each table",
        tags=["basic", "count"],
    ),
    EvalCase(
        case_id="sql_03",
        agent_type="sql",
        task="Show the schema for the users table",
        tags=["schema"],
    ),
    EvalCase(
        case_id="sql_04",
        agent_type="sql",
        task="Find users who placed more than 5 orders",
        expected_output="SELECT",
        tags=["join", "aggregate"],
    ),
    EvalCase(
        case_id="sql_05",
        agent_type="sql",
        task="Calculate total revenue per month",
        expected_output="GROUP BY",
        tags=["aggregate", "date"],
    ),
]

CODE_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="code_01",
        agent_type="code",
        task="Write a Python function to reverse a string and test it",
        expected_output="def reverse_string",
        tags=["basic"],
    ),
    EvalCase(
        case_id="code_02",
        agent_type="code",
        task="Debug this code: def add(a,b): return a-b",
        tags=["debug"],
    ),
    EvalCase(
        case_id="code_03",
        agent_type="code",
        task="Write a function that checks if a number is prime",
        expected_output="def is_prime",
        tags=["basic", "math"],
    ),
    EvalCase(
        case_id="code_04",
        agent_type="code",
        task="Implement a binary search algorithm with tests",
        expected_output="def binary_search",
        tags=["algorithm", "search"],
    ),
]
