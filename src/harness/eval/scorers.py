"""Scoring functions for evaluating agent output quality."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Container for a scorer's output.

    Attributes:
        score:   Float in [0.0, 1.0]; higher is better.
        method:  Name of the scoring method used.
        details: Optional human-readable explanation of the score.
    """

    score: float
    method: str
    details: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            self.score = max(0.0, min(1.0, self.score))


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


def score_exact_match(output: str, expected: str) -> float:
    """Return 1.0 if *expected* appears (case-insensitively) inside *output*.

    Args:
        output:   The agent's actual output text.
        expected: The expected substring to search for.

    Returns:
        1.0 if expected is found, 0.0 otherwise.
    """
    if not expected:
        return 1.0
    return 1.0 if expected.lower() in output.lower() else 0.0


def score_contains_all(output: str, expected_keywords: list[str]) -> float:
    """Return the fraction of keywords present in *output* (case-insensitive).

    Args:
        output:            The agent's actual output text.
        expected_keywords: List of keywords to check for.

    Returns:
        Float in [0.0, 1.0] representing how many keywords were found.
    """
    if not expected_keywords:
        return 1.0
    output_lower = output.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
    return found / len(expected_keywords)


def score_success_rate(results: list[Any]) -> float:
    """Return the fraction of AgentResult objects with success=True.

    Args:
        results: List of AgentResult (or any object with a .success bool).

    Returns:
        Float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if getattr(r, "success", False))
    return successes / len(results)


# ---------------------------------------------------------------------------
# SQL equivalence scorer
# ---------------------------------------------------------------------------


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: uppercase keywords, collapse whitespace."""
    sql = sql.strip()
    # Collapse all whitespace to single space
    sql = re.sub(r"\s+", " ", sql)
    # Uppercase SQL keywords
    keywords = [
        "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
        "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN",
        "ON", "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
        "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT",
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "AS", "BY", "LIMIT", "OFFSET", "UNION", "ALL", "EXISTS",
    ]
    for kw in keywords:
        sql = re.sub(r"\b" + kw + r"\b", kw, sql, flags=re.IGNORECASE)
    return sql.strip()


def score_sql_equivalence(sql1: str, sql2: str) -> float:
    """Compare two SQL strings for semantic equivalence.

    Uses sqlparse for AST-level comparison when available, falling back to
    normalized string comparison.

    Scoring:
        1.0  — Exactly equivalent (same normalized form or same AST tokens).
        0.5  — Similar structure (same statement type and overlapping tokens).
        0.0  — Different or incomparable.

    Args:
        sql1: First SQL string.
        sql2: Second SQL string.

    Returns:
        Float in {0.0, 0.5, 1.0}.
    """
    if not sql1 or not sql2:
        return 0.0

    norm1 = _normalize_sql(sql1)
    norm2 = _normalize_sql(sql2)

    if norm1 == norm2:
        return 1.0

    try:
        import sqlparse  # type: ignore

        parsed1 = sqlparse.parse(sql1)
        parsed2 = sqlparse.parse(sql2)

        if not parsed1 or not parsed2:
            return 0.0

        stmt1 = parsed1[0]
        stmt2 = parsed2[0]

        # Compare statement type
        type1 = stmt1.get_type()
        type2 = stmt2.get_type()
        if type1 != type2:
            return 0.0

        # Compare flattened token values (stripped of whitespace)
        def token_values(stmt) -> list[str]:
            return [
                t.normalized.upper()
                for t in stmt.flatten()
                if not t.is_whitespace
            ]

        tokens1 = token_values(stmt1)
        tokens2 = token_values(stmt2)

        if tokens1 == tokens2:
            return 1.0

        # Calculate token overlap
        set1 = set(tokens1)
        set2 = set(tokens2)
        if not set1 or not set2:
            return 0.0

        overlap = len(set1 & set2) / max(len(set1), len(set2))
        if overlap >= 0.8:
            return 0.5

        return 0.0

    except ImportError:
        logger.debug("sqlparse not available; falling back to string comparison")

    # Fallback: word-overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if not words1 or not words2:
        return 0.0
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    if overlap >= 0.9:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# LLM-based judge scorer
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are an evaluation judge for an AI agent system.

## Task
{task}

## Agent Output
{output}

## Expected Output (may be null)
{expected}

## Rubric
{rubric}

Evaluate the agent output on a scale from 0.0 to 1.0, where:
- 1.0 = Perfect: fully correct, complete, and well-reasoned
- 0.8 = Good: mostly correct with minor issues
- 0.6 = Acceptable: partially correct, some important parts missing
- 0.4 = Poor: shows some understanding but significant problems
- 0.2 = Very poor: mostly incorrect but shows minimal relevance
- 0.0 = Completely wrong or irrelevant

Respond ONLY with a JSON object in this exact format:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

_DEFAULT_RUBRIC = (
    "Evaluate correctness, completeness, and relevance to the task. "
    "If expected output is provided, check that the key content is present."
)


async def score_llm_judge(
    task: str,
    output: str,
    expected: str | None,
    llm_provider: Any,
    rubric: str | None = None,
) -> ScoreResult:
    """Use an LLM to evaluate agent output quality on a 0-1 scale.

    Constructs a structured judge prompt, calls the LLM provider, and parses
    the response JSON to extract score and reasoning.

    Args:
        task:         The original task given to the agent.
        output:       The agent's output to be evaluated.
        expected:     Optional ground-truth expected output (can be None).
        llm_provider: An LLMProvider-compatible object with a .complete() method.
        rubric:       Optional custom evaluation rubric.  Falls back to default.

    Returns:
        ScoreResult with score, method="llm_judge", and reasoning in details.

    Raises:
        Exception: Propagated from LLM provider on connectivity failure.
                   On parse failure, returns score=0.5 (uncertain).
    """
    effective_rubric = rubric or _DEFAULT_RUBRIC
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        task=task,
        output=output,
        expected=expected or "(none provided)",
        rubric=effective_rubric,
    )

    try:
        response = await llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            system="You are a strict but fair AI evaluation judge. Always respond with valid JSON.",
        )
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
            raw = raw.strip()

        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        reasoning = parsed.get("reasoning", "")
        score = max(0.0, min(1.0, score))

        return ScoreResult(
            score=score,
            method="llm_judge",
            details=reasoning,
        )

    except json.JSONDecodeError as exc:
        logger.warning("LLM judge response was not valid JSON: %s", exc)
        # Fallback: try to extract a number from the response
        try:
            raw_text = response.content if hasattr(response, "content") else ""
            match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", raw_text)
            if match:
                score = float(match.group(1))
                return ScoreResult(
                    score=max(0.0, min(1.0, score)),
                    method="llm_judge",
                    details=f"Extracted from non-JSON response: {raw_text[:200]}",
                )
        except Exception:
            pass
        return ScoreResult(score=0.5, method="llm_judge", details="Parse error; defaulting to 0.5")

    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return ScoreResult(
            score=0.5,
            method="llm_judge",
            details=f"LLM judge error: {exc}",
        )
