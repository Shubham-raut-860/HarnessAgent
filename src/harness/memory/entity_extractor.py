"""Entity extraction for Graph RAG — three-tier strategy.

Tier 1 (best):   sqlglot AST parsing — when the query looks like SQL
                 Zero false positives, extracts tables and columns precisely.

Tier 2 (good):   LLM extraction — for natural language questions
                 A tiny prompt to an already-available LLM. Uses the cheapest
                 model available. Skipped if no LLM provider is configured.

Tier 3 (fallback): Regex — always available, always last resort.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# SQL keyword detection — if any of these appear, try sqlglot first
_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|CREATE|DROP|"
    r"GROUP\s+BY|ORDER\s+BY|HAVING|UNION|WITH|LIMIT|ON|AND|OR|"
    r"INNER|LEFT|RIGHT|OUTER|CROSS)\b",
    re.IGNORECASE,
)

# English stopwords that are never graph entities
_STOP = {
    "the", "and", "for", "are", "not", "that", "this", "with", "from",
    "have", "all", "was", "been", "can", "will", "how", "what", "where",
    "which", "when", "who", "but", "list", "get", "show", "give", "find",
    "return", "select", "query", "using", "into", "data", "value", "values",
    "count", "total", "number", "average", "each", "per", "between", "last",
    "first", "next", "some", "many", "more", "less", "much", "most", "week",
    "month", "year", "day", "time", "date", "name", "type", "user",
}

# Regex fallback patterns (last resort)
_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")
_QUOTED_RE = re.compile(r'["\']([^"\']{2,})["\']')


def extract_from_sql(query: str) -> list[str]:
    """Parse SQL using sqlglot AST to extract table and column names.

    Returns precise results with zero false positives.
    Falls back to empty list if sqlglot cannot parse the query.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp

        # Try multiple dialects in case the SQL is dialect-specific
        for dialect in (None, "sqlite", "postgres", "mysql"):
            try:
                statements = sqlglot.parse(query, dialect=dialect, error_level=sqlglot.ErrorLevel.IGNORE)
                if not statements:
                    continue

                entities: list[str] = []
                seen: set[str] = set()

                for stmt in statements:
                    if stmt is None:
                        continue
                    # Tables referenced in FROM / JOIN clauses
                    for table in stmt.find_all(exp.Table):
                        name = table.name
                        if name and name.lower() not in _STOP and name not in seen:
                            seen.add(name)
                            entities.append(name)

                    # Column references (useful for column-level graph nodes)
                    for col in stmt.find_all(exp.Column):
                        col_name = col.name
                        if col_name and col_name.lower() not in _STOP and col_name not in seen:
                            seen.add(col_name)
                            entities.append(col_name)

                if entities:
                    return entities

            except Exception:
                continue

    except ImportError:
        logger.debug("sqlglot not installed, falling back to regex")
    except Exception as exc:
        logger.debug("sqlglot extraction failed: %s", exc)

    return []


async def extract_from_nl(query: str, llm_provider: Any | None) -> list[str]:
    """Ask the LLM to extract database entity names from a natural language query.

    Uses a minimal prompt. Returns an empty list if LLM is unavailable or fails.
    This is intentionally cheap — keep the prompt tiny.
    """
    if llm_provider is None:
        return []

    prompt = (
        "Extract database entity names (table names, column names) "
        f"from this query. Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Query: {query}\n\nEntities:"
    )

    try:
        response = await llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        content = response.content.strip()

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()

        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(e) for e in parsed if e and str(e).lower() not in _STOP]

    except Exception as exc:
        logger.debug("LLM entity extraction failed: %s", exc)

    return []


def extract_from_regex(query: str) -> list[str]:
    """Regex-based fallback entity extraction.

    Same as the original implementation. Noisy but always works.
    """
    entities: list[str] = []

    # Quoted strings first (highest confidence)
    for m in _QUOTED_RE.finditer(query):
        entities.append(m.group(1))

    # Generic identifiers, filtered by stopwords
    for m in _IDENTIFIER_RE.finditer(query):
        word = m.group(1)
        if word.lower() not in _STOP:
            entities.append(word)

    seen: set[str] = set()
    return [e for e in entities if not (e in seen or seen.add(e))]  # type: ignore[func-returns-value]


async def extract_entities(
    query: str,
    llm_provider: Any | None = None,
) -> list[str]:
    """Main entry point. Tries sqlglot first, LLM second, regex last.

    Args:
        query:        The user query (SQL or natural language).
        llm_provider: Optional LLM provider for NL extraction.

    Returns:
        Deduplicated list of entity names in confidence order.
    """
    # Tier 1: SQL AST parsing (fast, zero cost, zero false positives)
    if _SQL_KEYWORDS.search(query):
        sql_entities = extract_from_sql(query)
        if sql_entities:
            logger.debug("Entity extraction via sqlglot: %s", sql_entities)
            return sql_entities

    # Tier 2: LLM extraction for natural language
    if llm_provider is not None:
        nl_entities = await extract_from_nl(query, llm_provider)
        if nl_entities:
            logger.debug("Entity extraction via LLM: %s", nl_entities)
            return nl_entities

    # Tier 3: Regex fallback
    regex_entities = extract_from_regex(query)
    logger.debug("Entity extraction via regex: %s", regex_entities)
    return regex_entities
