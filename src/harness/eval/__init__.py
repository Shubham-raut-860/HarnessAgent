"""Eval module for Codex Harness — dataset, runner, scorers, and reports."""

from harness.eval.datasets import (
    EvalCase,
    EvalDataset,
    SQL_EVAL_CASES,
    CODE_EVAL_CASES,
)
from harness.eval.runner import EvalReport, EvalRunner
from harness.eval.scorers import (
    ScoreResult,
    score_exact_match,
    score_contains_all,
    score_llm_judge,
    score_sql_equivalence,
    score_success_rate,
)

__all__ = [
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "EvalRunner",
    "ScoreResult",
    "SQL_EVAL_CASES",
    "CODE_EVAL_CASES",
    "score_exact_match",
    "score_contains_all",
    "score_llm_judge",
    "score_sql_equivalence",
    "score_success_rate",
]
