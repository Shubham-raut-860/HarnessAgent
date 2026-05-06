"""Eval module for HarnessAgent — dataset, runner, scorers, and reports."""

from harness.eval.datasets import (
    CODE_EVAL_CASES,
    MULTI_AGENT_EVAL_CASES,
    SQL_EVAL_CASES,
    EvalCase,
    EvalDataset,
    MultiAgentEvalCase,
    MultiAgentEvalDataset,
)
from harness.eval.diagnostics import (
    AgentMetricSummary,
    CaseDiagnostic,
    EvalDiagnostics,
    build_diagnostics,
    classify_failure,
)
from harness.eval.runner import EvalReport, EvalRunner
from harness.eval.scorers import (
    ScoreResult,
    score_contains_all,
    score_exact_match,
    score_llm_judge,
    score_sql_equivalence,
    score_success_rate,
)

__all__ = [
    "CODE_EVAL_CASES",
    "MULTI_AGENT_EVAL_CASES",
    "SQL_EVAL_CASES",
    "AgentMetricSummary",
    "CaseDiagnostic",
    "EvalCase",
    "EvalDataset",
    "EvalDiagnostics",
    "EvalReport",
    "EvalRunner",
    "MultiAgentEvalCase",
    "MultiAgentEvalDataset",
    "ScoreResult",
    "build_diagnostics",
    "classify_failure",
    "score_contains_all",
    "score_exact_match",
    "score_llm_judge",
    "score_sql_equivalence",
    "score_success_rate",
]
