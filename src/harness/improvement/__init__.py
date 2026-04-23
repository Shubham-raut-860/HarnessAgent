"""Improvement module — Hermes self-healing loop, patch generation, error collection."""

from harness.improvement.error_collector import ErrorCollector, ErrorRecord
from harness.improvement.evaluator import EvalResult, Evaluator, PatchEvaluator
from harness.improvement.hermes import HermesLoop, PatchOutcome
from harness.improvement.patch_generator import Patch, PatchGenerator

__all__ = [
    "ErrorCollector",
    "ErrorRecord",
    "EvalResult",
    "Evaluator",
    "HermesLoop",
    "Patch",
    "PatchEvaluator",
    "PatchGenerator",
    "PatchOutcome",
]
