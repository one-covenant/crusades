"""Affinetes integration for Templar Tournament.

R2-Based Architecture:
- Validator owns the evaluation image (templar-eval)
- Miner's code is downloaded from their R2 at evaluation time
- Supports docker (local) and basilica (remote) modes
"""

from .runner import AffinetesRunner, EvaluationResult, R2Info

__all__ = ["AffinetesRunner", "EvaluationResult", "R2Info"]
