"""Affinetes integration for Templar Crusades.

URL-Based Architecture:
- Validator owns the evaluation image (templar-eval)
- Miner's code is downloaded from their committed URL
- Supports docker (local) and basilica (remote) modes
"""

from .runner import (
    AffinetesRunner,
    BasilicaDeploymentContext,
    EvaluationResult,
    cleanup_stale_eval_containers,
)

__all__ = [
    "AffinetesRunner",
    "BasilicaDeploymentContext",
    "EvaluationResult",
    "cleanup_stale_eval_containers",
]
