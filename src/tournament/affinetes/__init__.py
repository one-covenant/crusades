"""Affinetes integration for Templar Tournament.

This module provides a bridge to affinetes for running evaluations.
Supports two modes:
1. docker - Run locally using Docker (for testing)
2. basilica - Run remotely on Basilica GPUs (production)
"""

from .runner import AffinetesRunner, EvaluationResult

__all__ = ["AffinetesRunner", "EvaluationResult"]

