"""Verification system for validating miner submissions."""

from .config import VerificationConfig
from .errors import (
    LogitsMismatchError,
    LossMismatchError,
    TokenCountMismatchError,
    VerificationError,
)
from .reference import ReferenceExecutor, ReferenceResult
from .verifier import SandboxVerifier, VerificationResult

__all__ = [
    "VerificationConfig",
    "VerificationError",
    "LogitsMismatchError",
    "TokenCountMismatchError",
    "LossMismatchError",
    "ReferenceExecutor",
    "ReferenceResult",
    "SandboxVerifier",
    "VerificationResult",
]
