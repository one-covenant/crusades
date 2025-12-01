"""Verification configuration."""

from dataclasses import dataclass


@dataclass
class VerificationConfig:
    """Configuration for verification tolerances and settings."""

    # Logits comparison tolerances
    logits_atol: float = 1e-3  # Absolute tolerance
    logits_rtol: float = 1e-3  # Relative tolerance

    # Loss comparison tolerance
    loss_tolerance: float = 1e-3

    # Whether to verify model state after training
    verify_model_state: bool = False

    # Deterministic mode settings
    deterministic_mode: bool = True

    # Random seed for reproducibility
    random_seed: int = 42

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationConfig":
        """Create config from dictionary."""
        return cls(
            logits_atol=data.get("logits_atol", 1e-3),
            logits_rtol=data.get("logits_rtol", 1e-3),
            loss_tolerance=data.get("loss_tolerance", 1e-3),
            verify_model_state=data.get("verify_model_state", False),
            deterministic_mode=data.get("deterministic_mode", True),
            random_seed=data.get("random_seed", 42),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "logits_atol": self.logits_atol,
            "logits_rtol": self.logits_rtol,
            "loss_tolerance": self.loss_tolerance,
            "verify_model_state": self.verify_model_state,
            "deterministic_mode": self.deterministic_mode,
            "random_seed": self.random_seed,
        }
