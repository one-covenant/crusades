"""Verification configuration."""

from dataclasses import dataclass


@dataclass
class VerificationConfig:
    """Configuration for verification tolerances and settings."""

    # Output vector comparison tolerance (10% = 0.10)
    # Aggregate difference across all dimensions must be under this threshold
    # Note: GPU variance and BF16 precision can cause ~5-10% differences
    output_vector_tolerance: float = 0.10  # 10% aggregate difference allowed

    # Whether to verify model state after training
    verify_model_state: bool = False

    # Deterministic mode settings
    deterministic_mode: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationConfig":
        """Create config from dictionary."""
        return cls(
            output_vector_tolerance=data.get("output_vector_tolerance", 0.10),
            verify_model_state=data.get("verify_model_state", False),
            deterministic_mode=data.get("deterministic_mode", True),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "output_vector_tolerance": self.output_vector_tolerance,
            "verify_model_state": self.verify_model_state,
            "deterministic_mode": self.deterministic_mode,
        }
