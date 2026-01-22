"""Configuration management for templar-tournament (Chi/Affinetes Architecture)."""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseModel):
    """Storage settings."""

    database_url: str = "sqlite+aiosqlite:///tournament.db"


class VerificationConfig(BaseModel):
    """Verification tolerance settings."""

    output_vector_tolerance: float = 0.02  # 2% aggregate difference allowed
    deterministic_mode: bool = True


class HParams(BaseModel):
    """Hyperparameters loaded from hparams.json.
    
    Chi/Affinetes Architecture:
    - Miners build Docker images and commit to blockchain
    - Validators read commitments and evaluate via Docker/Basilica
    - All settings for evaluation are defined here
    """

    netuid: int = 2
    
    # Emissions distribution (winner-takes-some)
    burn_rate: float = 0.95  # 95% to validator, 5% to winner
    burn_uid: int = 1  # UID that receives burn portion (validator)

    # Evaluation settings
    evaluation_runs: int = 5  # Number of runs per submission (median taken)
    eval_steps: int = 5  # Training steps per evaluation
    eval_timeout: int = 600  # Max seconds per evaluation

    # Benchmark settings - model and data for evaluation
    benchmark_model_name: str = "Qwen/Qwen2.5-7B"
    benchmark_dataset_name: str = "HuggingFaceFW/fineweb"
    benchmark_data_samples: int = 10000  # Number of samples to load
    benchmark_sequence_length: int = 1024
    benchmark_batch_size: int = 8

    # Timing settings
    set_weights_interval_seconds: int = 600  # 10 minutes

    # Commitment settings
    reveal_blocks: int = 100  # Blocks until commitment is revealed
    min_blocks_between_commits: int = 100  # Rate limit: ~20 min between commits

    # Verification
    verification: VerificationConfig = Field(default_factory=VerificationConfig)

    # Storage (for evaluation records)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Self:
        """Load hyperparameters from JSON file."""
        if path is None:
            # Default to hparams/hparams.json relative to project root
            path = Path(__file__).parent.parent.parent.parent / "hparams" / "hparams.json"
        else:
            path = Path(path)

        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)


class Config(BaseSettings):
    """Runtime configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TOURNAMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Bittensor settings
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
     subtensor_network: str = "finney"  # Default to mainnet for production safety

    # Paths
    hparams_path: str = "hparams/hparams.json"

    # Debug
    debug: bool = False


# Global instances (lazy loaded)
_hparams: HParams | None = None
_config: Config | None = None


def get_hparams() -> HParams:
    """Get or create global HParams instance."""
    global _hparams
    if _hparams is None:
        config = get_config()
        _hparams = HParams.load(config.hparams_path)
    return _hparams


def get_config() -> Config:
    """Get or create global Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
