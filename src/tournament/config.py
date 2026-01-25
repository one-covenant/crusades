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


class DockerConfig(BaseModel):
    """Docker execution settings for validator evaluations.
    
    GPU device options:
    - "all": Use all available GPUs (default)
    - "0": Use only GPU 0
    - "0,1": Use GPUs 0 and 1
    - "none": Disable GPU (CPU only)
    """

    gpu_devices: str = "all"  # "all", "0", "0,1", "none"
    memory_limit: str = "32g"  # Docker memory limit
    shm_size: str = "8g"  # Shared memory size (important for PyTorch)


class HParams(BaseModel):
    """Hyperparameters loaded from hparams.json.
    
    URL-Based Architecture:
    - Miners host train.py at any URL and commit the URL to blockchain
    - Validators download from miner's URL and evaluate via Docker/Basilica
    - All settings are defined in hparams.json (no hardcoded defaults)
    """

    # Network settings
    netuid: int
    
    # Emissions distribution
    burn_rate: float
    burn_uid: int

    # Evaluation settings
    evaluation_runs: int
    eval_steps: int
    eval_timeout: int

    # Benchmark settings
    benchmark_model_name: str
    benchmark_dataset_name: str
    benchmark_dataset_split: str
    benchmark_data_samples: int
    benchmark_train_size: int
    benchmark_master_seed: int
    benchmark_sequence_length: int
    benchmark_batch_size: int

    # Timing settings
    set_weights_interval_seconds: int

    # Commitment settings (timelock encrypted via drand)
    reveal_blocks: int
    min_blocks_between_commits: int
    block_time: int

    # Docker execution settings
    docker: DockerConfig = Field(default_factory=DockerConfig)

    # Verification (nested config with defaults from JSON)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)

    # Storage (for evaluation records - not in hparams.json)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Self:
        """Load hyperparameters from JSON file.
        
        Args:
            path: Path to hparams.json file
            
        Returns:
            HParams instance with all values from JSON
            
        Raises:
            FileNotFoundError: If hparams.json doesn't exist
            ValidationError: If required fields are missing
        """
        if path is None:
            # Default to hparams/hparams.json relative to project root
            # config.py is at src/tournament/config.py, so 3 parents gets to project root
            path = Path(__file__).parent.parent.parent / "hparams" / "hparams.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"hparams.json not found at {path}")

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
