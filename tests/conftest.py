from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from tournament.sandbox.manager import SandboxManager, SandboxResult
from tournament.schemas import BenchmarkConfig
from tournament.verification.config import VerificationConfig
from tournament.verification.reference import ReferenceExecutor, ReferenceResult


@pytest.fixture
def mock_config():
    return VerificationConfig(
        output_vector_tolerance=0.01,  # 1% tolerance
    )

@pytest.fixture
def mock_benchmark_config():
    return BenchmarkConfig(
        batch_size=1,
        sequence_length=10,
        num_steps=10
    )

@pytest.fixture
def mock_reference_executor():
    executor = MagicMock(spec=ReferenceExecutor)
    # Mock result
    result = ReferenceResult(
        final_logits=torch.randn(1, 10),
        total_tokens=100,
        final_loss=0.5,
        initial_state={}
    )
    executor.execute.return_value = result
    executor.save_reference_outputs.return_value = {}
    return executor

@pytest.fixture
def mock_sandbox_manager():
    manager = AsyncMock(spec=SandboxManager)
    # Mock result
    result = SandboxResult(
        success=True,
        total_tokens=100,
        tokens_per_second=100.0,
        wall_time_seconds=1.0,
        stdout="Success",
        stderr="",
        exit_code=0
    )
    # Add attributes that might be dynamically added in real execution
    result.final_logits = torch.randn(1, 10) # Different from reference initially
    result.final_loss = 0.5

    manager.run.return_value = result
    return manager

