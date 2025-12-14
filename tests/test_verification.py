import pytest
import torch
from unittest.mock import MagicMock, AsyncMock
from tournament.verification.verifier import SandboxVerifier
from tournament.verification.errors import TokenCountMismatchError, LossMismatchError, LogitsMismatchError, SandboxExecutionError

@pytest.mark.asyncio
async def test_verification_success(mock_sandbox_manager, mock_reference_executor, mock_config):
    # Setup matching outputs
    ref_result = mock_reference_executor.execute()
    
    # Sandbox result matches reference
    sandbox_result = mock_sandbox_manager.run.return_value
    sandbox_result.final_logits = ref_result.final_logits
    sandbox_result.final_loss = ref_result.final_loss
    sandbox_result.total_tokens = ref_result.total_tokens
    
    verifier = SandboxVerifier(mock_sandbox_manager, mock_reference_executor, mock_config)
    result = await verifier.verify_and_benchmark("dummy_path.py")
    
    assert result.success
    assert result.tokens_per_second == 100.0  # 100 tokens / 1.0s
    assert result.total_tokens == 100

@pytest.mark.asyncio
async def test_token_mismatch(mock_sandbox_manager, mock_reference_executor, mock_config):
    ref_result = mock_reference_executor.execute()
    
    # Sandbox result has different tokens
    sandbox_result = mock_sandbox_manager.run.return_value
    sandbox_result.final_logits = ref_result.final_logits
    sandbox_result.final_loss = ref_result.final_loss
    sandbox_result.total_tokens = ref_result.total_tokens + 10 # Mismatch
    
    verifier = SandboxVerifier(mock_sandbox_manager, mock_reference_executor, mock_config)
    result = await verifier.verify_and_benchmark("dummy_path.py")
    
    assert not result.success
    assert result.error_type == "TokenCountMismatchError"

@pytest.mark.asyncio
async def test_output_vector_mismatch(mock_sandbox_manager, mock_reference_executor, mock_config):
    ref_result = mock_reference_executor.execute()
    
    # Sandbox result has different output vectors (outside 1% tolerance)
    sandbox_result = mock_sandbox_manager.run.return_value
    sandbox_result.final_logits = ref_result.final_logits * 2.0  # Way off (100% difference)
    sandbox_result.total_tokens = ref_result.total_tokens
    
    verifier = SandboxVerifier(mock_sandbox_manager, mock_reference_executor, mock_config)
    result = await verifier.verify_and_benchmark("dummy_path.py")
    
    assert not result.success
    assert result.error_type == "LogitsMismatchError"

@pytest.mark.asyncio
async def test_logits_mismatch(mock_sandbox_manager, mock_reference_executor, mock_config):
    ref_result = mock_reference_executor.execute()
    
    # Sandbox result has different logits
    sandbox_result = mock_sandbox_manager.run.return_value
    sandbox_result.final_logits = torch.randn_like(ref_result.final_logits) + 100 # Huge mismatch
    sandbox_result.final_loss = ref_result.final_loss
    sandbox_result.total_tokens = ref_result.total_tokens
    
    verifier = SandboxVerifier(mock_sandbox_manager, mock_reference_executor, mock_config)
    result = await verifier.verify_and_benchmark("dummy_path.py")
    
    assert not result.success
    assert result.error_type == "LogitsMismatchError"

@pytest.mark.asyncio
async def test_sandbox_failure(mock_sandbox_manager, mock_reference_executor, mock_config):
    # Sandbox execution fails
    sandbox_result = mock_sandbox_manager.run.return_value
    sandbox_result.success = False
    sandbox_result.error = "Runtime Error"
    
    verifier = SandboxVerifier(mock_sandbox_manager, mock_reference_executor, mock_config)
    result = await verifier.verify_and_benchmark("dummy_path.py")
    
    assert not result.success
    assert result.error_type == "SandboxExecutionError"

