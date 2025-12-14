"""
Tests for the Tournament Scoring System

These tests verify the core verification logic:
1. Token count matching (exact)
2. Loss matching (within tolerance)
3. Logits matching (within tolerance)
4. TPS calculation
"""

import pytest
import torch
from unittest.mock import MagicMock, AsyncMock

from tournament.verification.verifier import SandboxVerifier
from tournament.verification.config import VerificationConfig
from tournament.verification.errors import (
    TokenCountMismatchError,
    LossMismatchError, 
    LogitsMismatchError,
)
from tournament.core.protocols import SandboxResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def verification_config():
    """Standard verification config with production tolerances."""
    return VerificationConfig(
        output_vector_tolerance=0.01,  # 1% aggregate tolerance
    )


@pytest.fixture
def reference_result():
    """Simulated reference execution result."""
    class RefResult:
        final_logits = torch.randn(2, 127, 32000)  # (batch, seq, vocab)
        total_tokens = 1280
        final_loss = 2.5
        initial_state = {}
    return RefResult()


@pytest.fixture
def mock_reference_executor(reference_result):
    """Mock reference executor that returns consistent results."""
    executor = MagicMock()
    executor.execute.return_value = reference_result
    executor.save_reference_outputs.return_value = {}
    return executor


@pytest.fixture
def create_sandbox_result(reference_result):
    """Factory to create sandbox results with customizable differences."""
    def _create(
        token_diff: int = 0,
        loss_diff: float = 0.0,
        logits_diff: float = 0.0,
        success: bool = True,
    ):
        result = SandboxResult(
            success=success,
            tokens_per_second=1280.0,
            total_tokens=reference_result.total_tokens + token_diff,
            wall_time_seconds=1.0,
            exit_code=0 if success else 1,
            stdout="",
            stderr="",
            error=None if success else "Execution failed",
        )
        # Add verification fields
        result.final_loss = reference_result.final_loss + loss_diff
        result.final_logits = reference_result.final_logits + logits_diff
        return result
    return _create


# ============================================================================
# TOKEN COUNT TESTS
# ============================================================================

class TestTokenCountMatching:
    """Tests for exact token count verification."""

    @pytest.mark.asyncio
    async def test_exact_token_match_passes(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Token count must match exactly - same count passes."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(token_diff=0)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True
        assert result.total_tokens == 1280

    @pytest.mark.asyncio
    async def test_token_count_off_by_one_fails(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Even 1 token difference should fail."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(token_diff=1)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "TokenCountMismatchError"

    @pytest.mark.asyncio
    async def test_token_count_off_by_batch_fails(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Missing a whole batch of tokens should fail."""
        mock_sandbox = AsyncMock()
        # Missing one batch worth (batch_size=2, seq_len=128 = 256 tokens)
        mock_sandbox.run.return_value = create_sandbox_result(token_diff=-256)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "TokenCountMismatchError"

    @pytest.mark.asyncio
    async def test_extra_tokens_fails(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Extra tokens (cheating attempt) should fail."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(token_diff=1000)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "TokenCountMismatchError"


# ============================================================================
# LOSS MATCHING TESTS
# ============================================================================

class TestOutputVectorMatching:
    """Tests for output vector verification with aggregate tolerance."""

    @pytest.mark.asyncio
    async def test_exact_output_match_passes(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Exact output vector match passes."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.0)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_small_output_difference_passes(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Small output differences (under 1% aggregate) pass."""
        mock_sandbox = AsyncMock()
        # Small difference that's under 1% aggregate
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.001)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_output_at_tolerance_boundary_passes(
        self, mock_reference_executor, verification_config, reference_result, create_sandbox_result
    ):
        """Output difference at 1% boundary passes."""
        mock_sandbox = AsyncMock()
        
        # Create result with small difference (under 1%)
        result = create_sandbox_result(logits_diff=0.005)
        mock_sandbox.run.return_value = result
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        verify_result = await verifier.verify_and_benchmark("dummy.py")
        
        # Should pass (0.005 is much less than 1% aggregate)
        assert verify_result.success is True

    @pytest.mark.asyncio
    async def test_output_exceeds_tolerance_fails(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Output difference exceeding 1% aggregate fails."""
        mock_sandbox = AsyncMock()
        # Large difference exceeding 1%
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.1)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "LogitsMismatchError"

    @pytest.mark.asyncio
    async def test_large_output_difference_fails(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Large output difference (cheating attempt) fails."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=1.0)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "LogitsMismatchError"


# ============================================================================
# LOGITS MATCHING TESTS
# ============================================================================

class TestLogitsMatching:
    """Tests for logits tensor verification within tolerance."""

    @pytest.mark.asyncio
    async def test_exact_logits_match_passes(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Exact logits match passes."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.0)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_logits_within_tolerance_passes(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Small logits difference within tolerance passes."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.0005)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_logits_exceeds_tolerance_fails(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Logits difference exceeding tolerance fails."""
        mock_sandbox = AsyncMock()
        # Large difference that exceeds tolerance
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.1)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "LogitsMismatchError"

    @pytest.mark.asyncio
    async def test_random_logits_fails(
        self, mock_reference_executor, verification_config, reference_result
    ):
        """Completely random logits (cheating) fails."""
        mock_sandbox = AsyncMock()
        
        result = SandboxResult(
            success=True,
            tokens_per_second=1280.0,
            total_tokens=reference_result.total_tokens,
            wall_time_seconds=1.0,
            exit_code=0,
            stdout="",
            stderr="",
        )
        result.final_loss = reference_result.final_loss
        # Completely different random logits
        result.final_logits = torch.randn_like(reference_result.final_logits) * 100
        mock_sandbox.run.return_value = result
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        verify_result = await verifier.verify_and_benchmark("dummy.py")
        
        assert verify_result.success is False
        assert verify_result.error_type == "LogitsMismatchError"


# ============================================================================
# TPS CALCULATION TESTS
# ============================================================================

class TestTPSCalculation:
    """Tests for Tokens Per Second calculation."""

    @pytest.mark.asyncio
    async def test_tps_calculation_correct(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """TPS = total_tokens / wall_time_seconds."""
        mock_sandbox = AsyncMock()
        result = create_sandbox_result()
        result.total_tokens = 10000
        result.wall_time_seconds = 2.0  # TPS should be 5000
        mock_sandbox.run.return_value = result
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        # Override reference to match tokens
        mock_reference_executor.execute.return_value.total_tokens = 10000
        
        verify_result = await verifier.verify_and_benchmark("dummy.py")
        
        assert verify_result.success is True
        assert verify_result.tokens_per_second == 5000.0

    @pytest.mark.asyncio
    async def test_tps_with_fast_execution(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Fast execution = high TPS."""
        mock_sandbox = AsyncMock()
        result = create_sandbox_result()
        result.total_tokens = 100000
        result.wall_time_seconds = 0.5  # TPS should be 200000
        mock_sandbox.run.return_value = result
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        mock_reference_executor.execute.return_value.total_tokens = 100000
        
        verify_result = await verifier.verify_and_benchmark("dummy.py")
        
        assert verify_result.success is True
        assert verify_result.tokens_per_second == 200000.0

    @pytest.mark.asyncio
    async def test_tps_with_slow_execution(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Slow execution = low TPS."""
        mock_sandbox = AsyncMock()
        result = create_sandbox_result()
        result.total_tokens = 1000
        result.wall_time_seconds = 10.0  # TPS should be 100
        mock_sandbox.run.return_value = result
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        mock_reference_executor.execute.return_value.total_tokens = 1000
        
        verify_result = await verifier.verify_and_benchmark("dummy.py")
        
        assert verify_result.success is True
        assert verify_result.tokens_per_second == 100.0


# ============================================================================
# COMBINED VERIFICATION TESTS
# ============================================================================

class TestCombinedVerification:
    """Tests for the complete verification flow."""

    @pytest.mark.asyncio
    async def test_all_checks_pass(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """When all checks pass, verification succeeds."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(
            token_diff=0,
            loss_diff=0.0,
            logits_diff=0.0,
        )
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_token_check_fails_first(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Token count is checked first - fails before other checks."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(
            token_diff=100,  # Will fail
            loss_diff=0.0,   # Would pass
            logits_diff=0.0, # Would pass
        )
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "TokenCountMismatchError"

    @pytest.mark.asyncio
    async def test_sandbox_failure_handled(
        self, mock_reference_executor, verification_config, create_sandbox_result
    ):
        """Sandbox execution failure is handled gracefully."""
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(success=False)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, verification_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False
        assert result.error_type == "SandboxExecutionError"


# ============================================================================
# TOLERANCE CONFIGURATION TESTS
# ============================================================================

class TestToleranceConfiguration:
    """Tests for different tolerance settings."""

    @pytest.mark.asyncio
    async def test_strict_tolerance_rejects_small_diff(self, mock_reference_executor, create_sandbox_result):
        """Stricter tolerance (0.1%) rejects small differences."""
        strict_config = VerificationConfig(
            output_vector_tolerance=0.001,  # 0.1% - very strict
        )
        
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.005)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, strict_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is False

    @pytest.mark.asyncio
    async def test_loose_tolerance_accepts_larger_diff(self, mock_reference_executor, create_sandbox_result):
        """Looser tolerance (10%) accepts larger differences."""
        loose_config = VerificationConfig(
            output_vector_tolerance=0.1,  # 10% - very loose
        )
        
        mock_sandbox = AsyncMock()
        mock_sandbox.run.return_value = create_sandbox_result(logits_diff=0.05)
        
        verifier = SandboxVerifier(mock_sandbox, mock_reference_executor, loose_config)
        result = await verifier.verify_and_benchmark("dummy.py")
        
        assert result.success is True

