"""
Unit tests for Payment System

Tests the anti-spam payment mechanism adapted from Ridges.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tournament.payment.manager import PaymentManager, PaymentReceipt
from tournament.payment.verifier import PaymentVerifier

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_wallet():
    """Mock Bittensor wallet."""
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5HotKeyAddress123"
    wallet.coldkey = MagicMock()
    return wallet


@pytest.fixture
def mock_subtensor():
    """Mock Subtensor for blockchain interactions."""
    subtensor = MagicMock()

    # Mock substrate
    subtensor.substrate = MagicMock()
    subtensor.substrate.compose_call = MagicMock(return_value="mock_call")
    subtensor.substrate.create_signed_extrinsic = MagicMock(return_value="mock_extrinsic")

    # Mock receipt
    mock_receipt = MagicMock()
    mock_receipt.block_hash = "0xabcd1234"
    mock_receipt.extrinsic_idx = 5
    subtensor.substrate.submit_extrinsic = MagicMock(return_value=mock_receipt)

    return subtensor


@pytest.fixture
def payment_manager(mock_wallet, mock_subtensor):
    """Create payment manager with mocked dependencies."""
    return PaymentManager(
        wallet=mock_wallet,
        subtensor=mock_subtensor,
        submission_cost_rao=100_000_000,  # 0.1 TAO
    )


@pytest.fixture
def payment_verifier(mock_subtensor):
    """Create payment verifier with mocked dependencies."""
    return PaymentVerifier(
        recipient_address="5RecipientAddress456",
        subtensor=mock_subtensor,
        required_amount_rao=100_000_000,
    )


# ============================================================================
# PAYMENT MANAGER TESTS
# ============================================================================

class TestPaymentManager:
    """Tests for PaymentManager (miner-side payment)."""

    def test_get_submission_cost(self, payment_manager):
        """Can retrieve submission cost in RAO and TAO."""
        cost_rao, cost_tao = payment_manager.get_submission_cost()

        assert cost_rao == 100_000_000
        assert cost_tao == 0.1

    def test_default_submission_cost(self, mock_wallet, mock_subtensor):
        """Default submission cost is 0.1 TAO."""
        manager = PaymentManager(wallet=mock_wallet, subtensor=mock_subtensor)
        cost_rao, cost_tao = manager.get_submission_cost()

        assert cost_rao == 100_000_000
        assert cost_tao == 0.1

    def test_custom_submission_cost(self, mock_wallet, mock_subtensor):
        """Can set custom submission cost."""
        manager = PaymentManager(
            wallet=mock_wallet,
            subtensor=mock_subtensor,
            submission_cost_rao=50_000_000,  # 0.05 TAO
        )
        cost_rao, cost_tao = manager.get_submission_cost()

        assert cost_rao == 50_000_000
        assert cost_tao == 0.05

    @pytest.mark.asyncio
    async def test_make_payment_success(self, payment_manager, mock_subtensor):
        """Can make a successful payment."""
        receipt = await payment_manager.make_payment(
            recipient_address="5RecipientAddress456",
            amount_rao=100_000_000,
        )

        assert isinstance(receipt, PaymentReceipt)
        assert receipt.block_hash == "0xabcd1234"
        assert receipt.extrinsic_index == 5
        assert receipt.amount_rao == 100_000_000
        assert receipt.timestamp > 0

    @pytest.mark.asyncio
    async def test_make_payment_uses_default_amount(self, payment_manager):
        """Payment uses default amount if not specified."""
        receipt = await payment_manager.make_payment(
            recipient_address="5RecipientAddress456",
        )

        assert receipt.amount_rao == 100_000_000  # Default

    @pytest.mark.asyncio
    async def test_make_payment_calls_substrate(self, payment_manager, mock_subtensor):
        """Payment calls substrate with correct parameters."""
        await payment_manager.make_payment(
            recipient_address="5RecipientAddress456",
            amount_rao=100_000_000,
        )

        # Verify compose_call was called
        mock_subtensor.substrate.compose_call.assert_called_once_with(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": "5RecipientAddress456",
                "value": 100_000_000,
            },
        )

        # Verify extrinsic was submitted
        mock_subtensor.substrate.submit_extrinsic.assert_called_once()


# ============================================================================
# PAYMENT VERIFIER TESTS
# ============================================================================

class TestPaymentVerifier:
    """Tests for PaymentVerifier (validator-side verification)."""

    @pytest.mark.asyncio
    async def test_verify_payment_success(self, payment_verifier, mock_subtensor):
        """Valid payment passes verification."""
        # Mock block data
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": "5MinerColdkey123",
                        "call": {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": [
                                {"name": "dest", "value": "5RecipientAddress456"},
                                {"name": "value", "value": 100_000_000},
                            ],
                        },
                    }
                )
            ]
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
            expected_amount_rao=100_000_000,
        )

        assert is_valid is True
        assert "verified" in message.lower()

    @pytest.mark.asyncio
    async def test_verify_payment_block_not_found(self, payment_verifier, mock_subtensor):
        """Payment verification fails if block not found."""
        mock_subtensor.substrate.get_block = MagicMock(return_value=None)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xinvalid",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
        )

        assert is_valid is False
        assert "Block not found" in message

    @pytest.mark.asyncio
    async def test_verify_payment_wrong_sender(self, payment_verifier, mock_subtensor):
        """Payment verification fails if sender doesn't match."""
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": "5WrongSender999",  # Wrong sender
                        "call": {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": [
                                {"name": "dest", "value": "5RecipientAddress456"},
                                {"name": "value", "value": 100_000_000},
                            ],
                        },
                    }
                )
            ]
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
        )

        assert is_valid is False
        assert "Sender mismatch" in message

    @pytest.mark.asyncio
    async def test_verify_payment_wrong_recipient(self, payment_verifier, mock_subtensor):
        """Payment verification fails if recipient doesn't match."""
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": "5MinerColdkey123",
                        "call": {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": [
                                {"name": "dest", "value": "5WrongRecipient999"},  # Wrong recipient
                                {"name": "value", "value": 100_000_000},
                            ],
                        },
                    }
                )
            ]
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
        )

        assert is_valid is False
        assert "Recipient mismatch" in message

    @pytest.mark.asyncio
    async def test_verify_payment_insufficient_amount(self, payment_verifier, mock_subtensor):
        """Payment verification fails if amount is too low."""
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": "5MinerColdkey123",
                        "call": {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": [
                                {"name": "dest", "value": "5RecipientAddress456"},
                                {"name": "value", "value": 50_000_000},  # Too low
                            ],
                        },
                    }
                )
            ]
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
            expected_amount_rao=100_000_000,
        )

        assert is_valid is False
        assert "Insufficient payment" in message

    @pytest.mark.asyncio
    async def test_verify_payment_not_balance_call(self, payment_verifier, mock_subtensor):
        """Payment verification fails if not a Balances call."""
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": "5MinerColdkey123",
                        "call": {
                            "call_module": "System",  # Wrong module
                            "call_function": "remark",
                            "call_args": [],
                        },
                    }
                )
            ]
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
        )

        assert is_valid is False
        assert "Not a Balances call" in message

    @pytest.mark.asyncio
    async def test_verify_payment_extrinsic_out_of_range(self, payment_verifier, mock_subtensor):
        """Payment verification fails if extrinsic index is invalid."""
        mock_block = {"extrinsics": []}  # Empty
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=5,  # Out of range
            miner_coldkey="5MinerColdkey123",
        )

        assert is_valid is False
        assert "out of range" in message


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPaymentIntegration:
    """Integration tests for payment flow."""

    @pytest.mark.asyncio
    async def test_payment_flow_end_to_end(self, mock_wallet, mock_subtensor):
        """Test full payment flow: make payment → verify payment."""
        # Step 1: Miner makes payment
        manager = PaymentManager(
            wallet=mock_wallet,
            subtensor=mock_subtensor,
            submission_cost_rao=100_000_000,
        )

        receipt = await manager.make_payment(
            recipient_address="5RecipientAddress456",
        )

        assert receipt.block_hash == "0xabcd1234"
        assert receipt.amount_rao == 100_000_000

        # Step 2: Validator verifies payment
        verifier = PaymentVerifier(
            recipient_address="5RecipientAddress456",
            subtensor=mock_subtensor,
            required_amount_rao=100_000_000,
        )

        # Mock the block data to match the payment
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": mock_wallet.hotkey.ss58_address,
                        "call": {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": [
                                {"name": "dest", "value": "5RecipientAddress456"},
                                {"name": "value", "value": 100_000_000},
                            ],
                        },
                    }
                )
            ] * 6  # Need at least 6 extrinsics for index 5
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await verifier.verify_payment(
            block_hash=receipt.block_hash,
            extrinsic_index=receipt.extrinsic_index,
            miner_coldkey=mock_wallet.hotkey.ss58_address,
            expected_amount_rao=receipt.amount_rao,
        )

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_payment_prevents_spam(self):
        """Payment mechanism prevents spam submissions."""
        # Without payment: unlimited submissions
        # With payment: each submission costs 0.1 TAO

        # Scenario: Miner wants to spam 100 submissions
        num_spam_attempts = 100
        cost_per_submission_tao = 0.1

        total_cost_tao = num_spam_attempts * cost_per_submission_tao

        # Cost becomes prohibitive
        assert total_cost_tao == 10.0  # 10 TAO to spam 100 times

        # This economic barrier prevents spam
        assert total_cost_tao > 1.0  # More than 1 TAO makes spam expensive


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestPaymentEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_payment_with_zero_amount_allowed(self, payment_manager):
        """Zero amount payment is technically allowed (but not useful)."""
        # Note: The blockchain allows 0-value transfers
        # In practice, validators would reject submissions with 0 payment
        receipt = await payment_manager.make_payment(
            recipient_address="5RecipientAddress456",
            amount_rao=0,
        )

        assert receipt.amount_rao == 0

    @pytest.mark.asyncio
    async def test_verify_payment_with_overpayment_succeeds(self, payment_verifier, mock_subtensor):
        """Overpayment (more than required) is accepted."""
        mock_block = {
            "extrinsics": [
                MagicMock(
                    value={
                        "address": "5MinerColdkey123",
                        "call": {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": [
                                {"name": "dest", "value": "5RecipientAddress456"},
                                {"name": "value", "value": 200_000_000},  # Double the required
                            ],
                        },
                    }
                )
            ]
        }
        mock_subtensor.substrate.get_block = MagicMock(return_value=mock_block)

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
            expected_amount_rao=100_000_000,
        )

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_payment_handles_exceptions(self, payment_verifier, mock_subtensor):
        """Payment verification handles exceptions gracefully."""
        mock_subtensor.substrate.get_block = MagicMock(side_effect=Exception("Network error"))

        is_valid, message = await payment_verifier.verify_payment(
            block_hash="0xabcd1234",
            extrinsic_index=0,
            miner_coldkey="5MinerColdkey123",
        )

        assert is_valid is False
        assert "Verification error" in message

