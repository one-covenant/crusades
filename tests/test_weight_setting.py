"""
Unit tests for Weight Setting Logic

Tests the winner-takes-all weight setting with burn fallback.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tournament.chain.weights import WeightSetter
from tournament.chain.manager import ChainManager
from tournament.storage.database import Database
from tournament.storage.models import SubmissionModel
from tournament.core.protocols import SubmissionStatus


@pytest.fixture
def mock_chain():
    """Mock ChainManager."""
    chain = AsyncMock(spec=ChainManager)
    
    # Mock metagraph
    chain.metagraph = MagicMock()
    chain.metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "burn_hotkey"]
    chain.metagraph.uids = [0, 1, 2, 3]
    
    # Mock methods
    chain.sync_metagraph = AsyncMock()
    chain.is_registered = MagicMock(return_value=True)
    chain.get_uid_for_hotkey = MagicMock(side_effect=lambda h: {
        "hotkey1": 0,
        "hotkey2": 1,
        "hotkey3": 2,
        "burn_hotkey": 3,
    }.get(h))
    chain.set_weights = AsyncMock(return_value=(True, "Success"))
    
    return chain


@pytest.fixture
async def mock_db():
    """Mock Database."""
    db = AsyncMock(spec=Database)
    return db


@pytest.fixture
def winner_submission():
    """Create a winning submission."""
    sub = SubmissionModel(
        miner_hotkey="hotkey1",
        miner_uid=0,
        code_hash="hash1",
        bucket_path="path1",
    )
    sub.final_score = 10000.0
    sub.status = SubmissionStatus.FINISHED
    return sub


class TestWeightSetting:
    """Tests for winner-takes-all weight setting."""

    @pytest.mark.asyncio
    async def test_set_weights_to_winner(self, mock_chain, mock_db, winner_submission):
        """Winner receives 100% weight."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        assert success is True
        mock_chain.set_weights.assert_called_once_with(
            uids=[0],  # Winner's UID
            weights=[1.0],
        )

    @pytest.mark.asyncio
    async def test_no_winner_falls_back_to_burn(self, mock_chain, mock_db):
        """When no winner, emissions go to burn hotkey."""
        mock_db.get_top_submission = AsyncMock(return_value=None)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        assert success is True
        mock_chain.set_weights.assert_called_once_with(
            uids=[3],  # Burn hotkey UID
            weights=[1.0],
        )

    @pytest.mark.asyncio
    async def test_burn_mode_always_burns(self, mock_chain, mock_db, winner_submission):
        """When burn_enabled=True, always burn regardless of winner."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=True,
        )
        
        success, message = await setter.set_weights()
        
        assert success is True
        mock_chain.set_weights.assert_called_once_with(
            uids=[3],  # Burn hotkey UID
            weights=[1.0],
        )

    @pytest.mark.asyncio
    async def test_winner_not_registered_falls_back_to_burn(self, mock_chain, mock_db, winner_submission):
        """If winner is not registered on chain, fall back to burn."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        # Make winner appear unregistered
        mock_chain.is_registered = MagicMock(side_effect=lambda h: h == "burn_hotkey")
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        assert success is True
        # Should burn instead
        mock_chain.set_weights.assert_called_once_with(
            uids=[3],
            weights=[1.0],
        )

    @pytest.mark.asyncio
    async def test_no_burn_hotkey_configured(self, mock_chain, mock_db):
        """When no burn hotkey and no winner, return error."""
        mock_db.get_top_submission = AsyncMock(return_value=None)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey=None,  # No burn hotkey
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        assert success is False
        assert "No burn hotkey configured" in message

    @pytest.mark.asyncio
    async def test_burn_hotkey_not_registered(self, mock_chain, mock_db):
        """When burn hotkey not registered, return error."""
        mock_db.get_top_submission = AsyncMock(return_value=None)
        
        # Make burn hotkey appear unregistered
        mock_chain.is_registered = MagicMock(return_value=False)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        assert success is False
        assert "not registered" in message

    @pytest.mark.asyncio
    async def test_weight_setting_failure_handled(self, mock_chain, mock_db, winner_submission):
        """Handle chain weight setting failures gracefully."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        # Make weight setting fail
        mock_chain.set_weights = AsyncMock(return_value=(False, "Chain error"))
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        assert success is False
        assert "Chain error" in message

    @pytest.mark.asyncio
    async def test_metagraph_syncs_before_setting(self, mock_chain, mock_db, winner_submission):
        """Metagraph is synced before setting weights."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        await setter.set_weights()
        
        # Verify metagraph was synced
        mock_chain.sync_metagraph.assert_called_once()


class TestWeightSettingPriority:
    """Tests for weight setting priority logic."""

    @pytest.mark.asyncio
    async def test_priority_1_burn_mode_takes_precedence(self, mock_chain, mock_db, winner_submission):
        """Priority 1: Burn mode takes precedence over winner."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=True,
        )
        
        success, message = await setter.set_weights()
        
        # Should burn, not give to winner
        assert "Burn" in message

    @pytest.mark.asyncio
    async def test_priority_2_winner_gets_reward(self, mock_chain, mock_db, winner_submission):
        """Priority 2: Winner gets reward when burn mode off."""
        mock_db.get_top_submission = AsyncMock(return_value=winner_submission)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        # Should give to winner
        mock_chain.set_weights.assert_called_once()
        call_args = mock_chain.set_weights.call_args
        assert call_args[1]["uids"] == [0]  # Winner UID

    @pytest.mark.asyncio
    async def test_priority_3_fallback_to_burn(self, mock_chain, mock_db):
        """Priority 3: Fallback to burn when no winner."""
        mock_db.get_top_submission = AsyncMock(return_value=None)
        
        setter = WeightSetter(
            chain=mock_chain,
            database=mock_db,
            burn_hotkey="burn_hotkey",
            burn_enabled=False,
        )
        
        success, message = await setter.set_weights()
        
        # Should burn
        mock_chain.set_weights.assert_called_once()
        call_args = mock_chain.set_weights.call_args
        assert call_args[1]["uids"] == [3]  # Burn UID

