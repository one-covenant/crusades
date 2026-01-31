"""Weight setting logic with configurable burn rate."""

import logging

from ..config import get_config, get_hparams
from ..storage.database import Database
from .manager import ChainManager

logger = logging.getLogger(__name__)


class WeightSetter:
    """Sets weights on Bittensor: burn_rate% to validator, rest to rank 1 winner."""

    def __init__(
        self,
        chain: ChainManager,
        database: Database,
    ):
        self.chain = chain
        self.db = database
        self.config = get_config()
        self.hparams = get_hparams()

        self.burn_rate = self.hparams.burn_rate
        self.burn_uid = self.hparams.burn_uid

    async def set_weights(self) -> tuple[bool, str]:
        """Set weights: burn_rate% to validator, rest to rank 1 (with 1% threshold)."""
        # Sync metagraph to get latest state
        await self.chain.sync_metagraph()

        # Skip weight setting if metagraph sync failed
        if self.chain.metagraph is None:
            logger.warning("Metagraph not available - cannot set weights")
            logger.warning(
                "Possible causes: subtensor not running, network issues, or netuid doesn't exist"
            )
            return False, "Metagraph sync failed - cannot set weights"

        # Get leaderboard rank 1 (with 1% threshold applied)
        winner = await self.db.get_leaderboard_winner(threshold=0.01)

        # If no valid winner, all emissions go to burn_uid
        if winner is None:
            logger.info("No finished submissions - 100% to burn_uid")
            return await self._set_burn_only_weights("No finished submissions")

        # Verify miner is still registered
        winner_hotkey = winner.miner_hotkey
        if not self.chain.is_registered(winner_hotkey):
            logger.warning(f"Winner {winner_hotkey} not registered - 100% to burn_uid")
            return await self._set_burn_only_weights(f"Winner {winner_hotkey} not registered")

        # Get UID for winner
        winner_uid = self.chain.get_uid_for_hotkey(winner_hotkey)
        if winner_uid is None:
            logger.error(f"Could not get UID for {winner_hotkey} - 100% to burn_uid")
            return await self._set_burn_only_weights(f"Could not get UID for {winner_hotkey}")

        winner_score = winner.final_score or 0.0

        # Calculate weight distribution
        winner_weight = 1.0 - self.burn_rate  # e.g., 5%
        burn_weight = self.burn_rate  # e.g., 95%

        logger.info(
            f"Setting weights with burn_rate={self.burn_rate:.0%}:\n"
            f"  - UID {self.burn_uid} (validator): {burn_weight:.2f}\n"
            f"  - UID {winner_uid} (winner, score={winner_score:.2f}): {winner_weight:.2f}"
        )

        # Set weights for both burn_uid and winner
        uids = [self.burn_uid, winner_uid]
        weights = [burn_weight, winner_weight]

        # Handle case where winner IS the burn_uid (unlikely but possible)
        if winner_uid == self.burn_uid:
            uids = [self.burn_uid]
            weights = [1.0]
            logger.info(f"Winner is burn_uid - setting 100% to UID {self.burn_uid}")

        success, message = await self.chain.set_weights(
            uids=uids,
            weights=weights,
        )

        if success:
            logger.info(f"Weights set successfully: {dict(zip(uids, weights))}")
        else:
            logger.error(f"Failed to set weights: {message}")

        return (success, message)

    async def _set_burn_only_weights(self, reason: str) -> tuple[bool, str]:
        """Set 100% weights to burn_uid when no valid winner exists.

        Args:
            reason: Why we're giving all to burn_uid (for logging)

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Setting 100% weight to burn_uid {self.burn_uid} ({reason})")

        success, message = await self.chain.set_weights(
            uids=[self.burn_uid],
            weights=[1.0],
        )

        if success:
            logger.info(f"Weights set successfully: UID {self.burn_uid} -> 100%")
        else:
            logger.error(f"Failed to set weights: {message}")

        return (success, f"Burn only ({reason}): {message}" if success else message)
