"""Blockchain commitment reading for Chi/Affinetes architecture.

This module reads miner Docker image commitments from the Bittensor blockchain.
Validators use this to discover which submissions to evaluate.

Commitment format: "docker|<image_name>|<hash>"

For local testing, also supports reading from .local_commitments/ directory.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class MinerCommitment:
    """A miner's Docker image commitment from the blockchain."""
    
    uid: int
    hotkey: str
    image_name: str
    image_hash: str
    commit_block: int
    reveal_block: int
    is_revealed: bool
    raw_data: str
    
    @classmethod
    def from_chain_data(
        cls,
        uid: int,
        hotkey: str,
        data: str,
        commit_block: int,
        reveal_block: int,
        current_block: int,
    ) -> "MinerCommitment | None":
        """Parse commitment from blockchain data.
        
        Expected format: "docker|<image_name>|<hash>"
        
        Args:
            uid: Miner UID
            hotkey: Miner hotkey address
            data: Raw commitment data string
            commit_block: Block when committed
            reveal_block: Block when revealed
            current_block: Current blockchain block
            
        Returns:
            MinerCommitment if valid, None if invalid format
        """
        if not data:
            return None
        
        # Parse "docker|<image>|<hash>" format
        parts = data.split("|")
        
        if len(parts) != 3:
            logger.debug(f"Invalid commitment format from UID {uid}: {data[:50]}...")
            return None
        
        if parts[0] != "docker":
            logger.debug(f"Non-docker commitment from UID {uid}: {parts[0]}")
            return None
        
        image_name = parts[1]
        image_hash = parts[2]
        
        # Validate image name format
        if not cls._is_valid_image_name(image_name):
            logger.warning(f"Invalid Docker image name from UID {uid}: {image_name}")
            return None
        
        return cls(
            uid=uid,
            hotkey=hotkey,
            image_name=image_name,
            image_hash=image_hash,
            commit_block=commit_block,
            reveal_block=reveal_block,
            is_revealed=current_block >= reveal_block,
            raw_data=data,
        )
    
    @staticmethod
    def _is_valid_image_name(name: str) -> bool:
        """Validate Docker image name format."""
        # Basic validation: alphanumeric, dashes, underscores, slashes, colons, dots
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._\-/:]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$'
        return bool(re.match(pattern, name))


class CommitmentReader:
    """Reads miner commitments from the Bittensor blockchain or local files."""
    
    def __init__(
        self,
        subtensor: bt.subtensor | None = None,
        netuid: int = 1,
        network: str = "local",
        local_mode: bool = False,
        local_commits_dir: Path | None = None,
    ):
        """Initialize commitment reader.
        
        Args:
            subtensor: Bittensor subtensor instance (created if None)
            netuid: Subnet ID
            network: Network name (local, test, finney)
            local_mode: Read from local files instead of blockchain
            local_commits_dir: Directory for local commitments
        """
        self.netuid = netuid
        self.network = network
        self._subtensor = subtensor
        self._metagraph = None
        
        # Enable local mode for "local" network by default
        self.local_mode = local_mode or (network == "local")
        self.local_commits_dir = local_commits_dir or Path.cwd() / ".local_commitments"
    
    @property
    def subtensor(self) -> bt.subtensor:
        """Lazy-load subtensor connection."""
        if self._subtensor is None:
            self._subtensor = bt.subtensor(network=self.network)
        return self._subtensor
    
    @property
    def metagraph(self) -> bt.metagraph:
        """Lazy-load metagraph."""
        if self._metagraph is None:
            self._metagraph = bt.metagraph(netuid=self.netuid, network=self.network)
        return self._metagraph
    
    def sync(self) -> None:
        """Sync metagraph with blockchain."""
        logger.info(f"Syncing metagraph for subnet {self.netuid}...")
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            logger.info(f"Metagraph synced: {self.metagraph.n} neurons")
        except Exception as e:
            logger.warning(f"Metagraph sync failed: {e}")
    
    def get_current_block(self) -> int:
        """Get current blockchain block number."""
        return self.subtensor.get_current_block()
    
    def get_all_commitments(self) -> list[MinerCommitment]:
        """Get all miner commitments from blockchain.
        
        Returns:
            List of MinerCommitment objects
        """
        # For local testing, read from files
        if self.local_mode:
            return self._get_local_commitments()
        
        current_block = self.get_current_block()
        commitments = []
        
        # Use get_all_commitments which returns {hotkey: data_string}
        if hasattr(self.subtensor, 'get_all_commitments'):
            try:
                all_commits = self.subtensor.get_all_commitments(netuid=self.netuid)
                
                # Get UID mapping from metagraph
                hotkey_to_uid = {}
                try:
                    self.sync()
                    for uid in range(self.metagraph.n):
                        hotkey_to_uid[self.metagraph.hotkeys[uid]] = uid
                except Exception:
                    pass  # Continue without UID mapping
                
                for hotkey, data in all_commits.items():
                    uid = hotkey_to_uid.get(hotkey, 0)
                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        commit_block=0,  # Not available from this API
                        reveal_block=0,
                        current_block=current_block,
                    )
                    if commitment:
                        commitment.is_revealed = True  # All from get_all_commitments are visible
                        commitments.append(commitment)
                
                logger.info(f"Found {len(commitments)} commitments from chain")
                return commitments
                
            except Exception as e:
                logger.warning(f"get_all_commitments failed: {e}")
        
        # Fallback: Read commitments from each UID
        logger.info("Using fallback commitment reading...")
        
        for uid in range(self.metagraph.n):
            try:
                commitment = self.get_commitment_for_uid(uid, current_block)
                if commitment:
                    commitments.append(commitment)
            except Exception as e:
                logger.debug(f"Error reading commitment for UID {uid}: {e}")
        
        logger.info(f"Found {len(commitments)} commitments via fallback")
        return commitments
    
    def _get_local_commitments(self) -> list[MinerCommitment]:
        """Read commitments from local files (for testing).
        
        Returns:
            List of MinerCommitment objects from local files
        """
        commitments = []
        
        if not self.local_commits_dir.exists():
            logger.info(f"No local commitments directory: {self.local_commits_dir}")
            return commitments
        
        current_block = self.get_current_block()
        
        for commit_file in self.local_commits_dir.glob("*.json"):
            try:
                data = json.loads(commit_file.read_text())
                
                commitment = MinerCommitment.from_chain_data(
                    uid=data.get("uid", 1),
                    hotkey=data.get("hotkey", ""),
                    data=data.get("data", ""),
                    commit_block=data.get("commit_block", 0),
                    reveal_block=data.get("reveal_block", 0),
                    current_block=current_block,
                )
                
                if commitment:
                    # For local testing, always consider revealed
                    commitment.is_revealed = True
                    commitments.append(commitment)
                    logger.debug(f"Loaded local commitment: {commit_file.name}")
                    
            except Exception as e:
                logger.warning(f"Error reading local commitment {commit_file}: {e}")
        
        logger.info(f"Found {len(commitments)} local commitments")
        return commitments
    
    def get_commitment_for_uid(
        self,
        uid: int,
        current_block: int | None = None,
    ) -> MinerCommitment | None:
        """Get commitment for a specific UID.
        
        Args:
            uid: Miner UID
            current_block: Current block (fetched if None)
            
        Returns:
            MinerCommitment if found and revealed, None otherwise
        """
        if current_block is None:
            current_block = self.get_current_block()
        
        # Get hotkey for UID
        try:
            hotkey = self.metagraph.hotkeys[uid]
        except (IndexError, AttributeError):
            return None
        
        # Try to get commitment data
        try:
            if hasattr(self.subtensor, 'get_commitment'):
                result = self.subtensor.get_commitment(
                    netuid=self.netuid,
                    uid=uid,
                )
                
                if result:
                    return MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=result.get("data", ""),
                        commit_block=result.get("commit_block", current_block),
                        reveal_block=result.get("reveal_block", current_block),
                        current_block=current_block,
                    )
        except Exception as e:
            logger.debug(f"get_commitment failed for UID {uid}: {e}")
        
        return None
    
    def get_new_commitments_since(
        self,
        last_block: int,
    ) -> list[MinerCommitment]:
        """Get commitments revealed since a specific block.
        
        Useful for validators to process only new submissions.
        
        Args:
            last_block: Last processed block number
            
        Returns:
            List of newly revealed commitments
        """
        all_commitments = self.get_all_commitments()
        
        # Filter to only those revealed after last_block
        new_commitments = [
            c for c in all_commitments
            if c.reveal_block > last_block
        ]
        
        logger.info(f"Found {len(new_commitments)} new commitments since block {last_block}")
        return new_commitments
    
    def iter_commitments(self) -> Iterator[MinerCommitment]:
        """Iterate over all revealed commitments.
        
        Yields:
            MinerCommitment objects
        """
        for commitment in self.get_all_commitments():
            yield commitment


def get_commitment_reader(
    network: str = "local",
    netuid: int = 1,
) -> CommitmentReader:
    """Factory function to create a commitment reader.
    
    Args:
        network: Subtensor network
        netuid: Subnet ID
        
    Returns:
        Configured CommitmentReader
    """
    return CommitmentReader(netuid=netuid, network=network)

