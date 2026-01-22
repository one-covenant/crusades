"""Blockchain commitment reading for Templar Tournament.

R2-Based Architecture:
- Miners upload train.py to their own R2 bucket
- Miners commit R2 credentials + path to blockchain (timelock encrypted)
- After reveal, validators download using miner's R2 credentials
- Validator stores evaluated code in database (no validator R2 needed)

Commitment format:
  {
    "r2_endpoint": "https://xxx.r2.cloudflarestorage.com",
    "r2_bucket": "miner-bucket",
    "r2_key": "submissions/hotkey/timestamp/train.py",
    "r2_access_key": "...",
    "r2_secret_key": "..."
  }
"""

import json
import logging
from dataclasses import dataclass

import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class R2Credentials:
    """R2/S3 credentials from miner commitment."""
    
    endpoint: str
    bucket: str
    key: str
    access_key: str
    secret_key: str
    
    def is_valid(self) -> bool:
        """Check if credentials are complete."""
        return all([self.bucket, self.key, self.access_key, self.secret_key])


@dataclass
class MinerCommitment:
    """A miner's commitment from the blockchain.
    
    Contains R2 credentials for validator to download miner's train.py.
    """
    
    uid: int
    hotkey: str
    r2_credentials: R2Credentials | None
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
        
        Expects JSON format with R2 credentials:
        {
            "r2_endpoint": "...",
            "r2_bucket": "...",
            "r2_key": "...",
            "r2_access_key": "...",
            "r2_secret_key": "..."
        }
        
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
        
        data = data.strip()
        r2_credentials = None
        
        # Try JSON format
        if data.startswith("{"):
            try:
                parsed = json.loads(data)
                
                # R2 credentials format
                if "r2_bucket" in parsed or "r2_key" in parsed:
                    r2_credentials = R2Credentials(
                        endpoint=parsed.get("r2_endpoint", ""),
                        bucket=parsed.get("r2_bucket", ""),
                        key=parsed.get("r2_key", ""),
                        access_key=parsed.get("r2_access_key", ""),
                        secret_key=parsed.get("r2_secret_key", ""),
                    )
                    logger.debug(f"R2 credentials from UID {uid}: bucket={r2_credentials.bucket}")
                    
            except json.JSONDecodeError:
                logger.debug(f"Invalid JSON in commitment from UID {uid}")
        
        # If we couldn't parse R2 credentials, skip
        if r2_credentials is None:
            logger.debug(f"Could not parse commitment from UID {uid}: {data[:50]}...")
            return None
        
        return cls(
            uid=uid,
            hotkey=hotkey,
            r2_credentials=r2_credentials,
            commit_block=commit_block,
            reveal_block=reveal_block,
            is_revealed=current_block >= reveal_block,
            raw_data=data,
        )
    
    def has_valid_credentials(self) -> bool:
        """Check if this commitment has valid R2 credentials."""
        return self.r2_credentials is not None and self.r2_credentials.is_valid()


class CommitmentReader:
    """Reads miner commitments from the Bittensor blockchain."""
    
    def __init__(
        self,
        subtensor: bt.subtensor | None = None,
        netuid: int = 1,
        network: str = "finney",
    ):
        """Initialize commitment reader.
        
        Args:
            subtensor: Bittensor subtensor instance
            netuid: Subnet ID
            network: Network name (local, test, finney)
        """
        self.netuid = netuid
        self.network = network
        self._subtensor = subtensor
        self._metagraph = None
    
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
        
        Uses simple commit API (get_all_commitments) which works on all networks.
        
        Returns:
            List of MinerCommitment objects
        """
        current_block = self.get_current_block()
        commitments = []
        
        # Build hotkey to UID mapping
        hotkey_to_uid = {}
        try:
            self.sync()
            for uid in range(self.metagraph.n):
                hotkey_to_uid[self.metagraph.hotkeys[uid]] = uid
        except Exception:
            pass
        
        # Get all simple commitments
        if hasattr(self.subtensor, 'get_all_commitments'):
            try:
                all_commits = self.subtensor.get_all_commitments(netuid=self.netuid)
                logger.info(f"get_all_commitments returned {len(all_commits)} entries")
                
                for hotkey, data in all_commits.items():
                    uid = hotkey_to_uid.get(hotkey, 0)
                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        commit_block=0,
                        reveal_block=0,
                        current_block=current_block,
                    )
                    if commitment:
                        commitment.is_revealed = True
                        commitments.append(commitment)
                
                logger.info(f"Found {len(commitments)} commitments")
                return commitments
                
            except Exception as e:
                logger.warning(f"get_all_commitments failed: {e}")
        
        # Fallback: read from each UID individually
        logger.info("Using per-UID fallback commitment reading...")
        
        for uid in range(self.metagraph.n):
            try:
                commitment = self.get_commitment_for_uid(uid, current_block)
                if commitment:
                    commitments.append(commitment)
            except Exception as e:
                logger.debug(f"Error reading commitment for UID {uid}: {e}")
        
        logger.info(f"Found {len(commitments)} commitments via fallback")
        return commitments
    
    def get_commitment_for_uid(
        self,
        uid: int,
        current_block: int | None = None,
    ) -> MinerCommitment | None:
        """Get commitment for a specific UID.
        
        Uses simple commit API (get_commitment) which works on all networks.
        
        Args:
            uid: Miner UID
            current_block: Current block (fetched if None)
            
        Returns:
            MinerCommitment if found, None otherwise
        """
        if current_block is None:
            current_block = self.get_current_block()
        
        try:
            hotkey = self.metagraph.hotkeys[uid]
        except (IndexError, AttributeError):
            return None
        
        try:
            if hasattr(self.subtensor, 'get_commitment'):
                result = self.subtensor.get_commitment(
                    netuid=self.netuid,
                    uid=uid,
                )
                
                if result:
                    data = result if isinstance(result, str) else result.get("data", "")
                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        commit_block=current_block,
                        reveal_block=current_block,
                        current_block=current_block,
                    )
                    if commitment:
                        commitment.is_revealed = True
                        return commitment
        except Exception as e:
            logger.debug(f"get_commitment failed for UID {uid}: {e}")
        
        return None
    
    def get_new_commitments_since(
        self,
        last_block: int,
    ) -> list[MinerCommitment]:
        """Get commitments revealed since a specific block.
        
        Args:
            last_block: Last processed block number
            
        Returns:
            List of newly revealed commitments
        """
        all_commitments = self.get_all_commitments()
        
        # Filter to only those committed after last_block and revealed
        new_commitments = [
            c for c in all_commitments
            if c.commit_block > last_block and c.is_revealed
        ]
        
        logger.info(f"Found {len(new_commitments)} new commitments since block {last_block}")
        return new_commitments
    
    def get_valid_commitments(self) -> list[MinerCommitment]:
        """Get only commitments with valid R2 credentials.
        
        Returns:
            List of commitments with valid R2 credentials
        """
        all_commitments = self.get_all_commitments()
        return [c for c in all_commitments if c.has_valid_credentials()]


def get_commitment_reader(
    network: str = "finney",
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
