"""Blockchain commitment reading for Templar Tournament.

R2-Based Architecture:
- Miners commit R2 credentials + link to blockchain
- After reveal, validators can download train.py from miner's R2
- Commitment format (JSON):
  {
    "r2_endpoint": "https://xxx.r2.cloudflarestorage.com",
    "r2_bucket": "miner-bucket",
    "r2_key": "submissions/xxx/train.py",
    "r2_access_key": "...",
    "r2_secret_key": "...",
    "code_hash": "sha256..."
  }

For local testing, also supports reading from .local_commitments/ directory.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

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
        return all([
            self.endpoint,
            self.bucket,
            self.key,
            self.access_key,
            self.secret_key,
        ])


@dataclass
class MinerCommitment:
    """A miner's R2 commitment from the blockchain."""
    
    uid: int
    hotkey: str
    r2_credentials: R2Credentials | None
    code_hash: str
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
            "r2_secret_key": "...",
            "code_hash": "..."
        }
        
        Also supports legacy Docker format for backwards compatibility:
        - "docker|<image_name>|<hash>"
        - {"image": "...", "fingerprint": "..."}
        
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
        code_hash = ""
        
        # Try JSON format (R2 credentials)
        if data.startswith("{"):
            try:
                parsed = json.loads(data)
                
                # New R2 format
                if "r2_endpoint" in parsed or "r2_bucket" in parsed:
                    r2_credentials = R2Credentials(
                        endpoint=parsed.get("r2_endpoint", ""),
                        bucket=parsed.get("r2_bucket", ""),
                        key=parsed.get("r2_key", ""),
                        access_key=parsed.get("r2_access_key", ""),
                        secret_key=parsed.get("r2_secret_key", ""),
                    )
                    code_hash = parsed.get("code_hash", "")
                    
                    if not r2_credentials.is_valid():
                        logger.warning(f"Incomplete R2 credentials from UID {uid}")
                        # Still return commitment so it can be tracked
                
                # Legacy Docker format (Chi compatibility)
                elif "image" in parsed:
                    # Convert to legacy format for backwards compatibility
                    image_name = parsed.get("image") or parsed.get("image_url", "")
                    code_hash = parsed.get("fingerprint") or parsed.get("fp") or parsed.get("hash", "")
                    logger.debug(f"Legacy Docker format from UID {uid}: {image_name}")
                    
            except json.JSONDecodeError:
                logger.debug(f"Invalid JSON in commitment from UID {uid}")
        
        # Legacy pipe format: "docker|<image>|<hash>"
        elif "|" in data:
            parts = data.split("|")
            if len(parts) == 3 and parts[0] == "docker":
                logger.debug(f"Legacy Docker pipe format from UID {uid}")
                code_hash = parts[2]
        
        # If we couldn't parse anything useful, skip
        if r2_credentials is None and not code_hash:
            logger.debug(f"Could not parse commitment from UID {uid}: {data[:50]}...")
            return None
        
        return cls(
            uid=uid,
            hotkey=hotkey,
            r2_credentials=r2_credentials,
            code_hash=code_hash,
            commit_block=commit_block,
            reveal_block=reveal_block,
            is_revealed=current_block >= reveal_block,
            raw_data=data,
        )
    
    def has_r2_credentials(self) -> bool:
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
        """Get all miner commitments.
        
        Returns:
            List of MinerCommitment objects
        """
        if self.local_mode:
            return self._get_local_commitments()
        
        current_block = self.get_current_block()
        commitments = []
        
        # Try get_all_commitments first
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
                    pass
                
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
                
                logger.info(f"Found {len(commitments)} commitments from chain")
                return commitments
                
            except Exception as e:
                logger.warning(f"get_all_commitments failed: {e}")
        
        # Fallback: Read from each UID
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
                
                # Build commitment data string (JSON format)
                commitment_data = json.dumps({
                    "r2_endpoint": data.get("r2_endpoint", ""),
                    "r2_bucket": data.get("r2_bucket", ""),
                    "r2_key": data.get("r2_key", ""),
                    "r2_access_key": data.get("r2_access_key", ""),
                    "r2_secret_key": data.get("r2_secret_key", ""),
                    "code_hash": data.get("code_hash", ""),
                })
                
                commitment = MinerCommitment.from_chain_data(
                    uid=data.get("uid", 1),
                    hotkey=data.get("hotkey", ""),
                    data=commitment_data,
                    commit_block=data.get("commit_block", 0),
                    reveal_block=data.get("reveal_block", 0),
                    current_block=current_block,
                )
                
                if commitment:
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
    
    def get_r2_commitments(self) -> list[MinerCommitment]:
        """Get only commitments with valid R2 credentials.
        
        Returns:
            List of commitments with R2 credentials
        """
        all_commitments = self.get_all_commitments()
        return [c for c in all_commitments if c.has_r2_credentials()]


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
