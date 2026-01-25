"""Blockchain commitment reading for Templar Tournament.

URL-Based Architecture with Timelock Encryption:
- Miners host their train.py code at any URL (Gist, raw GitHub, etc.)
- Miners commit the code URL to blockchain using set_reveal_commitment()
- Commitments are timelock encrypted via drand
- After reveal_blocks, validators can read decrypted URL
- Validator fetches code from URL and evaluates

Commitment format:
  {
    "code_url": "https://example.com/train.py"
  }
"""

import json
import logging
from dataclasses import dataclass

import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class CodeUrlInfo:
    """Code URL information from miner commitment."""
    
    url: str
    
    def is_valid(self) -> bool:
        """Check if code URL is valid."""
        return bool(self.url) and (self.url.startswith("http://") or self.url.startswith("https://"))


@dataclass
class MinerCommitment:
    """A miner's commitment from the blockchain.
    
    Contains code URL for validator to fetch miner's train.py.
    """
    
    uid: int
    hotkey: str
    code_url_info: CodeUrlInfo | None
    reveal_block: int
    is_revealed: bool
    raw_data: str
    
    @classmethod
    def from_chain_data(
        cls,
        uid: int,
        hotkey: str,
        data: str,
        reveal_block: int,
        current_block: int,
    ) -> "MinerCommitment | None":
        """Parse commitment from blockchain data.
        
        Expects JSON format with code URL:
        {
            "code_url": "https://example.com/train.py"
        }
        
        Also supports legacy format:
        {
            "gist_url": "https://gist.githubusercontent.com/user/abc123/raw"
        }
        
        Args:
            uid: Miner UID
            hotkey: Miner hotkey address
            data: Raw commitment data string (decrypted JSON)
            reveal_block: Block when commitment was revealed
            current_block: Current blockchain block
            
        Returns:
            MinerCommitment if valid, None if invalid format
        """
        if not data:
            return None
        
        data = data.strip()
        code_url_info = None
        
        # Parse JSON format
        if data.startswith("{"):
            try:
                parsed = json.loads(data)
                
                # New format: code_url
                if "code_url" in parsed:
                    code_url_info = CodeUrlInfo(url=parsed["code_url"])
                    logger.debug(f"Code URL from UID {uid}: {code_url_info.url[:50]}...")
                # Legacy format: gist_url
                elif "gist_url" in parsed:
                    code_url_info = CodeUrlInfo(url=parsed["gist_url"])
                    logger.debug(f"Legacy gist URL from UID {uid}: {code_url_info.url[:50]}...")
                    
            except json.JSONDecodeError:
                logger.debug(f"Invalid JSON in commitment from UID {uid}")
        
        # If we couldn't parse code URL info, skip
        if code_url_info is None:
            logger.debug(f"Could not parse commitment from UID {uid}: {data[:50]}...")
            return None
        
        return cls(
            uid=uid,
            hotkey=hotkey,
            code_url_info=code_url_info,
            reveal_block=reveal_block,
            is_revealed=current_block >= reveal_block,
            raw_data=data,
        )
    
    def has_valid_code_url(self) -> bool:
        """Check if this commitment has a valid code URL."""
        return self.code_url_info is not None and self.code_url_info.is_valid()


class CommitmentReader:
    """Reads timelock-encrypted miner commitments from the Bittensor blockchain.
    
    Uses get_all_revealed_commitments() to read commitments that have been
    decrypted after their reveal block (set via set_reveal_commitment).
    """
    
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
        """Lazy-load metagraph using subtensor (like templar)."""
        if self._metagraph is None:
            self._metagraph = self.subtensor.metagraph(netuid=self.netuid)
        return self._metagraph
    
    def sync(self) -> None:
        """Sync metagraph with blockchain."""
        logger.info(f"Syncing metagraph for subnet {self.netuid}...")
        self.metagraph.sync(subtensor=self.subtensor)
        logger.info(f"Metagraph synced: {self.metagraph.n} neurons")
    
    def get_current_block(self) -> int:
        """Get current blockchain block number."""
        return self.subtensor.get_current_block()
    
    def _build_hotkey_to_uid_map(self) -> dict[str, int]:
        """Build mapping from hotkey to UID."""
        hotkey_to_uid: dict[str, int] = {}
        self.sync()
        for uid in range(self.metagraph.n):
            hotkey_to_uid[self.metagraph.hotkeys[uid]] = uid
        return hotkey_to_uid
    
    def _parse_revealed_result(self, result) -> tuple[int, str]:
        """Parse result from get_revealed_commitment APIs.
        
        Result format: ((reveal_block1, data1), (reveal_block2, data2), ...) 
        - Tuple of tuples, one per commitment
        - Returns the LATEST commitment (highest block number)
        
        Returns:
            Tuple of (reveal_block, data) for the latest commitment
        """
        if not result:
            return 0, ""
        
        if isinstance(result, tuple):
            # Find the latest commitment (highest block number)
            latest_block = 0
            latest_data = ""
            
            for item in result:
                if isinstance(item, tuple) and len(item) >= 2:
                    block = item[0]
                    data = item[1] if item[1] else ""
                    if block > latest_block:
                        latest_block = block
                        latest_data = data
            
            if latest_block > 0:
                return latest_block, latest_data
        
        return 0, str(result) if result else ""
    
    def get_all_commitments(self) -> list[MinerCommitment]:
        """Get all revealed miner commitments from blockchain.
        
        Uses get_all_revealed_commitments() to read timelock-encrypted commits
        that have been decrypted after their reveal block.
        
        Returns:
            List of MinerCommitment objects with valid code URLs
        """
        current_block = self.get_current_block()
        commitments = []
        
        # Build hotkey to UID mapping (may fail on some localnet versions)
        hotkey_to_uid = self._build_hotkey_to_uid_map()
        
        # Get all revealed commitments (timelock decrypted)
        if hasattr(self.subtensor, 'get_all_revealed_commitments'):
            try:
                logger.info("Reading revealed commitments from blockchain...")
                all_revealed = self.subtensor.get_all_revealed_commitments(netuid=self.netuid)
                logger.info(f"Found {len(all_revealed)} revealed commitments on chain")
                
                for hotkey, result in all_revealed.items():
                    uid = hotkey_to_uid.get(hotkey, 0)
                    reveal_block, data = self._parse_revealed_result(result)
                    
                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        reveal_block=reveal_block,
                        current_block=current_block,
                    )
                    if commitment and commitment.has_valid_code_url():
                        commitment.is_revealed = True
                        commitments.append(commitment)
                        logger.debug(f"Valid commitment from {hotkey[:16]}... (UID {uid})")
                
                logger.info(f"Found {len(commitments)} valid commitments with code URLs")
                return commitments
                    
            except Exception as e:
                logger.error(f"Failed to read revealed commitments: {e}")
        else:
            logger.error("Subtensor does not support get_all_revealed_commitments()")
        
        return commitments
    
    def get_commitment_for_hotkey(
        self,
        hotkey: str,
        current_block: int | None = None,
    ) -> MinerCommitment | None:
        """Get revealed commitment for a specific hotkey.
        
        Args:
            hotkey: Miner hotkey address
            current_block: Current block (fetched if None)
            
        Returns:
            MinerCommitment if found and valid, None otherwise
        """
        if current_block is None:
            current_block = self.get_current_block()
        
        # Get UID from metagraph
        uid = 0
        try:
            hotkey_to_uid = self._build_hotkey_to_uid_map()
            uid = hotkey_to_uid.get(hotkey, 0)
        except Exception:
            pass
        
        # Get revealed commitment
        if hasattr(self.subtensor, 'get_revealed_commitment_by_hotkey'):
            try:
                result = self.subtensor.get_revealed_commitment_by_hotkey(
                    netuid=self.netuid,
                    hotkey_ss58_address=hotkey,
                )
                
                if result:
                    reveal_block, data = self._parse_revealed_result(result)
                    
                    commitment = MinerCommitment.from_chain_data(
                        uid=uid,
                        hotkey=hotkey,
                        data=data,
                        reveal_block=reveal_block,
                        current_block=current_block,
                    )
                    if commitment:
                        commitment.is_revealed = True
                        return commitment
                        
            except Exception as e:
                logger.debug(f"get_revealed_commitment_by_hotkey failed for {hotkey[:16]}...: {e}")
        
        return None
    
    def get_new_commitments_since(self, last_block: int) -> list[MinerCommitment]:
        """Get commitments revealed since a specific block.
        
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
