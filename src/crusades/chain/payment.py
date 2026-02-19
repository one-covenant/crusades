"""On-chain payment verification for miner submissions.

Verifies that miners have paid the required submission fee by transferring
alpha tokens to the coldkey that owns the burn_uid's hotkey on the subnet.

The destination address is derived at runtime from the metagraph:
  burn_uid → metagraph.hotkeys[burn_uid] → get_hotkey_owner() → coldkey

The verification scans a configurable window of blocks around the commitment
for a SubtensorModule.transfer_stake extrinsic from the miner's coldkey to
the owner's coldkey.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class PaymentInfo:
    """Verified payment details from an on-chain transfer_stake extrinsic."""

    block_hash: str
    extrinsic_index: int
    alpha_amount: int
    coldkey: str


def get_hotkey_owner(
    subtensor: bt.subtensor, hotkey_ss58: str, block: int | None = None
) -> str | None:
    """Look up which coldkey owns a hotkey.

    Args:
        subtensor: Subtensor connection
        hotkey_ss58: The hotkey to look up
        block: Optional block number for historical lookup

    Returns:
        The owner coldkey SS58 address, or None if lookup fails
    """
    try:
        return subtensor.get_hotkey_owner(hotkey_ss58=hotkey_ss58, block=block)
    except Exception as e:
        logger.error(f"Failed to look up coldkey for hotkey {hotkey_ss58[:16]}...: {e}")
        return None


def resolve_payment_address(subtensor: bt.subtensor, netuid: int, burn_uid: int) -> str | None:
    """Derive the payment destination coldkey from burn_uid via the metagraph.

    Resolves: burn_uid → hotkey → coldkey owner.

    Returns:
        The SS58 coldkey address, or None if resolution fails.
    """
    try:
        metagraph = subtensor.metagraph(netuid)
        if burn_uid >= len(metagraph.hotkeys):
            logger.error(
                f"burn_uid {burn_uid} out of range (metagraph has {len(metagraph.hotkeys)} hotkeys)"
            )
            return None
        burn_hotkey = metagraph.hotkeys[burn_uid]
        coldkey = get_hotkey_owner(subtensor, burn_hotkey)
        if coldkey is None:
            logger.error(f"Could not resolve coldkey owner for burn hotkey {burn_hotkey[:16]}...")
            return None
        logger.debug(f"Payment address resolved: burn_uid {burn_uid} → {coldkey[:16]}...")
        return coldkey
    except Exception as e:
        logger.error(f"Failed to resolve payment address from burn_uid {burn_uid}: {e}")
        return None


def _check_extrinsic_failed(
    subtensor: bt.subtensor, block_hash: str, extrinsic_index: int, retries: int = 2
) -> bool:
    """Check if an extrinsic in a block failed by examining events.

    Retries on transient RPC failures to avoid rejecting valid payments
    due to network hiccups.
    """
    for attempt in range(1 + retries):
        try:
            events = subtensor.substrate.get_events(block_hash=block_hash)
            for event in events:
                if event.get("extrinsic_idx") != extrinsic_index:
                    continue
                module = event["event"]["module_id"]
                event_id = event["event"]["event_id"]
                if module == "System" and event_id == "ExtrinsicFailed":
                    return True
            return False
        except Exception as e:
            if attempt < retries:
                logger.warning(
                    f"Could not check extrinsic events (attempt {attempt + 1}/{1 + retries}): {e}"
                )
                time.sleep(1)
                continue
            logger.error(
                f"Could not check extrinsic events after {1 + retries} attempts: {e}. "
                f"Assuming failed to be safe."
            )
            return True


def _scan_block_for_transfer_stake(
    subtensor: bt.subtensor,
    block_hash: str,
    miner_coldkey: str,
    payment_address: str,
    netuid: int,
    min_amount: int = 0,
) -> PaymentInfo | None:
    """Scan a single block for a matching transfer_stake extrinsic.

    Looks for SubtensorModule.transfer_stake calls from the miner's coldkey
    to the payment address (owner's coldkey) on the correct subnet.

    Args:
        subtensor: Subtensor connection
        block_hash: Hash of the block to scan
        miner_coldkey: Expected source coldkey
        payment_address: Expected destination coldkey (burn_uid owner)
        netuid: Expected subnet ID
        min_amount: Minimum alpha amount required (reject payments below this)

    Returns:
        PaymentInfo if a valid payment found, None otherwise
    """
    try:
        block = subtensor.substrate.get_block(block_hash=block_hash)
    except Exception as e:
        logger.debug(f"Could not fetch block {block_hash[:16]}...: {e}")
        return None

    extrinsics = block.get("extrinsics", [])

    for idx, extrinsic in enumerate(extrinsics):
        try:
            ext_value = extrinsic.value if hasattr(extrinsic, "value") else extrinsic
            call = ext_value.get("call", {})

            call_module = call.get("call_module", "")
            call_function = call.get("call_function", "")

            if call_module != "SubtensorModule" or call_function != "transfer_stake":
                continue

            raw_address = ext_value.get("address", "")
            if isinstance(raw_address, dict):
                sender = raw_address.get("Id", "")
            else:
                sender = raw_address
            if sender != miner_coldkey:
                continue

            call_args = {arg["name"]: arg["value"] for arg in call.get("call_args", [])}

            dest_coldkey = call_args.get("destination_coldkey")
            if isinstance(dest_coldkey, dict):
                dest_coldkey = dest_coldkey.get("Id", "")
            if dest_coldkey != payment_address:
                logger.debug(
                    f"Extrinsic {idx}: transfer_stake to {str(dest_coldkey)[:16]}... "
                    f"instead of {payment_address[:16]}... — skipping"
                )
                continue

            origin_netuid = call_args.get("origin_netuid")
            if origin_netuid is None or int(origin_netuid) != int(netuid):
                continue

            alpha_amount = call_args.get("alpha_amount", 0)
            if alpha_amount <= 0:
                continue

            if alpha_amount < min_amount:
                logger.debug(
                    f"Extrinsic {idx}: alpha_amount {alpha_amount} below "
                    f"minimum {min_amount} — skipping"
                )
                continue

            if _check_extrinsic_failed(subtensor, block_hash, idx):
                logger.debug(f"Found matching transfer_stake at index {idx} but it failed")
                continue

            return PaymentInfo(
                block_hash=block_hash,
                extrinsic_index=idx,
                alpha_amount=alpha_amount,
                coldkey=miner_coldkey,
            )

        except Exception as e:
            logger.debug(f"Error parsing extrinsic {idx}: {e}")
            continue

    return None


def _scan_block_range(
    subtensor: bt.subtensor,
    start_block: int,
    end_block: int,
    miner_coldkey: str,
    payment_address: str,
    netuid: int,
    min_amount: int = 0,
) -> PaymentInfo | None:
    """Scan a range of blocks (end_block down to start_block) for a matching transfer_stake."""
    for block_num in range(end_block, start_block - 1, -1):
        try:
            block_hash = subtensor.get_block_hash(block_num)
            if block_hash is None:
                continue

            payment = _scan_block_for_transfer_stake(
                subtensor=subtensor,
                block_hash=block_hash,
                miner_coldkey=miner_coldkey,
                payment_address=payment_address,
                netuid=netuid,
                min_amount=min_amount,
            )

            if payment is not None:
                logger.info(
                    f"Found valid payment at block {block_num} "
                    f"(extrinsic {payment.extrinsic_index}, "
                    f"{payment.alpha_amount} alpha)"
                )
                return payment

        except Exception as e:
            logger.debug(f"Error scanning block {block_num}: {e}")
            continue

    return None


def verify_payment_on_chain(
    subtensor: bt.subtensor,
    miner_coldkey: str,
    commitment_block: int,
    payment_address: str,
    netuid: int,
    scan_blocks: int = 200,
    fast_scan_blocks: int = 15,
    min_amount: int = 0,
) -> PaymentInfo | None:
    """Scan a range of blocks for a valid transfer_stake payment from the miner.

    Uses a two-phase approach: first checks the most recent blocks (where
    the payment is most likely to be), then falls back to a full scan.

    Args:
        subtensor: Subtensor connection
        miner_coldkey: The miner's coldkey that should have sent the transfer
        commitment_block: The block the commitment was revealed at
        payment_address: Destination coldkey SS58 to check transfers against
        netuid: Expected subnet ID
        scan_blocks: How many blocks back to scan (full range)
        fast_scan_blocks: How many recent blocks to check first
        min_amount: Minimum alpha amount required (reject payments below this)

    Returns:
        PaymentInfo if valid payment found, None otherwise
    """
    start_block = max(0, commitment_block - scan_blocks)
    end_block = commitment_block
    fast_boundary = max(end_block - fast_scan_blocks + 1, start_block)

    logger.info(
        f"Scanning blocks {start_block}-{end_block} for transfer_stake payment "
        f"from {miner_coldkey[:16]}... to {payment_address[:16]}... on netuid {netuid}"
    )

    # Fast pass: most miners pay shortly before committing
    payment = _scan_block_range(
        subtensor, fast_boundary, end_block, miner_coldkey, payment_address, netuid, min_amount
    )
    if payment is not None:
        return payment

    # Full scan: check remaining older blocks
    if fast_boundary > start_block:
        logger.debug(f"Fast scan miss, scanning remaining blocks {start_block}-{fast_boundary - 1}")
        payment = _scan_block_range(
            subtensor,
            start_block,
            fast_boundary - 1,
            miner_coldkey,
            payment_address,
            netuid,
            min_amount,
        )
        if payment is not None:
            return payment

    logger.warning(
        f"No valid payment found in blocks {start_block}-{end_block} from {miner_coldkey[:16]}..."
    )
    return None


async def verify_payment_on_chain_async(
    subtensor: bt.subtensor,
    miner_coldkey: str,
    commitment_block: int,
    payment_address: str,
    netuid: int,
    scan_blocks: int = 200,
    min_amount: int = 0,
) -> PaymentInfo | None:
    """Async wrapper for verify_payment_on_chain."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: verify_payment_on_chain(
            subtensor=subtensor,
            miner_coldkey=miner_coldkey,
            commitment_block=commitment_block,
            payment_address=payment_address,
            netuid=netuid,
            scan_blocks=scan_blocks,
            min_amount=min_amount,
        ),
    )


def verify_payment_direct(
    subtensor: bt.subtensor,
    block_number: int,
    extrinsic_index: int,
    miner_coldkey: str,
    payment_address: str,
    netuid: int,
    min_amount: int = 0,
) -> PaymentInfo | None:
    """O(1) payment verification using a miner-provided extrinsic reference.

    Instead of scanning a range of blocks, fetches a single block and validates
    that the extrinsic at the given index is a valid transfer_stake from the
    miner to the payment address.  This is secure because the blockchain data
    itself is the source of truth — a miner cannot forge an extrinsic.

    Args:
        subtensor: Subtensor connection
        block_number: Block containing the payment extrinsic
        extrinsic_index: Index of the extrinsic within that block
        miner_coldkey: Expected source coldkey
        payment_address: Expected destination coldkey
        netuid: Expected subnet ID
        min_amount: Minimum alpha amount required

    Returns:
        PaymentInfo if the extrinsic is a valid payment, None otherwise
    """
    try:
        block_hash = subtensor.get_block_hash(block_number)
        if block_hash is None:
            logger.warning(f"Could not get hash for block {block_number}")
            return None
    except Exception as e:
        logger.warning(f"Failed to get block hash for {block_number}: {e}")
        return None

    try:
        block = subtensor.substrate.get_block(block_hash=block_hash)
    except Exception as e:
        logger.warning(f"Could not fetch block {block_number}: {e}")
        return None

    extrinsics = block.get("extrinsics", [])
    if extrinsic_index >= len(extrinsics):
        logger.warning(
            f"Extrinsic index {extrinsic_index} out of range "
            f"(block {block_number} has {len(extrinsics)} extrinsics)"
        )
        return None

    extrinsic = extrinsics[extrinsic_index]
    try:
        ext_value = extrinsic.value if hasattr(extrinsic, "value") else extrinsic
        call = ext_value.get("call", {})

        if (
            call.get("call_module") != "SubtensorModule"
            or call.get("call_function") != "transfer_stake"
        ):
            logger.warning(f"Extrinsic at {block_number}:{extrinsic_index} is not a transfer_stake")
            return None

        raw_address = ext_value.get("address", "")
        sender = raw_address.get("Id", "") if isinstance(raw_address, dict) else raw_address
        if sender != miner_coldkey:
            logger.warning(f"Extrinsic sender {sender[:16]}... != expected {miner_coldkey[:16]}...")
            return None

        call_args = {arg["name"]: arg["value"] for arg in call.get("call_args", [])}

        dest_coldkey = call_args.get("destination_coldkey")
        if isinstance(dest_coldkey, dict):
            dest_coldkey = dest_coldkey.get("Id", "")
        if dest_coldkey != payment_address:
            logger.warning(
                f"Extrinsic destination {str(dest_coldkey)[:16]}... "
                f"!= expected {payment_address[:16]}..."
            )
            return None

        origin_netuid = call_args.get("origin_netuid")
        if origin_netuid is None or int(origin_netuid) != int(netuid):
            logger.warning(f"Extrinsic netuid {origin_netuid} != expected {netuid}")
            return None

        alpha_amount = call_args.get("alpha_amount", 0)
        if alpha_amount <= 0:
            return None

        if alpha_amount < min_amount:
            logger.warning(f"Extrinsic alpha_amount {alpha_amount} < min {min_amount}")
            return None

        if _check_extrinsic_failed(subtensor, block_hash, extrinsic_index):
            logger.warning(f"Extrinsic at {block_number}:{extrinsic_index} failed on-chain")
            return None

        logger.info(
            f"Direct verification OK: block {block_number} extrinsic {extrinsic_index} "
            f"({alpha_amount} alpha)"
        )
        return PaymentInfo(
            block_hash=block_hash,
            extrinsic_index=extrinsic_index,
            alpha_amount=alpha_amount,
            coldkey=miner_coldkey,
        )

    except Exception as e:
        logger.warning(f"Error validating extrinsic at {block_number}:{extrinsic_index}: {e}")
        return None


async def verify_payment_direct_async(
    subtensor: bt.subtensor,
    block_number: int,
    extrinsic_index: int,
    miner_coldkey: str,
    payment_address: str,
    netuid: int,
    min_amount: int = 0,
) -> PaymentInfo | None:
    """Async wrapper for verify_payment_direct."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: verify_payment_direct(
            subtensor=subtensor,
            block_number=block_number,
            extrinsic_index=extrinsic_index,
            miner_coldkey=miner_coldkey,
            payment_address=payment_address,
            netuid=netuid,
            min_amount=min_amount,
        ),
    )


def find_payment_extrinsic(
    subtensor: bt.subtensor,
    miner_coldkey: str,
    payment_address: str,
    netuid: int,
    lookback_blocks: int = 5,
) -> tuple[int, int] | None:
    """Locate the miner's own transfer_stake extrinsic in recent blocks.

    Called by the miner immediately after a successful transfer_stake to
    discover the block number and extrinsic index, which are then embedded
    in the commitment for O(1) validator verification.

    Args:
        subtensor: Subtensor connection
        miner_coldkey: The miner's coldkey that sent the transfer
        payment_address: Destination coldkey
        netuid: Subnet ID
        lookback_blocks: How many recent blocks to search

    Returns:
        (block_number, extrinsic_index) if found, None otherwise
    """
    current_block = subtensor.get_current_block()
    for block_num in range(current_block, max(0, current_block - lookback_blocks), -1):
        try:
            block_hash = subtensor.get_block_hash(block_num)
            if block_hash is None:
                continue
            payment = _scan_block_for_transfer_stake(
                subtensor=subtensor,
                block_hash=block_hash,
                miner_coldkey=miner_coldkey,
                payment_address=payment_address,
                netuid=netuid,
            )
            if payment is not None:
                return (block_num, payment.extrinsic_index)
        except Exception:
            continue
    return None
