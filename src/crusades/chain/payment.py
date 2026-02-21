"""On-chain payment verification for miner submissions.

Verifies that miners have paid the required submission fee by transferring
alpha tokens to the coldkey that owns the burn_uid's hotkey on the subnet.

The destination address is derived at runtime from the metagraph:
  burn_uid → metagraph.hotkeys[burn_uid] → get_hotkey_owner() → coldkey

The miner embeds the payment extrinsic reference (block number + index) in
the commitment data. The validator performs an O(1) direct lookup to verify
the specific SubtensorModule.transfer_stake extrinsic on-chain.
"""

import asyncio
import concurrent.futures
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
    subtensor: bt.subtensor,
    block_hash: str,
    extrinsic_index: int,
    retries: int = 2,
    rpc_timeout: int = 30,
) -> bool:
    """Check if an extrinsic in a block failed by examining events.

    Retries on transient RPC failures to avoid rejecting valid payments
    due to network hiccups. Each RPC call is capped at ``rpc_timeout``
    seconds to prevent a hung node from blocking the validator indefinitely.
    """
    for attempt in range(1 + retries):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(subtensor.substrate.get_events, block_hash=block_hash)
            events = future.result(timeout=rpc_timeout)
            pool.shutdown(wait=False)
            for event in events:
                if event.get("extrinsic_idx") != extrinsic_index:
                    continue
                ev = event.get("event", {})
                module = ev.get("module_id", "")
                event_id = ev.get("event_id", "")
                if module == "System" and event_id == "ExtrinsicFailed":
                    return True
            return False
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
            logger.warning(
                f"get_events timed out after {rpc_timeout}s (attempt {attempt + 1}/{1 + retries})"
            )
            if attempt >= retries:
                logger.error("get_events timed out on all attempts. Assuming failed to be safe.")
                return True
            time.sleep(1)
        except Exception as e:
            pool.shutdown(wait=False)
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
    rpc_timeout: int = 30,
    rpc_retries: int = 2,
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
        rpc_timeout: Seconds before an RPC call is considered hung
        rpc_retries: Retry count for transient RPC failures

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
            sender = sender or ""
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

            if _check_extrinsic_failed(
                subtensor, block_hash, idx, retries=rpc_retries, rpc_timeout=rpc_timeout
            ):
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


def verify_payment_direct(
    subtensor: bt.subtensor,
    block_number: int,
    extrinsic_index: int,
    miner_coldkey: str,
    payment_address: str,
    netuid: int,
    min_amount: int = 0,
    rpc_timeout: int = 30,
    rpc_retries: int = 2,
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
        rpc_timeout: Seconds before an RPC call is considered hung
        rpc_retries: Retry count for transient RPC failures

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
    if extrinsic_index < 0 or extrinsic_index >= len(extrinsics):
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
        sender = sender or ""
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

        if _check_extrinsic_failed(
            subtensor, block_hash, extrinsic_index, retries=rpc_retries, rpc_timeout=rpc_timeout
        ):
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
    rpc_timeout: int = 30,
    rpc_retries: int = 2,
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
            rpc_timeout=rpc_timeout,
            rpc_retries=rpc_retries,
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
        except Exception as e:
            logger.debug(f"Error scanning block {block_num} for payment extrinsic: {e}")
            continue
    return None
