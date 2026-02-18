"""On-chain payment verification for miner submissions.

Verifies that miners have staked the required submission fee (in TAO, converted
to alpha) into the subnet before their submission is evaluated.

The verification scans a configurable window of blocks around the commitment
for an add_stake extrinsic from the miner's coldkey to the subnet.
"""

import asyncio
import logging
from dataclasses import dataclass

import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class PaymentInfo:
    """Verified payment details from an on-chain staking extrinsic."""

    block_hash: str
    extrinsic_index: int
    amount_rao: int
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


def _check_extrinsic_failed(subtensor: bt.subtensor, block_hash: str, extrinsic_index: int) -> bool:
    """Check if an extrinsic in a block failed by examining events."""
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
        logger.warning(f"Could not check extrinsic events: {e}")
        return True  # Assume failed if we can't verify


def _scan_block_for_stake(
    subtensor: bt.subtensor,
    block_hash: str,
    miner_coldkey: str,
    netuid: int,
    min_amount_rao: int,
) -> PaymentInfo | None:
    """Scan a single block for a matching add_stake extrinsic.

    Looks for SubtensorModule.add_stake calls from the miner's coldkey
    to the correct subnet with sufficient amount.

    Args:
        subtensor: Subtensor connection
        block_hash: Hash of the block to scan
        miner_coldkey: Expected source coldkey
        netuid: Expected subnet ID
        min_amount_rao: Minimum required stake amount in RAO

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

            if call_module != "SubtensorModule" or call_function != "add_stake":
                continue

            sender = ext_value.get("address", "")
            if sender != miner_coldkey:
                continue

            call_args = {arg["name"]: arg["value"] for arg in call.get("call_args", [])}

            ext_netuid = call_args.get("netuid")
            if ext_netuid != netuid:
                continue

            amount = call_args.get("amount_staked", 0)
            if amount < min_amount_rao:
                continue

            if _check_extrinsic_failed(subtensor, block_hash, idx):
                logger.debug(f"Found matching stake extrinsic at index {idx} but it failed")
                continue

            return PaymentInfo(
                block_hash=block_hash,
                extrinsic_index=idx,
                amount_rao=amount,
                coldkey=miner_coldkey,
            )

        except Exception as e:
            logger.debug(f"Error parsing extrinsic {idx}: {e}")
            continue

    return None


def verify_payment_on_chain(
    subtensor: bt.subtensor,
    miner_coldkey: str,
    commitment_block: int,
    netuid: int,
    min_amount_rao: int,
    scan_blocks: int = 200,
) -> PaymentInfo | None:
    """Scan a range of blocks for a valid staking payment from the miner.

    Searches backwards from the commitment block for an add_stake extrinsic
    that matches the miner's coldkey, the correct subnet, and the minimum amount.

    Args:
        subtensor: Subtensor connection
        miner_coldkey: The miner's coldkey that should have staked
        commitment_block: The block the commitment was revealed at
        netuid: Subnet the stake should target
        min_amount_rao: Minimum stake amount required (in RAO)
        scan_blocks: How many blocks back to scan

    Returns:
        PaymentInfo if valid payment found, None otherwise
    """
    start_block = max(0, commitment_block - scan_blocks)
    end_block = commitment_block

    logger.info(
        f"Scanning blocks {start_block}-{end_block} for stake payment "
        f"from {miner_coldkey[:16]}... (min {min_amount_rao} RAO on netuid {netuid})"
    )

    for block_num in range(end_block, start_block - 1, -1):
        try:
            block_hash = subtensor.get_block_hash(block_num)
            if block_hash is None:
                continue

            payment = _scan_block_for_stake(
                subtensor=subtensor,
                block_hash=block_hash,
                miner_coldkey=miner_coldkey,
                netuid=netuid,
                min_amount_rao=min_amount_rao,
            )

            if payment is not None:
                logger.info(
                    f"Found valid payment at block {block_num} "
                    f"(extrinsic {payment.extrinsic_index}, {payment.amount_rao} RAO)"
                )
                return payment

        except Exception as e:
            logger.debug(f"Error scanning block {block_num}: {e}")
            continue

    logger.warning(
        f"No valid payment found in blocks {start_block}-{end_block} from {miner_coldkey[:16]}..."
    )
    return None


async def verify_payment_on_chain_async(
    subtensor: bt.subtensor,
    miner_coldkey: str,
    commitment_block: int,
    netuid: int,
    min_amount_rao: int,
    scan_blocks: int = 200,
) -> PaymentInfo | None:
    """Async wrapper for verify_payment_on_chain."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        verify_payment_on_chain,
        subtensor,
        miner_coldkey,
        commitment_block,
        netuid,
        min_amount_rao,
        scan_blocks,
    )
