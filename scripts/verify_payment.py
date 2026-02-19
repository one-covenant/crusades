#!/usr/bin/env python3
"""Verify a miner's submission payment on-chain given a block hash.

Usage:
    # Verify payment by block hash (localnet)
    uv run scripts/verify_payment.py 0x807813b15d3b3fcb... --network local

    # Verify on mainnet
    uv run scripts/verify_payment.py 0x807813b15d3b3fcb... --network finney

    # Also check by block number
    uv run scripts/verify_payment.py --block-number 5505 --network local
"""

import argparse
import json
import sys
from pathlib import Path

import bittensor as bt

from crusades.chain.payment import resolve_payment_address


def load_hparams():
    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    if hparams_path.exists():
        with open(hparams_path) as f:
            return json.load(f)
    return {}


def inspect_block(sub, block_hash=None, block_number=None):
    """Fetch a block and return all extrinsics with decoded details."""
    if block_hash:
        block = sub.substrate.get_block(block_hash=block_hash)
    elif block_number is not None:
        block_hash = sub.substrate.get_block_hash(block_number)
        block = sub.substrate.get_block(block_hash=block_hash)
    else:
        raise ValueError("Provide either block_hash or block_number")

    return block, block_hash


def find_transfer_stake_payments(block, payment_address, netuid):
    """Find all transfer_stake extrinsics in a block targeting the payment address."""
    payments = []
    extrinsics = block.get("extrinsics", [])

    for idx, ext in enumerate(extrinsics):
        try:
            call = ext.value.get("call", {})
            call_module = call.get("call_module", "")
            call_function = call.get("call_function", "")

            if call_module == "SubtensorModule" and call_function == "transfer_stake":
                params = {p["name"]: p["value"] for p in call.get("call_args", [])}

                dest_coldkey = params.get("destination_coldkey", "")
                if isinstance(dest_coldkey, dict):
                    dest_coldkey = dest_coldkey.get("Id", dest_coldkey)

                hotkey = params.get("hotkey", "")
                if isinstance(hotkey, dict):
                    hotkey = hotkey.get("Id", hotkey)

                origin_netuid = params.get("origin_netuid")
                dest_netuid = params.get("destination_netuid")
                alpha_amount = params.get("alpha_amount", 0)

                sender = ext.value.get("address", "unknown")
                if isinstance(sender, dict):
                    sender = sender.get("Id", sender)

                is_match = str(dest_coldkey) == str(payment_address) and (
                    origin_netuid is None or int(origin_netuid) == int(netuid)
                )

                payments.append(
                    {
                        "extrinsic_index": idx,
                        "sender_coldkey": sender,
                        "dest_coldkey": dest_coldkey,
                        "hotkey": hotkey,
                        "origin_netuid": origin_netuid,
                        "dest_netuid": dest_netuid,
                        "alpha_amount": alpha_amount,
                        "matches_payment": is_match,
                    }
                )
        except Exception as e:
            payments.append(
                {
                    "extrinsic_index": idx,
                    "error": str(e),
                }
            )

    return payments


def main():
    parser = argparse.ArgumentParser(description="Verify a miner's submission payment on-chain")
    parser.add_argument("block_hash", nargs="?", help="Block hash to inspect")
    parser.add_argument(
        "--block-number", type=int, help="Block number to inspect (alternative to hash)"
    )
    parser.add_argument(
        "--network", default="finney", help="Network: finney, local, test (default: finney)"
    )
    parser.add_argument(
        "--payment-address",
        type=str,
        help="Override payment dest address (default: derived from burn_uid)",
    )
    parser.add_argument("--netuid", type=int, help="Subnet UID (default: from hparams.json)")
    parser.add_argument("--burn-uid", type=int, help="Burn UID (default: from hparams.json)")
    args = parser.parse_args()

    if not args.block_hash and args.block_number is None:
        parser.error("Provide a block hash or --block-number")

    hparams = load_hparams()
    netuid = args.netuid or hparams.get("netuid", 2)
    burn_uid = args.burn_uid if args.burn_uid is not None else hparams.get("burn_uid", 0)

    print(f"\nConnecting to {args.network}...")
    sub = bt.subtensor(network=args.network)

    # Resolve payment address: explicit override or derive from burn_uid
    payment_address = args.payment_address
    if not payment_address:
        print(f"  Resolving payment address from burn_uid {burn_uid} on subnet {netuid}...")
        payment_address = resolve_payment_address(sub, netuid, burn_uid)
        if not payment_address:
            print(f"  ERROR: Could not resolve payment address from burn_uid {burn_uid}")
            sys.exit(1)

    print("=" * 60)
    print("PAYMENT VERIFICATION")
    print("=" * 60)
    print(f"  Network:         {args.network}")
    print(f"  Netuid:          {netuid}")
    print(f"  Burn UID:        {burn_uid}")
    print(f"  Payment address: {payment_address}")

    print("\nFetching block...")
    try:
        block, block_hash = inspect_block(
            sub,
            block_hash=args.block_hash,
            block_number=args.block_number,
        )
    except Exception as e:
        print(f"  ERROR: Could not fetch block: {e}")
        sys.exit(1)

    header = block.get("header", {})
    block_num = header.get("number", args.block_number or "?")
    print(f"  Block number: {block_num}")
    print(f"  Block hash:   {block_hash}")

    extrinsics = block.get("extrinsics", [])
    print(f"  Extrinsics:   {len(extrinsics)}")

    print(f"\n{'=' * 60}")
    print("ALL EXTRINSICS IN BLOCK")
    print(f"{'=' * 60}")
    for idx, ext in enumerate(extrinsics):
        try:
            call = ext.value.get("call", {})
            module = call.get("call_module", "?")
            function = call.get("call_function", "?")
            sender = ext.value.get("address", "")
            if isinstance(sender, dict):
                sender = sender.get("Id", str(sender))
            signed = "signed" if sender else "unsigned"
            print(f"  [{idx}] {module}.{function} ({signed})")
            if sender:
                print(f"       sender: {sender[:20]}...")
        except Exception as e:
            print(f"  [{idx}] ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("ALPHA TRANSFER PAYMENTS (transfer_stake to payment address)")
    print(f"{'=' * 60}")
    payments = find_transfer_stake_payments(block, payment_address, netuid)
    valid_payments = [p for p in payments if "error" not in p]

    if not valid_payments:
        print("  No transfer_stake extrinsics found in this block.")

    for p in valid_payments:
        match_str = "YES" if p["matches_payment"] else "NO"

        print(f"\n  Extrinsic #{p['extrinsic_index']}:")
        print(f"    Sender (coldkey):  {p['sender_coldkey']}")
        print(f"    Dest coldkey:      {p['dest_coldkey']}")
        print(f"    Hotkey:            {p['hotkey']}")
        print(f"    Origin netuid:     {p['origin_netuid']}")
        print(f"    Dest netuid:       {p['dest_netuid']}")
        print(f"    Alpha amount:      {p['alpha_amount']}")
        print(f"    Matches payment:   {match_str}")

        if p["matches_payment"]:
            print("\n    >>> VALID PAYMENT <<<")
        else:
            print("\n    >>> NOT A PAYMENT (different destination or netuid) <<<")

    valid = [p for p in valid_payments if p["matches_payment"]]

    # If no payment found, scan nearby blocks
    scan_range = 5
    found_block_num = None
    found_block_hash = None
    if not valid and block_num != "?":
        print(f"\n  No payment in block {block_num}. Scanning ±{scan_range} nearby blocks...")
        for offset in range(-scan_range, scan_range + 1):
            if offset == 0:
                continue
            try:
                nearby_num = int(block_num) + offset
                nearby_hash = sub.substrate.get_block_hash(nearby_num)
                nearby_block = sub.substrate.get_block(block_hash=nearby_hash)
                nearby_payments = find_transfer_stake_payments(
                    nearby_block, payment_address, netuid
                )
                nearby_valid = [
                    p for p in nearby_payments if "error" not in p and p["matches_payment"]
                ]
                if nearby_valid:
                    valid = nearby_valid
                    found_block_num = nearby_num
                    found_block_hash = nearby_hash
                    print(f"  >>> Found payment in block {nearby_num} (offset {offset:+d})")
                    for p in valid:
                        print(f"      Extrinsic #{p['extrinsic_index']}: {p['sender_coldkey']}")
                        print(f"      Alpha: {p['alpha_amount']}")
                    break
            except Exception as e:
                print(f"    (block {nearby_num}: fetch failed — {e})")
                continue

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    if valid:
        p = valid[0]
        actual_block = found_block_num or block_num
        actual_hash = found_block_hash or block_hash
        print("  PAYMENT VERIFIED")
        print(f"  Miner coldkey: {p['sender_coldkey']}")
        print(f"  Alpha amount:  {p['alpha_amount']}")
        print(f"  Block:         {actual_block}")
        print(f"  Block hash:    {actual_hash}")
        print(f"  Extrinsic:     #{p['extrinsic_index']}")
        if found_block_num:
            print(
                f"\n  NOTE: Payment was in block {found_block_num}, not the reported {block_num}."
            )
        print("\n  To refund, transfer equivalent TAO to coldkey:")
        print(f"    {p['sender_coldkey']}")
    else:
        print("  NO VALID PAYMENT FOUND")
        print(f"  Searched block {block_num} ± {scan_range} blocks.")


if __name__ == "__main__":
    main()
