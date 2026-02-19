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


def load_hparams():
    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    if hparams_path.exists():
        with open(hparams_path) as f:
            return json.load(f)
    return {}


def resolve_burn_hotkey(sub, netuid, burn_uid):
    metagraph = sub.metagraph(netuid)
    if burn_uid >= len(metagraph.hotkeys):
        return None
    return metagraph.hotkeys[burn_uid]


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


def find_stake_payments(block, netuid, burn_hotkey):
    """Find all add_stake extrinsics in a block targeting the burn hotkey."""
    payments = []
    extrinsics = block.get("extrinsics", [])

    for idx, ext in enumerate(extrinsics):
        try:
            call = ext.value.get("call", {})
            call_module = call.get("call_module", "")
            call_function = call.get("call_function", "")

            if call_module == "SubtensorModule" and call_function == "add_stake":
                params = {p["name"]: p["value"] for p in call.get("call_args", [])}

                hotkey_raw = params.get("hotkey", "")
                if isinstance(hotkey_raw, dict):
                    hotkey_raw = hotkey_raw.get("Id", hotkey_raw)

                target_netuid = params.get("netuid", None)
                amount_raw = params.get("amount_staked", 0)

                sender = ext.value.get("address", "unknown")
                if isinstance(sender, dict):
                    sender = sender.get("Id", sender)

                is_match = str(hotkey_raw) == str(burn_hotkey) and (
                    target_netuid is None or int(target_netuid) == int(netuid)
                )

                payments.append(
                    {
                        "extrinsic_index": idx,
                        "sender_coldkey": sender,
                        "target_hotkey": hotkey_raw,
                        "target_netuid": target_netuid,
                        "amount_rao": amount_raw,
                        "amount_tao": amount_raw / 1e9 if amount_raw else 0,
                        "matches_burn": is_match,
                        "call_function": call_function,
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
    parser.add_argument("--netuid", type=int, help="Subnet UID (default: from hparams.json)")
    parser.add_argument("--burn-uid", type=int, help="Burn UID (default: from hparams.json)")
    args = parser.parse_args()

    if not args.block_hash and args.block_number is None:
        parser.error("Provide a block hash or --block-number")

    hparams = load_hparams()
    netuid = args.netuid or hparams.get("netuid", 2)
    burn_uid = args.burn_uid if args.burn_uid is not None else hparams.get("burn_uid", 0)
    fee_rao = hparams.get("payment", {}).get("fee_rao", 100_000_000)

    print("=" * 60)
    print("PAYMENT VERIFICATION")
    print("=" * 60)
    print(f"  Network:      {args.network}")
    print(f"  Netuid:       {netuid}")
    print(f"  Burn UID:     {burn_uid}")
    print(f"  Required fee: {fee_rao:,} RAO ({fee_rao / 1e9:.4f} TAO)")

    print(f"\nConnecting to {args.network}...")
    sub = bt.subtensor(network=args.network)

    burn_hotkey = resolve_burn_hotkey(sub, netuid, burn_uid)
    if not burn_hotkey:
        print(f"  ERROR: Could not resolve burn UID {burn_uid} on subnet {netuid}")
        sys.exit(1)
    print(f"  Burn hotkey:  {burn_hotkey}")

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
    print("STAKE PAYMENTS (add_stake to burn address)")
    print(f"{'=' * 60}")
    payments = find_stake_payments(block, netuid, burn_hotkey)
    stake_payments = [p for p in payments if "error" not in p]

    if not stake_payments:
        print("  No add_stake extrinsics found in this block.")

    for p in stake_payments:
        match_str = "YES" if p["matches_burn"] else "NO"
        sufficient = p["amount_rao"] >= fee_rao if p["amount_rao"] else False
        sufficient_str = "YES" if sufficient else "NO"

        print(f"\n  Extrinsic #{p['extrinsic_index']}:")
        print(f"    Sender (coldkey): {p['sender_coldkey']}")
        print(f"    Target hotkey:    {p['target_hotkey']}")
        print(f"    Target netuid:    {p['target_netuid']}")
        print(f"    Amount:           {p['amount_rao']:,} RAO ({p['amount_tao']:.4f} TAO)")
        print(f"    Matches burn:     {match_str}")
        print(f"    Sufficient:       {sufficient_str} (required: {fee_rao:,} RAO)")

        if p["matches_burn"] and sufficient:
            print("\n    >>> VALID PAYMENT <<<")
        elif p["matches_burn"] and not sufficient:
            print(f"\n    >>> UNDERPAYMENT (paid {p['amount_rao']:,}, need {fee_rao:,}) <<<")
        else:
            print("\n    >>> NOT A BURN PAYMENT (different target) <<<")

    valid = [p for p in stake_payments if p["matches_burn"] and p.get("amount_rao", 0) >= fee_rao]

    # If no payment found, scan nearby blocks (miner CLI often reports a
    # slightly later block than where the extrinsic actually landed).
    SCAN_RANGE = 5
    found_block_num = None
    found_block_hash = None
    if not valid and block_num != "?":
        print(f"\n  No payment in block {block_num}. Scanning ±{SCAN_RANGE} nearby blocks...")
        for offset in range(-SCAN_RANGE, SCAN_RANGE + 1):
            if offset == 0:
                continue
            try:
                nearby_num = int(block_num) + offset
                nearby_hash = sub.substrate.get_block_hash(nearby_num)
                nearby_block = sub.substrate.get_block(block_hash=nearby_hash)
                nearby_payments = find_stake_payments(nearby_block, netuid, burn_hotkey)
                nearby_valid = [
                    p
                    for p in nearby_payments
                    if "error" not in p and p["matches_burn"] and p.get("amount_rao", 0) >= fee_rao
                ]
                if nearby_valid:
                    valid = nearby_valid
                    found_block_num = nearby_num
                    found_block_hash = nearby_hash
                    print(f"  >>> Found payment in block {nearby_num} (offset {offset:+d})")
                    for p in valid:
                        print(f"      Extrinsic #{p['extrinsic_index']}: {p['sender_coldkey']}")
                        print(f"      Amount: {p['amount_rao']:,} RAO ({p['amount_tao']:.4f} TAO)")
                    break
            except Exception:
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
        print(f"  Amount:        {p['amount_rao']:,} RAO ({p['amount_tao']:.4f} TAO)")
        print(f"  Block:         {actual_block}")
        print(f"  Block hash:    {actual_hash}")
        print(f"  Extrinsic:     #{p['extrinsic_index']}")
        if found_block_num:
            print(
                f"\n  NOTE: Payment was in block {found_block_num}, not the reported {block_num}."
            )
            print("  This is normal — add_stake() can report a slightly later block.")
        print(f"\n  To refund, send {p['amount_tao']:.4f} TAO to coldkey:")
        print(f"    {p['sender_coldkey']}")
    else:
        print("  NO VALID PAYMENT FOUND")
        print(f"  Searched block {block_num} ± {SCAN_RANGE} blocks.")


if __name__ == "__main__":
    main()
