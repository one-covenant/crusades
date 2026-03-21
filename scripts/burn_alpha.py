#!/usr/bin/env python3
"""Burn submission fee alpha via the on-chain burn_alpha extrinsic.

This script is intended for the coldkey owner (not the validator process)
to manually burn collected submission fees. The burn_alpha extrinsic must
be signed by the coldkey that holds the alpha stake.

Usage:
    # Burn a specific amount (in rao) on finney
    uv run scripts/burn_alpha.py --amount 50000000

    # Burn on localnet with a custom wallet
    uv run scripts/burn_alpha.py --amount 50000000 --network local --wallet-name my_wallet

    # Dry run (compose the call but don't submit)
    uv run scripts/burn_alpha.py --amount 50000000 --dry-run

    # Override hotkey/netuid instead of reading from hparams
    uv run scripts/burn_alpha.py --amount 50000000 --netuid 2 --hotkey 5GHFz...
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


def resolve_hotkey_for_uid(subtensor: bt.subtensor, netuid: int, uid: int) -> str | None:
    """Get the hotkey SS58 address for a given UID on a subnet."""
    try:
        metagraph = subtensor.metagraph(netuid=netuid)
        if uid < len(metagraph.hotkeys):
            return metagraph.hotkeys[uid]
    except Exception as e:
        print(f"  ERROR resolving hotkey for UID {uid}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Burn submission fee alpha on-chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/burn_alpha.py --amount 50000000
  uv run scripts/burn_alpha.py --amount 50000000 --network local
  uv run scripts/burn_alpha.py --amount 50000000 --dry-run
        """,
    )
    parser.add_argument(
        "--amount", type=int, required=True, help="Amount of alpha to burn (in rao)"
    )
    parser.add_argument(
        "--network", default="finney", help="Network: finney, local, test (default: finney)"
    )
    parser.add_argument("--netuid", type=int, help="Subnet UID (default: from hparams.json)")
    parser.add_argument("--burn-uid", type=int, help="Burn UID (default: from hparams.json)")
    parser.add_argument(
        "--hotkey", type=str, help="Hotkey SS58 to burn on (default: resolved from burn_uid)"
    )
    parser.add_argument("--wallet-name", default="default", help="Wallet name (default: 'default')")
    parser.add_argument(
        "--wallet-hotkey", default="default", help="Wallet hotkey name (default: 'default')"
    )
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets", help="Wallet path")
    parser.add_argument("--dry-run", action="store_true", help="Compose the call but don't submit")
    args = parser.parse_args()

    hparams = load_hparams()
    netuid = args.netuid or hparams.get("netuid", 2)
    burn_uid = args.burn_uid if args.burn_uid is not None else hparams.get("burn_uid", 0)

    if args.amount <= 0:
        print("ERROR: --amount must be positive")
        sys.exit(1)

    print(f"\nConnecting to {args.network}...")
    sub = bt.subtensor(network=args.network)

    hotkey = args.hotkey
    if not hotkey:
        print(f"  Resolving hotkey for burn_uid {burn_uid} on subnet {netuid}...")
        hotkey = resolve_hotkey_for_uid(sub, netuid, burn_uid)
        if not hotkey:
            print(f"  ERROR: Could not resolve hotkey for UID {burn_uid}")
            sys.exit(1)

    print("=" * 60)
    print("BURN ALPHA")
    print("=" * 60)
    print(f"  Network:     {args.network}")
    print(f"  Netuid:      {netuid}")
    print(f"  Burn UID:    {burn_uid}")
    print(f"  Hotkey:      {hotkey}")
    print(f"  Amount:      {args.amount} rao")
    print(f"  Wallet:      {args.wallet_name}")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 60)

    call = sub.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burn_alpha",
        call_params={
            "hotkey": hotkey,
            "amount": args.amount,
            "netuid": netuid,
        },
    )

    if args.dry_run:
        print("\n  [DRY RUN] Call composed successfully:")
        print("    Module:   SubtensorModule")
        print("    Function: burn_alpha")
        print(f"    Params:   hotkey={hotkey[:20]}..., amount={args.amount}, netuid={netuid}")
        print("\n  No transaction submitted.")
        return

    print(f"\n  Loading wallet '{args.wallet_name}'...")
    wallet = bt.wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
        path=args.wallet_path,
    )
    print(f"  Coldkey: {wallet.coldkeypub.ss58_address}")

    confirm = input(f"\n  Burn {args.amount} alpha on netuid {netuid}? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    print("\n  Submitting burn_alpha extrinsic...")
    try:
        success, msg = sub.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if success:
            print(f"\n  SUCCESS: Burned {args.amount} alpha")
            print(f"  Message: {msg}")
        else:
            print(f"\n  FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
