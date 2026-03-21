#!/usr/bin/env python3
"""Burn submission fee alpha via the on-chain burn_alpha extrinsic.

This script is intended for the coldkey owner (not the validator process)
to manually burn collected submission fees. The burn_alpha extrinsic must
be signed by the coldkey that holds the alpha stake.

Note: --amount is specified in rao (1 alpha = 1,000,000,000 rao).

Usage:
    # Burn on finney
    uv run scripts/burn_alpha.py --amount 50000000 --network finney --netuid 3 --burn-uid 1 --wallet-name default --wallet-hotkey default

    # Burn on localnet
    uv run scripts/burn_alpha.py --amount 1000000 --network local --netuid 2 --burn-uid 0 --wallet-name V1 --wallet-hotkey V1
"""

import argparse
import sys

import bittensor as bt


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
  uv run scripts/burn_alpha.py --amount 50000000 --network finney --netuid 39 --burn-uid 0 --wallet-name default --wallet-hotkey default
  uv run scripts/burn_alpha.py --amount 1000000 --network local --netuid 2 --burn-uid 0 --wallet-name V1 --wallet-hotkey V1
        """,
    )
    parser.add_argument(
        "--amount", type=int, required=True, help="Amount of alpha to burn (in rao)"
    )
    parser.add_argument("--network", type=str, required=True, help="Network: finney, local, test")
    parser.add_argument("--netuid", type=int, required=True, help="Subnet UID")
    parser.add_argument(
        "--burn-uid", type=int, help="UID to resolve hotkey from (mutually exclusive with --hotkey)"
    )
    parser.add_argument(
        "--hotkey", type=str, help="Hotkey SS58 to burn on (mutually exclusive with --burn-uid)"
    )
    parser.add_argument("--wallet-name", default="default", help="Wallet name (default: 'default')")
    parser.add_argument(
        "--wallet-hotkey", default="default", help="Wallet hotkey name (default: 'default')"
    )
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets", help="Wallet path")
    args = parser.parse_args()

    if not args.hotkey and args.burn_uid is None:
        parser.error("Either --hotkey or --burn-uid must be provided")

    if args.amount <= 0:
        print("ERROR: --amount must be positive")
        sys.exit(1)

    print(f"\nConnecting to {args.network}...")
    sub = bt.subtensor(network=args.network)

    hotkey = args.hotkey
    if not hotkey:
        print(f"  Resolving hotkey for burn_uid {args.burn_uid} on subnet {args.netuid}...")
        hotkey = resolve_hotkey_for_uid(sub, args.netuid, args.burn_uid)
        if not hotkey:
            print(f"  ERROR: Could not resolve hotkey for UID {args.burn_uid}")
            sys.exit(1)

    print("=" * 60)
    print("BURN ALPHA")
    print("=" * 60)
    print(f"  Network:     {args.network}")
    print(f"  Netuid:      {args.netuid}")
    print(f"  Hotkey:      {hotkey}")
    print(f"  Amount:      {args.amount} rao")
    print(f"  Wallet:      {args.wallet_name}")
    print("=" * 60)

    print(f"\n  Loading wallet '{args.wallet_name}'...")
    wallet = bt.wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
        path=args.wallet_path,
    )
    coldkey_ss58 = wallet.coldkeypub.ss58_address
    print(f"  Coldkey: {coldkey_ss58}")

    print("\n  Querying alpha stake before burn...")
    before_stake = None
    try:
        stake_info = sub.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=coldkey_ss58,
            hotkey_ss58=hotkey,
            netuids=[args.netuid],
        )
        if args.netuid in stake_info:
            before_stake = stake_info[args.netuid].stake
            print(f"  Alpha stake BEFORE: {before_stake}")
        else:
            print(f"  No stake found on netuid {args.netuid}")
    except Exception as e:
        print(f"  Could not query stake: {e}")

    call = sub.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burn_alpha",
        call_params={
            "hotkey": hotkey,
            "amount": args.amount,
            "netuid": args.netuid,
        },
    )

    confirm = (
        input(f"\n  Burn {args.amount} rao alpha on netuid {args.netuid}? [y/N] ").strip().lower()
    )
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
            print(f"\n  SUCCESS: Burned {args.amount} rao alpha")
            print(f"  Message: {msg}")
        else:
            print(f"\n  FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print("\n  Querying alpha stake after burn...")
    try:
        stake_info = sub.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=coldkey_ss58,
            hotkey_ss58=hotkey,
            netuids=[args.netuid],
        )
        if args.netuid in stake_info:
            after_stake = stake_info[args.netuid].stake
            print(f"  Alpha stake AFTER:  {after_stake}")
            if before_stake is not None:
                print(
                    f"  Difference:         {before_stake - after_stake} (expected: ~{args.amount} rao)"
                )
        else:
            print(f"  No stake info after burn on netuid {args.netuid}")
    except Exception as e:
        print(f"  Could not query stake after burn: {e}")


if __name__ == "__main__":
    main()
