"""Miner CLI for submitting training code to the tournament.

SECURITY NOTE: Miners submit code to validator API, NOT directly to storage.
Validators handle storage privately to prevent cheating.
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

import bittensor as bt
import httpx

from tournament.chain.manager import ChainManager
from tournament.payment.manager import PaymentManager
from tournament.pipeline.validator import CodeValidator


async def submit_code(
    code_path: Path,
    wallet: bt.wallet,
    skip_validation: bool = False,
    skip_payment: bool = False,
    payment_recipient: str | None = None,
    validator_api_url: str = "http://localhost:8000",
) -> str | None:
    """Submit training code to validator via API.
    
    SECURITY: Code is sent to validator's API endpoint, not directly to storage.
    This prevents miners from accessing or manipulating the storage layer.

    Args:
        code_path: Path to train.py file
        wallet: Bittensor wallet for signing
        skip_validation: Skip local code validation
        skip_payment: Skip payment (for testing only)
        payment_recipient: Validator's payment address
        validator_api_url: Validator API endpoint

    Returns:
        Submission ID if successful, None otherwise
    """
    # Read code
    if not code_path.exists():
        print(f"Error: File not found: {code_path}")
        return None

    code = code_path.read_text()

    # Validate code locally
    if not skip_validation:
        validator = CodeValidator()
        result = validator.validate(code)
        if not result.valid:
            print("Code validation failed:")
            for error in result.errors:
                print(f"  - {error}")
            return None
        print("Code validation passed")

    # Calculate code hash
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    print(f"Code hash: {code_hash}")

    # Initialize chain manager
    chain = ChainManager(wallet=wallet)
    await chain.sync_metagraph()

    # Check if miner is registered
    hotkey = wallet.hotkey.ss58_address
    if not chain.is_registered(hotkey):
        print(f"Error: Hotkey {hotkey} is not registered on subnet {chain.netuid}")
        return None

    uid = chain.get_uid_for_hotkey(hotkey)
    print(f"Miner UID: {uid}")

    # Payment for submission (anti-spam)
    payment_receipt = None
    if not skip_payment:
        if payment_recipient is None:
            print("Error: Payment recipient address required (use --payment-recipient)")
            print("For testing, use --skip-payment")
            return None

        payment_manager = PaymentManager(wallet=wallet, subtensor=chain.subtensor)
        cost_rao, cost_tao = payment_manager.get_submission_cost()

        # Confirm payment
        print(f"\n💰 Submission Cost: {cost_rao:,} RAO ({cost_tao:.4f} TAO)")
        print(f"📍 Payment Recipient: {payment_recipient}")

        confirm = input("\nProceed with payment? (y/n): ").strip().lower()
        if confirm != "y":
            print("Payment cancelled. Submission aborted.")
            return None

        try:
            print("Processing payment...")
            payment_receipt = await payment_manager.make_payment(
                recipient_address=payment_recipient,
            )
            print("✅ Payment confirmed!")
            print(f"   Block: {payment_receipt.block_hash}")
            print(f"   Extrinsic: {payment_receipt.extrinsic_index}")
        except Exception as e:
            print(f"Error: Payment failed: {e}")
            return None
    else:
        print("⚠️  Skipping payment (testing mode)")

    # Submit code to validator API
    print(f"\nSubmitting code to validator API: {validator_api_url}")
    
    # Prepare submission data
    submission_data = {
        "code": code,
        "code_hash": code_hash,
        "miner_hotkey": hotkey,
        "miner_uid": uid,
        "payment_block_hash": payment_receipt.block_hash if payment_receipt else None,
        "payment_extrinsic_index": payment_receipt.extrinsic_index if payment_receipt else None,
        "payment_amount_rao": payment_receipt.amount_rao if payment_receipt else None,
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{validator_api_url}/api/submissions",
                json=submission_data,
            )
            
            if response.status_code != 200:
                print(f"Error: Validator API returned {response.status_code}")
                print(f"       {response.text}")
                return None
            
            result = response.json()
            submission_id = result.get("submission_id")
            
            print("✅ Submission accepted by validator")
            print(f"\n📋 Submission Details:")
            print(f"   ID: {submission_id}")
            print(f"   Code Hash: {code_hash[:16]}...")
            print(f"   Status: pending")
            
            print(f"\n📊 Track your submission:")
            print(f"   curl {validator_api_url}/api/submissions/{submission_id}")
            
            return submission_id
            
    except httpx.RequestError as e:
        print(f"Error: Failed to connect to validator API")
        print(f"       {e}")
        print(f"\n💡 Make sure validator is running at: {validator_api_url}")
        return None
    except Exception as e:
        print(f"Error: Submission failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Submit training code to templar-tournament")
    parser.add_argument(
        "code_path",
        type=Path,
        help="Path to train.py file",
    )
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Wallet name",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip local code validation",
    )
    parser.add_argument(
        "--skip-payment",
        action="store_true",
        help="Skip payment (for local testing only)",
    )
    parser.add_argument(
        "--payment-recipient",
        type=str,
        default=None,
        help="SS58 address to send payment to (validator address)",
    )
    parser.add_argument(
        "--validator-api",
        type=str,
        default="http://localhost:8000",
        help="Validator API endpoint (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    # Initialize wallet
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    # Run submission
    submission_id = asyncio.run(
        submit_code(
            code_path=args.code_path,
            wallet=wallet,
            skip_validation=args.skip_validation,
            skip_payment=args.skip_payment,
            payment_recipient=args.payment_recipient,
            validator_api_url=args.validator_api,
        )
    )

    if submission_id:
        print("\nSubmission successful!")
        print(f"Track status at: /submissions/{submission_id}/status")
        sys.exit(0)
    else:
        print("\nSubmission failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
