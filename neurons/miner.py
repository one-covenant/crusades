"""Miner CLI for Templar Tournament.

R2-Based Architecture:
1. miner upload <train.py>  - Upload your code to R2 bucket
2. miner commit             - Commit R2 credentials + link to blockchain
3. Validator downloads from R2 and evaluates

Environment Variables (R2 config from .env):
  TOURNAMENT_R2_ACCOUNT_ID      - Cloudflare account ID
  TOURNAMENT_R2_BUCKET_NAME     - Bucket name
  TOURNAMENT_R2_ACCESS_KEY_ID   - Access key ID
  TOURNAMENT_R2_SECRET_ACCESS_KEY - Secret access key
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bittensor as bt

from tournament.config import HParams

try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# =============================================================================
# R2 CONFIG
# =============================================================================

def get_r2_config() -> dict | None:
    """Get R2 configuration from environment variables (.env file).
    
    Reads from TOURNAMENT_R2_* environment variables.
    
    Returns:
        Dict with R2 config or None if not configured
    """
    account_id = os.getenv("TOURNAMENT_R2_ACCOUNT_ID")
    bucket = os.getenv("TOURNAMENT_R2_BUCKET_NAME")
    access_key = os.getenv("TOURNAMENT_R2_ACCESS_KEY_ID")
    secret_key = os.getenv("TOURNAMENT_R2_SECRET_ACCESS_KEY")
    
    # Build endpoint from account ID
    endpoint = None
    if account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    
    if not all([endpoint, bucket, access_key, secret_key]):
        return None
    
    return {
        "endpoint": endpoint,
        "bucket": bucket,
        "access_key": access_key,
        "secret_key": secret_key,
    }


# =============================================================================
# R2 UPLOAD
# =============================================================================

def upload_to_r2(
    file_path: Path,
    hotkey: str,
    r2_config: dict | None = None,
) -> tuple[bool, str | dict]:
    """Upload train.py to R2 bucket.
    
    Args:
        file_path: Path to train.py file
        hotkey: Miner's hotkey (for unique path)
        r2_config: R2 config dict (from get_r2_config)
        
    Returns:
        Tuple of (success, result_dict or error_message)
    """
    if not HAS_BOTO3:
        return False, "boto3 not installed. Run: pip install boto3"
    
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    # Get R2 config from env if not provided
    if r2_config is None:
        r2_config = get_r2_config()
    
    if r2_config is None:
        return False, (
            "R2 not configured. Set in .env file:\n"
            "  TOURNAMENT_R2_ACCOUNT_ID\n"
            "  TOURNAMENT_R2_BUCKET_NAME\n"
            "  TOURNAMENT_R2_ACCESS_KEY_ID\n"
            "  TOURNAMENT_R2_SECRET_ACCESS_KEY"
        )
    
    # Read and validate file
    try:
        code = file_path.read_text()
    except Exception as e:
        return False, f"Failed to read file: {e}"
    
    # Basic validation
    if "def inner_steps" not in code:
        return False, "train.py must contain 'def inner_steps' function"
    
    # Generate unique key per miner
    timestamp = int(time.time())
    hotkey_short = hotkey[:16] if len(hotkey) > 16 else hotkey
    r2_key = f"submissions/{hotkey_short}/{timestamp}/train.py"
    
    print(f"Uploading to R2...")
    print(f"   Bucket: {r2_config['bucket']}")
    print(f"   Key: {r2_key}")
    
    try:
        # Create S3 client for R2
        s3 = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint"],
            aws_access_key_id=r2_config["access_key"],
            aws_secret_access_key=r2_config["secret_key"],
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3},
            ),
        )
        
        # Upload file
        s3.put_object(
            Bucket=r2_config["bucket"],
            Key=r2_key,
            Body=code.encode(),
            ContentType="text/x-python",
        )
        
        print(f"\nUpload successful!")
        
        # Result includes all info needed for commitment
        result = {
            "r2_endpoint": r2_config["endpoint"],
            "r2_bucket": r2_config["bucket"],
            "r2_key": r2_key,
            "r2_access_key": r2_config["access_key"],
            "r2_secret_key": r2_config["secret_key"],
        }
        
        return True, result
        
    except Exception as e:
        return False, f"R2 upload failed: {e}"


# =============================================================================
# BLOCKCHAIN COMMIT
# =============================================================================

def commit_to_chain(
    wallet: bt.wallet,
    r2_info: dict,
    network: str = "finney",
) -> tuple[bool, dict | str]:
    """Commit R2 credentials + link to blockchain.
    
    The commitment contains all info needed for validator to download your code:
    - R2 endpoint, bucket, key
    - R2 credentials (access_key, secret_key)
    
    After reveal_blocks (from hparams.json), this info becomes visible to validators.
    netuid and reveal_blocks are read from hparams.json (not user-controlled).
    
    Args:
        wallet: Bittensor wallet
        r2_info: Dict with R2 upload info from upload_to_r2()
        network: Subtensor network (finney, test, or local)
        
    Returns:
        Tuple of (success, result_dict or error_message)
    """
    # Load netuid and reveal_blocks from hparams (not user-controlled)
    hparams = HParams.load()
    netuid = hparams.netuid
    blocks_until_reveal = hparams.reveal_blocks
    
    print(f"\nCommitting to blockchain...")
    print(f"   Network: {network}")
    print(f"   Subnet: {netuid} (from hparams.json)")
    print(f"   Hotkey: {wallet.hotkey.ss58_address}")
    print(f"   Reveal blocks: {blocks_until_reveal} (from hparams.json)")
    
    # Create commitment data with R2 credentials
    # Validator will use these to download train.py from miner's R2
    commitment_data = json.dumps({
        "r2_endpoint": r2_info["r2_endpoint"],
        "r2_bucket": r2_info["r2_bucket"],
        "r2_key": r2_info["r2_key"],
        "r2_access_key": r2_info["r2_access_key"],
        "r2_secret_key": r2_info["r2_secret_key"],
    }, separators=(',', ':'))
    
    print(f"   Commitment size: {len(commitment_data)} bytes")
    
    # Connect to blockchain
    print(f"\nConnecting to {network}...")
    try:
        subtensor = bt.subtensor(network=network)
        current_block = subtensor.get_current_block()
        print(f"   Current block: {current_block}")
    except Exception as e:
        return False, f"Failed to connect to {network}: {e}"
    
    # Committing to blockchain using set_reveal_commitment (timelock encrypted)
    # Data is encrypted until reveal_block, then validators can read it
    print(f"\nCommitting to chain...")
    
    try:
        success = False
        
        if hasattr(subtensor, 'set_reveal_commitment'):
            try:
                success = subtensor.set_reveal_commitment(
                    wallet=wallet,
                    netuid=netuid,
                    data=commitment_data,
                    blocks_until_reveal=blocks_until_reveal,
                )
            except Exception as e:
                return False, f"Commit failed: {e}"
        else:
            return False, "Subtensor does not support set_reveal_commitment()"
        
        if success:
            commit_block = subtensor.get_current_block()
            reveal_block = commit_block + blocks_until_reveal
            
            result = {
                **r2_info,
                "commit_block": commit_block,
                "reveal_block": reveal_block,
                "hotkey": wallet.hotkey.ss58_address,
                "netuid": netuid,
            }
            
            print(f"\nCommitment successful!")
            print(f"   Commit block: {commit_block}")
            print(f"   Reveal block: {reveal_block}")
            print(f"\nValidators will evaluate after block {reveal_block}")
            
            return True, result
        else:
            return False, "Commitment transaction failed"
            
    except Exception as e:
        return False, f"Blockchain error: {e}"


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_upload(args):
    """Upload train.py to R2."""
    # Get wallet for hotkey
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    hotkey = wallet.hotkey.ss58_address
    
    # Get R2 config from .env
    r2_config = get_r2_config()
    
    # Override with CLI args if provided
    if args.r2_endpoint:
        r2_config = r2_config or {}
        r2_config["endpoint"] = args.r2_endpoint
    if args.r2_bucket:
        r2_config = r2_config or {}
        r2_config["bucket"] = args.r2_bucket
    if args.r2_access_key:
        r2_config = r2_config or {}
        r2_config["access_key"] = args.r2_access_key
    if args.r2_secret_key:
        r2_config = r2_config or {}
        r2_config["secret_key"] = args.r2_secret_key
    
    success, result = upload_to_r2(
        file_path=args.train_path,
        hotkey=hotkey,
        r2_config=r2_config,
    )
    
    if success:
        print(f"\n✓ Upload complete!")
        print(f"\nTo commit, use the submit command which uploads and commits in one step:")
        print(f"  uv run python -m neurons.miner submit <train.py> --wallet.name {args.wallet_name} --wallet.hotkey {args.wallet_hotkey}")
        return 0
    else:
        print(f"\nUpload failed: {result}")
        return 1


def cmd_commit(args):
    """Commit R2 info to blockchain.
    
    Note: This command requires --upload-info with the upload result JSON.
    For most users, use 'submit' command instead which handles upload + commit.
    """
    if not args.upload_info:
        print("Error: --upload-info is required")
        print("\nFor easier usage, use the 'submit' command which uploads and commits in one step:")
        print("  uv run python -m neurons.miner submit <train.py> --wallet.name <name> --wallet.hotkey <hotkey>")
        return 1
    
    upload_info_file = Path(args.upload_info)
    
    if not upload_info_file.exists():
        print(f"Error: Upload info file not found: {upload_info_file}")
        return 1
    
    try:
        r2_info = json.loads(upload_info_file.read_text())
    except Exception as e:
        print(f"Error reading upload info: {e}")
        return 1
    
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    
    success, result = commit_to_chain(
        wallet=wallet,
        r2_info=r2_info,
        network=args.network,
    )
    
    if success:
        print(f"\n✓ Commitment successful!")
        print(f"   Commit block: {result['commit_block']}")
        print(f"   Reveal block: {result['reveal_block']}")
        return 0
    else:
        print(f"\nCommit failed: {result}")
        return 1


def cmd_submit(args):
    """Upload and commit in one step."""
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    hotkey = wallet.hotkey.ss58_address
    
    # Get R2 config from .env
    r2_config = get_r2_config()
    
    # Override with CLI args if provided
    if args.r2_endpoint:
        r2_config = r2_config or {}
        r2_config["endpoint"] = args.r2_endpoint
    if args.r2_bucket:
        r2_config = r2_config or {}
        r2_config["bucket"] = args.r2_bucket
    if args.r2_access_key:
        r2_config = r2_config or {}
        r2_config["access_key"] = args.r2_access_key
    if args.r2_secret_key:
        r2_config = r2_config or {}
        r2_config["secret_key"] = args.r2_secret_key
    
    # Upload
    print("=" * 60)
    print("STEP 1: UPLOAD TO R2")
    print("=" * 60)
    
    success, r2_info = upload_to_r2(
        file_path=args.train_path,
        hotkey=hotkey,
        r2_config=r2_config,
    )
    
    if not success:
        print(f"\nUpload failed: {r2_info}")
        return 1
    
    # Commit
    print("\n" + "=" * 60)
    print("STEP 2: COMMIT TO BLOCKCHAIN")
    print("=" * 60)
    
    success, result = commit_to_chain(
        wallet=wallet,
        r2_info=r2_info,
        network=args.network,
    )
    
    if success:
        print("\n" + "=" * 60)
        print("SUBMISSION COMPLETE!")
        print("=" * 60)
        print(f"\nYour code is committed to the blockchain.")
        print(f"Validators will evaluate after block {result['reveal_block']}")
        return 0
    else:
        print(f"\nCommit failed: {result}")
        return 1


def cmd_status(args):
    """Check blockchain status and connection."""
    try:
        print(f"\nConnecting to {args.network}...")
        subtensor = bt.subtensor(network=args.network)
        current_block = subtensor.get_current_block()
        
        # Load hparams for netuid
        hparams = HParams.load()
        
        print(f"\n✓ Connected to blockchain")
        print(f"   Network: {args.network}")
        print(f"   Current block: {current_block}")
        print(f"   Subnet: {hparams.netuid}")
        print(f"   Reveal blocks: {hparams.reveal_blocks}")
        
        # Check if subnet exists
        if subtensor.subnet_exists(hparams.netuid):
            print(f"\n✓ Subnet {hparams.netuid} exists")
            meta = bt.metagraph(netuid=hparams.netuid, network=args.network)
            print(f"   Neurons: {meta.n.item()}")
        else:
            print(f"\n⚠ Subnet {hparams.netuid} does not exist on {args.network}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Templar Tournament Miner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test locally first (in local_test directory)
  cd local_test && uv run python train.py

  # Submit to blockchain (uploads to R2 and commits)
  uv run python -m neurons.miner submit local_test/train.py \\
      --wallet.name miner --wallet.hotkey default --network finney

  # Check blockchain status
  uv run python -m neurons.miner status --network finney

Settings from hparams.json (not user-controlled):
  netuid        - Subnet ID
  reveal_blocks - Blocks until commitment revealed

Environment Variables (set in .env):
  TOURNAMENT_R2_ACCOUNT_ID        - Cloudflare account ID
  TOURNAMENT_R2_BUCKET_NAME       - Bucket name  
  TOURNAMENT_R2_ACCESS_KEY_ID     - Access key ID
  TOURNAMENT_R2_SECRET_ACCESS_KEY - Secret access key
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # R2 credential arguments (shared helper)
    def add_r2_args(p):
        p.add_argument("--r2-endpoint", help="R2/S3 endpoint URL (override .env)")
        p.add_argument("--r2-bucket", help="Bucket name (override .env)")
        p.add_argument("--r2-access-key", help="Access key ID (override .env)")
        p.add_argument("--r2-secret-key", help="Secret access key (override .env)")
    
    # UPLOAD command
    upload_parser = subparsers.add_parser("upload", help="Upload train.py to R2")
    upload_parser.add_argument("train_path", type=Path, help="Path to train.py")
    upload_parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    upload_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    add_r2_args(upload_parser)
    upload_parser.set_defaults(func=cmd_upload)
    
    # COMMIT command
    commit_parser = subparsers.add_parser("commit", help="Commit R2 info to blockchain")
    commit_parser.add_argument("--upload-info", help="Path to upload info JSON")
    commit_parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    commit_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    commit_parser.add_argument("--network", default="finney", help="Network: finney (mainnet), test, or local")
    commit_parser.set_defaults(func=cmd_commit)
    
    # SUBMIT command (upload + commit)
    submit_parser = subparsers.add_parser("submit", help="Upload and commit in one step")
    submit_parser.add_argument("train_path", type=Path, help="Path to train.py")
    add_r2_args(submit_parser)
    submit_parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    submit_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    submit_parser.add_argument("--network", default="finney", help="Network: finney (mainnet), test, or local")
    submit_parser.set_defaults(func=cmd_submit)
    
    # STATUS command
    status_parser = subparsers.add_parser("status", help="Check commitment status")
    status_parser.add_argument("--network", default="finney", help="Network: finney (mainnet), test, or local")
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
