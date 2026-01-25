"""Miner CLI for Templar Tournament.

URL-Based Architecture:
1. Host your train.py code at any URL (Gist, Pastebin, raw GitHub, etc.)
2. Run: miner submit <code_url>
3. The URL is timelock encrypted on blockchain
4. After reveal, validators fetch and evaluate your code

The URL acts as a secret - only those who know it can access.
Timelock encryption keeps it hidden until reveal_blocks pass.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

import bittensor as bt

from tournament.config import HParams


def validate_code_url(url: str) -> tuple[bool, str]:
    """Validate that the URL points to a SINGLE valid train.py file.
    
    Accepts ANY URL that returns valid Python code with inner_steps function.
    Examples: GitHub Gist, raw GitHub, Pastebin, any HTTP/HTTPS URL.
    
    IMPORTANT: Must be a single file URL, not a folder/directory or repo.
    
    Args:
        url: The URL to validate
        
    Returns:
        Tuple of (is_valid, error_message_or_validated_url)
    """
    if not url:
        return False, "URL cannot be empty"
    
    # Must be HTTP or HTTPS
    if not url.startswith("http://") and not url.startswith("https://"):
        return False, "URL must start with http:// or https://"
    
    # Block obvious directory/folder URLs
    blocked_patterns = [
        "/tree/",      # GitHub repo tree
        "/blob/",      # GitHub blob without raw - redirect to raw
        "?tab=",       # GitHub tab navigation  
        "/commits",    # GitHub commits page
        "/pulls",      # GitHub PRs
        "/issues",     # GitHub issues
        "/actions",    # GitHub actions
    ]
    for pattern in blocked_patterns:
        if pattern in url.lower():
            return False, f"URL appears to be a folder/page, not a single file. Use raw file URL."
    
    # For GitHub blob URLs, suggest raw format
    if "github.com" in url and "/blob/" in url:
        return False, "Use raw.githubusercontent.com URL instead of github.com/blob/"
    
    # For GitHub Gist URLs, convert to raw format
    final_url = url
    if "gist.github.com" in url and "/raw" not in url:
        final_url = url.replace("gist.github.com", "gist.githubusercontent.com")
        if not final_url.endswith("/raw"):
            final_url = final_url.rstrip("/") + "/raw"
    
    # Verify the URL is accessible and contains valid code
    try:
        req = urllib.request.Request(final_url, headers={"User-Agent": "templar-tournament"})
        with urllib.request.urlopen(req, timeout=10) as response:
            code = response.read().decode("utf-8")
            
            # Check it's not HTML (indicates folder/page, not file)
            if "<html" in code.lower()[:500] or "<!doctype html" in code.lower()[:500]:
                return False, "URL returns HTML page, not a code file. Use raw file URL."
            
            # Check it's not JSON (could be API response listing files)
            if code.strip().startswith("{") and '"files"' in code[:500]:
                return False, "URL returns JSON (possibly file listing), not a single code file."
            
            # Basic validation - must contain inner_steps
            if "def inner_steps" not in code:
                return False, "Code must contain 'def inner_steps' function"
            
            # Size sanity check (single file should be < 100KB typically)
            if len(code) > 500_000:  # 500KB max
                return False, f"File too large ({len(code)} bytes). Max 500KB for single train.py"
            
            print(f"   ✓ URL accessible ({len(code)} bytes)")
            print(f"   ✓ Single file detected")
            print(f"   ✓ Contains inner_steps function")
            
    except urllib.error.HTTPError as e:
        return False, f"Cannot access URL: HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"Cannot access URL: {e.reason}"
    except Exception as e:
        return False, f"Error validating URL: {e}"
    
    return True, final_url


def commit_to_chain(
    wallet: bt.wallet,
    code_url: str,
    network: str = "finney",
) -> tuple[bool, dict | str]:
    """Commit code URL to blockchain (timelock encrypted).
    
    The URL is encrypted via drand and only revealed after reveal_blocks.
    This keeps your code location private until evaluation time.
    
    Args:
        wallet: Bittensor wallet
        code_url: URL containing train.py code
        network: Subtensor network (finney, test, or local)
        
    Returns:
        Tuple of (success, result_dict or error_message)
    """
    # Load settings from hparams
    hparams = HParams.load()
    netuid = hparams.netuid
    blocks_until_reveal = hparams.reveal_blocks
    block_time = hparams.block_time
    
    print(f"\nCommitting to blockchain...")
    print(f"   Network: {network}")
    print(f"   Subnet: {netuid} (from hparams.json)")
    print(f"   Hotkey: {wallet.hotkey.ss58_address}")
    print(f"   Reveal blocks: {blocks_until_reveal} (from hparams.json)")
    print(f"   Block time: {block_time}s (from hparams.json)")
    
    # Commitment data - just the code URL!
    commitment_data = json.dumps({
        "code_url": code_url,
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
    
    # Commit using timelock encryption (drand)
    print(f"\nCommitting to chain...")
    print(f"   Using set_reveal_commitment (timelock encrypted)")
    
    try:
        if not hasattr(subtensor, 'set_reveal_commitment'):
            return False, "Subtensor does not support set_reveal_commitment()"
        
        success, reveal_round = subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commitment_data,
            blocks_until_reveal=blocks_until_reveal,
            block_time=block_time,
        )
        
        if success:
            commit_block = subtensor.get_current_block()
            reveal_block = commit_block + blocks_until_reveal
            
            result = {
                "code_url": code_url,
                "commit_block": commit_block,
                "reveal_block": reveal_block,
                "reveal_round": reveal_round,
                "hotkey": wallet.hotkey.ss58_address,
                "netuid": netuid,
            }
            
            print(f"\n✓ Commitment successful!")
            print(f"   Commit block: {commit_block}")
            print(f"   Reveal block: {reveal_block}")
            print(f"   Reveal round: {reveal_round}")
            print(f"\nValidators will evaluate after block {reveal_block}")
            
            return True, result
        else:
            return False, "Commitment transaction failed"
            
    except Exception as e:
        return False, f"Blockchain error: {e}"


def cmd_submit(args):
    """Submit a code URL to the tournament."""
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    
    print("=" * 60)
    print("TEMPLAR TOURNAMENT - SUBMIT CODE")
    print("=" * 60)
    print(f"\nWallet: {args.wallet_name}/{args.wallet_hotkey}")
    print(f"Hotkey: {wallet.hotkey.ss58_address}")
    
    # Validate code URL
    print(f"\n--- STEP 1: VALIDATE CODE URL ---")
    print(f"Validating URL...")
    print(f"   URL: {args.code_url}")
    
    valid, result = validate_code_url(args.code_url)
    if not valid:
        print(f"\n✗ Invalid URL: {result}")
        return 1
    
    final_url = result
    print(f"   Final URL: {final_url}")
    
    # Commit to blockchain
    print(f"\n--- STEP 2: COMMIT TO BLOCKCHAIN ---")
    
    success, result = commit_to_chain(
        wallet=wallet,
        code_url=final_url,
        network=args.network,
    )
    
    if success:
        print("\n" + "=" * 60)
        print("SUBMISSION COMPLETE!")
        print("=" * 60)
        print(f"\nYour code URL is now timelock encrypted on the blockchain.")
        print(f"After block {result['reveal_block']}, validators will:")
        print(f"  1. Decrypt and retrieve your code URL")
        print(f"  2. Fetch your train.py code")
        print(f"  3. Evaluate and score your submission")
        print(f"\n⚠️  Do NOT delete or modify your code until evaluation is complete!")
        return 0
    else:
        print(f"\n✗ Commit failed: {result}")
        return 1


def cmd_status(args):
    """Check blockchain status and connection."""
    try:
        print(f"\nConnecting to {args.network}...")
        subtensor = bt.subtensor(network=args.network)
        current_block = subtensor.get_current_block()
        
        # Load hparams
        hparams = HParams.load()
        
        print(f"\n✓ Connected to blockchain")
        print(f"   Network: {args.network}")
        print(f"   Current block: {current_block}")
        print(f"   Subnet: {hparams.netuid}")
        print(f"   Reveal blocks: {hparams.reveal_blocks}")
        print(f"   Block time: {hparams.block_time}s")
        
        # Check if subnet exists
        if subtensor.subnet_exists(hparams.netuid):
            print(f"\n✓ Subnet {hparams.netuid} exists")
            try:
                meta = bt.metagraph(netuid=hparams.netuid, network=args.network)
                print(f"   Neurons: {meta.n.item()}")
            except Exception as e:
                print(f"   (Could not fetch neuron count: {e})")
        else:
            print(f"\n⚠ Subnet {hparams.netuid} does not exist on {args.network}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def cmd_validate(args):
    """Validate a code URL without submitting."""
    print("=" * 60)
    print("VALIDATE CODE URL")
    print("=" * 60)
    print(f"\nValidating: {args.code_url}")
    
    valid, result = validate_code_url(args.code_url)
    
    if valid:
        print(f"\n✓ URL is valid!")
        print(f"   Final URL: {result}")
        print(f"\nTo submit, run:")
        print(f"  uv run python -m neurons.miner submit '{result}' --wallet.name <name> --wallet.hotkey <hotkey>")
        return 0
    else:
        print(f"\n✗ Invalid: {result}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Templar Tournament Miner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How to Submit:

  1. Host your train.py code at any URL:
     - GitHub Gist (secret recommended)
     - Raw GitHub file
     - Pastebin or any paste service
     - Any HTTP/HTTPS URL that returns the code

  2. Submit to the tournament:
     uv run python -m neurons.miner submit <code_url> \\
         --wallet.name miner --wallet.hotkey default --network finney

  3. Your code URL is timelock encrypted - validators can only see it
     after reveal_blocks pass.

Examples:
  # Validate a URL without submitting
  uv run python -m neurons.miner validate https://example.com/train.py

  # Submit to mainnet
  uv run python -m neurons.miner submit https://example.com/train.py \\
      --wallet.name miner --wallet.hotkey default --network finney

  # Check blockchain status
  uv run python -m neurons.miner status --network finney

Settings from hparams.json:
  netuid        - Subnet ID
  reveal_blocks - Blocks until commitment revealed
  block_time    - Seconds per block (for drand calculation)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # SUBMIT command
    submit_parser = subparsers.add_parser("submit", help="Submit a code URL to the tournament")
    submit_parser.add_argument("code_url", help="URL containing your train.py code")
    submit_parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    submit_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    submit_parser.add_argument("--network", default="finney", help="Network: finney (mainnet), test, or local")
    submit_parser.set_defaults(func=cmd_submit)
    
    # VALIDATE command
    validate_parser = subparsers.add_parser("validate", help="Validate a code URL without submitting")
    validate_parser.add_argument("code_url", help="URL to validate")
    validate_parser.set_defaults(func=cmd_validate)
    
    # STATUS command
    status_parser = subparsers.add_parser("status", help="Check blockchain status")
    status_parser.add_argument("--network", default="finney", help="Network: finney (mainnet), test, or local")
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
