"""Miner CLI for Templar Tournament (Chi/Affinetes Architecture).

Flow:
1. miner build <train.py>  - Build Docker image with your optimized code
2. miner push              - Push image to registry (optional for local testing)
3. miner commit            - Commit image URL to blockchain
4. Validator reads chain → runs via affinetes → scores your submission
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import bittensor as bt


# =============================================================================
# DOCKER BUILD COMMAND
# =============================================================================

def build_docker_image(
    train_path: Path,
    image_name: str,
    image_tag: str = "latest",
    env_path: Path | None = None,
    no_cache: bool = False,
) -> tuple[bool, str]:
    """Build Docker image with miner's train.py.
    
    Args:
        train_path: Path to miner's train.py file
        image_name: Docker image name (e.g., "templar-submission")
        image_tag: Docker image tag (e.g., "v1" or "latest")
        env_path: Path to environment files (defaults to environments/templar/)
        no_cache: Force rebuild without cache
        
    Returns:
        Tuple of (success, image_full_name or error_message)
    """
    if not train_path.exists():
        return False, f"File not found: {train_path}"
    
    # Find environment directory
    if env_path is None:
        candidates = [
            Path(__file__).parent.parent / "environments" / "templar",
            Path.cwd() / "environments" / "templar",
            Path.cwd().parent / "environments" / "templar",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "Dockerfile").exists():
                env_path = candidate
                break
        
        if env_path is None:
            return False, "Could not find environments/templar/ directory with Dockerfile"
    
    required_files = ["Dockerfile", "env.py", "requirements.txt"]
    for f in required_files:
        if not (env_path / f).exists():
            return False, f"Missing required file: {env_path / f}"
    
    print(f"📦 Building Docker image: {image_name}:{image_tag}")
    print(f"   Environment: {env_path}")
    print(f"   Train.py: {train_path}")
    
    with tempfile.TemporaryDirectory(prefix="templar_build_") as tmp_dir:
        build_ctx = Path(tmp_dir)
        
        # Copy environment files
        for f in required_files:
            shutil.copy(env_path / f, build_ctx / f)
        
        # Copy miner's train.py
        shutil.copy(train_path, build_ctx / "train.py")
        
        # Calculate code hash
        code = train_path.read_text()
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        # Add metadata labels
        dockerfile_path = build_ctx / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()
        labels = f"""
# Metadata labels
LABEL templar.code_hash="{code_hash}"
LABEL templar.build_time="{int(time.time())}"
"""
        dockerfile_path.write_text(dockerfile_content + labels)
        
        # Build Docker image
        full_image_name = f"{image_name}:{image_tag}"
        cmd = ["docker", "build", "-t", full_image_name]
        
        if no_cache:
            cmd.append("--no-cache")
        
        cmd.append(str(build_ctx))
        
        print(f"\n🔨 Running: {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, cwd=build_ctx)
            
            if result.returncode != 0:
                return False, f"Docker build failed with exit code {result.returncode}"
            
            print("-" * 60)
            print(f"\n✅ Successfully built: {full_image_name}")
            print(f"   Code hash: {code_hash[:16]}...")
            
            inspect_result = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", full_image_name],
                capture_output=True,
                text=True,
            )
            if inspect_result.returncode == 0:
                image_id = inspect_result.stdout.strip()[:19]
                print(f"   Image ID: {image_id}")
            
            return True, full_image_name
            
        except FileNotFoundError:
            return False, "Docker not found. Please install Docker."
        except Exception as e:
            return False, f"Docker build error: {e}"


# =============================================================================
# PUSH TO REGISTRY COMMAND
# =============================================================================

def push_to_registry(
    image_name: str,
    registry: str | None = None,
) -> tuple[bool, str]:
    """Push Docker image to registry.
    
    Args:
        image_name: Local image name (e.g., "templar-submission:v1")
        registry: Registry URL (e.g., "docker.io/username" or "localhost:5000")
        
    Returns:
        Tuple of (success, remote_image_name or error_message)
    """
    if registry:
        remote_name = f"{registry}/{image_name}"
        
        print(f"\n📤 Pushing to registry: {registry}")
        print(f"   Local: {image_name}")
        print(f"   Remote: {remote_name}")
        
        # Tag image
        tag_cmd = ["docker", "tag", image_name, remote_name]
        tag_result = subprocess.run(tag_cmd, capture_output=True, text=True)
        
        if tag_result.returncode != 0:
            return False, f"Failed to tag image: {tag_result.stderr}"
        
        # Push image
        push_cmd = ["docker", "push", remote_name]
        push_result = subprocess.run(push_cmd, capture_output=False, text=True)
        
        if push_result.returncode != 0:
            return False, f"Failed to push image"
        
        print(f"\n✅ Pushed: {remote_name}")
        return True, remote_name
    else:
        print(f"\n📦 Local image (no registry push): {image_name}")
        print(f"   For local testing, validators will use this image directly")
        return True, image_name


# =============================================================================
# BLOCKCHAIN COMMIT COMMAND
# =============================================================================

def commit_to_chain(
    wallet: bt.wallet,
    image_name: str,
    netuid: int,
    network: str = "local",
    blocks_until_reveal: int = 100,
    local_mode: bool = False,
) -> tuple[bool, dict | str]:
    """Commit Docker image URL to blockchain using commit-reveal.
    
    Args:
        wallet: Bittensor wallet
        image_name: Full Docker image name (e.g., "registry/name:tag")
        netuid: Subnet ID
        network: Subtensor network (local, test, finney)
        blocks_until_reveal: Blocks until commitment is revealed
        local_mode: Use local file instead of blockchain (for testing)
        
    Returns:
        Tuple of (success, result_dict or error_message)
    """
    print(f"\n🔗 Committing image...")
    print(f"   Image: {image_name}")
    print(f"   Network: {network}")
    print(f"   Subnet: {netuid}")
    
    image_hash = hashlib.sha256(image_name.encode()).hexdigest()[:16]
    commitment_data = f"docker|{image_name}|{image_hash}"
    
    try:
        subtensor = bt.subtensor(network=network)
        current_block = subtensor.get_current_block()
        print(f"   Current block: {current_block}")
    except Exception:
        current_block = int(time.time())
        print(f"   Using timestamp as block: {current_block}")
    
    # For local testing, use file-based commitments
    if local_mode or network == "local":
        print(f"\n📝 Using LOCAL FILE mode for testing...")
        
        commit_block = current_block
        reveal_block = commit_block + blocks_until_reveal
        
        result = {
            "image": image_name,
            "image_hash": image_hash,
            "commit_block": commit_block,
            "reveal_block": reveal_block,
            "hotkey": wallet.hotkey.ss58_address,
            "uid": 1,  # Default UID for local testing
            "netuid": netuid,
            "data": commitment_data,
        }
        
        # Save to local commitments directory
        local_commits_dir = Path.cwd() / ".local_commitments"
        local_commits_dir.mkdir(exist_ok=True)
        
        commit_file = local_commits_dir / f"{commit_block}_{wallet.hotkey.ss58_address[:8]}.json"
        commit_file.write_text(json.dumps(result, indent=2))
        
        print(f"\n✅ Local commitment saved!")
        print(f"   File: {commit_file}")
        print(f"   Commit block: {commit_block}")
        print(f"   Hotkey: {wallet.hotkey.ss58_address}")
        
        return True, result
    
    # Production: Use blockchain
    print(f"\n📝 Commitment data: {commitment_data}")
    
    try:
        success = False
        if hasattr(subtensor, 'set_reveal_commitment'):
            try:
                print(f"   Trying set_reveal_commitment...")
                success = subtensor.set_reveal_commitment(
                    wallet=wallet,
                    netuid=netuid,
                    data=commitment_data,
                    blocks_until_reveal=blocks_until_reveal,
                )
            except Exception as e:
                print(f"   ⚠️  set_reveal_commitment failed: {e}")
                success = False
        
        if not success and hasattr(subtensor, 'commit'):
            print(f"   Using simple commit...")
            try:
                success = subtensor.commit(
                    wallet=wallet,
                    netuid=netuid,
                    data=commitment_data,
                )
            except Exception as e:
                return False, f"Commit failed: {e}"
        
        if success:
            commit_block = subtensor.get_current_block()
            reveal_block = commit_block + blocks_until_reveal
            
            result = {
                "image": image_name,
                "image_hash": image_hash,
                "commit_block": commit_block,
                "reveal_block": reveal_block,
                "hotkey": wallet.hotkey.ss58_address,
                "netuid": netuid,
            }
            
            print(f"\n✅ Commitment successful!")
            print(f"   Commit block: {commit_block}")
            print(f"   Reveal block: {reveal_block}")
            
            return True, result
        else:
            return False, "Commitment transaction failed"
            
    except Exception as e:
        return False, f"Blockchain error: {e}"


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_build(args):
    """Build Docker image with miner's train.py."""
    success, result = build_docker_image(
        train_path=args.train_path,
        image_name=args.image_name,
        image_tag=args.image_tag,
        no_cache=args.no_cache,
    )
    
    if success:
        print(f"\n🎉 Image ready: {result}")
        print(f"\nNext steps:")
        print(f"  1. (Optional) Push to registry:")
        print(f"     python -m neurons.miner push {result} --registry docker.io/yourname")
        print(f"  2. Commit to blockchain:")
        print(f"     python -m neurons.miner commit --image {result} --wallet.name <name> --wallet.hotkey <hotkey>")
        return 0
    else:
        print(f"\n❌ Build failed: {result}")
        return 1


def cmd_push(args):
    """Push Docker image to registry."""
    success, result = push_to_registry(
        image_name=args.image_name,
        registry=args.registry,
    )
    
    if success:
        print(f"\n🎉 Image pushed: {result}")
        print(f"\nNext step:")
        print(f"  python -m neurons.miner commit --image {result} --wallet.name <name> --wallet.hotkey <hotkey>")
        return 0
    else:
        print(f"\n❌ Push failed: {result}")
        return 1


def cmd_commit(args):
    """Commit Docker image to blockchain."""
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    
    success, result = commit_to_chain(
        wallet=wallet,
        image_name=args.image_name,
        netuid=args.netuid,
        network=args.network,
        blocks_until_reveal=args.reveal_blocks,
    )
    
    if success:
        print(f"\n🎉 Commitment recorded on blockchain!")
        print(f"\nValidators will evaluate your submission after block {result['reveal_block']}")
        
        # Save commitment info
        commit_file = Path.home() / ".templar" / "commits" / f"{result['commit_block']}.json"
        commit_file.parent.mkdir(parents=True, exist_ok=True)
        commit_file.write_text(json.dumps(result, indent=2))
        print(f"\nCommitment saved: {commit_file}")
        
        return 0
    else:
        print(f"\n❌ Commit failed: {result}")
        return 1


def cmd_status(args):
    """Check commitment status on blockchain."""
    try:
        subtensor = bt.subtensor(network=args.network)
        current_block = subtensor.get_current_block()
        
        print(f"\n📊 Blockchain Status")
        print(f"   Network: {args.network}")
        print(f"   Current block: {current_block}")
        
        commits_dir = Path.home() / ".templar" / "commits"
        if commits_dir.exists():
            commits = sorted(commits_dir.glob("*.json"), reverse=True)[:5]
            
            if commits:
                print(f"\n📝 Recent Commitments:")
                for commit_file in commits:
                    data = json.loads(commit_file.read_text())
                    commit_block = data.get("commit_block", 0)
                    reveal_block = data.get("reveal_block", 0)
                    image = data.get("image", "unknown")
                    
                    if current_block >= reveal_block:
                        status = "✅ REVEALED"
                    else:
                        blocks_left = reveal_block - current_block
                        status = f"⏳ Hidden ({blocks_left} blocks left)"
                    
                    print(f"   Block {commit_block}: {image[:40]}... - {status}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Templar Tournament Miner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build Docker image with your optimized train.py
  python -m neurons.miner build train.py --image templar-submission --tag v1

  # Push to registry (optional, for production)
  python -m neurons.miner push templar-submission:v1 --registry docker.io/myuser

  # Commit to blockchain
  python -m neurons.miner commit --image templar-submission:v1 --wallet.name miner --wallet.hotkey default

  # Check commitment status
  python -m neurons.miner status --network local
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # BUILD command
    build_parser = subparsers.add_parser("build", help="Build Docker image with your train.py")
    build_parser.add_argument("train_path", type=Path, help="Path to your train.py file")
    build_parser.add_argument("--image", dest="image_name", default="templar-submission", help="Docker image name")
    build_parser.add_argument("--tag", dest="image_tag", default="latest", help="Docker image tag")
    build_parser.add_argument("--no-cache", action="store_true", help="Build without cache")
    build_parser.set_defaults(func=cmd_build)
    
    # PUSH command
    push_parser = subparsers.add_parser("push", help="Push image to registry")
    push_parser.add_argument("image_name", help="Local image name (e.g., templar-submission:v1)")
    push_parser.add_argument("--registry", help="Registry URL (e.g., docker.io/username)")
    push_parser.set_defaults(func=cmd_push)
    
    # COMMIT command
    commit_parser = subparsers.add_parser("commit", help="Commit image URL to blockchain")
    commit_parser.add_argument("--image", dest="image_name", required=True, help="Docker image name")
    commit_parser.add_argument("--wallet.name", dest="wallet_name", default="default", help="Wallet name")
    commit_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default", help="Wallet hotkey")
    commit_parser.add_argument("--netuid", type=int, default=1, help="Subnet ID")
    commit_parser.add_argument("--network", default="local", help="Subtensor network")
    commit_parser.add_argument("--reveal-blocks", type=int, default=100, help="Blocks until reveal")
    commit_parser.set_defaults(func=cmd_commit)
    
    # STATUS command
    status_parser = subparsers.add_parser("status", help="Check commitment status")
    status_parser.add_argument("--network", default="local", help="Subtensor network")
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
