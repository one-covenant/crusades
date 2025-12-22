# /// script
# dependencies = [
#   "basilica-sdk>=0.10.0",
#   "affinetes",
# ]
# ///

"""
Deploy tournament sandbox to Basilica.

Usage:
    # Deploy with defaults
    python deploy.py

    # Deploy with custom name
    python deploy.py --name my-sandbox

    # Deploy with custom GPU
    python deploy.py --gpu-model A100

    # Deploy and wait for ready
    python deploy.py --wait

Prerequisites:
    - BASILICA_API_TOKEN environment variable set
    - Image pushed to ghcr.io/one-covenant/templar-tournament-sandbox:latest
"""

import argparse
import sys
from typing import Optional

from basilica import BasilicaClient, GpuRequirementsSpec, ResourceRequirements

IMAGE = "ghcr.io/one-covenant/templar-tournament-sandbox:latest"
DEFAULT_NAME = "templar-tournament-sandbox"
DEFAULT_GPU_COUNT = 1
DEFAULT_GPU_MEMORY_GB = 24
DEFAULT_GPU_MODELS = ["A100", "H100", "RTX4090"]
DEFAULT_MEMORY = "32Gi"
DEFAULT_CPU = "4"
PORT = 8000


def deploy(
    name: str = DEFAULT_NAME,
    gpu_count: int = DEFAULT_GPU_COUNT,
    gpu_models: Optional[list[str]] = None,
    min_gpu_memory_gb: int = DEFAULT_GPU_MEMORY_GB,
    wait: bool = True,
    timeout: int = 600,
) -> str:
    """
    Deploy tournament sandbox to Basilica.

    Args:
        name: Deployment name
        gpu_count: Number of GPUs
        gpu_models: Acceptable GPU models (e.g., ["A100", "H100"])
        min_gpu_memory_gb: Minimum GPU memory in GB
        wait: Wait for deployment to be ready
        timeout: Timeout in seconds

    Returns:
        Deployment URL
    """
    client = BasilicaClient()

    effective_gpu_models = gpu_models if gpu_models else DEFAULT_GPU_MODELS

    gpu_spec = GpuRequirementsSpec(
        count=gpu_count,
        model=effective_gpu_models,
        min_cuda_version="12.0",
        min_gpu_memory_gb=min_gpu_memory_gb,
    )

    resources = ResourceRequirements(
        cpu=DEFAULT_CPU,
        memory=DEFAULT_MEMORY,
        gpus=gpu_spec,
    )

    print(f"Deploying {name} with image {IMAGE}")
    print(f"  GPU: {gpu_count}x {effective_gpu_models} (min {min_gpu_memory_gb}GB VRAM)")
    print(f"  Memory: {DEFAULT_MEMORY}")
    print(f"  Port: {PORT}")

    response = client.create_deployment(
        instance_name=name,
        image=IMAGE,
        port=PORT,
        cpu=DEFAULT_CPU,
        memory=DEFAULT_MEMORY,
        gpu_count=gpu_count,
        gpu_models=effective_gpu_models,
        min_gpu_memory_gb=min_gpu_memory_gb,
        public=True,
    )

    print(f"Deployment created: {response.instance_name}")
    print(f"URL: {response.url}")

    if wait:
        print("Waiting for deployment to be ready...")
        from basilica import Deployment

        deployment = Deployment._from_response(client, response)
        deployment.wait_until_ready(timeout=timeout)
        deployment.refresh()
        print(f"Deployment ready at: {deployment.url}")
        return deployment.url

    return response.url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy tournament sandbox to Basilica"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"Deployment name (default: {DEFAULT_NAME})",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=DEFAULT_GPU_COUNT,
        help=f"Number of GPUs (default: {DEFAULT_GPU_COUNT})",
    )
    parser.add_argument(
        "--gpu-model",
        action="append",
        dest="gpu_models",
        help="Acceptable GPU model (can be specified multiple times)",
    )
    parser.add_argument(
        "--min-gpu-memory",
        type=int,
        default=DEFAULT_GPU_MEMORY_GB,
        help=f"Minimum GPU memory in GB (default: {DEFAULT_GPU_MEMORY_GB})",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for deployment to be ready",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)",
    )

    args = parser.parse_args()

    try:
        url = deploy(
            name=args.name,
            gpu_count=args.gpu_count,
            gpu_models=args.gpu_models,
            min_gpu_memory_gb=args.min_gpu_memory,
            wait=not args.no_wait,
            timeout=args.timeout,
        )
        print(f"\nDeployment URL: {url}")
        print(f"\nTo evaluate code:")
        print(f'  curl -X POST {url}/evaluate -H "Content-Type: application/json" \\')
        print(
            f'    -d \'{{"code": "...", "model_name": "Qwen/Qwen2.5-0.5B-Instruct"}}\''
        )
        print(f"\nTo check health:")
        print(f"  curl {url}/health")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
