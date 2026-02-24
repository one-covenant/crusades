"""
Local Validator Simulation Tool.

Test your train.py exactly as the production validator does, inside the
same Docker container.

1. Build the docker image (run from repo root):

   docker build --network=host -f environments/templar/Dockerfile \
       --no-cache -t templar-eval:latest .

2. Run the simulation:

    docker run --gpus all -it --rm \
        -v "$(pwd)/local_test/train.py":/test/train.py \
        -v "$(pwd)/local_test/simulate_validator.py":/test/simulate.py \
        -v "$(pwd)/hparams/hparams.json":/app/hparams.json \
        -e PYTHONPATH=/app \
        templar-eval:latest \
        python3 /test/simulate.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.append("/app")

from env import Actor


def _sanitize(obj):
    """Make diagnostics JSON-serializable (handles tensors etc.)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "detach"):
        return obj.cpu().detach().tolist()
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


async def simulate():
    print("Starting Simulated Validator Test...")

    hparams_path = Path("/app/hparams.json")
    if not hparams_path.exists():
        hparams_path = Path("/app/hparams/hparams.json")

    if not hparams_path.exists():
        print("ERROR: hparams.json not found. See docstring for launch command.")
        sys.exit(1)

    print(f"Loading hparams from: {hparams_path}")
    with open(hparams_path) as f:
        hb = json.load(f)

    code_path = Path("/test/train.py")
    if not code_path.exists():
        print("ERROR: train.py not found at /test/train.py. See docstring for launch command.")
        sys.exit(1)

    print(f"Loading miner code from: {code_path}")
    code = code_path.read_text()

    actor = Actor()

    result = await actor.evaluate(
        task_id=1337,
        seed="local:test:1",
        model_url=hb["benchmark_model_name"],
        data_url=hb["benchmark_dataset_name"],
        steps=hb["eval_steps"],
        batch_size=hb["benchmark_batch_size"],
        sequence_length=hb["benchmark_sequence_length"],
        data_samples=hb["benchmark_data_samples"],
        timeout=hb["eval_timeout"],
        code=code,
        use_random_init=True,
        min_trainable_params_ratio=1.0,
        max_loss_difference=hb["verification"]["max_loss_difference"],
        min_params_changed_ratio=hb["verification"]["min_params_changed_ratio"],
        gradient_norm_ratio_max=hb["verification"]["gradient_norm_ratio_max"],
        weight_relative_error_max=hb["verification"]["weight_relative_error_max"],
        timer_divergence_threshold=hb["verification"]["timer_divergence_threshold"],
        gpu_peak_tflops=hb["mfu"]["gpu_peak_tflops"],
        max_plausible_mfu=hb["mfu"]["max_plausible_mfu"],
        min_mfu=hb["mfu"]["min_mfu"],
        require_cuda_timing=True,
    )

    print()
    print("=" * 50)
    print("VALIDATOR RESULT")
    print("=" * 50)
    print(f"Success: {result['success']}")
    if not result["success"]:
        print(f"Error: {result.get('error')}")
        print(f"Error Code: {result.get('error_code')}")
    print(f"MFU: {result.get('mfu', 0.0):.2f}%")
    print(f"TPS: {result.get('tps', 0.0):.2f}")
    print()
    print("Diagnostics:")
    print(json.dumps(_sanitize(result.get("diagnostics", {})), indent=2))
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(simulate())
