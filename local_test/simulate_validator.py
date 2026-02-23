"""
Local Validator Simulation Tool.

Use this script to test your train.py exactly as the production validator does.

1. Build the docker image (run from repo root):
   docker build --network=host -f environments/templar/Dockerfile --no-cache -t templar-eval:latest .

2. Run the simulation:
   docker run --gpus all -it --rm \
     -v $(pwd)/local_test/train.py:/test/train.py \
     -v $(pwd)/local_test/simulate_validator.py:/test/simulate.py \
     -v $(pwd)/hparams/hparams.json:/app/hparams.json \
     -e PYTHONPATH=/app \
     templar-eval:latest \
     python3 /test/simulate.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the paths required to import the validator's environment
# Use /app for the image's env.py and stubs
sys.path.append("/app")

from env import Actor

async def simulate():
    print("🚀 Starting Simulated Validator Test...")
    
    # 1. Load HParams (from the image's path or mounted path)
    hparams_path = Path("/app/hparams.json") 
    if not hparams_path.exists():
        hparams_path = Path("/app/hparams/hparams.json")
        
    print(f"Loading hparams from: {hparams_path}")
    with open(hparams_path) as f:
        hb = json.load(f)
    
    # 2. Load your WIP code
    # We will mount this to /test/train.py
    code_path = Path("/test/train.py")
    print(f"Loading miner code from: {code_path}")
    code = code_path.read_text()
    
    # 3. Setup Actor
    actor = Actor()
    
    # 4. Run Evaluation (Exact same signature as the validator)
    result = await actor.evaluate(
        task_id=1337,
        seed="local:test:1",
        model_url=hb["benchmark_model_name"],
        data_url=hb["benchmark_dataset_name"],
        steps=hb["eval_steps"],
        batch_size=hb["benchmark_batch_size"],
        sequence_length=hb["benchmark_sequence_length"],
        code=code,
        # Pass all the verification thresholds from hparams
        max_loss_difference=hb["verification"]["max_loss_difference"],
        gradient_norm_ratio_max=hb["verification"]["gradient_norm_ratio_max"],
        weight_relative_error_max=hb["verification"]["weight_relative_error_max"],
        min_params_changed_ratio=hb["verification"]["min_params_changed_ratio"],
        # MFU settings
        gpu_peak_tflops=hb["mfu"]["gpu_peak_tflops"],
        min_mfu=hb["mfu"]["min_mfu"],
    )
    
    # 5. Output Result
    print("\n" + "="*50)
    print("VALIDATOR RESULT")
    print("="*50)
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Error: {result.get('error')}")
        print(f"Error Code: {result.get('error_code')}")
    
    print(f"MFU: {result.get('mfu', 0.0):.2f}%")
    print(f"TPS: {result.get('tps', 0.0):.2f}")
    
    print("\nDiagnostics:")
    # Clean up large tensors from diagnostics for display
    diag = result.get('diagnostics', {})
    print(json.dumps(diag, indent=2))
    print("="*50)

if __name__ == "__main__":
    asyncio.run(simulate())
