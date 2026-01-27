"""
Miners can improve upon this to achieve higher TPS!

Usage:
    1. Run setup: uv run local_test/setup_benchmark.py
    2. Test locally: uv run local_test/train.py
    3. Submit when ready!

Version: Optimized Baseline v2
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""

    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int  # Total tokens processed across all steps
    final_loss: float  # Loss value from last training step


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Optimize this function for maximum TPS.

    Args:
        model: Pre-loaded 7B model (already on device, in train mode)
        data_iterator: Iterator yielding batches of shape (batch_size, seq_len)
        optimizer: Pre-configured AdamW optimizer
        num_steps: Number of training steps to run (typically 5)
        device: Target device (cuda or cpu)

    Returns:
        InnerStepsResult with outputs for verification
    """
    # Ensure deterministic behavior (REQUIRED for verification)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    # Pre-fetch first batch
    batch = next(data_iterator)
    batch = batch.to(device, dtype=torch.long, non_blocking=True)

    # Optimized training loop
    for step in range(num_steps):
        # Prepare inputs and labels from current batch
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Pre-fetch next batch while computing (hide transfer latency)
        if step < num_steps - 1:
            next_batch = next(data_iterator)
            next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)

        # Forward pass with bfloat16 autocasting
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Efficient loss computation - reshape in place
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        # Backward pass
        loss.backward()

        # Optimizer step with set_to_none (faster than zero_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

        # Swap to pre-fetched batch
        if step < num_steps - 1:
            batch = next_batch

    # Sync GPU before returning (ensures accurate timing)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_hparams() -> dict:
    """Load hparams.json configuration."""
    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    if hparams_path.exists():
        with open(hparams_path) as f:
            return json.load(f)
    return {}


# =============================================================================
# DIRECT TEST - Run this file to see baseline performance
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING train.py - Optimized Baseline Performance")
    print("=" * 70)
    print()

    # Load configuration from hparams.json
    hparams = load_hparams()
    batch_size = hparams.get("benchmark_batch_size", 16)
    num_steps = hparams.get("eval_steps", 5)
    num_evals = hparams.get("evaluation_runs", 5)

    print("Configuration from hparams.json:")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per eval: {num_steps}")
    print(f"   Evaluations: {num_evals}")
    print()

    # Check if model and data exist (relative to project root)
    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    if not model_path.exists():
        print("Model not found at benchmark/model/")
        print("   Run: uv run local_test/setup_benchmark.py")
        exit(1)

    if not data_path.exists():
        print("Data not found at benchmark/data/train.pt")
        print("   Run: uv run local_test/setup_benchmark.py")
        exit(1)

    print("Loading model from benchmark/model/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"   GPUs available: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print()

    # Load model with optimizations
    # Try flash_attention_2 first, fall back to sdpa (PyTorch native)
    attn_impl = "sdpa"  # PyTorch's native scaled dot product attention
    try:
        import flash_attn  # noqa: F401 - checking availability

        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    model.train()

    print(" Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   Gradient checkpointing: enabled")
    print(f"   Attention: {attn_impl}")
    print()

    print(" Loading data from benchmark/data/train.pt")
    data = torch.load(data_path, weights_only=True)
    print(f"   Samples: {data.shape[0]:,}")
    print(f"   Sequence length: {data.shape[1]}")
    print()

    # Create data iterator (batch_size from hparams)
    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    # Create optimizer with fused=True for faster updates
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        fused=use_fused,  # Fused optimizer for CUDA
    )
    print(f" Optimizer: AdamW (fused={use_fused})")
    print()

    # Warmup run
    print("Warmup run...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("   Warmup complete")
    print()

    # Reset optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    # Run evaluations and take median (like validators do)
    print(f"Running {num_evals} evaluations ({num_steps} steps each)...")
    tps_scores = []

    for eval_num in range(num_evals):
        data_iterator = create_iterator()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        result = inner_steps(
            model=model,
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_steps=num_steps,
            device=device,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall_time = time.perf_counter() - start_time
        tps = result.total_tokens / wall_time
        tps_scores.append(tps)

        print(
            f"   Eval {eval_num + 1}/{num_evals}: {tps:,.0f} TPS ({wall_time:.2f}s, loss={result.final_loss:.4f})"
        )

    # Calculate median
    sorted_scores = sorted(tps_scores)
    median_tps = sorted_scores[len(sorted_scores) // 2]

    print()
    print("=" * 70)
    print(f"RESULTS (Median of {num_evals} evaluations)")
    print("=" * 70)
    print(f"   Tokens per eval: {result.total_tokens:,}")
    print(f"   Median TPS: {median_tps:,.0f}")
    print(f"   Min TPS: {min(tps_scores):,.0f}")
    print(f"   Max TPS: {max(tps_scores):,.0f}")
    print(f"   Final loss: {result.final_loss:.4f}")
    print()
    print("Tips to improve TPS:")
    print("   - Try torch.compile (uncomment in model loading section)")
    print("   - Experiment with different batch sizes (hparams.json)")
    print("   - Try different activation checkpointing strategies")
    print("   - Use torchtitan components for advanced optimizations")
    print("   - Multi-GPU: Model uses device_map='auto' for distribution")
    print()
