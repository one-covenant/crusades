"""
Example training code for Templar Tournament.

Usage:
    1. Edit and optimize train.py
    2. Test: uv run python -m tournament.test_local train.py
    3. Submit when ready!
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""
    
    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int           # Total tokens processed across all steps
    final_loss: float           # Loss value from last training step


def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """Run training for num_steps and return results.
    
    OPTIMIZE THIS FUNCTION to maximize your TPS (tokens per second)!
    
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
    
    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    
    # Basic training loop - OPTIMIZE THIS!
    for step in range(num_steps):
        # Get next batch
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)
        
        # Prepare inputs and labels for next-token prediction
        input_ids = batch[:, :-1]  # Input: all except last token
        labels = batch[:, 1:]      # Target: all except first token
        
        # Forward pass with bfloat16
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()
    
    # Sync GPU before returning (ensures accurate timing)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )

# =============================================================================
# REMEMBER:
# - Test locally first (FREE): uv run python -m tournament.test_local train.py
# - Your outputs must match reference within 10% tolerance
# - Token count must be exact
# - Submission costs 0.1 TAO
# - Version 2: Testing updated submission flow
# =============================================================================


# =============================================================================
# DIRECT TEST - Run this file to see baseline performance
# =============================================================================
if __name__ == "__main__":
    import time
    from pathlib import Path
    
    print("="*70)
    print("TESTING train.py - Baseline Performance")
    print("="*70)
    print()
    
    # Check if model and data exist
    model_path = Path("benchmark/model")
    data_path = Path("benchmark/data/train.pt")
    
    if not model_path.exists():
        print("❌ Model not found at benchmark/model/")
        print("   Run: uv run python scripts/setup_benchmark.py")
        exit(1)
    
    if not data_path.exists():
        print("❌ Data not found at benchmark/data/train.pt")
        print("   Run: uv run python scripts/setup_benchmark.py")
        exit(1)
    
    print("✅ Loading model from benchmark/model/")
    from transformers import AutoModelForCausalLM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",  # Let accelerate handle device placement
        trust_remote_code=True,
    )
    model.train()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    print("✅ Loading data from benchmark/data/train.pt")
    data = torch.load(data_path, weights_only=True)
    print(f"   Samples: {data.shape[0]:,}")
    print(f"   Sequence length: {data.shape[1]}")
    print()
    
    # Create data iterator
    def create_iterator():
        idx = 0
        batch_size = 8
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx
    
    data_iterator = create_iterator()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup run (to trigger any compilation, cache population)
    print("🔥 Warmup run (to stabilize timings)...")
    torch.cuda.synchronize()
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=1, device=device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("   Warmup complete")
    print()
    
    # Reset optimizer and model state
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    data_iterator = create_iterator()
    
    # Run 3 evaluations and average (like validators do)
    print("🔄 Running 3 evaluations (5 steps each)...")
    tps_scores = []
    
    for eval_num in range(3):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = inner_steps(
            model=model,
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_steps=5,
            device=device,
        )
        
        torch.cuda.synchronize()
        wall_time = time.perf_counter() - start_time
        tps = result.total_tokens / wall_time
        tps_scores.append(tps)
        
        print(f"   Eval {eval_num + 1}/3: {tps:,.1f} TPS ({wall_time:.2f}s)")
    
    avg_tps = sum(tps_scores) / len(tps_scores)
    
    print()
    print("✅ Complete!")
    print()
    print("📊 Results (Averaged over 3 evaluations):")
    print(f"   Tokens per eval: {result.total_tokens:,}")
    print(f"   Average TPS: {avg_tps:,.1f}")
    print(f"   Min TPS: {min(tps_scores):,.1f}")
    print(f"   Max TPS: {max(tps_scores):,.1f}")
    print(f"   Final loss: {result.final_loss:.4f}")
    print()
    print("💡 This is your BASELINE. Now optimize to increase TPS!")
    print("   Test: uv run python -m tournament.test_local train.py")
    print("   Submit: uv run python -m neurons.miner train.py ...")
