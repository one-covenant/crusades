"""
Optimized training implementation for Templar Crusades.

Targets 60%+ MFU via torch.compile with reduce-overhead (CUDA graphs),
autocast, and elimination of per-step CPU-GPU synchronization points.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function.

    All fields are verified by the validator:
    - final_logits: Must be a 3D tensor (batch, seq_len-1, vocab), NOT None
    - total_tokens: Should equal batch_size * seq_len * num_steps
    - final_loss: Must be a positive float, close to reference loss
    """

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Run training steps and return results.

    Optimizations over baseline:
    - torch.compile with reduce-overhead: CUDA graphs eliminate per-kernel
      launch overhead. Warmup absorbs compilation; timed run reuses cache.
    - torch.autocast: keeps all ops in bfloat16, avoiding upcasts.
    - Metrics extracted only after the loop to avoid per-step CUDA syncs.
    """
    compiled_model = torch.compile(model, mode="reduce-overhead")

    total_tokens = 0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = compiled_model(input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()

    # Extract metrics once after the loop -- avoids CPU-GPU sync on every step
    return InnerStepsResult(
        final_logits=logits.detach().float(),
        total_tokens=total_tokens,
        final_loss=loss.item(),
    )
