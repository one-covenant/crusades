"""
Reference training implementation for Templar Crusades.

This is the baseline implementation. Miners should optimize it for maximum MFU
(Model FLOPs Utilization) while passing all verification checks.

=== SUBMISSION ===

The validator only calls the inner_steps function.

=== VERIFICATION RULES ===

Your inner_steps function MUST:
  - Use the provided optimizer (call optimizer.step() and optimizer.zero_grad())
  - Process ALL tokens in each batch (no truncation)
  - Return actual final_logits tensor (not None)
  - Return logits with correct shape: (batch_size, seq_len - 1, vocab_size)
  - Produce gradients that closely match the reference implementation
  - Train all model parameters (don't freeze layers)
  - Call optimizer.step() for each training step

Your inner_steps function MUST NOT:
  - Access optimizer internals (e.g., optimizer.optimizer)
  - Truncate or skip parts of input sequences
  - Return None for final_logits
  - Report inflated token counts
  - Modify the model's requires_grad settings
  - Modify torch backend settings (deterministic, benchmark, SDP toggles, etc.)
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

    This is the function the validator calls. It receives:
    - model: Pre-loaded model (already on device, in train mode, with gradient checkpointing)
    - data_iterator: Infinite iterator yielding batches of shape (batch_size, seq_len)
    - optimizer: Pre-configured AdamW optimizer (wrapped by validator for gradient capture)
    - num_steps: Number of training steps to run (must complete all of them)
    - device: Target device (cuda or cpu)

    The validator measures wall_time of this function and calculates:
        MFU = (6 * model_params * batch_size * seq_len * num_steps) / (wall_time * gpu_peak_tflops)

    Higher MFU = you completed the same training faster = better score.

    Returns:
        InnerStepsResult with outputs for verification
    """
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
