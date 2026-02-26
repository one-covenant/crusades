"""Basic DDP train.py for testing multi-GPU parallelism.

Single-GPU: uses the validator-provided optimizer directly.
Multi-GPU: wraps model with DDP and creates own optimizer.
No torch.compile — kept simple for debugging.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    if num_gpus > 1:
        model = DDP(model, device_ids=[device.index])

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)
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
        final_logits = logits.detach()
        final_loss = loss.item()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
