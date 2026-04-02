# Reference: Single-GPU strategy (no get_strategy)
#
# NOTE: Only works when the validator uses docker_num_gpus=1.
# With docker_num_gpus=4 (current production), miners MUST use a multi-GPU
# strategy (FSDP, DDP, TP, or mixed).  This file is kept as a reference.
#
# Memory-safe for large-vocab tokenizers (262K) via chunked cross-entropy
# and gradient checkpointing.

from dataclasses import dataclass

import torch


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def _chunked_cross_entropy(logits, labels, chunk_size=4096):
    """Compute cross-entropy in chunks to avoid materializing full vocab logits."""
    logits_flat = logits.reshape(-1, logits.size(-1))
    labels_flat = labels.reshape(-1)
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    n_valid = torch.tensor(0, device=logits.device, dtype=torch.long)
    for i in range(0, logits_flat.size(0), chunk_size):
        chunk_logits = logits_flat[i : i + chunk_size]
        chunk_labels = labels_flat[i : i + chunk_size]
        mask = chunk_labels != -100
        if mask.any():
            total_loss += torch.nn.functional.cross_entropy(
                chunk_logits[mask], chunk_labels[mask], reduction="sum"
            )
            n_valid += mask.sum()
    return total_loss / n_valid.clamp(min=1)


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = model.to(dtype=torch.bfloat16)

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
        loss = _chunked_cross_entropy(logits, labels)

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
