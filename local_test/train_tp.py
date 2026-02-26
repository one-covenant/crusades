"""TP train.py -- tensor-parallel strategy.

Declares get_strategy() -> "tp". All ranks receive the same data
(replicated batches). The model's linear layers are sharded column/row-wise
across ranks. Must gather full params before returning.

Uses PyTorch's native tensor_parallel APIs (torch.distributed.tensor_parallel).
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def get_strategy():
    return "tp"


def _apply_tp(model, device_mesh):
    """Apply tensor parallelism to transformer attention and MLP layers."""
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            parallelize_module(
                module,
                device_mesh,
                {
                    "q_proj": ColwiseParallel(),
                    "k_proj": ColwiseParallel(),
                    "v_proj": ColwiseParallel(),
                    "o_proj": RowwiseParallel(),
                },
            )
        if hasattr(module, "gate_proj") and hasattr(module, "down_proj"):
            parallelize_module(
                module,
                device_mesh,
                {
                    "gate_proj": ColwiseParallel(),
                    "up_proj": ColwiseParallel(),
                    "down_proj": RowwiseParallel(),
                },
            )
    return model


def _gather_full_params(model, world_size):
    """Gather sharded TP parameters back to full tensors on all ranks."""
    state = {}
    for name, param in model.named_parameters():
        p = param.data
        if hasattr(p, "full_tensor"):
            p = p.full_tensor()
        elif hasattr(p, "to_local"):
            local = p.to_local()
            gathered = [torch.zeros_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            p = torch.cat(gathered, dim=0)
        state[name] = p
    return state


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    if num_gpus > 1:
        mesh = init_device_mesh("cuda", (num_gpus,))
        model = _apply_tp(model, mesh)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=num_gpus == 1,
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

    if num_gpus > 1:
        full_state = _gather_full_params(model, num_gpus)
        current_state = model.state_dict()
        for k in current_state:
            if k in full_state:
                current_state[k] = full_state[k]
        model.load_state_dict(current_state, strict=False)

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
