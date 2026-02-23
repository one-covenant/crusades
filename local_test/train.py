from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


_COMPILED_FN = {}
_PREPARED_MODEL_IDS = set()
_TORCH_CONFIGURED = False


def _configure_torch():
    global _TORCH_CONFIGURED
    if _TORCH_CONFIGURED:
        return
    _TORCH_CONFIGURED = True


def _prepare_model(model):
    model_id = id(model)
    if model_id in _PREPARED_MODEL_IDS:
        return
    _PREPARED_MODEL_IDS.add(model_id)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if hasattr(model, "config"):
        model.config.use_cache = False
        try:
            model.config.output_hidden_states = False
        except Exception:
            pass
        try:
            model.config.output_attentions = False
        except Exception:
            pass


def _get_compiled_fn(model):
    key = id(model)
    if key in _COMPILED_FN:
        return _COMPILED_FN[key]

    def fwd_bwd(input_ids, labels):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        loss.backward()
        return logits, loss

    try:
        compiled = torch.compile(fwd_bwd, mode="reduce-overhead", dynamic=False)
    except Exception:
        compiled = fwd_bwd

    _COMPILED_FN[key] = compiled
    return compiled


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    """Optimized training loop for maximum MFU on A100.

    When num_gpus > 1 the caller (env.py) launches via torchrun and
    each rank gets its own copy of this function.  The distributed
    process group is already initialized by the eval environment.
    Wrap the model with DDP/FSDP here to leverage multiple GPUs.

    ``device`` is ``torch.device("cuda:N")`` where N is the local rank,
    so ``device.index`` gives the integer GPU index.  ``os`` is forbidden
    in miner code — use ``device.index`` instead of ``os.environ``.

    Example DDP usage::

        from torch.nn.parallel import DistributedDataParallel as DDP

        if num_gpus > 1:
            model = DDP(model, device_ids=[device.index])

    Example FSDP usage::

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if num_gpus > 1:
            model = FSDP(model, device_id=device.index)
    """

    _configure_torch()
    _prepare_model(model)

    step_fn = _get_compiled_fn(model)

    # Prefetch and pre-split all batches (contiguous copies happen here,
    # outside the tight training loop)
    all_inputs = []
    all_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        all_inputs.append(batch[:, :-1].contiguous())
        all_labels.append(batch[:, 1:].contiguous())
        tokens_per_batch = batch.numel()

    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad
    total_tokens = num_steps * tokens_per_batch
    last_step = num_steps - 1
    final_logits = None
    final_loss_val = 0.0

    for step in range(num_steps):
        logits, loss = step_fn(all_inputs[step], all_labels[step])
        opt_step()
        opt_zero(set_to_none=True)

        if step == last_step:
            final_logits = logits.detach()
            final_loss_val = loss.item()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss_val,
    )
