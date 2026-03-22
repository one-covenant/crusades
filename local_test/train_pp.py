# Reference: Pipeline Parallelism (PP) strategy
#
# Topology: dp_size=2, tp_size=1, pp_size=2 (requires 4 GPUs)
#   - 2 pipeline replicas, each with 2 stages
#   - Ranks [0,1] form pipeline replica 0 (stage 0 on rank 0, stage 1 on rank 1)
#   - Ranks [2,3] form pipeline replica 1 (stage 0 on rank 2, stage 1 on rank 3)
#   - Different pipeline replicas get different data (data-parallel across DP dim)
#   - Within a pipeline, stages share the same data and pass activations
#
# Uses manual pipeline scheduling (no torch.distributed.pipelining dependency).
# Each pipeline stage holds a contiguous slice of transformer layers.
# Forward: stage 0 computes embeddings + first half of layers, sends activations.
#          stage 1 receives activations, computes second half + lm_head, computes loss.
# Backward: reverse order with gradient communication.
#
# Requirements for verification:
#   - get_strategy() returning {"dp_size": 2, "tp_size": 1, "pp_size": 2}
#   - Return InnerStepsResult with final_logits, total_tokens, final_loss
#   - Must return final_state: gathered full model state on rank 0

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None


def get_strategy():
    return {"dp_size": 2, "tp_size": 1, "pp_size": 2}


def _split_model_layers(model):
    """Split a HuggingFace causal LM into (embed + first half layers) and (second half layers + head)."""
    layers = model.model.layers
    n = len(layers)
    mid = n // 2
    return list(range(mid)), list(range(mid, n))


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if hasattr(model, "config"):
        model.config.use_cache = False

    strategy = get_strategy()
    pp_size = strategy["pp_size"]
    dp_size = strategy["dp_size"]

    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))

    pp_rank = local_rank % pp_size
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1

    pp_peer = local_rank + 1 if is_first_stage else local_rank - 1

    layer_indices_0, layer_indices_1 = _split_model_layers(model)
    my_layer_indices = layer_indices_0 if is_first_stage else layer_indices_1

    all_layers = list(model.model.layers)
    n_layers = len(all_layers)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = model.to(device)

    for i, layer in enumerate(all_layers):
        if i not in my_layer_indices:
            for p in layer.parameters():
                p.requires_grad_(False)
                p.data = p.data.to("meta") if hasattr(torch, "meta") else p.data.new_empty(0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if dp_size > 1:
        dp_ranks = [r for r in range(num_gpus) if (r % pp_size) == pp_rank]
        dp_group = dist.new_group(dp_ranks)
    else:
        dp_group = None

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

    hidden_size = model.config.hidden_size

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        bs, seq_len = input_ids.shape

        if is_first_stage:
            hidden = model.model.embed_tokens(input_ids)

            if hasattr(model.model, "rotary_emb"):
                pass

            for idx in my_layer_indices:
                hidden = all_layers[idx](hidden)[0]

            dist.send(hidden.detach().contiguous(), dst=pp_peer)

            recv_grad = torch.zeros_like(hidden)
            dist.recv(recv_grad, src=pp_peer)
            hidden.backward(recv_grad)

        else:
            hidden = torch.zeros(bs, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
            hidden.requires_grad_(True)
            dist.recv(hidden, src=pp_peer)
            hidden = hidden.detach().requires_grad_(True)

            h = hidden
            for idx in my_layer_indices:
                h = all_layers[idx](h)[0]

            h = model.model.norm(h)
            logits = model.lm_head(h)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()

            dist.send(hidden.grad.contiguous(), dst=pp_peer)

            final_logits = logits.detach()
            final_loss = loss.item()

        if dp_group is not None:
            for p in trainable_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=dp_group)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()

    if is_last_stage and final_logits is None:
        final_logits = torch.zeros(1, device=device)

    if not is_last_stage:
        final_logits = torch.zeros(1, device=device)
        final_loss = 0.0

    full_state = _gather_pp_state(
        model, all_layers, my_layer_indices, n_layers, pp_rank, pp_peer, is_first_stage, rank
    )

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )


def _gather_pp_state(
    model, all_layers, my_layer_indices, n_layers, pp_rank, pp_peer, is_first_stage, global_rank
):
    """Gather full model state dict across pipeline stages onto rank 0."""
    my_state = {}

    if is_first_stage:
        for k, v in model.model.embed_tokens.state_dict().items():
            my_state[f"model.embed_tokens.{k}"] = v.detach().cpu().clone()

    for idx in my_layer_indices:
        for k, v in all_layers[idx].state_dict().items():
            my_state[f"model.layers.{idx}.{k}"] = v.detach().cpu().clone()

    if not is_first_stage:
        for k, v in model.model.norm.state_dict().items():
            my_state[f"model.norm.{k}"] = v.detach().cpu().clone()
        for k, v in model.lm_head.state_dict().items():
            my_state[f"lm_head.{k}"] = v.detach().cpu().clone()

    if not dist.is_initialized():
        return my_state

    if is_first_stage:
        keys_and_shapes = []
        for k, v in my_state.items():
            keys_and_shapes.append((k, v.shape, v.dtype))
        obj_list = [keys_and_shapes]
        dist.broadcast_object_list(obj_list, src=dist.get_rank())

        peer_obj = [None]
        dist.broadcast_object_list(peer_obj, src=pp_peer)
        peer_keys = peer_obj[0]

        for k, shape, dtype in peer_keys:
            buf = torch.empty(shape, dtype=dtype)
            dist.recv(buf, src=pp_peer)
            my_state[k] = buf

        return my_state if global_rank == 0 else None
    else:
        peer_obj = [None]
        dist.broadcast_object_list(peer_obj, src=pp_peer)

        keys_and_shapes = []
        for k, v in my_state.items():
            keys_and_shapes.append((k, v.shape, v.dtype))
        obj_list = [keys_and_shapes]
        dist.broadcast_object_list(obj_list, src=dist.get_rank())

        for k, v in my_state.items():
            dist.send(v.contiguous(), dst=pp_peer)

        return None
