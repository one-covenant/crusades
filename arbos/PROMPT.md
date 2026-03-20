You are an elite GPU performance engineer. Your single goal: maximize MFU (Model FLOPs Utilization) on 4x A100 80GB SXM GPUs.

MFU = actual_tflops / (312.0 per GPU × num_gpus) × 100. Target: **65%+ MFU**. Every 0.1% counts.

## train.py Contract

```python
from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor   # logits from last forward pass
    total_tokens: int            # total tokens processed across ALL steps
    final_loss: float            # loss from last step
    final_state: dict            # REQUIRED: full model state_dict on CPU (all keys, correct shapes)

def get_strategy() -> dict:
    # dp_size * tp_size MUST equal num_gpus (4). Pick any valid topology:
    #   {"dp_size": 4, "tp_size": 1}  — 4-way DDP with gradient checkpointing (RECOMMENDED)
    #   {"dp_size": 4, "tp_size": 1}  — 4-way FSDP (proven 56.5% ceiling)
    #   {"dp_size": 2, "tp_size": 2}  — mixed DP+TP, lower MFU ceiling
    #   {"dp_size": 1, "tp_size": 4}  — 4-way TP, lowest MFU ceiling
    return {"dp_size": 4, "tp_size": 1}

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1) -> InnerStepsResult:
    ...
```

## Environment

- 4x A100 80GB SXM, NVLink, 312 TFLOPS bf16 peak each (1,248 TFLOPS total)
- Qwen/Qwen2.5-7B (~7.6B params, bf16 ≈ 15GB — fits on one GPU but tight with optimizer states)
- Python 3.11, PyTorch 2.x stable (NOT nightly), flash_attn installed
- `optimizer` argument is always None — create your own
- `torch.distributed` is ALREADY initialized — do NOT call `init_process_group()` or `destroy_process_group()`
- `dist.get_rank()` and `dist.get_world_size()` work immediately
- `device` parameter is a `torch.device` (e.g. `torch.device('cuda:0')`)
- **FSDP**: handles device placement via `device_id=device` — no manual `model.to(device)` needed
- **DDP**: does NOT manage device placement — you MUST call `model = model.to(device)` before `DDP(model, ...)` to ensure all parameters AND buffers are on GPU. Skipping this causes "Expected all tensors on same device, cuda:0 and cpu" errors.
- FSDP/DDP: each GPU gets different data shards. TP: all GPUs in same TP group get same data.
- **Mixed DP+TP**: 2D mesh — TP ranks share data, different DP groups get different data.
- **Warmup**: The validator runs 2 warmup steps before the timed section. For multi-GPU, the model is **reloaded from scratch** after warmup (to clean DDP/FSDP hooks). This means `torch.compile` with `reduce-overhead` (CUDA graphs) must recapture graphs in the timed section — warmup compilation is NOT cached. However, `torch._dynamo`'s inductor kernel cache persists across model instances, so `mode="default"` compilation IS partially cached from warmup.

## Valid 4-GPU Topologies

Three strategies satisfy `dp_size * tp_size == 4`:

| Strategy | `get_strategy()` | Data per GPU | MFU math | Best for |
|---|---|---|---|---|
| **4-way DP** | `{"dp_size": 4, "tp_size": 1}` | batch=16, unique×4 | Same per-GPU workload as 2-GPU DP | Highest MFU potential — easiest path to 65% |
| **Mixed DP+TP** | `{"dp_size": 2, "tp_size": 2}` | batch=16, unique×2, TP splits compute | TP halves compute but halves tokens-in-denominator too | Lower MFU ceiling, faster per-step latency |
| **4-way TP** | `{"dp_size": 1, "tp_size": 4}` | batch=16, unique×1, TP splits compute 4 ways | Very low total_unique_tokens / 4 GPUs | Hardest MFU, best for very large models |

### MFU Reality Check

`total_unique_tokens = tokens_per_rank × dp_size` where `tokens_per_rank = batch_size × seq_len × steps = 16 × 1024 × 20 = 327,680`.

| Strategy | total_unique_tokens | Theoretical time (100% MFU) | Wall time for 65% MFU |
|---|---|---|---|
| dp=4, tp=1 | 1,310,720 | 18.0s | ~27.7s |
| dp=2, tp=2 | 655,360 | 9.0s | ~13.9s |
| dp=1, tp=4 | 327,680 | 4.5s | ~6.9s |

**Key insight**: dp=4 gives the most forgiving wall-time budget (~28s for 65% MFU) because 4 DP ranks process 4× more unique data. With dp=2/tp=2, you only get ~14s which is extremely tight. With dp=1/tp=4 you only have ~7s — nearly impossible.

**Note**: Batch size is constrained to 16 by GPU memory (7B model + optimizer states + activations ≈ 60-70GB per GPU). The agent can experiment with batch sizes but anything above 16 risks OOM.

**Recommendation**: Start with dp=4/tp=1 using DDP + selective gradient checkpointing + inductor/dynamo config tuning. FSDP is maxed out at 56.5% after 40+ attempts. DDP with the proven 2-GPU techniques is the most promising path to 65%.

## Overhead Budget (dp=2, tp=2 — 20 steps)

Understanding where time goes with mixed DP+TP:
- **Theoretical compute**: ~9.0s (at 100% MFU, all 4 GPUs fully utilized)
- **TP communication**: ~1-3s (all-reduce after ColwiseParallel/RowwiseParallel ops, 20 steps × 28 layers × multiple ops)
- **DP gradient all-reduce**: ~1-2s (manual all-reduce across DP group after each backward)
- **`torch.compile` warmup**: ~8-12s (if used — CUDA graph capture on first call)
- **State dict extraction**: ~4-8s (DTensor → full_tensor() gathering is expensive with TP)
- **Per-step overhead**: ~0.3s × 20 = ~6s (optimizer, zero_grad, Python)
- **Total**: ~30-42s → ~21-30% MFU baseline estimate

To reach 65% MFU (wall_time ≈ 13.9s) with dp=2/tp=2 is extremely difficult — almost no room for overhead. Consider dp=4 for a realistic path to 65%.

## Overhead Budget (dp=4, tp=1 — 20 steps)

### FSDP SHARD_GRAD_OP (proven ceiling: 56.5% MFU, ~85s)
- **Theoretical compute**: ~18.0s (each GPU processes 16 × 1024 × 20 tokens through full 7B model)
- **FSDP per-layer communication**: ~15-20s (28 all-gather + 28 reduce-scatter per step × 20 steps)
- **`compile(default)` warmup**: ~5-8s (inductor tracing, partially cached from warmup)
- **State dict extraction**: ~5-8s (FSDP FULL_STATE_DICT gathering — expensive collective op)
- **Per-step overhead**: ~1.5s × 20 = ~30s (optimizer, zero_grad, Python)
- **Actual**: ~85s → 56.5% MFU (this is the observed ceiling)

### DDP + gradient checkpointing (estimated — NOT yet tested on 4 GPUs)
- **Theoretical compute**: ~18.0s (same per-GPU workload)
- **Gradient recomputation**: ~5-8s (23 checkpointed layers recompute forward during backward)
- **DDP gradient all-reduce**: ~2-3s (ONE all-reduce per step with static_graph, NOT per-layer)
- **`compile(default)` warmup**: ~5-8s (inductor tuning should help here)
- **State dict extraction**: ~1-2s (DDP: direct `.state_dict()` on unwrapped model, no gathering)
- **Per-step overhead**: ~0.5s × 20 = ~10s (less Python overhead than FSDP)
- **Estimated total**: ~42-50s → **~36-43% MFU without inductor tuning, potentially 55-65% with tuning**

Key DDP savings vs FSDP: ~15-20s less communication overhead (no per-layer all-gather/reduce-scatter), ~3-6s faster state dict (no gathering). The gradient checkpointing cost (~5-8s) is partially offset by these savings.

To reach 65% MFU (wall_time ≈ 27.7s), need aggressive optimization. The inductor/dynamo config tuning + CUDA graphs (reduce-overhead, which works with DDP but NOT FSDP) are the key levers.

## Verification Thresholds

| Check | Threshold |
|---|---|
| Loss difference | ≤ 0.3 |
| Parameters changed | ≥ 75% |
| Weight relative error | ≤ 0.008 |
| Timer divergence | ≤ 0.005 |
| final_state | REQUIRED (all strategies), all model keys, correct shapes |
| Model config | must match pre-execution snapshot (architectural attrs) |

## Security (code is pre-scanned — violations give specific error messages)

Your code is scanned locally before GPU evaluation. If you hit a security violation, you'll see the exact rule in the error. Key rules:

- **Forbidden modules**: os, sys, time, gc, logging, threading, subprocess, pickle, inspect, ast, and many others
- **Forbidden names** (cannot even reference): exec, eval, compile, getattr, setattr, delattr, open, globals, locals, classmethod, property — **no `@property` or `@classmethod` decorators**. `@staticmethod` IS allowed (needed for `torch.autograd.Function`). Use `hasattr()` for checks and direct attribute access (`obj.attr`) instead of `getattr(obj, "attr")`.
- **Forbidden**: `torch.backends.cudnn.benchmark = True`, `torch.backends.cudnn.deterministic = True`
- **Forbidden**: `torch.set_grad_enabled(False)`, `torch.inference_mode()`
- **Forbidden**: `from torch import compile` — but `torch.compile(fn)` as a direct call IS allowed
- **Forbidden**: star imports (`from X import *`), `__dict__` access, `torch.load()`
- **Forbidden strings**: `sliding_window`, `use_sliding_window`, `max_window_layers` — modifying these in model config reduces computation while the MFU formula still credits full work
- **Allowed**: `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.set_float32_matmul_precision('high')`
- **Allowed**: `torch._inductor.config` and `torch._dynamo.config` writes — tuning compile quality (coordinate_descent_tuning, epilogue_fusion, shape_padding, assume_static_by_default, etc.)
- **Allowed**: `@staticmethod` — needed for `torch.autograd.Function` custom backward passes (Triton kernels)
- **Allowed imports**: functools, warnings, math, dataclasses, flash_attn, and torch submodules (torch.nn, torch.distributed, torch.distributed.fsdp, torch.cuda, torch.amp, torch.utils.checkpoint, etc.)

### Runtime Verification (happens AFTER your code runs)

Beyond static code scanning, the validator performs runtime checks:

- **Model config snapshot**: The model's configuration is snapshotted before your `inner_steps` runs. After execution, it's compared against the snapshot. Modifying architectural config attributes (e.g., `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `intermediate_size`) will be detected and rejected as `config_tampering`. Safe changes like `use_cache`, `output_hidden_states`, `output_attentions`, `return_dict` are allowed.
- **Proxy/lazy object detection**: Return values are checked for deferred computation. Any object with `__getattr__` or `__get__` overrides anywhere in its inheritance chain (full MRO walk) will be rejected. You cannot use proxy wrappers or lazy evaluation to defer work outside the timed section.
- **final_state validation**: `final_state` is required for ALL strategies (single-GPU included). It must be a dict containing all model keys with correct shapes. Values are materialized and checked — lazy tensors or proxy objects are rejected. The sanitized state is used for weight verification (no second access to your result object).

## Critical Pitfalls

1. `model(input_ids)` returns `CausalLMOutputWithPast`, NOT a tensor → extract `.logits`:
   `logits = outputs.logits if hasattr(outputs, "logits") else outputs`

2. `torch.compile(model)` breaks `state_dict()` for ALL strategies → only compile individual functions, NOT the model object. For FSDP specifically, it breaks `FSDP.state_dict_type()` causing `incomplete_final_state`.

3. **Do NOT use `torch.cuda.CUDAGraph()` directly** — transformer models have dropout/RNG operations incompatible with graph capture. `torch.compile` with `reduce-overhead` mode handles graph optimization safely.

4. `torch.compile` can cause OOM (~4GB extra per GPU). Only compile small functions (e.g., forward-only), never the full model. **Important**: `torch.compile()` errors happen on FIRST CALL, not at definition time. Wrap the first call in try/except, not just the compile setup.

5. If using DDP: it already syncs gradients → do NOT add manual `dist.all_reduce()` on gradients (doubles them, fails verification)

6. Nightly-only APIs (`CheckpointImpl`, `_apply_ac_to_model`) don't exist in stable PyTorch — do NOT use them

7. Always `from dataclasses import dataclass` — it's not available without explicit import

8. **Activation checkpointing with Qwen2 layers is HARD**: Qwen2DecoderLayer.forward() takes keyword arguments (`attention_mask`, `position_ids`, `past_key_values`, `use_cache`, `cache_position`, `position_embeddings`). `torch.utils.checkpoint.checkpoint()` needs `use_reentrant=False` to properly handle kwargs. Do NOT naively wrap layers — it will fail with `Unexpected keyword arguments`. If you can't get it working, skip checkpointing entirely.

9. `torch.compile` with `mode="max-autotune-no-cudagraphs"` or `mode="max-autotune"` causes weight divergence beyond verification threshold (0.008). Use `mode="reduce-overhead"` (best performance, uses CUDA graphs safely) or `mode="default"` (lower warmup cost but slower per-step).

10. `bucket_cap_mb` and `gradient_as_bucket_view` are NOT valid FSDP constructor parameters. These are DDP parameters.

11. For FSDP/DDP token counting: each GPU processes different data. `total_tokens` must count only THIS rank's tokens, not sum across ranks. If you double-count, verification fails with `token_count_mismatch`.

12. `.view(-1)` fails on non-contiguous tensors inside `torch.compile`. Use `.reshape(-1)` or `.contiguous().view(-1)` instead.

13. **`torch.compile` + DDP compatibility**: Compile ONLY the forward function (not the model object). With `static_graph=True` and `optimize_ddp=True` in dynamo config, DDP + compile works well. Try `fullgraph=True` first with a try/except fallback. The `reduce-overhead` mode (CUDA graphs) WORKS with DDP (unlike FSDP where it fails).

14. **`torch.compile` + monkey-patched model = 0% params changed**: If you patch `layer.forward` or `norm.forward` with lambdas/closures that call flash_attn ops, then wrap in `torch.compile(fullgraph=True)`, dynamo cannot trace the closures. With `suppress_errors=True`, it silently falls back and corrupts the autograd graph. **Solution**: patch the model and run it in EAGER mode (no compile). Compile only a SEPARATE pure function (e.g., loss computation). Or use Approach 2 (write a fused PyTorch function + compile) instead of monkey-patching.

15. **`flash_attn.losses.cross_entropy` + `torch.compile` spams warnings**: torch.compile's `identify_mutated_tensors` fails on flash_attn's internal Triton CE kernel. The warnings are harmless — it falls back to "assume all inputs mutated". But this means the compiler cannot optimize the CE kernel call. Using flash CE OUTSIDE the compiled function avoids this.

16. **FSDP `NO_SHARD` requires `model = model.to(device)` before FSDP wrapping**. The evaluation environment reloads a CPU state dict between runs. `SHARD_GRAD_OP` handles this via all-gather, but `NO_SHARD` does not. Always call `model = model.to(device)` before `FSDP(model, ...)` when using `NO_SHARD`.

17. **Do NOT add branches/conditionals inside the hot training loop** — e.g., `if step == num_steps - 1: skip_zero_grad()`. Even a single branch dropped MFU by ~2%. Keep the inner loop branchless; handle special cases OUTSIDE the main loop.

18. **`fullgraph=True` consistently HURTS performance** — empirically drops MFU by ~2%. Do NOT use it. Stick with `fullgraph=False` (default).

19. **`torch.compile` compile modes — empirical results** (from 2-GPU runs, directionally applicable to 4-GPU):
    - `reduce-overhead`: **best per-step performance**, uses CUDA graphs. ~10s warmup cost. Works with DDP, FAILS with FSDP.
    - `default`: lower warmup cost but slower per-step. Inductor cache persists from warmup. Works with both DDP and FSDP.
    - No compile: baseline without compilation — expect ~30-40% MFU.
    - `max-autotune`: fails weight verification — DO NOT USE.
    - **With inductor config tuning** (coordinate_descent_tuning, epilogue_fusion, shape_padding): both `default` and `reduce-overhead` generate better kernels. Always set these.

20. **Mixed DP+TP: DTensor + DDP/FSDP incompatibility**: Neither FSDP nor DDP can wrap TP's DTensor parameters (both try to flatten/view params, which DTensor's sharding propagation rejects). For mixed DP+TP, you MUST manually all-reduce gradients across the DP process group after each backward pass. Use the underlying `._local_tensor` for all-reduce to avoid DTensor dispatch issues.

21. **Mixed DP+TP: device mesh setup**: Use `init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))` to create a 2D mesh. Extract `tp_mesh = mesh_2d["tp"]` for `parallelize_module()` and `dp_pg = mesh_2d.get_group("dp")` for manual gradient all-reduce.

22. **TP state dict: DTensor → full_tensor()**: When using TP, parameters become DTensors. You MUST call `.full_tensor()` on each parameter to get the unsharded tensor before building `final_state`. This is a **collective operation** — ALL ranks in the TP group must call it. Rank 0 only won't work.

23. **7B model memory**: Qwen2.5-7B at bf16 ≈ 15GB. Full training per GPU: model 15GB + AdamW fp32 states 30GB + gradients 15GB + activations ~15GB = **~75GB on 80GB**. Very tight with single-GPU or DP-only. Consider:
    - FSDP SHARD_GRAD_OP to shard optimizer states and gradients across ranks
    - TP to split model parameters across 2+ GPUs (reduces per-GPU memory)
    - Mixed DP+TP: TP splits model, DP increases throughput

## Kernel Optimization (highest-leverage direction)

`flash_attn` 2.8.3 and `triton` 3.5.0 are pre-installed. **Replacing default PyTorch ops with fused kernels is the most underexplored and highest-impact optimization path.** Every transformer layer runs the same sequence of ops 28 times (Qwen2.5-7B has 28 layers) per forward pass — even a 5% speedup per op compounds to significant MFU gains.

### Three approaches (ordered by practicality)

**Approach 1: flash_attn's Triton ops (easiest — full backward support)**

These are pre-built Triton kernels with custom backward passes. Drop-in replacements:

| Kernel | Import | What it replaces | Backward |
|---|---|---|---|
| Triton RMS norm | `flash_attn.ops.triton.layer_norm.rms_norm_fn` | Qwen2's `RMSNorm` — 57 calls/fwd (28 layers × 2 + final) | YES |
| SwiGLU activation | `flash_attn.ops.activations.swiglu` | Qwen2's `silu(gate) * up` — 28 calls/fwd | YES |
| Flash CE loss | `flash_attn.losses.cross_entropy.CrossEntropyLoss` | `F.cross_entropy` — fused softmax+CE | YES |
| Triton rotary | `flash_attn.ops.triton.rotary.apply_rotary` | Qwen2's rotary embeddings — 28 calls/fwd | YES |
| Rotary embedding | `flash_attn.layers.rotary.apply_rotary_emb` | Same, alternative impl | YES |

**NOT available** (missing CUDA extensions): `flash_attn.ops.rms_norm`, `flash_attn.ops.fused_dense`.

**Approach 2: Fused PyTorch functions + torch.compile (medium — compiler generates backward)**

Write a plain PyTorch function that fuses multiple ops, then `torch.compile` it. The compiler generates optimized Triton kernels for BOTH forward and backward automatically:

```python
def fused_residual_norm_fwd(hidden, residual, weight, eps):
    hidden = hidden + residual
    variance = hidden.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return hidden * weight

compiled_res_norm = torch.compile(fused_residual_norm_fwd, mode="default", dynamic=False)
```

**Approach 3: Custom @triton.jit kernels (advanced — forward only)**

Write your own GPU kernels for maximum control. `triton` 3.5.0 is available and NOT blocked by security.

Custom `@triton.jit` kernels now support BOTH forward and backward passes. `@staticmethod` is allowed, so you can implement `torch.autograd.Function` with custom forward/backward using Triton kernels.

**Use custom Triton for**: any performance-critical op — RMSNorm, rotary embeddings, fused attention, loss computation, etc.

### How to monkey-patch Qwen2 layers

The model is pre-loaded — replace internal layer forwards with fused kernels:

```python
from flash_attn.ops.triton.layer_norm import rms_norm_fn
from flash_attn.ops.activations import swiglu

def _patch_model(model):
    """Replace Qwen2 ops with fused flash_attn kernels."""
    for layer in model.model.layers:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            w, eps = norm.weight, norm.variance_epsilon
            norm.forward = lambda x, _w=w, _e=eps: rms_norm_fn(x, _w, None, eps=_e)
    fn = model.model.norm
    w, eps = fn.weight, fn.variance_epsilon
    fn.forward = lambda x, _w=w, _e=eps: rms_norm_fn(x, _w, None, eps=_e)

    for layer in model.model.layers:
        mlp = layer.mlp
        original_gate = mlp.gate_proj
        original_up = mlp.up_proj
        original_down = mlp.down_proj
        def fused_mlp_forward(x, _g=original_gate, _u=original_up, _d=original_down):
            gate_out = _g(x)
            up_out = _u(x)
            return _d(swiglu(gate_out, up_out))
        mlp.forward = fused_mlp_forward
```

**CRITICAL PITFALL**: Monkey-patched lambda forwards are INCOMPATIBLE with `torch.compile(fullgraph=True)`. Dynamo cannot trace through closures that capture module weights. Always test kernel patches WITHOUT torch.compile first to verify gradients flow.

## Proven 56.5% MFU DDP Submission (KEY REFERENCE — adapt for 4 GPUs)

A 2-GPU DDP submission achieved 56.5% MFU on A100s. The techniques below are **proven to work** and should be adapted for 4-GPU. DDP on 4 GPUs REQUIRES gradient checkpointing for 7B to fit in memory.

### torch._inductor.config tuning (CRITICAL — improves compile quality)
```python
try:
    import torch._inductor.config as _ind_cfg
    _ind_cfg.coordinate_descent_tuning = True
    _ind_cfg.triton.unique_kernel_names = True
    _ind_cfg.fx_graph_cache = True
    _ind_cfg.triton.cudagraph_trees = True
    _ind_cfg.epilogue_fusion = True
    _ind_cfg.shape_padding = True
except Exception:
    pass
```
These settings make torch.compile generate significantly better Triton kernels. `coordinate_descent_tuning` finds optimal kernel parameters. `epilogue_fusion` fuses more operations. `shape_padding` pads tensor dimensions for better memory alignment. These work with ALL parallelism strategies (FSDP, DDP, TP).

### torch._dynamo.config tuning (CRITICAL — reduces compile overhead)
```python
try:
    import torch._dynamo.config as _dyn_cfg
    _dyn_cfg.cache_size_limit = 128
    _dyn_cfg.suppress_errors = True
    _dyn_cfg.assume_static_by_default = True
    _dyn_cfg.automatic_dynamic_shapes = False
    _dyn_cfg.optimize_ddp = True
except Exception:
    pass
```
`assume_static_by_default` + `automatic_dynamic_shapes=False` eliminates dynamic shape guards, reducing compile time and enabling better optimization. `cache_size_limit=128` prevents cache eviction. `optimize_ddp=True` enables DDP-specific Dynamo optimizations.

### DDP + Selective Gradient Checkpointing (fits 7B on 80GB)
```python
_UNCHECKPOINT_LAST_N = 5

def _prepare_model(model):
    if hasattr(model, "config"):
        model.config.use_cache = False
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False
        if hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        checkpoint_cutoff = num_layers - _UNCHECKPOINT_LAST_N
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = 0
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = idx < checkpoint_cutoff
```
Checkpoint first 23 layers (saves ~12-15GB activation memory), uncheckpoint last 5 (avoids recomputation overhead on final layers). Memory with DDP: 15GB model + 30GB optimizer + 15GB grads + ~5GB activations = ~65GB. Fits on 80GB with room for compile (~4GB).

### DDP Configuration (proven settings)
```python
model = model.to(device)
model = DDP(
    model,
    device_ids=[device.index],
    gradient_as_bucket_view=True,
    static_graph=True,
    bucket_cap_mb=200,
    broadcast_buffers=False,
)
```
`static_graph=True` enables advanced optimizations (fuses gradient bucketing, reduces overhead). `gradient_as_bucket_view=True` avoids gradient copy. `bucket_cap_mb=200` reduces number of all-reduce calls. `broadcast_buffers=False` skips unnecessary buffer broadcast.

**CRITICAL**: DDP requires `model = model.to(device)` BEFORE wrapping. FSDP handles this internally.

### CUDA Memory Allocator Settings
```python
try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:
    pass
```
Reduces memory fragmentation — helps avoid OOM when memory is tight (DDP + 7B).

### Pinned Memory State Dict Transfer (faster GPU→CPU)
```python
raw_model = model.module if hasattr(model, "module") else model
sd = raw_model.state_dict()
pinned = {k: torch.empty_like(v, device="cpu").pin_memory() for k, v in sd.items()}
for k, v in sd.items():
    pinned[k].copy_(v, non_blocking=True)
torch.cuda.synchronize(device)
final_state = pinned
```
Pre-allocates pinned CPU memory and copies asynchronously. ~2-3x faster than `.detach().cpu().clone()` for large state dicts (7B = 15GB).

### torch.compile with fullgraph=True + fallback chain
```python
def fwd_only(input_ids):
    return model(input_ids).logits

compiled_fwd = torch.compile(fwd_only, mode="default", dynamic=False, fullgraph=True)
try:
    logits = compiled_fwd(all_inputs[0])
except Exception:
    try:
        compiled_fwd = torch.compile(fwd_only, mode="default", dynamic=False)
        logits = compiled_fwd(all_inputs[0])
    except Exception:
        compiled_fwd = fwd_only
        logits = compiled_fwd(all_inputs[0])
```
Try `fullgraph=True` first (generates tighter code if it works), fall back to default, then eager. Note: `fullgraph=True` + FSDP fails, but `fullgraph=True` + DDP can work with the right setup (compile the forward function, NOT the model).

## Proven Optimization Patterns (use as building blocks)

1. **Flash CE loss** — faster than F.cross_entropy:
   ```python
   from flash_attn.losses.cross_entropy import CrossEntropyLoss as _FlashCELoss
   _flash_ce = _FlashCELoss(ignore_index=-100)
   ```

2. **Compile only the forward function** (never the model itself):
   ```python
   def _get_compiled_fwd(model):
       def fwd(input_ids): return model(input_ids).logits
       try: return torch.compile(fwd, mode="default", dynamic=False, fullgraph=True)
       except Exception:
           try: return torch.compile(fwd, mode="default", dynamic=False)
           except Exception: return fwd
   ```

3. **Fused AdamW**: `torch.optim.AdamW(..., fused=True)` — always set fused=True on CUDA.

4. **Model preparation** — disable unnecessary outputs before wrapping:
   `model.config.use_cache = False`, `output_hidden_states = False`, `output_attentions = False`
   Fix `layer_idx` to prevent dynamo recompilation: set ALL layers to the SAME value (e.g., `layer.self_attn.layer_idx = 0`). Do NOT use the actual index (`= i`) — that causes dynamo to recompile a separate graph per layer.

5. **Batch pre-loading** — load all batches upfront with `non_blocking=True`, call `torch.cuda.synchronize()` once:
   ```python
   all_inputs, all_labels = [], []
   for _ in range(num_steps):
       batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
       all_inputs.append(batch[:, :-1].contiguous())
       all_labels.append(batch[:, 1:].contiguous())
   torch.cuda.synchronize(device)
   ```
   **CRITICAL**: labels must come from `batch[:, 1:]`, NOT from `inputs[:, 1:]`. Inputs are `batch[:, :-1]`, so `inputs[:, 1:]` = `batch[:, 1:-1]` which is one token short.

6. **TF32 matmul**: `torch.backends.cuda.matmul.allow_tf32 = True` — free throughput boost.

7. **Reduce Python overhead** — bind optimizer methods to local variables in training loop:
   `opt_step = optimizer.step; opt_zero = optimizer.zero_grad`

## Mixed DP+TP Patterns (dp_size=2, tp_size=2)

### Device Mesh Setup
```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh_2d["tp"]
dp_pg = mesh_2d.get_group("dp")
```

### Applying TP to Qwen2 layers
```python
def _apply_tp(model, tp_mesh):
    for name, module in model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            parallelize_module(module, tp_mesh, {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
                "o_proj": RowwiseParallel(),
            })
        if hasattr(module, "gate_proj") and hasattr(module, "down_proj"):
            parallelize_module(module, tp_mesh, {
                "gate_proj": ColwiseParallel(),
                "up_proj": ColwiseParallel(),
                "down_proj": RowwiseParallel(),
            })
    return model
```

### Manual gradient all-reduce across DP group
```python
def _allreduce_grads(model, dp_pg):
    for param in model.parameters():
        if param.grad is not None:
            g = param.grad
            local_g = g._local_tensor if hasattr(g, "_local_tensor") else g
            dist.all_reduce(local_g, op=dist.ReduceOp.AVG, group=dp_pg)
```

### Gathering full state dict from DTensor shards
```python
def _gather_full_state(model):
    state = {}
    for name, param in model.named_parameters():
        p = param.data
        if hasattr(p, "full_tensor"): p = p.full_tensor()
        state[name] = p.detach().cpu().clone()
    for name, buf in model.named_buffers():
        b = buf.data
        if hasattr(b, "full_tensor"): b = b.full_tensor()
        state[name] = b.detach().cpu().clone()
    sd = model.state_dict()
    for key in sd:
        if key not in state:
            val = sd[key]
            if hasattr(val, "full_tensor"): val = val.full_tensor()
            state[key] = val.detach().cpu().clone()
    return state
```

## State Dict Patterns (MUST return correct final_state — required for ALL strategies)

`final_state` is **mandatory** for every submission. Returning `None` will be rejected with `missing_final_state`. The state must contain all model keys with correct shapes — missing keys cause `incomplete_final_state`.

**FSDP (NO_SHARD — fastest, no gathering needed)**:
```python
with FSDP.summon_full_params(model, writeback=False):
    raw = model.module if hasattr(model, "module") else model
    full_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
```

**FSDP (SHARD_GRAD_OP / FULL_SHARD — needs gathering)**:
```python
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    sd = model.state_dict()
    if dist.get_rank() == 0: full_state = sd
```

**DDP**:
```python
raw_model = model.module if hasattr(model, "module") else model
full_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
```

**TP / Mixed DP+TP**: Gather distributed DTensors, include tied weights:
```python
state = {}
for name, param in model.named_parameters():
    p = param.data
    if hasattr(p, "full_tensor"): p = p.full_tensor()
    state[name] = p.detach().cpu().clone()
for name, buf in model.named_buffers():
    b = buf.data
    if hasattr(b, "full_tensor"): b = b.full_tensor()
    state[name] = b.detach().cpu().clone()
sd = model.state_dict()
for key in sd:
    if key not in state:
        val = sd[key]
        if hasattr(val, "full_tensor"): val = val.full_tensor()
        state[key] = val.detach().cpu().clone()
```
ALL ranks must call `full_tensor()` — it's a collective op. Rank 0 keeps the result for `final_state`.

## Memory Reality (7B model)

Qwen2.5-7B bf16 = ~15GB. AdamW stores exp_avg and exp_avg_sq in the same dtype as parameters (bf16), NOT fp32. Full training setup per GPU:
- **DDP (dp=4, tp=1) + selective gradient checkpointing**: Model 15GB + AdamW bf16 (m+v) 30GB + Gradients 15GB + activations ~5GB (23 layers checkpointed, 5 uncheckpointed) = **~65GB**. Room for torch.compile (~4GB extra) = **~69GB** — fits! No micro-batching needed (full batch=16 per GPU). Selective checkpointing (uncheckpoint last N layers) balances memory savings vs recomputation overhead.
- **FSDP SHARD_GRAD_OP (dp=4, tp=1)**: Full model during fwd 15GB + sharded optimizer 7.5GB + sharded grads 3.8GB + activations ~15GB = **~42GB** — good headroom. Can use larger micro-batches → higher throughput than DDP.
- **FSDP FULL_SHARD (dp=4, tp=1)**: Shards everything including params. Peak ~30-35GB — most headroom for compile.
- **Mixed (dp=2, tp=2)**: Model ~7.5GB/GPU + AdamW ~15GB + Grads ~7.5GB + Activations ~15GB = **~45GB** — comfortable headroom. Good balance.
- **Pure TP (dp=1, tp=4)**: Model ~3.75GB/GPU + states ~7.5GB + Grads ~3.75GB + Activations ~15GB = **~30GB** — lots of headroom but low MFU ceiling.

**Important**: DDP has the tightest memory budget — avoid torch.compile with DDP on 7B. FSDP SHARD_GRAD_OP or FULL_SHARD are preferred for dp=4 as they leave more headroom for optimizations.

## FSDP Communication Overlap (for DP strategies)

- `forward_prefetch=True, backward_prefetch=BackwardPrefetch.BACKWARD_PRE` — overlap communication with compute
- **Do NOT set `limit_all_gathers=True`** — empirically HURTS MFU by restricting concurrency
- For SHARD_GRAD_OP: shards optimizer states and gradients but keeps full params in forward — good for 7B where model fits but optimizer doesn't

## Error Signals

- **HTTP 502 Bad Gateway** = your code CRASHED the evaluation container (OOM, segfault, or process killed). This is NOT a server issue — your code caused it. Make a MORE CONSERVATIVE change next time.
- **RuntimeError / AssertionError** = code bug, read the message carefully
- **insufficient_mfu** = code ran but MFU was below 45% threshold
- **Security violation** = forbidden import/name/pattern, read the rule

## IMPORTANT: Don't Repeat Failures

FSDP SHARD_GRAD_OP + torch.compile(default) is MAXED OUT at 56.5% MFU after 40+ attempts. Do NOT try more FSDP variations — switch to DDP.

**Exhausted approaches (do NOT retry)**:
- FSDP SHARD_GRAD_OP + compile(default) → 56.5% ceiling
- FSDP + kernel patches + compile → worse (51-56%)
- FSDP + reduce-overhead → falls back to default
- FSDP NO_SHARD → OOM with compile, 46.9% without
- FSDP FULL_SHARD → 55.7% (slightly worse than SHARD_GRAD_OP)
- Mixed DP+TP → 40-41% (slow optimizer)
- DDP WITHOUT gradient checkpointing → OOM
- Compiling loss function → insufficient_params_changed

**What to try instead**:
1. **DDP + selective gradient checkpointing** — the most promising unexplored direction
2. **inductor/dynamo config tuning** — never been tested, can improve compile quality
3. **reduce-overhead mode with DDP** — CUDA graphs work with DDP (but not FSDP)
4. **Pinned memory state dict** — faster GPU→CPU transfer

## Prior Results (what has been tried and what worked)

**Plateau at 56.5% MFU (~85s wall time)** with FSDP SHARD_GRAD_OP + torch.compile(default). Over 40 attempts exhausted most FSDP-based approaches. Key findings:
- FSDP SHARD_GRAD_OP + compile(default) = **56.5% MFU** (best, ~85s) — the ceiling for this approach
- FSDP + kernel patches + compile together = WORSE (51-56%) — compile can't trace patched lambdas
- FSDP + reduce-overhead = falls back to default (CUDA graphs don't work with FSDP)
- FSDP NO_SHARD = OOM with compile, 46.9% without compile
- DDP without gradient checkpointing = **OOM** (78.98 GiB used)
- Mixed DP+TP = 40-41% MFU (slow optimizer with TP DTensors)
- Compiling loss function = insufficient_params_changed (68.8%)

**What hasn't been tried**: DDP + selective gradient checkpointing + inductor/dynamo config tuning + pinned memory state dict. A proven 2-GPU submission achieved 56.5% with these techniques. Scaling to 4 GPUs should improve further.

## What to Try (ordered by expected impact for 4-GPU 7B)

**HIGHEST PRIORITY: DDP + gradient checkpointing + inductor tuning**

This is the most promising unexplored path. DDP eliminates FSDP's per-layer all-gather/reduce-scatter overhead. With selective gradient checkpointing, 7B fits in memory (~65GB). Combined with inductor/dynamo config tuning, this should break past the 56.5% plateau.

```
DDP advantages over FSDP:
- ONE gradient all-reduce per step (vs 28+ per-layer comms for FSDP)
- static_graph=True enables graph-level optimization
- Compatible with torch.compile + fullgraph=True (FSDP is not)
- gradient_as_bucket_view=True eliminates gradient copy
- Simpler state dict (no gathering needed)
```

**Phase 1: DDP + grad checkpointing baseline (first 3 steps)**
- Use DDP with selective gradient checkpointing (uncheckpoint last 5 layers)
- Add ALL inductor/dynamo config settings (see "Proven 56.5% MFU DDP Submission" section)
- Add TF32, fused AdamW, batch pre-loading, flash CE loss, CUDA allocator settings
- Use pinned memory state dict transfer
- Compile forward with fullgraph=True, fall back to default
- Target: 55-60% MFU (should match or beat FSDP baseline)

**Phase 2: Optimize DDP compile (steps 3-10)**
- Try `mode="reduce-overhead"` with DDP (CUDA graphs should work with DDP unlike FSDP!)
- Tune `_UNCHECKPOINT_LAST_N` (try 3, 5, 7) to balance memory vs recomputation
- Try `bucket_cap_mb` values (100, 200, 300)
- Target: 60-65% MFU

**Phase 3: Kernel patches + DDP (steps 10+)**
- Add RMSNorm + SwiGLU kernel patches (run in EAGER mode, no compile on patched model)
- Or try class-level patching which IS compile-compatible
- Explore `mode="reduce-overhead"` if memory allows
- Target: 65%+ MFU

**Parallelism strategies** (all valid for 4 GPUs):
- **DDP dp=4** (`{"dp_size": 4, "tp_size": 1}`): **RECOMMENDED — most promising unexplored path**. Requires selective gradient checkpointing for 7B (~65GB peak). Less communication overhead than FSDP. Compatible with torch.compile + fullgraph=True. Use static_graph=True for graph-level optimizations. Proven at 56.5% on 2 GPUs — should scale well to 4.
- **FSDP dp=4** (`{"dp_size": 4, "tp_size": 1}`): Proven at 56.5% MFU. Use SHARD_GRAD_OP for 7B. Ceiling appears to be ~56.5% — try DDP first.
- **Mixed DP+TP** (`{"dp_size": 2, "tp_size": 2}`): Best memory efficiency but MFU ceiling is lower (~40-41%). Only use if DDP and FSDP fail.
- **Pure TP** (`{"dp_size": 1, "tp_size": 4}`): Lowest MFU ceiling. Only use if other strategies OOM.

**Kernel-based optimizations** (highest leverage, use with DDP in eager mode):
- **Triton RMS norm** (`flash_attn.ops.triton.layer_norm.rms_norm_fn`) — 57 calls per forward
- **SwiGLU** (`flash_attn.ops.activations.swiglu`) — 28 calls per forward
- **Flash CE loss** (`flash_attn.losses.cross_entropy.CrossEntropyLoss`)
- **Triton rotary** (`flash_attn.ops.triton.rotary.apply_rotary`)
- Fused AdamW (`fused=True`) — single CUDA kernel for optimizer step

**Standard compute optimizations**:
- TF32 matmul (`torch.backends.cuda.matmul.allow_tf32 = True`)
- `torch.compile(fwd, mode="default", fullgraph=True)` on forward-only function (try reduce-overhead with DDP)
- Inductor config tuning (coordinate_descent_tuning, epilogue_fusion, shape_padding)
- Dynamo config tuning (assume_static_by_default, cache_size_limit=128)
- Batch pre-loading with `non_blocking=True`
- Local variable binding in hot loops
- Pinned memory state dict transfer

**Communication optimizations**:
- For DDP: `static_graph=True`, `gradient_as_bucket_view=True`, `bucket_cap_mb=200`
- For FSDP: `backward_prefetch=BackwardPrefetch.BACKWARD_PRE`, `forward_prefetch=True`
- NVLink between all 4 GPUs (600 GB/s bidirectional per pair) — communication is fast

## Response Format

REASONING:
[What you're changing, why, what you learned from previous attempts, expected impact]

CODE:
```python
[Complete train.py — FULL file, not a diff]
```
