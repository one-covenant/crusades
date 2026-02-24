# Templar Crusades

**MFU Crusades on Bittensor** - Miners compete to optimize training code for maximum MFU (Model FLOPs Utilization).

## How It Works

```
┌───────────────────────────────────────────────────────────────────────────────── ┐
│                              Crusades FLOW                                       │
│                                                                                  │
│   MINER                        BLOCKCHAIN                      VALIDATOR         │
│     │                                                              │             │
│     │  1. Host train.py at URL                                     │             │
│     │     (Gist, Pastebin, etc)                                    │             │
│     │                                                              │             │
│     ├──▶ 2. Submit URL ─────────▶ set_reveal_commitment            │             │
│     │                             (timelock encrypted)             │             │
│     │                                    │                         │             │
│     │                                    │ (wait reveal_blocks)    │             │
│     │                                    ▼                         │             │
│     │                              3. Decrypted ◀───────────────── ┤ Read        │
│     │                                                              │             │
│     │                                                     4. Download code       │
│     │                                                        from URL            │
│     │                                                              │             │
│     │                                                     5. Runs in Container   │
│     │                                                        (X eval runs)       │
│     │                                                              │             │
│     │                                                     6. Calculate MFU       │
│     │                                                        (median score)      │
│     │                                                              │             │
│     │                                                     7. Set weights         │
│                                                                                  │
└───────────────────────────────────────────────────────────────────────────────── ┘
```

## Quick Start

### Prerequisites

```bash
# Clone and setup
git clone https://github.com/one-covenant/crusades
cd crusades
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Create .env (for HuggingFace access)
echo "HF_TOKEN=hf_your_token" > .env
```

---

## For Miners

### 1. Setup & Test Locally

```bash
# Download model & data for local testing
uv run local_test/setup_benchmark.py

# Test your train.py locally (performance test)
uv run local_test/train.py

# Verify your submission to avoid potential failures during validator checks
uv run local_test/verify.py
```

### Simulate the Validator (Recommended)

Test your `train.py` inside the **exact same Docker container** the production validator uses. This gives MFU numbers that closely match the tournament leaderboard and avoids environment dependency mismatches.

```bash
# Build the eval image (from repo root)
docker build --network=host -f environments/templar/Dockerfile \
    --no-cache -t templar-eval:latest .

# Run the simulation (requires an A100 GPU or equivalent)
# Note: MFU/benchmark results will only closely match leaderboard numbers
# when executed on the same GPU model (A100). Runs on other hardware may
# produce divergent MFU results.
docker run --gpus '"device=0"' -it --rm \
    -v "$(pwd)/local_test/train.py":/test/train.py \
    -v "$(pwd)/local_test/simulate_validator.py":/test/simulate.py \
    -v "$(pwd)/hparams/hparams.json":/app/hparams.json \
    -v "$(pwd)/environments/templar/env.py":/app/env.py \
    -v "$(pwd)/src/crusades/core/security_defs.py":/app/crusades/core/security_defs.py \
    -e PYTHONPATH=/app \
    templar-eval:latest \
    python3 /test/simulate.py
```

This runs the full evaluation pipeline: security scan, reference baseline, warmup, timed eval, gradient/weight verification, and MFU calculation — all with the same thresholds as production.

### 2. Host Your Code

Host your `train.py` at any URL that returns raw code:
- **GitHub Gist** (recommended - use secret gist for privacy)
- **Raw GitHub file** (use raw.githubusercontent.com)
- **Pastebin** or any paste service
- Any HTTP/HTTPS URL

### 3. Submit to Crusades

```bash

# Submit to mainnet
uv run -m neurons.miner submit "https://gist.github.com/user/gist_id" \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network finney

# Submit to localnet (testing)
uv run -m neurons.miner submit "https://gist.github.com/user/gist_id" \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network local
```

**Parameters**: `--wallet.name`, `--wallet.hotkey`, `--network` (finney/test/local)

---

## For Validators

See [docs/Validator.md](docs/Validator.md) for detailed validator setup.

---

## train.py Requirements

Your `train.py` must implement the `inner_steps` function.

### Function Signature

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    ...
    return InnerStepsResult(final_logits=..., total_tokens=..., final_loss=...)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Pre-loaded model on the correct device, in train mode |
| `data_iterator` | `Iterator[torch.Tensor]` | Infinite iterator yielding `(batch_size, seq_len)` tensors. In multi-GPU mode each rank receives non-overlapping batches. |
| `optimizer` | `Optimizer \| None` | Validator-provided optimizer for single-GPU. **`None` when `num_gpus > 1`** -- miner must create their own. |
| `num_steps` | `int` | Number of training steps to complete |
| `device` | `torch.device` | Target device (`cuda:0`, `cuda:1`, etc.). Use `device.index` for local rank. |
| `num_gpus` | `int` | Number of GPUs. `1` = single-GPU, `>1` = multi-GPU with `torchrun` (process group already initialized). |

### Single-GPU Baseline

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor  # 3D: (batch, seq_len-1, vocab) - NOT None
    total_tokens: int           # Total tokens processed across all steps
    final_loss: float           # Loss from last step (must be > 0)

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
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
```

### Multi-GPU Example (DDP)

When `num_gpus > 1`, `optimizer` is `None`. The miner wraps the model and creates its own optimizer:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    if num_gpus > 1:
        model = DDP(model, device_ids=[device.index])

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    # ... training loop (same as single-GPU) ...
```

Any parallelism strategy works: DDP, FSDP, Tensor Parallelism, Pipeline Parallelism, or hybrids. The process group is already initialized by `torchrun` before `inner_steps` is called.

### Rules

**General (all modes):**
- Process all tokens in each batch, return valid `final_logits` (not `None`)
- Train all model parameters (no freezing layers)
- Don't alias `torch` (e.g., `import torch as t`) -- the scanner needs the literal name

**Single-GPU (`num_gpus == 1`):**
- Use the provided `optimizer` directly (`optimizer.step()` / `optimizer.zero_grad()`)
- The validator captures gradients through the optimizer wrapper for verification

**Multi-GPU (`num_gpus > 1`):**
- `optimizer` is `None` -- create your own after wrapping the model
- You may use any parallelism: DDP, FSDP, TP, PP, or combinations
- The original `model` object must have full (unsharded) parameters when `inner_steps` returns (DDP does this automatically; FSDP/TP require explicit parameter gathering)
- Gradient verification is skipped; the validator verifies via loss and final weight comparison

A static security scanner blocks dangerous patterns (forbidden imports, monkey-patching, timer tampering, etc.). See [`src/crusades/core/security_defs.py`](src/crusades/core/security_defs.py) for the full blocklist. Run `uv run local_test/verify.py` before submitting to catch violations early.

Any genuine optimization is fair game -- `torch.compile`, mixed precision, Flash Attention, Triton kernels, CUDA Graphs, custom loss functions, and more. If the scanner rejects something that should be allowed, report to us.

> **Note:** The validator scans **all** code including `if __name__ == "__main__":` blocks. Use a separate test script if you need locally-useful imports (like `pathlib`) for testing.

---

## Configuration

Key settings in `hparams/hparams.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `netuid` | 3 | Subnet ID |
| `evaluation_runs` | 5 | Runs per submission (median taken) |
| `eval_steps` | 5 | Training steps per evaluation |
| `benchmark_model_name` | Qwen/Qwen2.5-3B | Model for evaluation |
| `benchmark_batch_size` | 8 | Batch size for evaluation |
| `docker.num_gpus` | 1 | Number of GPUs (1 = single-GPU, >1 = multi-GPU via `torchrun`) |

---

## TUI Dashboard

Monitor crusades activity in real-time with the terminal dashboard.

```bash
# Connect to the official Crusades API
uv run -m crusades.tui --url 69.19.137.219:8080
```

### Features

- Leaderboard with MFU scores
- Recent submissions and their status
- MFU history chart
- Validator status
- View submission code (after evaluation)
