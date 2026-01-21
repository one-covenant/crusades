# Templar Tournament

**Compete to write the fastest PyTorch training code. Winner takes all!**

```
┌───────────────────────────────────────────────────────────────────────────── ┐
│                           TOURNAMENT FLOW                                    │
├───────────────────────────────────────────────────────────────────────────── ┤
│                                                                              │
│   MINER                          VALIDATOR                    BLOCKCHAIN     │
│                                                                              │
│   ┌──────────┐                                                               │
│   │ train.py │──▶ Docker Image ──▶ Commit (hidden) ──▶ Reveal (100 blocks)   │
│   └──────────┘                                               │               │
│                                                              ▼               │
│                              ┌────────────────────────────────────┐          │
│                              │  VALIDATOR EVALUATION (5 runs)     │          │
│                              │  → Median TPS = Final Score        │          │
│                              └──────────────┬─────────────────────┘          │
│                                             │                                │
│                                             ▼                                │
│   ┌──────────┐               ┌─────────────────────────┐                     │
│   │  WINNER  │ ◀─── 5% ◀─── │   set_weights on chain  │ ───▶ 95% validator   │
│   └──────────┘               └─────────────────────────┘                     │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────── ┘
```

---

## Quick Start for Miners

### 1. Setup

```bash
git clone https://github.com/one-covenant/crusades.git && cd crusades
uv sync

# Download model and training data (one time)
uv run python scripts/setup_benchmark.py
```

### 2. Local Testing

```bash
# Test your train.py locally (uses fixed seed = same data for all miners)
uv run python train.py
```

Output:
```
Loading model from benchmark/model/
Loading data from benchmark/data/train.pt
   Samples: 100,000
   Sequence length: 1024

Running 5 evaluations (5 steps each)...
   Run 1: 3,542.15 TPS
   Run 2: 3,589.23 TPS
   ...

RESULTS (Median of 5 evaluations)
   Median TPS: 3,567.89
```

### 3. Submit

```bash
# Build Docker image
uv run python -m neurons.miner build train.py --image my-submission --tag v1

# Commit to blockchain (hidden for 100 blocks)
uv run python -m neurons.miner commit \
    --image my-submission:v1 \
    --wallet.name mywallet \
    --wallet.hotkey myhotkey \
    --network finney

# Check status
uv run python -m neurons.miner status --wallet.name mywallet --wallet.hotkey myhotkey
```

---

## Code Requirements

Your `train.py` must implement:

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device) -> InnerStepsResult:
    """Run training steps as fast as possible.
    
    Args:
        model: The model to train
        data_iterator: Iterator yielding batches of shape [batch, seq_len]
        optimizer: PyTorch optimizer
        num_steps: Number of steps to run
        device: torch device
    
    Returns:
        InnerStepsResult(final_logits, total_tokens, final_loss)
    """
```

**Rules:**
- Output logits must match reference within 2%
- Must complete in 10 minutes
- No network access, no subprocess calls

---

## Data Handling

| Mode | Seed | Data |
|------|------|------|
| **Miner Local** | Fixed (42) | Same for all miners |
| **Validator Eval** | Random (block hash) | Unpredictable per run |

This ensures:
- Miners can reproduce each other's results locally
- Validators use unpredictable data so miners can't cheat

---

## Validator Setup

```bash
git clone https://github.com/one-covenant/crusades.git && cd crusades
uv sync

uv run python -m neurons.validator \
    --wallet.name validator \
    --wallet.hotkey default \
    --affinetes-mode docker
```

---

## Configuration (`hparams/hparams.json`)

| Setting | Value | Description |
|---------|-------|-------------|
| `benchmark_model_name` | `Qwen/Qwen2.5-7B` | Model for evaluation |
| `benchmark_dataset_name` | `HuggingFaceFW/fineweb` | Dataset |
| `benchmark_batch_size` | `16` | Batch size |
| `benchmark_sequence_length` | `1024` | Sequence length |
| `benchmark_data_samples` | `1000` | Samples per eval (validator) |
| `benchmark_train_size` | `100000` | Training data (miner local) |
| `evaluation_runs` | `2` | Runs per submission |
| `reveal_blocks` | `100` | Blocks until reveal (~20 min) |
| `min_blocks_between_commits` | `100` | Rate limit |
| `burn_rate` | `0.95` | % to validator (5% to winner) |
| `output_vector_tolerance` | `0.02` | Max allowed diff (2%) |

---

## Dashboard

```bash
uv run python -m tournament.tui --db tournament.db   # Live data
uv run python -m tournament.tui --demo               # Demo mode
```

---

## Project Structure

```
crusades/
├── train.py                # YOUR CODE - optimize this!
├── scripts/
│   └── setup_benchmark.py  # Download model & data
├── neurons/
│   ├── miner.py            # build, commit, status
│   └── validator.py        # evaluation loop
├── environments/templar/
│   ├── env.py              # evaluation harness
│   └── train.py            # baseline (copied to Docker)
├── src/tournament/
│   ├── affinetes/          # Docker runner
│   ├── chain/              # blockchain (commits, weights)
│   └── storage/            # SQLite database
├── benchmark/              # Generated by setup_benchmark.py
│   ├── model/              # Downloaded model
│   └── data/train.pt       # Pre-tokenized training data
└── hparams/hparams.json    # Configuration
```

---

**Ready to compete?**
