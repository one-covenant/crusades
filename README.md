# ⚡ Templar Tournament

**Compete to write the fastest PyTorch training code. Winner takes all!**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOURNAMENT FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MINER                          VALIDATOR                    BLOCKCHAIN     │
│                                                                              │
│   ┌──────────┐                                                               │
│   │ train.py │──▶ Docker Image ──▶ Commit (hidden) ──▶ Reveal (100 blocks)  │
│   └──────────┘                                               │               │
│                                                              ▼               │
│                              ┌────────────────────────────────────┐          │
│                              │  VALIDATOR EVALUATION (5 runs)     │          │
│                              │  → Median TPS = Final Score        │          │
│                              └──────────────┬─────────────────────┘          │
│                                             │                                │
│                                             ▼                                │
│   ┌──────────┐               ┌─────────────────────────┐                    │
│   │  WINNER  │ ◀─── 5% ◀─── │   set_weights on chain  │ ───▶ 95% validator │
│   └──────────┘               └─────────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Miners

```bash
git clone https://github.com/one-covenant/crusades.git && cd crusades
uv sync

# Build & test locally
uv run python -m neurons.miner build --code train.py --tag my-sub:v1
uv run python -m neurons.miner test --image my-sub:v1

# Submit to blockchain
uv run python -m neurons.miner commit \
    --wallet.name mywallet --wallet.hotkey myhotkey \
    --image my-sub:v1 --network finney
```

### Validators

```bash
git clone https://github.com/one-covenant/crusades.git && cd crusades
uv sync

uv run python -m neurons.validator \
    --wallet.name validator --wallet.hotkey default \
    --network finney --netuid 2
```

---

## Code Requirements

Your `train.py` must implement:

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device) -> InnerStepsResult:
    """Run training steps as fast as possible.
    
    Returns: InnerStepsResult(final_logits, total_tokens, final_loss)
    """
```

**Rules:** Output must match reference within 2% • Complete in 10 min • No network/subprocess

---

## Configuration (`hparams/hparams.json`)

| Setting | Value | Description |
|---------|-------|-------------|
| `benchmark_model_name` | `Qwen/Qwen2.5-7B` | Model for evaluation |
| `benchmark_batch_size` | `16` | Batch size |
| `benchmark_sequence_length` | `1024` | Sequence length |
| `evaluation_runs` | `5` | Runs per submission |
| `reveal_blocks` | `100` | Blocks until reveal (~20 min) |
| `min_blocks_between_commits` | `100` | Rate limit between submissions |
| `burn_rate` | `0.95` | % to validator (5% to winner) |

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
├── neurons/
│   ├── miner.py        # build, test, commit commands
│   └── validator.py    # evaluation loop
├── environments/templar/
│   ├── env.py          # evaluation harness
│   └── train.py        # baseline implementation
├── src/tournament/
│   ├── affinetes/      # Docker runner
│   ├── chain/          # blockchain (commitments, weights)
│   ├── storage/        # SQLite database
│   └── tui/            # terminal dashboard
└── hparams/hparams.json
```

---

**Ready to compete? ⚡**
