# Templar Tournament

**TPS Competition on Bittensor** - Miners compete to optimize training code for maximum Tokens Per Second.

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TOURNAMENT FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   MINER                      BLOCKCHAIN                  VALIDATOR      │
│                                                                         │
│   1. Setup                                                              │
│      └─▶ local_test/setup_benchmark.py                                  │
│          (downloads model + data)                                       │
│                                                                         │
│   2. Optimize train.py                                                  │
│      └─▶ local_test/train.py                                            │
│          (test locally, maximize TPS)                                   │
│                                                                         │
│   3. Upload to R2                                                       │
│      └─▶ miner upload train.py ──────▶ R2 bucket                        │
│                                                                         │
│   4. Commit                                                             │
│      └─▶ miner commit ───────────────▶ Blockchain                       │
│                                        (hidden 100 blocks)              │
│                                              │                          │
│                                              ▼                          │
│                                        5. Revealed ──▶ 6. Validator     │
│                                                          reads commit   │
│                                                              │          │
│                                                              ▼          │
│                                                       7. Downloads      │
│                                                          from R2        │
│                                                              │          │
│                                                              ▼          │
│                                                       8. Evaluates      │
│                                                          ( runs)       │
│                                                          Median TPS     │
│                                                              │          │
│   ┌────────┐                                                 ▼          │
│   │ WINNER │ ◀──── 5% ◀────────────────────────── 9. set_weights        │
│   └────────┘                                         (95% validator)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start (Miner)

### Step 1: Setup Environment

```bash
cd templar-tournament

# Create .env with R2 credentials
cat > .env << 'EOF'
TOURNAMENT_R2_ACCOUNT_ID=your_cloudflare_account_id
TOURNAMENT_R2_BUCKET_NAME=your_bucket_name
TOURNAMENT_R2_ACCESS_KEY_ID=your_access_key
TOURNAMENT_R2_SECRET_ACCESS_KEY=your_secret_key
HF_TOKEN=hf_your_token_here
EOF

# Load environment
export $(cat .env | grep -v '^#' | xargs)
```

### Step 2: Download Model & Data

```bash
uv run python local_test/setup_benchmark.py
```

Creates:
- `benchmark/model/` - Qwen2.5-7B model
- `benchmark/data/train.pt` - Pre-tokenized data (10,000 samples)

### Step 3: Optimize train.py

Edit `local_test/train.py` - the key function to optimize:

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Optimize this function for maximum TPS."""
    # Your optimizations here!
    return InnerStepsResult(
        final_logits=logits,
        total_tokens=total_tokens,
        final_loss=loss,
    )
```

Test locally:
```bash
cd local_test && uv run python train.py
```

### Step 4: Upload to R2

```bash
uv run python -m neurons.miner upload local_test/train.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey
```

### Step 5: Commit to Blockchain

```bash
uv run python -m neurons.miner commit \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network finney \
    --netuid 3
```

Or do both in one step:
```bash
uv run python -m neurons.miner submit local_test/train.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network finney \
    --netuid 3
```

### Step 6: Check Status

```bash
uv run python -m neurons.miner status --network finney
```

## Miner Commands

| Command | Description |
|---------|-------------|
| `miner test train.py` | Validate train.py structure |
| `miner upload train.py` | Upload to R2 bucket |
| `miner commit` | Commit R2 info to blockchain |
| `miner submit train.py` | Upload + commit in one step |
| `miner status` | Check commitment status |

## Configuration (hparams.json)

| Setting | Value | Description |
|---------|-------|-------------|
| `benchmark_model_name` | Qwen/Qwen2.5-7B | Model used for evaluation |
| `benchmark_batch_size` | 16 | Batch size |
| `eval_steps` | 5 | Steps per evaluation run |
| `evaluation_runs` | 2 | Number of runs (median = score) |
| `reveal_blocks` | 100 | Blocks before commit revealed |
| `min_blocks_between_commits` | 100 | Rate limit between submissions |

## Local Testing (No Blockchain)

```bash
# Upload
export $(cat .env | grep -v '^#' | xargs)
uv run python -m neurons.miner upload local_test/train.py \
    --wallet.name templar_test --wallet.hotkey M1

# Commit locally (saves to .local_commitments/)
uv run python -m neurons.miner commit \
    --wallet.name templar_test --wallet.hotkey M1 \
    --network local --netuid 3
```

## Project Structure

```
templar-tournament/
├── local_test/
│   ├── setup_benchmark.py   # Download model + data
│   └── train.py             # Your optimized code
├── neurons/
│   ├── miner.py             # Miner CLI
│   └── validator.py         # Validator (coming next)
├── hparams/
│   └── hparams.json         # Configuration
└── .env                     # R2 credentials (not in git)
```

## Validator Setup

*Coming in next section...*
