# Templar Tournament

**TPS Competition on Bittensor** - Miners compete to optimize training code for maximum Tokens Per Second.

## Architecture

```
MINER                           BLOCKCHAIN                      VALIDATOR
  │                                                                  │
  ├─▶ Upload train.py to R2                                          │
  │                                                                  │
  ├─▶ Commit R2 credentials ──────▶ set_reveal_commitment            │
  │                                 (timelock encrypted)             │
  │                                        │                         │
  │                                        ▼ (after reveal_blocks)   │
  │                                   Decrypted ◀────────────────────┤ Read commitments
  │                                                                  │
  │                                                           Download from R2
  │                                                                  │
  │                                                           Evaluate in Docker
  │                                                                  │
  │                                                           Set weights
  │                                                                  ▼
  └──────────────────────── Rewards ◀────────────────────────── Winner gets 5%
```

## Quick Start

### Prerequisites

```bash
# Clone and setup
cd templar-tournament
uv sync

# Create .env
cat > .env << 'EOF'
TOURNAMENT_R2_ACCOUNT_ID=your_cloudflare_account_id
TOURNAMENT_R2_BUCKET_NAME=your_bucket_name
TOURNAMENT_R2_ACCESS_KEY_ID=your_access_key
TOURNAMENT_R2_SECRET_ACCESS_KEY=your_secret_key
HF_TOKEN=hf_your_token
EOF

export $(cat .env | grep -v '^#' | xargs)
```

### Miner

```bash
# 1. Download model & data
uv run python local_test/setup_benchmark.py

# 2. Optimize train.py and test locally
cd local_test && uv run python train.py

# 3. Submit to tournament
uv run python -m neurons.miner submit local_test/train.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey

# 4. Check status
uv run python -m neurons.miner status
```

### Validator

```bash
uv run python -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker
```

## Miner Commands

| Command | Description |
|---------|-------------|
| `miner upload <file>` | Upload train.py to R2 |
| `miner commit` | Commit to blockchain |
| `miner submit <file>` | Upload + commit (recommended) |
| `miner status` | Check commitment status |

**Parameters**: `--wallet.name`, `--wallet.hotkey`, `--network` (default: finney)

Note: `netuid` and `reveal_blocks` are controlled by `hparams.json`, not CLI.

## Configuration

Key settings in `hparams/hparams.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `netuid` | 3 | Subnet ID |
| `reveal_blocks` | 100 | Blocks before reveal (~20 min) |
| `min_blocks_between_commits` | 10 | Rate limit between submissions |
| `evaluation_runs` | 2 | Runs per submission (median taken) |
| `eval_steps` | 3 | Training steps per evaluation |
| `benchmark_model_name` | Qwen/Qwen2.5-7B | Model for evaluation |
| `benchmark_batch_size` | 16 | Batch size for evaluation |
| `burn_rate` | 0.95 | Emissions to validator (95%) |

## train.py Requirements

Your `train.py` must implement:

```python
def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Optimize this function for maximum TPS."""
    # Your optimizations here
    return InnerStepsResult(
        final_logits=logits,      # Last step logits
        total_tokens=total_tokens, # Total tokens processed
        final_loss=loss,          # Final loss value
    )
```

## Project Structure

```
templar-tournament/
├── neurons/
│   ├── miner.py          # Miner CLI
│   └── validator.py      # Validator
├── local_test/
│   ├── setup_benchmark.py
│   └── train.py          # Template to optimize
├── environments/templar/
│   └── env.py            # Docker evaluation environment
├── hparams/
│   └── hparams.json      # Configuration
└── src/tournament/
    ├── chain/            # Blockchain interactions
    ├── affinetes/        # Docker evaluation
    └── storage/          # Database
```

## TUI Dashboard

```bash
uv run python -m tournament.tui --db tournament.db
```

## Local Testing

Localnet doesn't have drand integration, so commit-reveal won't decrypt. Use mock mode:

```bash
# Start localnet with 12-second blocks
docker run -d --name subtensor-localnet \
    -p 9944:9944 -p 9945:9945 \
    ghcr.io/opentensor/subtensor-localnet:v2.0.11 False

# Run validator with mock commitments (bypasses drand)
export MOCK_COMMITMENTS=1
uv run python -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --skip-blockchain-check \
    --affinetes-mode docker
```

Update `MOCK_COMMITMENT_DATA` in `src/tournament/chain/commitments.py` with your R2 credentials.

## Testing on Testnet

For full E2E testing with real commit-reveal, use testnet (has drand):

```bash
uv run python -m neurons.miner submit local_test/train.py \
    --wallet.name your_wallet --wallet.hotkey your_hotkey --network test
```

## License

MIT
