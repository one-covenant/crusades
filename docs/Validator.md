# Validator Guide

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VALIDATOR (always runs locally)                    │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                    validator.py main loop                           │    │
│   │                                                                     │    │
│   │  1. Read commitments from blockchain                               │    │
│   │  2. Download miner code from committed URLs                        │    │
│   │  3. Call affinetes_runner.evaluate(code=...)                       │    │
│   │  4. Set weights based on TPS scores                                │    │
│   └───────────────────────────┬────────────────────────────────────────┘    │
│                               │                                              │
│                     ┌─────────┴─────────┐                                   │
│                     │  --affinetes-mode │                                   │
│                     └─────────┬─────────┘                                   │
│                               │                                              │
│           ┌───────────────────┴───────────────────┐                         │
│           │                                       │                         │
│           ▼                                       ▼                         │
│   ┌───────────────┐                      ┌───────────────┐                  │
│   │    DOCKER     │                      │   BASILICA    │                  │
│   │    MODE       │                      │     MODE      │                  │
│   └───────┬───────┘                      └───────┬───────┘                  │
│           │                                       │                         │
└───────────┼───────────────────────────────────────┼─────────────────────────┘
            │                                       │
            ▼                                       ▼
   ┌─────────────────┐                    ┌─────────────────────┐
   │  LOCAL DOCKER   │                    │   BASILICA CLOUD    │
   │                 │                    │                     │
   │  • Spawns       │                    │  • Sends code to    │
   │    container    │                    │    remote API       │
   │  • Uses YOUR    │                    │  • Remote GPUs run  │
   │    GPU          │                    │    evaluation       │
   │  • Runs         │                    │  • Returns TPS      │
   │    templar-eval │                    │    score            │
   │    image        │                    │                     │
   └─────────────────┘                    └─────────────────────┘
```

| Mode | Where evaluation runs | Best for |
|------|----------------------|----------|
| `docker` | Your local GPU | Self-hosted, full control |
| `basilica` | Remote cloud GPU | No local GPU (requires API key) |

---

## Run Validator

### Mainnet (Production)

```bash
cd /path/to/templar-tournament

# Start validator
SUBTENSOR_NETWORK=finney uv run python -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

### Localnet (Testing)

```bash
cd /path/to/templar-tournament

# Start validator on localnet
SUBTENSOR_NETWORK=local uv run python -m neurons.validator \
    --wallet.name templar_test \
    --wallet.hotkey V1 \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

### Testnet

```bash
cd /path/to/templar-tournament

# Start validator on testnet
SUBTENSOR_NETWORK=test uv run python -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

---

## Prerequisites

### 1. Build Evaluation Image

```bash
cd environments/templar
docker build -t templar-eval:latest .
```

### 2. Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### 3. Configure hparams.json

Edit `hparams/hparams.json` for your setup:

```json
{
    "netuid": 1,
    "burn_uid": 0,
    "docker": {
        "gpu_devices": "0,1,2,3",
        "memory_limit": "32g",
        "shm_size": "8g"
    }
}
```

| Setting | Description |
|---------|-------------|
| `gpu_devices` | GPUs to use: `"0"`, `"0,1"`, `"all"`, or `"none"` |
| `memory_limit` | Container memory limit |
| `shm_size` | Shared memory for PyTorch DataLoader |

---

## Manage

```bash
# View logs (includes Docker evaluation progress)
tail -f logs/validator.log

# Check if running
ps aux | grep neurons.validator

# Stop
pkill -f neurons.validator
# or Ctrl+C if running in foreground
```

### Log Output Example

During evaluation, you'll see Docker container logs streamed in real-time:

```
INFO | Running Docker evaluation
INFO |    Code size: 9876 bytes
INFO |    Docker command: docker run --rm -v /tmp/train_abc.py...
INFO |    [DOCKER] Loading model: Qwen/Qwen2.5-7B
INFO |    [DOCKER] Loading dataset: HuggingFaceFW/fineweb (samples=10000)
INFO |    [DOCKER] Validator mode: seed=12345678 (from 1:3:1737...)
INFO |    [DOCKER] Loaded data: shape=torch.Size([10000, 1024])
INFO |    [DOCKER] Running miner's inner_steps...
INFO | Run 1 PASSED: 3,622.26 TPS
```

---

## Monitor (TUI)

```bash
uv run python -m tournament.tui --db tournament.db
```

Shows:
- Leaderboard with TPS scores
- Recent submissions
- TPS history chart
- Evaluation queue

---

## Networks Summary

| Network | Environment Variable | Use Case |
|---------|---------------------|----------|
| Mainnet | `SUBTENSOR_NETWORK=finney` | Production |
| Testnet | `SUBTENSOR_NETWORK=test` | Testing with real commit-reveal |
| Localnet | `SUBTENSOR_NETWORK=local` | Development |

---

## Troubleshooting

### "Too soon to commit weights"

Normal rate limiting. Validators can only set weights once every ~100-360 blocks.

### Docker permission denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

---

## View Submission Code

Miner code is stored in the database after evaluation for verification purposes.

### List All Submissions

```bash
uv run python scripts/view_submission.py
```

Output:
```
================================================================================
SUBMISSION           UID   STATUS     TPS          CODE       SUBMITTED
================================================================================
commit_15118_3       3     evaluating N/A          N/A        2026-01-25 09:34:06
commit_14954_1       1     finished   3616.38      9791 bytes 2026-01-25 09:01:13
commit_9303_1        1     finished   3622.26      10042 bytes 2026-01-25 08:46:29
```

### View Specific Submission

```bash
# View submission details and code
uv run python scripts/view_submission.py commit_9303_1
```

### Save Code to File

```bash
# Save miner's code to a file for review
uv run python scripts/view_submission.py commit_9303_1 --save
# Creates: commit_9303_1_train.py
```

### Direct Database Query

```bash
# Using Python
uv run python -c "
import sqlite3
conn = sqlite3.connect('tournament.db')
cur = conn.cursor()
cur.execute('SELECT code_content FROM submissions WHERE submission_id=?', ('commit_9303_1',))
print(cur.fetchone()[0])
"
```
