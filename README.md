# ⚡ Templar Tournament

Compete to write the fastest PyTorch training code. Winner-takes-all: Highest TPS → 100% of subnet emissions.

**Live Dashboard:** Check with your validator for their dashboard URL

---

## 🎯 Overview

**The Challenge:** Optimize training to achieve maximum tokens-per-second (TPS)

- **Task**: Run 5 training steps on Qwen2.5-3B
- **Metric**: TPS = 40,960 tokens / wall time
- **Reward**: #1 on leaderboard gets 100% emissions
- **Cost**: 0.1 TAO per submission

---

## 🚀 For Miners

### Setup
```bash
# 1. Clone and install
git clone https://github.com/one-covenant/templar-tournament.git
cd templar-tournament
uv sync

# 2. Download model and data (~10GB, 15-30 mins)
uv run python scripts/setup_miner.py
```

### Test Locally
```bash
# Test baseline (~4,300 TPS)
uv run python train.py

# Validate before submitting
uv run python -m tournament.test_local train.py
```

### Submit to Validator
```bash
# One-time: Register on subnet 2
btcli subnet register --netuid 2 --wallet.name mywallet --wallet.hotkey myhotkey

# Submit code (costs 0.1 TAO)
uv run python -m neurons.miner train.py \
    --wallet.name mywallet \
    --wallet.hotkey myhotkey \
    --payment-recipient <validator_hotkey> \
    --validator-api <validator_url>

# Example:
uv run python -m neurons.miner train.py \
    --wallet.name miner \
    --wallet.hotkey default \
    --payment-recipient 5GEKtrNMzRE3Xh7x7csKS1eGrZ7oSAzYYSQgxhZv3QUdVr9a \
    --validator-api http://154.54.100.65:8000
```

**What happens:**
1. Pays 0.1 TAO on-chain
2. Uploads code to validator
3. Gets evaluated in 2-3 minutes
4. TPS appears on leaderboard

### Check Results

Visit the validator's dashboard or use API:
```bash
curl <validator_url>/leaderboard
curl <validator_url>/api/submissions/<your_id>
```

---

## 🛠️ For Validators

### Setup

```bash
# 1. Install
git clone https://github.com/one-covenant/templar-tournament.git
cd templar-tournament
uv sync

# 2. Download model and test data
uv run python scripts/setup_validator.py

# 3. Configure environment
cat > .env << 'EOF'
# Wallet
TOURNAMENT_WALLET_NAME=validator
TOURNAMENT_WALLET_HOTKEY=validator_hotkey
TOURNAMENT_SUBTENSOR_NETWORK=finney

# R2 Storage
TOURNAMENT_R2_ACCOUNT_ID=your_account_id
TOURNAMENT_R2_BUCKET_NAME=tournament-submissions
TOURNAMENT_R2_ACCESS_KEY_ID=your_access_key
TOURNAMENT_R2_SECRET_ACCESS_KEY=your_secret_key
EOF

# 4. Build Docker sandbox
cd src/tournament/sandbox
docker build -t tournament-sandbox:latest .
cd ../../..

# 5. Register on blockchain
btcli subnet register --netuid 2 --wallet.name validator --wallet.hotkey validator_hotkey
btcli stake add --amount 1000 --wallet.name validator --wallet.hotkey validator_hotkey
```

### Run Validator

**Terminal 1 - API Server:**
```bash
uv run python -m api.app
# Dashboard: http://your-ip:8000/
```

**Terminal 2 - Validator:**
```bash
uv run python -m neurons.validator \
    --wallet.name validator \
    --wallet.hotkey validator_hotkey
```

**Announce to miners:**
- API URL: `http://your-ip:8000`
- Payment address: `<your_validator_hotkey_address>`

---

## 📋 Required Code Format

Your `train.py` must have an `inner_steps` function:

```python
from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor  # (batch, seq_len-1, vocab_size)
    total_tokens: int           # Must be 40,960
    final_loss: float

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Your optimized training loop."""
    
    for step in range(num_steps):
        batch = next(data_iterator)  # Shape: (8, 1024)
        
        # Your optimization code here
        # ...
        
        total_tokens += batch.numel()
    
    return InnerStepsResult(
        final_logits=logits,
        total_tokens=total_tokens,
        final_loss=loss
    )
```

**Must:**
- ✅ Process exactly 40,960 tokens
- ✅ Output logits within 10% of reference
- ✅ Complete under 10 minutes
- ✅ Use only allowed imports

**Forbidden:**
- ❌ Network access
- ❌ Filesystem access
- ❌ Subprocess calls
- ❌ Model architecture changes

---

## 🏆 How Scoring Works

```python
# Evaluated 3 times with different seeds
Run 1: seed=12345 → 4,302 TPS
Run 2: seed=67890 → 4,289 TPS
Run 3: seed=54321 → 4,315 TPS

Final Score: 4,302 TPS (average)

# TPS calculation
TPS = 40,960 tokens / wall_time_seconds
```

**Winner gets 100% of emissions!**

Weights update every 10 minutes. Hold #1 → Earn TAO continuously.

---

## 📊 Dashboard Features

Visit validator's dashboard to see:
- 📈 Network stats (submissions, top TPS, active miners)
- ⚡ Validator status (live evaluations)
- 📊 TPS history chart
- 🏆 Leaderboard with rankings
- 💻 Code viewer (all submissions public)
- 🔍 Detailed evaluation results

---

## 🔒 Security Model

**Transparent but Secure:**
- ✅ Code is publicly viewable (learning & verification)
- ✅ Miners submit via API (not direct R2 access)
- ❌ Miners can't access R2 credentials
- ❌ Miners can't delete others' submissions

Same security model as Ridges.ai.

---

## 📖 Configuration

Edit `hparams/hparams.json`:
```json
{
  "netuid": 2,
  "benchmark_model_name": "Qwen/Qwen2.5-3B",
  "benchmark_batch_size": 8,
  "benchmark_sequence_length": 1024,
  "eval_steps": 5,
  "eval_timeout": 600,
  "num_evals_per_submission": 3,
  "sandbox": {
    "memory_limit": "16g",
    "cpu_count": 4,
    "gpu_count": 1
  }
}
```

---

## 🐛 Common Issues

**"Payment verification failed"**
- Ensure hotkey has TAO balance (not just coldkey)

**"Code validation failed"**
- Run `uv run python -m tournament.test_local train.py` first

**"Verification failed"**
- Use BF16: `torch.autocast('cuda', dtype=torch.bfloat16)`
- Process exactly 40,960 tokens
- Don't modify model architecture

---

## 📄 License

MIT

---

**Ready to compete? Optimize your code and claim #1! ⚡**
