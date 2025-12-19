# Templar Tournament

Compete to write the fastest PyTorch training code. Winner-takes-all: Highest TPS → 100% of subnet emissions.

---

## For Miners - Quick Start

### Step 1: Setup
```bash
# Clone and install
git clone https://github.com/one-covenant/templar-tournament.git
cd templar-tournament
uv sync

# Download official 7B model and training data (~20GB, 30-60 mins)
uv run python scripts/setup_miner.py
```

**Downloads:**
- Model: Qwen2.5-7B (~15GB, 7.6B parameters)
- Training data: 100k samples (seed=42, deterministic)

---

### Step 2: Test Baseline
```bash
uv run python train.py
```

**Output:**
```
📊 Results (Averaged over 3 evaluations):
   Average TPS: 4,302
```

This is your unoptimized baseline (~4,300 TPS).

---

### Step 3: Optimize Your Code

Edit `train.py` to improve TPS:

```python
# Easy win: Add torch.compile() at start of inner_steps
compiled_model = torch.compile(model, mode="reduce-overhead")

# Use compiled_model instead of model
outputs = compiled_model(input_ids)
```

Test after each change:
```bash
uv run python train.py  # See new TPS
```

---

### Step 4: Validate Before Submitting
```bash
uv run python -m tournament.test_local train.py
```

Checks:
- ✅ Code syntax valid
- ✅ inner_steps function exists  
- ✅ No forbidden imports

---

### Step 5: Register on Subnet

```bash
# Register on Templar subnet (netuid 3)
btcli subnet register \
    --netuid 3 \
    --wallet.name mywallet \
    --wallet.hotkey myhotkey
```

**Cost:** ~0.01 TAO (one-time)

---

### Step 6: Submit Your Code (Costs 0.1 TAO)

```bash
# Submit to validator
uv run python -m neurons.miner train.py \
    --wallet.name mywallet \
    --wallet.hotkey myhotkey \
    --payment-recipient <validator_hotkey_address> \
    --validator-api <validator_api_url>
```

**What happens:**
1. ✅ Pays 0.1 TAO on-chain to validator
2. ✅ Sends code content to validator API  
3. ✅ Validator stores in private R2 bucket
4. ✅ Validator evaluates in Docker sandbox
5. ✅ Your TPS score published to leaderboard

---

### Step 7: Check Results

```bash
# Check your submission
curl <validator_url>/api/submissions/<submission_id>

# Check leaderboard
curl <validator_url>/leaderboard
```

**Response:**
```json
{
    "submission_id": "abc-123",
    "status": "finished", 
    "final_score": 18543.2,
    "miner_uid": 42
}
```

---

### Step 8: Win Emissions!

If you're #1 → You get 100% of subnet emissions!

```bash
# Check your emissions
btcli wallet overview --netuid 3 --wallet.name mywallet
```

**Winner updates every 10 minutes.**

---

## The Competition

- **Task**: Run 5 training steps as fast as possible
- **Model**: Qwen2.5-7B (everyone uses same)
- **Data**: Miners test on train.pt, Validators evaluate on test.pt (hidden)
- **Metric**: TPS = 40,960 tokens / wall_time
- **Winner**: Highest average TPS across 3 evaluations

---

## Verification

Validator runs your code with **random seed** and checks:

1. **Token count**: Exactly 40,960 tokens
2. **Output vectors**: Within 1% of reference
3. **Timeout**: Under 10 minutes

**Pass all 3 → Get TPS score**

---

## For Validators

### Requirements
- Registered on subnet (netuid 3) with stake
- Docker with GPU support
- Private R2 bucket
- Public API endpoint

### Setup

**Step 1: Download Model + Test Data**
```bash
# Downloads model + test.pt (private evaluation data)
uv run python scripts/setup_validator.py
```

**Step 2: Configure R2 Credentials**
```bash
cat > .env << 'EOF'
TOURNAMENT_SUBTENSOR_NETWORK=finney
TOURNAMENT_R2_ACCOUNT_ID=your_r2_account_id
TOURNAMENT_R2_ACCESS_KEY_ID=your_access_key  
TOURNAMENT_R2_SECRET_ACCESS_KEY=your_secret_key
TOURNAMENT_R2_BUCKET_NAME=validator-submissions
EOF
```

**Step 3: Register & Stake**
```bash
# Register
btcli subnet register --netuid 3 --wallet.name validator --wallet.hotkey validator_hotkey

# Stake (required to set weights)
btcli stake add --netuid 3 --wallet.name validator --wallet.hotkey validator_hotkey --amount 100
```

**Step 4: Build Docker Sandbox**
```bash
cd src/tournament/sandbox
docker build -t tournament-sandbox:latest .
```

**Step 5: Start Services**

Terminal 1 - API:
```bash
uv run python -m api.app
# Miners submit to: http://your-server-ip:8000
```

Terminal 2 - Validator:
```bash
uv run python -m neurons.validator \
    --wallet.name validator \
    --wallet.hotkey validator_hotkey
```

**Step 6: Announce API**

Share with miners:
- API URL: `http://your-server.com:8000`
- Payment address: `<your_validator_hotkey_address>`

---

## Data Separation

- **Miners**: train.pt (samples 0-99,999, seed=42)
- **Validators**: test.pt (samples 100,000-199,999, seed=42)
- **Zero overlap** - Prevents overfitting
- **Deterministic** - Everyone gets same samples

---

## License

MIT
