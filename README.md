# Templar Tournament
Compete to write the fastest PyTorch training code. Winner-takes-all: Highest TPS → 100% of subnet emissions.

## For Miners


### Step 1: Setup
```bash
# Clone and install
git clone https://github.com/one-covenant/templar-tournament.git
cd templar-tournament
uv sync

# Download official 7B model and dataset (~20GB, 30-60 mins)
uv run python scripts/setup_benchmark.py
```

This downloads:
- **Model**: Qwen2.5-7B (~15GB, 7B parameters, publicly accessible)
- **Data**: FineWeb dataset (100k samples, ~80MB tokenized)


### Step 2: Optimize the Code
Edit `train.py` (included in the repo) and optimize it. The file includes a basic implementation you can improve.


### Step 3: Test Your Code
**Quick baseline test** (see current performance):
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True uv run python train.py 2>&1
```

**Validate before submitting** (check structure):
```bash
uv run python -m tournament.test_local train.py
```
    ✅ Checks code syntax (no errors)
    ✅ Verifies inner_steps function exists
    ✅ Verifies InnerStepsResult class exists
    ✅ Checks for forbidden imports (os, socket, etc.)
    ❌ Does NOT run the code
    ❌ Does NOT measure TPS
**Iterate and optimize until you're happy!**


### Step 4: Register on Subnet
```bash
btcli subnet register --netuid 3 --wallet.name mywallet --wallet.hotkey myhotkey
```

### Step 5: Submit (Costs 0.1 TAO)
```bash
uv run python -m neurons.miner train.py \
    --wallet.name mywallet \
    --wallet.hotkey myhotkey \
    --payment-recipient <validator_address> \
    --validator-api http://validator-url:8000
```

### Step 6: Check Results
```bash
curl http://validator-url:8000/api/submissions/<submission_id>
```

Response:
```json
{
    "submission_id": "abc-123",
    "status": "finished",
    "final_score": 18543.2,
    "miner_uid": 42
}
```

### Step 7: Win Emissions
If your TPS is highest → You get 100% of subnet emissions!

Check earnings:
```bash
btcli wallet overview --netuid 3 --wallet.name mywallet
```

---

## The Competition

### What You're Optimizing
- **Task**: Run 5 training steps on 7B model
- **Goal**: Maximum tokens per second (TPS)
- **Model**: Qwen2.5-7B (7 billion parameters)
- **Data**: FineWeb dataset (100,000 samples × 1024 tokens)
- **Format**: Batch size 8, sequence length 1024
- **Per Eval**: 5 steps × 8 batch × 1024 seq = 40,960 tokens

### How You Win
- **Metric**: TPS = total_tokens / wall_time
- **Winner**: Highest average TPS across 3 evaluations
- **Reward**: 100% of subnet emissions
- **Updates**: Every 10 minutes


## Verification (How We Check Correctness)

Your code runs in a Docker sandbox with a **random seed**. Validator compares your outputs against a hidden reference:

### **3 Verification Checks:**

**1. Token Count (EXACT match)**
```
Expected: 819,200 tokens
Your output: must be exactly 819,200
Fail if: different by even 1 token
```
This ensures you processed all batches correctly.

**2. Output Vectors (1% tolerance)**
```
Expected: reference_logits from last forward pass  
Your output: final_logits from your last forward pass
Tolerance: 1% aggregate difference
Fail if: aggregate_diff > 1%
```
This ensures your training actually matches the reference.

**3. Timeout (10 minutes max)**
```
Fail if: execution takes > 600 seconds
```

### **Pass All 3 → Get Your TPS Score**

If verification passes:
- Your TPS = total_tokens / wall_time
- TPS saved to leaderboard
- Highest TPS wins 100% emissions

If verification fails:
- No TPS score
- You see error message
- No refund (test locally first!)

---

## Rules

### ✅ You Can Change
- How you run the training loop
- Any PyTorch optimizations
- Custom CUDA kernels
- Memory management
- Data loading strategy

### ❌ You Cannot Change
- Model (must use official 7B model)
- Dataset (must use official FineWeb data)
- Loss function (must be cross_entropy)
- Number of steps (must complete all 100)
- Input/output format
- Label preparation (next-token prediction)

---

## Security

### What You Can Access
- ✅ Official 7B model (from HuggingFace)
- ✅ Official dataset (from HuggingFace)
- ✅ Reference implementation (for matching outputs)
- ✅ Local testing (unlimited, free)

### What You Cannot Access
- ❌ Validator's storage credentials
- ❌ Other miners' code
- ❌ Internal evaluation details
- ❌ Exact validation seeds

### How Evaluation Works
1. You submit code to validator API (not direct storage)
2. Validator stores your code privately
3. Validator runs your code in isolated Docker sandbox
4. Validator compares outputs vs reference (with random seed)
5. Validator measures TPS externally
6. You see final TPS score only

---

## Costs

- **Registration**: ~0.01-0.1 TAO (one-time)
- **Per Submission**: 0.1 TAO
- **Local Testing**: FREE (unlimited)

**Tip**: Test thoroughly locally before submitting to save TAO!

---

## For Validators

### Requirements
- Registered on subnet NetUID 3
- Docker with GPU support  
- NVIDIA A100 GPU (or equivalent)
- Private R2/S3 bucket
- Sufficient stake for weight setting

### Setup
```bash
# Configure private credentials
cat > .env << 'EOF'
TOURNAMENT_SUBTENSOR_NETWORK=finney
TOURNAMENT_R2_ACCOUNT_ID=your_account
TOURNAMENT_R2_ACCESS_KEY_ID=your_key
TOURNAMENT_R2_SECRET_ACCESS_KEY=your_secret
TOURNAMENT_R2_BUCKET_NAME=private-submissions
EOF

# Download same model/data as miners
uv run python scripts/setup_benchmark.py

# Start API
uv run python -m api.app &

# Run validator
uv run python -m neurons.validator \
    --wallet.name validator \
    --wallet.hotkey validator_hotkey \
    --burn-hotkey <fallback_address>
```

---

## API Endpoints

```bash
# Submit code (via miner CLI)
POST /api/submissions

# Check status
GET /api/submissions/{id}

# View leaderboard
GET /leaderboard

# Health check
GET /health
```

---

## FAQ

**Q: Can I test before paying?**  
A: Yes! Local testing is free and unlimited.

**Q: How many times should I submit?**  
A: Test locally first. Most miners submit 3-5 times.

**Q: Can I see other miners' code?**  
A: No. All submissions are private.

**Q: What if my submission fails?**  
A: No refund. Always test locally first.

**Q: How do I know what TPS to beat?**  
A: Check leaderboard: `curl http://validator:8000/leaderboard`

**Q: Can I use a different model?**  
A: No. Everyone must use the official 7B model for fairness.

---

## Support

- **GitHub**: https://github.com/one-covenant/templar-tournament
- **Issues**: https://github.com/one-covenant/templar-tournament/issues

---

## License

MIT
