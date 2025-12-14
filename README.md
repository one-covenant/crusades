# Templar Tournament

A Bittensor subnet (NetUID 3) where miners compete to write the **fastest training code**. Winner-takes-all: the miner with the highest verified TPS receives 100% of subnet emissions.

---

## What is This?

A speed competition for PyTorch training code. You optimize a training loop to run as fast as possible while producing correct outputs.

### The Challenge:
- **Given**: A model + training data
- **Task**: Run 100 training steps as fast as possible
- **Metric**: Tokens Per Second (TPS)
- **Winner**: Highest TPS gets 100% of emissions

---

## How It Works

### 1. You Write Optimized Training Code

Create a `train.py` with this function:

```python
from collections.abc import Iterator
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor  # Output from last forward pass
    total_tokens: int           # Total tokens processed
    final_loss: float           # Loss from last step

def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """Execute num_steps training steps. Return results."""
    
    total_tokens = 0
    
    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device)
        
        # Your optimized training loop here
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()
    
    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
```

### 2. Submit Your Code

```bash
# Pay 0.1 TAO and submit
uv run python -m neurons.miner train.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --payment-recipient <validator_address>
```

### 3. Get Evaluated

Validators will:
1. Run your code in an isolated Docker sandbox
2. Measure execution time (external timing)
3. Verify outputs match reference (within 1% tolerance)
4. Calculate your TPS = tokens / time

### 4. Win Rewards

If you have the highest TPS:
- You receive 100% of subnet emissions
- Updates every 10 minutes

---

## Verification Rules

Your code MUST produce outputs that match the reference implementation:

### ✅ Pass Criteria:
1. **Token count**: Exact match (process all data)
2. **Output vectors**: Within 1% aggregate difference

### ❌ Fail Criteria:
- Skipping batches (wrong token count)
- Producing different outputs (> 1% difference)
- Timeout (> 10 minutes)
- Crashes or errors

---

## What You Can Optimize

### ✅ Allowed Optimizations:
- `torch.compile()` - Kernel fusion
- Flash Attention - Faster transformer operations
- Custom CUDA kernels
- Data prefetching (pinned memory, non-blocking transfers)
- Fused optimizers
- Memory-efficient operations

### ❌ Cannot Change:
- Model architecture
- Loss function (must use cross_entropy)
- Input/output format
- Number of training steps (must complete all)
- Label preparation (next-token prediction)

---

## Submission Cost

**Cost**: 0.1 TAO (100,000,000 RAO) per submission

**Why**: Prevents spam submissions. This ensures:
- Miners test locally before submitting
- Validators only evaluate serious attempts
- GPU resources used efficiently

**For local testing**: Use `--skip-payment` flag

---

## Example Optimizations

### Basic (Slow):
```python
def inner_steps(model, data_iterator, optimizer, num_steps, device):
    for step in range(num_steps):
        batch = next(data_iterator).to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                               batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return result

# TPS: ~50,000
```

### Optimized (Fast):
```python
def inner_steps(model, data_iterator, optimizer, num_steps, device):
    # Optimization 1: Compile model
    model = torch.compile(model, mode="reduce-overhead")
    
    # Optimization 2: Prefetch first batch
    current_batch = next(data_iterator).to(device, non_blocking=True)
    
    for step in range(num_steps):
        batch = current_batch
        
        # Optimization 3: Prefetch next while computing
        if step < num_steps - 1:
            current_batch = next(data_iterator).to(device, non_blocking=True)
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                   batch[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # Slightly faster
    
    return result

# TPS: ~150,000 (3x faster!)
```

---

## Configuration

### Benchmark Settings (`hparams/hparams.json`):
- **Model**: 150M parameters
- **Sequence Length**: 1024 tokens
- **Batch Size**: 8
- **Training Steps**: 100 per evaluation
- **Timeout**: 600 seconds (10 minutes)
- **Evaluations**: 3 per submission (averaged)

### Verification Tolerance:
- **Output Vector Tolerance**: 1% aggregate difference
- **Token Count**: Exact match required

### Hardware:
- **GPUs**: 1x A100 (or equivalent)
- **RAM**: 16GB limit
- **CPUs**: 4 cores

---

## API Endpoints

Query the leaderboard and submission status:

```bash
# Health check
curl http://validator-url:8000/health

# Current leaderboard
curl http://validator-url:8000/leaderboard

# Your submission status
curl http://validator-url:8000/submissions/<submission_id>

# Your evaluation results
curl http://validator-url:8000/submissions/<submission_id>/evaluations
```

---

## Local Testing

Test your code before submitting:

```bash
# 1. Clone repo
git clone https://github.com/your-org/templar-tournament.git
cd templar-tournament

# 2. Install dependencies
uv sync

# 3. Run unit tests
uv run pytest tests/

# 4. Test your submission (skip payment for local testing)
uv run python -m neurons.miner your_train.py \
    --wallet.name test \
    --wallet.hotkey test \
    --skip-payment
```

---

## For Validators

### Running a Validator:

```bash
uv run python -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --burn-hotkey <fallback_address>
```

### Requirements:
- Registered on subnet NetUID 3
- Docker installed with GPU support
- NVIDIA GPU (A100 recommended)
- Sufficient stake for weight setting

---

## Project Structure

```
templar-tournament/
├── neurons/
│   ├── validator.py        # Validator main loop
│   └── miner.py            # Miner submission CLI
├── src/tournament/
│   ├── verification/       # Output verification logic
│   ├── sandbox/            # Docker isolation
│   ├── payment/            # Anti-spam payment system
│   ├── storage/            # Database (submissions, evaluations)
│   └── chain/              # Weight setting (winner-takes-all)
├── benchmark/
│   ├── model/              # Training model
│   ├── data/               # Training data
│   └── reference_inner_steps.py  # Reference implementation
├── tests/                  # 82 unit tests
├── hparams/
│   └── hparams.json        # Configuration
└── api/                    # REST API (leaderboard)
```

---

## Security

Your code runs in an isolated Docker container with:
- ✅ No network access
- ✅ Read-only filesystem
- ✅ Resource limits (16GB RAM, 4 CPUs, 256 processes)
- ✅ 10-minute timeout
- ✅ Non-root user
- ✅ Automatic cleanup

You **cannot**:
- Access validator's files
- Install malware
- Consume unlimited resources
- Interfere with other submissions

---

## FAQ

### Q: How much does it cost to submit?
**A**: 0.1 TAO per submission (~$20-50 depending on TAO price)

### Q: How many times should I submit?
**A**: Test locally first. Most miners submit 5-10 times to find optimal code.

### Q: What if my submission fails verification?
**A**: You'll get a detailed error message explaining what went wrong. No refund on failed submissions.

### Q: How is the winner determined?
**A**: Highest average TPS across 3 evaluations. Winner-takes-all.

### Q: Can I use custom CUDA kernels?
**A**: Yes, as long as outputs match reference within 1% tolerance.

### Q: What model/data is used?
**A**: 150M parameter language model with tokenized training data (details TBD).

### Q: How often are weights set?
**A**: Every 10 minutes, based on current leaderboard.

---

## Support

- **Repository**: https://github.com/your-org/templar-tournament
- **Issues**: https://github.com/your-org/templar-tournament/issues
- **Discord**: [Link TBD]

---

## License

MIT
