# Validation System Documentation

This document explains how the validation system works in the Templar Tournament for Bittensor subnet 3.

## Overview

The tournament implements a competitive validation system where miners submit training code and compete based on **Tokens Per Second (TPS)** performance. The system ensures correctness through output verification and prevents cheating through multiple security mechanisms.

## What Gets Validated

Miners submit training code (`train.py`) containing an `inner_steps` function. The system validates:

- **Code syntax and structure** - Must be valid Python
- **Required function** - Must contain `inner_steps(model, data_iterator, optimizer, num_steps, device)`
- **Security** - No forbidden imports (`os`, `subprocess`, `socket`, `pickle`, `ctypes`, etc.)
- **Output correctness** - Logits, loss, and token count must match reference
- **Performance** - Tokens per second throughput

## Validation Flow

### Stage 1: Code Validation

**File:** `src/tournament/pipeline/validator.py`

The `CodeValidator` class performs static analysis:

1. **AST Parsing** - Checks for syntax errors
2. **Function Check** - Verifies `inner_steps` function exists with correct signature
3. **Import Check** - Blocks dangerous imports that could compromise security

```python
FORBIDDEN_IMPORTS = [
    'os', 'subprocess', 'socket', 'urllib', 'requests',
    'ctypes', 'pickle', 'multiprocessing', 'threading', ...
]
```

### Stage 2: Payment Verification

**File:** `src/tournament/payment/verifier.py`

Anti-spam measure requiring payment before evaluation:

- Default cost: 0.1 TAO (100,000,000 RAO)
- Verified via blockchain block hash + extrinsic index
- Confirms correct recipient, amount, and sender

### Stage 3: Benchmarking & Verification

**File:** `src/tournament/verification/verifier.py`

This is the core validation process orchestrated by `SandboxVerifier.verify_and_benchmark()`:

#### Step 1: Run Reference Implementation

```
1. Load model checkpoint from benchmark_model_path
2. Create data iterator from benchmark_data_path
3. Set deterministic mode with random seed
4. Execute reference inner_steps function
5. Capture: final_logits, total_tokens, final_loss
6. Save reference outputs to temp directory
```

#### Step 2: Run Miner Code in Sandbox

```
1. Copy miner's code to sandbox temp directory
2. Launch Docker container (network_mode="none")
3. Load model with SAME seed as reference
4. Execute miner's inner_steps function
5. Measure wall_time and capture outputs
6. Return SandboxResult (tokens, logits, loss, timing)
```

#### Step 3: Verify Outputs Match Reference

| Check | Requirement | Purpose |
|-------|-------------|---------|
| Token count | **Exact match** | Prevents batch skipping or early termination |
| Logits | Aggregate difference < 1% | Allows bf16 precision variance while detecting wrong outputs |
| Loss | Informational only | Reference loss used for comparison |

#### Step 4: Calculate Performance

```
TPS = total_tokens / wall_time_seconds
```

### Stage 4: Multi-Validator Scoring

**File:** `src/tournament/storage/database.py`

Each submission requires multiple independent evaluations:

1. At least `num_evals_per_submission` (default: 3) validators evaluate
2. Each validator runs independently with their own sandbox
3. Results stored with `evaluator_hotkey` for traceability
4. Final score = **average TPS** across successful evaluations

This prevents single-validator manipulation through Byzantine-resilient consensus.

### Stage 5: Weight Setting

**File:** `src/tournament/chain/weights.py`

Every `set_weights_interval_seconds` (default: 600 seconds):

1. Query `get_top_submission()` - highest `final_score`
2. If valid winner exists and is registered:
   - Set `weight = 1.0` to winner's UID
   - **Winner-takes-all**: 100% of emissions go to top miner
3. If no valid winner:
   - Set `weight = 1.0` to burn hotkey (emissions burned)

## Submission Lifecycle

```
PENDING
   │
   ▼
VALIDATING ──(code fails)──► FAILED_VALIDATION
   │
   ▼
EVALUATING ──(verification fails)──► FAILED_VALIDATION
   │
   ▼
FINISHED (with final_score)
```

## Security Mechanisms

### 1. Random Seed Verification

Each evaluation uses a **randomly generated seed** at evaluation time:

- Prevents pre-computation attacks
- Same seed used for both reference AND miner code
- Ensures exact reproducibility for comparison

```python
seed = random.randint(0, 2**32 - 1)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### 2. Deterministic Mode

Forces reproducible execution:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 3. Docker Sandbox Isolation

Miner code runs in heavily restricted containers:

| Restriction | Value |
|-------------|-------|
| Network | `none` (completely isolated) |
| Memory limit | 16GB |
| PID limit | 256 |
| Filesystem | Read-only except output directory |

### 4. Exact Token Count Verification

Token count must match **exactly** (zero tolerance):

- Detected via `batch.numel()` across all steps
- Catches batch skipping, early termination, or modified data loading

### 5. Output Vector Verification

Logits compared using aggregate difference:

```python
aggregate_diff = mean_diff / mean_abs_value
if aggregate_diff > output_vector_tolerance:  # 0.01 = 1%
    raise LogitsMismatchError(...)
```

The 1% tolerance accounts for bf16 floating-point precision differences while still detecting:
- Modified architectures
- Wrong precision settings
- Incorrect forward pass implementations

### 6. Code Validation (Pre-Execution)

AST-based analysis prevents:
- Syntax errors reaching execution
- Missing required functions
- Dangerous imports

### 7. Timeout Protection

- `eval_timeout = 600` seconds (10 minutes)
- Container killed if exceeded
- Prevents infinite loops and resource hogging

### 8. Multiple Independent Evaluators

- At least 3 validators evaluate each submission
- Each uses independent sandbox instance
- Results averaged to prevent single-validator manipulation
- Evaluator hotkey stored for audit trail

### 9. Payment Requirement

- Blockchain-verified payment required
- Prevents submission flooding/spam
- Amount configurable via `submission_cost_rao`

## Key Files Reference

| File | Purpose |
|------|---------|
| `neurons/validator.py` | Main validator node orchestrator |
| `src/tournament/pipeline/validator.py` | Code syntax validation |
| `src/tournament/verification/verifier.py` | Output verification & benchmarking |
| `src/tournament/verification/reference.py` | Reference implementation executor |
| `src/tournament/verification/config.py` | Verification configuration |
| `src/tournament/sandbox/manager.py` | Docker sandbox orchestration |
| `src/tournament/sandbox/runner.py` | In-container execution script |
| `src/tournament/chain/weights.py` | Weight setting logic |
| `src/tournament/payment/verifier.py` | Payment verification |
| `src/tournament/storage/database.py` | Submission & evaluation storage |
| `src/tournament/storage/models.py` | Database models |
| `src/tournament/config.py` | Tournament configuration |

## Configuration

### Tournament Parameters (`src/tournament/config.py`)

```python
class HParams:
    num_evals_per_submission: int = 3      # Evaluations required
    eval_steps: int = 100                   # Training steps per eval
    eval_timeout: int = 600                 # Max execution time (seconds)
    benchmark_sequence_length: int = 1024   # Sequence length
    benchmark_batch_size: int = 8           # Batch size
    set_weights_interval_seconds: int = 600 # Weight update frequency
    submission_cost_rao: int = 100_000_000  # Anti-spam payment (0.1 TAO)
```

### Verification Parameters (`src/tournament/verification/config.py`)

```python
class VerificationConfig:
    output_vector_tolerance: float = 0.01   # 1% aggregate difference allowed
    verify_model_state: bool = False        # State verification (disabled)
    deterministic_mode: bool = True         # Reproducibility enforcement
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     MINER SUBMISSION                            │
│               (train.py with inner_steps function)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   CODE VALIDATION    │
                  │   - Syntax check     │
                  │   - Function check   │
                  │   - Import check     │
                  └──────────┬───────────┘
                             │ PASS
                             ▼
             ┌───────────────────────────────┐
             │    PAYMENT VERIFICATION       │
             │    - Blockchain check         │
             │    - Amount verification      │
             └───────────────┬───────────────┘
                             │ PASS
                             ▼
    ┌────────────────────────────────────────────────────────────┐
    │       VERIFICATION (3x Independent Validators)             │
    │                                                            │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 1. RUN REFERENCE                                     │  │
    │  │    - Load model + data                               │  │
    │  │    - Set random seed                                 │  │
    │  │    - Execute reference inner_steps                   │  │
    │  │    - Capture outputs (logits, tokens, loss)          │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                                                            │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 2. RUN MINER (Docker Sandbox)                        │  │
    │  │    - Isolated container (no network)                 │  │
    │  │    - Same seed as reference                          │  │
    │  │    - Execute miner's inner_steps                     │  │
    │  │    - Measure wall time                               │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                                                            │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 3. VERIFY OUTPUTS                                    │  │
    │  │    - Token count: EXACT match required               │  │
    │  │    - Logits: <1% aggregate difference                │  │
    │  │    - Calculate TPS = tokens / wall_time              │  │
    │  └──────────────────────────────────────────────────────┘  │
    └────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │           SCORING             │
                 │   final_score = avg TPS       │
                 │   (min 3 successful evals)    │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │   WEIGHT SETTING (10-min)     │
                 │   - Get top submission        │
                 │   - Winner gets weight = 1.0  │
                 │   - 100% emissions to winner  │
                 └───────────────────────────────┘
```

## Error Types

The system provides detailed error feedback to miners:

| Error | Cause |
|-------|-------|
| `TokenCountMismatchError` | Token count doesn't match reference |
| `LogitsMismatchError` | Output vectors differ by >1% |
| `LossMismatchError` | Loss value differs significantly |
| `SandboxExecutionError` | Runtime crash, timeout, OOM |
| `MissingFunctionError` | `inner_steps` function not found |
| `InvalidReturnTypeError` | Wrong return type from `inner_steps` |

## Key Design Principles

1. **Reproducibility** - Random seeds and deterministic mode ensure exact comparison
2. **Security** - Sandbox isolation prevents malicious code execution
3. **Byzantine Resilience** - Multiple validators prevent manipulation
4. **Transparency** - TPS scoring is simple and verifiable
5. **Winner-Takes-All** - Strong incentive for optimization
6. **Anti-Spam** - Payment requirement prevents flooding
