# Arbos — Crusades MFU Optimization Agent

An autonomous agent loop that continuously improves `train.py` for maximum MFU (Model FLOPs Utilization) on the Templar Crusades Bittensor subnet.

Inspired by [unconst/Arbos](https://github.com/unconst/Arbos).

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Agent Loop                                            │
│   ┌───────────────────────────────────────────────┐     │
│   │ 1. Read current best train.py                 │     │
│   │ 2. LLM suggests improvement ───── LLM API     │     │
│   │ 3. Validate generated code                    │     │
│   │ 4. Test on GPUs ──── Basilica or local Docker │     │
│   │ 5. MFU improved?                              │     │
│   │    YES → Save, log, notify                    │     │
│   │    NO  → Revert, try different approach       │     │
│   │ 6. Loop                                       │     │
│   └───────────────────────────────────────────────┘     │
│                                                         │
│   The agent NEVER auto-submits.                         │
│   You review the results and submit manually.           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- `httpx` (already in Crusades dependencies)
- One LLM API key (see below)
- **Basilica mode**: `basilica-sdk` + Basilica API token
- **Local mode**: Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) + A100 GPUs

## Setup

### 1. Copy the environment file

```bash
cp arbos/.env.example arbos/.env
```

### 2. Set your API keys

Edit `arbos/.env` with your keys:

```bash
# Pick ONE LLM provider and set its key:

# Option A: Chutes (cheapest, ~$5-15/day for 24/7 running)
LLM_PROVIDER=chutes
CHUTES_API_KEY=your-chutes-key-here

# Option B: OpenRouter (access to Claude, GPT, etc.)
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=your-openrouter-key-here

# Option C: Anthropic (direct Claude access, best quality)
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your-anthropic-key-here

# Basilica (only needed for cloud mode, not needed for --local)
BASILICA_API_TOKEN=your-basilica-token-here
```

### 3. Build the Docker image (for local mode)

If using `--local`, build the evaluation image first:

```bash
docker build --network=host --no-cache \
  -f environments/templar/Dockerfile \
  -t templar-eval:latest .
```

### 4. Run

```bash
# === Local Docker (recommended — uses your own GPUs, no cloud costs) ===

# 4 GPUs (default from hparams.json docker.num_gpus)
uv run arbos/agent.py --train-py local_test/train_fsdp.py --local

# Specific GPU devices
uv run arbos/agent.py --train-py local_test/train_fsdp.py --local --gpu-devices 4,5,6,7

# Custom Docker image
uv run arbos/agent.py --train-py local_test/train_fsdp.py --local --docker-image my-eval:v2

# === Basilica cloud (default when --local is not set) ===

# Miner (default — pays 0.05 TAO submission fee)
uv run arbos/agent.py --train-py local_test/train.py

# Validator team (burn_uid 3 — no fee)
uv run arbos/agent.py --train-py local_test/train.py --mode validator

# === Other options ===

# Dry run (test LLM generation only, no GPU costs)
uv run arbos/agent.py --train-py local_test/train.py --dry-run

# Limit to N steps
uv run arbos/agent.py --train-py local_test/train.py --max-steps 20
```

When you see **"READY FOR SUBMISSION"** in the logs:
1. Check the saved file in `arbos/best_submissions/`
2. Host it at a URL (GitHub Gist, etc.)
3. Submit:
```bash
uv run -m neurons.miner submit <url> --wallet.name <name> --wallet.hotkey <hotkey>
```

## Output

### Logs

All activity is logged to both console and `arbos/logs/`:

```
[2026-03-13 14:22:01] ============================================================
[2026-03-13 14:22:01]   STEP 23
[2026-03-13 14:22:01] ============================================================
[2026-03-13 14:22:01] Strategy: Switch to SHARD_GRAD_OP with reduce-overhead compile
[2026-03-13 14:22:06] Sending for evaluation (local-docker)...
[2026-03-13 14:25:43] Result: MFU=57.40% | TPS=12841.3 | Tokens=327680 | Time=215.4s
[2026-03-13 14:25:43]
[2026-03-13 14:25:43] ************************************************************
[2026-03-13 14:25:43]   NEW BEST MFU: 57.40% (+1.20%)
[2026-03-13 14:25:43]   Improvement #4 at step 23
[2026-03-13 14:25:43]   Change: Switch to SHARD_GRAD_OP with reduce-overhead compile
[2026-03-13 14:25:43] ************************************************************
[2026-03-13 14:25:43]   Saved to: arbos/best_submissions/train_v23_57.4.py
[2026-03-13 14:25:43]
[2026-03-13 14:25:43]   READY FOR SUBMISSION
[2026-03-13 14:25:43]   File: arbos/best_submissions/train_v23_57.4.py
[2026-03-13 14:25:43]   Submit (0.05 TAO fee):
[2026-03-13 14:25:43]     1. Host the file at a URL (e.g. GitHub Gist)
[2026-03-13 14:25:43]     2. uv run -m neurons.miner submit <url> ...
```

### Saved Files

```
arbos/
├── best_submissions/
│   ├── train_latest.py          # Always the current best
│   ├── train_v8_57.4.py         # Each improvement saved with MFU
│   ├── train_v15_58.1.py
│   └── ...
├── runs/
│   ├── step_0001_candidate.py   # Every candidate (pass or fail)
│   ├── step_0002_candidate.py
│   └── ...
├── logs/
│   └── agent_20260313_142200.log
└── state.json                    # Persistent state (survives restarts)
```

### State Persistence

The agent saves state to `arbos/state.json` after every step. If you stop and restart, it resumes from where it left off — same best MFU, same history, same step count.

To start fresh, delete `arbos/state.json`.

## Architecture

```
arbos/
├── agent.py              Main loop, CLI, state management, logging
├── llm_client.py         Multi-provider LLM client (Chutes/OpenRouter/Anthropic)
├── tester.py             Evaluation backends: LocalDockerTester + BasilicaTester
├── security_scanner.py   AST-based pre-validation (catches violations before GPU eval)
├── PROMPT.md             LLM system prompt (editable — see Customization below)
├── .env.example          Configuration template
└── README.md             This file
```

### Evaluation Backends

| Backend | Flag | GPUs | Cost | Latency |
|---------|------|------|------|---------|
| **Local Docker** | `--local` | Your own A100s | Free (electricity) | ~3-5 min/eval |
| **Basilica** | *(default)* | Cloud A100s | Per-evaluation GPU time | ~5-8 min/eval (includes deploy) |

Both backends run the exact same `env.py` Actor — results are identical.

## CLI Reference

```
usage: agent.py --train-py FILE [options]

Required:
  --train-py FILE          Path to initial train.py to optimize

Evaluation backend (pick one):
  --local                  Use local Docker (your GPUs, no cloud costs)
  (default)                Use Basilica cloud

Local Docker options (only with --local):
  --num-gpus N             Number of GPUs (default: docker.num_gpus from hparams)
  --gpu-devices IDS        Specific GPU IDs, e.g. "4,5" or "0,1,2,3"
  --docker-image IMAGE     Custom Docker image (default: from hparams)

General:
  --mode {miner,validator} Running mode (default: miner)
  --max-steps N            Stop after N steps (default: run forever)
  --dry-run                Skip evaluation, test LLM generation only
  --env-file PATH          Path to .env file (default: arbos/.env)
```

## LLM Provider Comparison

| Provider | Key | Models | Cost | Best For |
|----------|-----|--------|------|----------|
| Chutes | `CHUTES_API_KEY` | Kimi K2.5, GLM-5, MiniMax | ~$0.14-0.60/M tokens | 24/7 running, budget |
| OpenRouter | `OPENROUTER_API_KEY` | Claude, GPT, Gemini, etc. | Varies by model | Flexibility |
| Anthropic | `ANTHROPIC_API_KEY` | Claude Sonnet/Opus | ~$3-15/M tokens | Best code quality |

## FAQ

**Q: Does the agent auto-submit?**
No. It saves improved versions and logs a notification. You decide when to submit.

**Q: Can I resume after stopping?**
Yes. State is saved to `arbos/state.json`. Just run the same command again.

**Q: Should I use `--local` or Basilica?**
Use `--local` if you have A100 GPUs on your machine — it's free and faster (no deployment spin-up). Use Basilica if you don't have local GPUs.

**Q: What if evaluation fails?**
The agent retries with exponential backoff. Consecutive failures trigger delays up to 2 minutes.

**Q: How much does Basilica cost?**
Each evaluation creates a fresh 4x A100 80GB SXM deployment, runs the test, then deletes it. Cost is per-evaluation based on GPU time.

**Q: What's the difference between miner and validator mode?**
Only the submission instructions in the logs. Miner mode says "0.05 TAO fee"; validator mode says "burn_uid 3, no fee". The optimization loop is identical.

**Q: How do I use specific GPUs for local mode?**
Use `--gpu-devices 4,5` to select GPUs 4 and 5, or `--num-gpus 4` to use the first 4. The `--gpu-devices` flag maps to Docker's `--gpus "device=4,5"` syntax.

**Q: Can I customize the LLM prompt?**
Yes. Edit `arbos/PROMPT.md` — the agent reloads it on every LLM call. This is the main lever for guiding what optimizations the LLM tries. Add domain knowledge, warn about pitfalls, or steer toward specific strategies.
