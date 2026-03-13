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
│   │ 4. Test on Basilica ─────────── Remote A100s  │     │
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
- `basilica-sdk` (already in Crusades dependencies)
- `httpx` (already in Crusades dependencies)
- One LLM API key (see below)
- One Basilica API token

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

# Basilica (required for GPU testing)
BASILICA_API_TOKEN=your-basilica-token-here
```

### 3. Provide your starting train.py

The agent needs a starting file to improve upon. Use any valid `train.py`:

```bash
# Use the reference FSDP implementation
python arbos/agent.py --train-py local_test/train_fsdp.py

# Or use your own
python arbos/agent.py --train-py /path/to/your/train.py
```

## Usage

### For Miners

```bash
# Start the optimization loop
python arbos/agent.py --train-py local_test/train_fsdp.py --mode miner

# When you see "READY FOR SUBMISSION":
# 1. Check the saved file in arbos/best_submissions/
# 2. Host it at a URL (GitHub Gist, etc.)
# 3. Submit: uv run -m neurons.miner submit <url> --wallet.name <name> --wallet.hotkey <hotkey>
```

### For Validator Team

```bash
# Same agent, different mode (no submission fee from burn_uid 3)
python arbos/agent.py --train-py local_test/train_fsdp.py --mode validator

# When you see "READY FOR SUBMISSION":
# Submit from burn_uid 3 wallet (no fee)
```

### Dry Run (No GPU Costs)

Test the LLM code generation without Basilica:

```bash
python arbos/agent.py --train-py local_test/train_fsdp.py --dry-run
```

Candidates are saved to `arbos/runs/` for manual review.

### Limit Steps

```bash
python arbos/agent.py --train-py local_test/train_fsdp.py --max-steps 20
```

## Output

### Logs

All activity is logged to both console and `arbos/logs/`:

```
[2026-03-13 14:22:01] ============================================================
[2026-03-13 14:22:01]   STEP 23
[2026-03-13 14:22:01] ============================================================
[2026-03-13 14:22:01] Strategy: Switch to SHARD_GRAD_OP with reduce-overhead compile
[2026-03-13 14:22:06] Sending to Basilica for evaluation...
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
├── agent.py           Main loop, CLI, state management, logging
├── llm_client.py      Multi-provider LLM client (Chutes/OpenRouter/Anthropic)
├── tester.py          Basilica deployment management and evaluation
├── .env.example       Configuration template
└── README.md          This file
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

**Q: What if Basilica is unavailable?**
The agent retries with exponential backoff. Consecutive failures trigger delays up to 2 minutes.

**Q: How much does Basilica cost?**
Each evaluation deploys 2x A100 80GB for the test duration. The deployment is reused across multiple tests within its TTL (2 hours by default).

**Q: What's the difference between miner and validator mode?**
Only the submission instructions in the logs. Miner mode says "0.05 TAO fee"; validator mode says "burn_uid 3, no fee". The optimization loop is identical.
