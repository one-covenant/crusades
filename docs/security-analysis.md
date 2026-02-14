# Crusades Security & Validation Analysis

## Architecture Overview

This is a **Bittensor subnet** where miners compete to write the fastest `inner_steps()` training function. The metric is **MFU (Model FLOPs Utilization)** — how efficiently you use the GPU for transformer training.

### Flow

1. **Miner** hosts `train.py` at a URL (Gist, raw GitHub, etc.)
2. **Miner** commits that URL to the blockchain via `set_reveal_commitment()` — **timelock encrypted** via drand so nobody can see the URL until `reveal_blocks` (100 blocks ≈ 20 min) pass
3. **Validator** reads revealed commitments, downloads the code, runs it inside a sandboxed Docker container (or Basilica remote GPU)
4. Validator compares miner's results against a **reference implementation** run with the same data/model/seed
5. Winner-takes-all: top MFU score gets `5%` of emissions (95% burned to `burn_uid`)

---

## The Validation Pipeline

### Layer 1: Commitment & URL Security

**SSRF Protection** (`src/crusades/chain/commitments.py`):

- URL is resolved to IP and checked against private/internal ranges:
  - `127.0.0.0/8` (loopback)
  - `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (private)
  - `169.254.0.0/16` (link-local / cloud metadata)
  - IPv6 equivalents (`::1/128`, `fc00::/7`, `fe80::/10`)
- Redirect targets are also validated via custom `SSRFSafeRedirectHandler`
- Max file size: 500KB
- Rejects HTML pages and JSON file listings
- Must contain `def inner_steps`

### Layer 2: AST Static Analysis

Core sandbox gatekeeper in `environments/templar/env.py` — `_validate_code_structure()` / `_scan_for_dangerous_patterns()`.

#### Forbidden Imports (~35 modules)

```
ctypes, _ctypes, gc, subprocess, sys, os, pathlib, io, socket, http,
urllib, requests, shutil, tempfile, signal, threading, multiprocessing,
inspect, ast, dis, code, codeop, compileall, pickle, shelve, marshal,
builtins, _builtins, operator, types, codecs, base64, pdb, pprint,
env, __main__, miner_train, logging
```

Also blocks `importlib`, `torch.utils.cpp_extension`.

The `env` module is blocked to prevent miners from importing the validator's evaluation environment and tampering with timer references. `logging` is blocked because `logging.Logger.manager.loggerDict` holds references to all loggers — traversable to reach the env module's logger and back to the env module itself. `__main__` and `miner_train` prevent self-import tricks.

#### Forbidden Builtins (~15)

```
setattr, getattr, delattr, vars, dir, globals, locals, type, memoryview,
open, chr, ord, breakpoint, input, classmethod, staticmethod, property
```

#### Forbidden AST Patterns

| Pattern | Reason |
|---------|--------|
| `exec()`, `eval()`, `compile()` | Arbitrary code execution (except `torch.compile`) |
| `__import__()`, `importlib` | Dynamic import bypass |
| `__setattr__`, `__delattr__` | Attribute manipulation |
| `__class__` assignment | Type confusion attacks |
| `__dict__`, `__globals__`, `__code__`, `__func__`, `__self__` | Object introspection |
| `__subclasses__`, `__bases__`, `__mro__` | Class hierarchy traversal |
| Frame introspection (`f_globals`, `f_builtins`, `f_locals`, `f_back`, `tb_frame`) | Stack frame access |
| `time.perf_counter`, `time.monotonic` | Timer tampering |
| `cudnn.deterministic`, `cudnn.benchmark` (store) | Backend setting manipulation |
| `enable_flash_sdp`, `enable_mem_efficient_sdp`, `enable_math_sdp` | Attention backend manipulation |
| `gc.get_objects`, `gc.get_referrers`, `gc.get_referents` | Finding env module in memory |
| `.optimizer` access (except `self.optimizer`) | Unwrapping gradient capture |
| `__name__` reassignment | Module identity confusion |
| `__slots__` modification | Object layout tampering |
| `torch.utils.cpp_extension` | Arbitrary C/C++ compilation |
| `torch._C` access | Low-level C++ bindings escape hatch |
| `torch._dynamo.config.*` writes | Masking compilation errors (e.g. `suppress_errors`) |
| `torch._inductor.config.*` writes | Inductor backend manipulation |

#### String-Level Scanning

Checks `ast.Constant` string values AND `bytes([...]).decode()` / `b"...".decode()` patterns for forbidden strings:

```
__setattr__, __delattr__, __class__, perf_counter, _perf_counter,
captured_gradients, _opt_impl, _grad_snapshot_gpu, step_count,
GradientCapturingOptimizer, __import__, importlib, sys.modules,
setattr, getattr, delattr, __builtins__, __dict__, __globals__,
__code__, __func__, __self__, globals, locals, _cuda_synchronize,
_monotonic, __traceback__, tb_frame, f_globals, f_builtins, f_locals,
f_back, co_consts, co_names, __getattribute__, builtins,
operator.attrgetter, operator.methodcaller, attrgetter, methodcaller,
_CACHE, initial_state, _hidden_modules, _sensitive_keys
```

This catches string-based evasion like `getattr(x, "__" + "dict__")` and byte-encoded forbidden strings.

#### Attribute-Name Scanning

In addition to string constants, the scanner checks every `ast.Attribute` node's `.attr` value against the forbidden strings list. This catches patterns like `_e._perf_counter` where `_perf_counter` appears as an attribute name, not a string literal.

### Layer 3: Docker Sandbox

```
docker run --rm \
  --network none \              # NO network access
  --cap-drop ALL \              # Drop all Linux capabilities
  --security-opt no-new-privileges \
  --read-only \                 # Read-only filesystem
  --pids-limit 1024 \           # Limit process spawning
  --tmpfs /tmp:rw,exec,nosuid,size=4g \         # Writable /tmp (torch.compile)
  --tmpfs /home/appuser/.triton:rw,exec,size=2g # Triton cache
  --memory 80g --shm-size 32g \
  -v miner_train.py:/app/scripts/miner_train.py:ro  # Read-only mount
```

Key properties:
- `--network none`: No internet access inside container
- `--read-only`: Filesystem is read-only (except tmpfs)
- `--cap-drop ALL`: No Linux capabilities
- `--no-new-privileges`: Can't escalate
- `--pids-limit 1024`: Can't fork-bomb
- Miner code is mounted **read-only**
- Model/data are **pre-cached** in the Docker image (no downloads needed)
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set in image

### Layer 4: Runtime Verification (The Anti-Cheat)

The validator runs **two passes** with identical model state, data, and seed:

#### Pass 1 — Reference Run

Standard training loop with the validator's own code (`_run_reference()`). Captures gradients on the final step before `optimizer.step()`.

#### Pass 2 — Miner's Code

The optimizer passed to the miner is a `GradientCapturingOptimizer` — a **transparent wrapper** that:

1. Counts `step()` calls
2. On the **final step**, snapshots all gradients on GPU before the real `optimizer.step()` runs
3. Blocks access to private attributes (`_opt_impl`, `_grad_snapshot_gpu`, etc.)
4. Is **read-only after init** — `__setattr__` raises `AttributeError`
5. Uses `__slots__` to prevent attribute injection
6. Custom `__getattribute__` that only exposes `_PUBLIC_ATTRS`

#### Verification Checks (in order)

| # | Check | Threshold | What It Catches |
|---|-------|-----------|----------------|
| 1 | Token count == expected | Exact match | Skipping batches, changing batch size |
| 2 | Loss valid & close to reference | `\|diff\| < 0.3` | NaN injection, wrong loss function, skipping backward |
| 3 | Gradient relative error | `\|g - g_ref\| / \|g_ref\| < 6%` | Modified backward pass, frozen layers, gradient manipulation |
| 4 | Final weight relative error | `\|w - w_ref\| / \|w_ref\| < 0.8%` | Wrong optimizer, skipped steps, modified optimizer state |
| 5 | All layers have gradients | 100% coverage | Layer freezing |
| 6 | Parameter elements changed | ≥80% | No-op training |
| 7 | Sequence length matches expected | Exact match | Sequence truncation (cheap speedup) |
| 8 | Logits shape is 3D | `(batch, seq, vocab)` | Returning dummy data |
| 9 | Backend settings unchanged | Exact match | Disabling flash attention, changing precision |
| 10 | Timer integrity (3-way) | All pairs within 10% | Timer function replacement (see below) |
| 11 | Timer function identity | `id()` unchanged | Direct timer reference replacement |
| 12 | Optimizer wrapper type unchanged | `type(opt) is GradientCapturingOptimizer` | Replacing the wrapper |
| 13 | MFU sanity cap | ≤75% | Physically impossible scores from any timing bypass |

#### Timing Integrity (Three-Way Cross-Check)

Timing uses **three independent sources**, compared pairwise:

1. **`time.perf_counter`** — saved reference at module load time
2. **`time.monotonic`** — saved reference at module load time
3. **CUDA events** — recorded on the GPU command stream, immune to Python-level patches

```python
_perf_counter = time.perf_counter
_monotonic = time.monotonic
_perf_counter_id = id(_perf_counter)  # Identity check
_monotonic_id = id(_monotonic)
```

After miner code returns:
1. **Identity check**: `id(_perf_counter)` and `id(_monotonic)` must match the saved values — detects direct reference replacement
2. **Three-way divergence check**: if any pair of `(perf_counter, monotonic, cuda_events)` diverges by >10%, the submission fails with `timer_tampering`
3. **CUDA event time is canonical**: used as the authoritative wall time since it cannot be manipulated from Python

The `env` module is hidden from `sys.modules` during miner code execution (both warmup and timed eval), and timer references are restored after each evaluation via `_reset_torch_state()`.

#### Module Isolation

The `env` module is hidden from `sys.modules` at **two points**:

1. During `exec_module` (miner import time) — prevents top-level `import env`
2. During `inner_steps()` execution (both warmup and timed eval) — prevents runtime `import env`

```python
_hidden = {}
for key in list(sys.modules):
    if key == "env" or key == __name__ or key.endswith(".env"):
        _hidden[key] = sys.modules.pop(key)
try:
    miner_result = miner_module.inner_steps(...)
finally:
    sys.modules.update(_hidden)
```

Since `sys` is a forbidden import and `env` is also a forbidden import, miners cannot access `sys.modules` to find the env module.

---

## Known Attacks

### Timer Manipulation via `env` Module Import (Discovered & Fixed)

**Severity**: Critical — achieved 100% MFU (physically impossible)

**How it worked**:

1. Miner's `inner_steps()` called a setup function that did `import env as _e`
2. Grabbed the real timer references: `_rp = _e._perf_counter`, `_rm = _e._monotonic`
3. Replaced them with wrappers that report 50% of actual elapsed time
4. Both timers compressed equally, so the perf-vs-monotonic integrity check passed
5. Set `_tc["phase"] = 1` at the end of `inner_steps()` to trigger compression on the next timer read

**Why it evaded each defense**:

| Defense | Why It Failed |
|---------|---------------|
| Forbidden modules list | `env` was not in the list |
| AST scan for `time.perf_counter` | Attacker used `_e._perf_counter` (different object path) |
| AST scan for `time.monotonic` | Attacker used `_e._monotonic` (different object path) |
| Forbidden strings scan | Only checked `ast.Constant` string literals, not `ast.Attribute.attr` names |
| Timer integrity (perf vs mono) | Both timers compressed by same factor — ratio preserved |
| Module hiding | `env` was restored to `sys.modules` before `inner_steps()` ran |

**Fixes applied**:

1. Added `env`, `__main__`, `miner_train`, `logging` to forbidden modules
2. Added attribute-name scanning (`ast.Attribute.attr`) against forbidden strings
3. `env` module now hidden from `sys.modules` during `inner_steps()` execution (not just import time)
4. Added CUDA event-based timing as third untamperable timer source
5. Added timer function identity check (`id()` comparison)
6. Added MFU sanity cap (75%) as safety net

---

## Potential Bypass Vectors

### 1. `torch` Internals as Attack Surface — MITIGATED

`torch` is fully allowed. Previously concerning vectors:
- `torch._C` — **now blocked** by AST scanner
- `torch._dynamo.config` / `torch._inductor.config` writes — **now blocked** by AST scanner
- `torch.ops` — custom operator registry (low risk, legitimate use)
- `torch.Tensor.storage()`, `.data_ptr()` — raw memory access

**Mitigation**: `ctypes` and `memoryview` are blocked, `torch._C` is now blocked. `torch.ops` remains allowed for legitimate operator registration.

### 2. No Code Hash in Commitment — REAL RISK

The commitment contains only the URL, not a hash of the code. After committing, the miner can **change the code at the URL** before the validator downloads it.

**Mitigation**: 
- First-committer-wins logic prevents URL reuse
- Rate limiting (`min_blocks_between_commits: 300`)
- Code is downloaded once and evaluated — can't change mid-evaluation
- **TODO**: Consider adding a code hash to the commitment for integrity verification

### 3. Gradient/Weight Tolerance is Exploitable — SPECULATIVE

- Gradient check allows **6% relative error**
- Weight check allows **0.8% relative error**

These tolerances exist because BF16 arithmetic is inherently noisy. An attacker could:
- Do correct computation with micro-optimizations that cause gradients to drift within tolerance
- Skip a tiny fraction of computation (e.g., approximate attention) and still pass

**Note**: This is partially the intended use case — miners optimize for speed while staying within bounds.

### 4. `torch.compile` with Custom Backends — LOW RISK

`torch.compile` is explicitly allowed. `/tmp` has `exec` permission for compiled kernels. A miner could use custom Triton kernels. This is **intended and legitimate** optimization.

### 5. AST Bypass via Non-Literal String Construction — PARTIALLY MITIGATED

The string scan checks `ast.Constant` values, `bytes().decode()`, and now `ast.Attribute.attr` names. Runtime string construction via list joins could still evade it:

```python
name = "".join(["s", "e", "t", "a", "t", "t", "r"])
```

**Mitigation**: Since `getattr`, `setattr`, `exec`, `eval` are all blocked as callable builtins, constructing the string alone isn't useful without a way to call it dynamically.

**TODO**: Consider scanning `str.join()` calls on lists of single-character constants.

### 6. `__getattr__` Override on User Classes — LOW RISK

Defining `__getattr__` on a miner-created class is not explicitly blocked (only accessing `__getattribute__` attribute is). A miner could intercept attribute access on their own objects.

**Mitigation**: Unlikely to be useful for bypassing the optimizer wrapper since it uses `object.__getattribute__` directly.

### 7. Race Condition in Threshold Update — LOW RISK

The immediate threshold update saves winner identity FIRST, then updates threshold. If the process crashes between these operations, the threshold won't be bumped but the winner is recorded.

**Mitigation**: The weight setter has fallback logic that detects this state and recovers.

### 8. `GradientCapturingOptimizer.__getattr__` Fallback — LOW RISK

The fallback `__getattr__` forwards non-private attributes to the real optimizer. If `AdamW` has any public method that could modify state affecting gradient capture, it would be accessible.

**Mitigation**: The `param_groups` setter is forwarded (miners could change learning rate), but this would cause weight verification to fail.

### 9. Pre-cached Data is Architecture-Specific — NOT A VULNERABILITY

The model architecture (`Qwen/Qwen2.5-3B`) and dataset (`fineweb`) are fixed and known. Miners can pre-compute optimized kernels offline. This is **legitimate optimization** and the whole point of the competition.

### 10. `logging` Module Traversal — MITIGATED

`logging.Logger.manager.loggerDict` holds references to all loggers. A miner could traverse from the logging module to the env module's logger and back to the env module itself.

**Mitigation**: `logging` is now a forbidden import.

---

## Security Posture Summary

### Strong

- Docker sandbox (`--network none`, read-only, dropped capabilities)
- AST scanner — comprehensive, blocks most escape vectors including attribute-name scanning
- Three-way timer integrity check (perf_counter + monotonic + CUDA events)
- Timer function identity verification (`id()` comparison)
- MFU sanity cap (75%) — safety net against any timing bypass
- Gradient + weight verification — strong mathematical proof-of-work
- `GradientCapturingOptimizer` — `__slots__`, read-only after init, private attr blocking
- SSRF protection with redirect validation
- String-level + attribute-level scanning catches obfuscation attempts
- Module isolation during both import and execution phases
- `torch._C` and dynamo/inductor config writes blocked
- `env`, `logging`, `__main__`, `miner_train` all forbidden imports

### Moderate Concerns

- No code hash in commitment (URL-only) — miner can swap code post-commit
- 6% gradient tolerance + 0.8% weight tolerance provides room for approximate computation
- `torch.ops` remains allowed (legitimate use but potential for custom operator abuse)
- `exec` permission on tmpfs (required for torch.compile) is a necessary evil

### Low Concerns

- Race conditions in threshold updates (has fallback logic)
- `__getattr__` on user classes (can't reach protected objects)
- Optimizer fallback forwarding (weight check catches abuse)

---

## Recommendations

### Completed

1. ~~**Block `torch._C` access**~~: Done — `_C` attribute access on `torch` is now blocked in AST scanner.
2. ~~**Scan attribute names**~~: Done — `ast.Attribute.attr` values are now checked against forbidden strings.
3. ~~**Block `env` module import**~~: Done — `env`, `logging`, `__main__`, `miner_train` added to forbidden modules.
4. ~~**CUDA event timing**~~: Done — three-way cross-check with untamperable GPU-level timing.
5. ~~**Timer identity verification**~~: Done — `id()` comparison detects reference replacement.
6. ~~**MFU sanity cap**~~: Done — physically impossible scores (>75%) are rejected.

### Open

1. **Add code hash to commitment**: Include `sha256(code)` alongside the URL in the commitment JSON. Validator verifies hash matches downloaded code before evaluation.

2. **Scan for `str.join()` obfuscation**: Detect patterns like `"".join([...single chars...])` that construct forbidden strings at runtime.

3. **Consider tightening gradient tolerance**: As the competition matures and legitimate optimizations plateau, reducing the 6% threshold would narrow the window for approximate computation cheats.

4. **Add `torch.ops` to monitored attributes**: While legitimate use exists, it could be used to register custom operators that bypass Python-level restrictions.

5. **Log and audit `torch.compile` usage**: Track whether miners use `torch.compile` and what backends they target, to detect unusual compilation patterns.

6. **Add seccomp profiles**: Further restrict syscalls inside Docker beyond `--cap-drop ALL` for defense-in-depth.
