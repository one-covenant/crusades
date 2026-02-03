"""
Templar TPS Evaluation Environment (Validator-Owned)

URL-Based Architecture:
- This env.py is owned by the VALIDATOR, not the miner
- Miner commits a URL pointing to their train.py code
- Validator downloads code from URL and passes it to this env
- This ensures miners can't tamper with evaluation logic

Flow:
1. Validator reads miner commitment (contains code URL)
2. Validator downloads train.py from URL
3. Validator calls env.evaluate(code="...", ...)
4. This Actor runs benchmark and returns TPS
"""

import ast
import gc
import hashlib
import importlib.util
import logging
import os
import random
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

# Setup logging - only for this module, don't configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)

# Suppress noisy loggers from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Configuration from environment variables
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "1") == "1"
EVAL_SEQUENCE_LENGTH = int(os.getenv("EVAL_SEQUENCE_LENGTH", "1024"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/templar_eval"))

# =============================================================================
# SECURITY: Import Sandboxing
# =============================================================================
# Modules that miners are NOT allowed to import (security risk)
BLOCKED_MODULES = frozenset(
    [
        # System access
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        # Network access
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "asyncio",  # Could be used for network
        # File system
        "io",
        "tempfile",
        "glob",
        # Code execution
        "exec",
        "eval",
        "compile",
        "code",
        "codeop",
        # Process control
        "multiprocessing",
        "threading",
        "concurrent",
        # Dangerous builtins access
        "builtins",
        "__builtins__",
        # Pickle (code execution risk)
        "pickle",
        "cPickle",
        "dill",
        "cloudpickle",
    ]
)

# Modules that miners ARE allowed to import
ALLOWED_MODULES = frozenset(
    [
        # PyTorch and ML
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.optim",
        "torch.cuda",
        "torch.amp",
        "torch.autograd",
        "torch.distributed",
        "torch.utils",
        # Math/Science
        "math",
        "random",
        "numpy",
        # Data structures
        "collections",
        "dataclasses",
        "typing",
        "functools",
        "itertools",
        # Transformers (for model access)
        "transformers",
        # Time (for benchmarking, read-only)
        "time",
    ]
)

_original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _sandboxed_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Sandboxed import that blocks dangerous modules."""
    # Get the top-level module name
    top_module = name.split(".")[0]

    # Check if explicitly blocked
    if top_module in BLOCKED_MODULES or name in BLOCKED_MODULES:
        raise ImportError(
            f"Import of '{name}' is blocked for security reasons. "
            f"Allowed modules: torch, numpy, math, random, collections, dataclasses, typing, functools, itertools, transformers, time"
        )

    # Allow if in whitelist or is a submodule of allowed
    is_allowed = top_module in ALLOWED_MODULES or any(
        name.startswith(allowed + ".") for allowed in ALLOWED_MODULES
    )

    if not is_allowed:
        # Log warning but allow (some modules may be needed)
        logger.warning(f"Miner importing non-whitelisted module: {name}")

    return _original_import(name, globals, locals, fromlist, level)


def _enable_import_sandbox():
    """Enable the import sandbox for miner code execution."""
    import builtins

    builtins.__import__ = _sandboxed_import
    logger.info("Import sandbox ENABLED - blocked modules: os, sys, subprocess, socket, etc.")


def _disable_import_sandbox():
    """Disable the import sandbox and restore original import."""
    import builtins

    builtins.__import__ = _original_import
    logger.info("Import sandbox DISABLED")


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Captured gradient information for verification."""

    grad_norm: float  # L2 norm of all gradients
    grad_vector: torch.Tensor | None = None  # Flattened gradient vector for cosine sim
    layers_with_grad: int = 0  # Layers with non-zero gradients
    total_layers: int = 0  # Total trainable layers


# Global cache for model (data is NOT cached for validators)
# NOTE: initial_state stores full model weights on CPU for verification.
# For large models (e.g., 70B), ensure sufficient CPU RAM (~140GB for bf16).
# This is necessary for params_changed verification and model reset between evals.
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
}


def _load_miner_module(train_path: Path):
    """Dynamically load miner's train.py as a module with import sandboxing.

    Args:
        train_path: Path to train.py

    Returns:
        Loaded module with inner_steps function

    Security:
        - Enables import sandbox before loading (blocks os, sys, subprocess, etc.)
        - Sandbox remains active during module execution
    """
    spec = importlib.util.spec_from_file_location("miner_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["miner_train"] = module

    # Enable import sandbox before executing miner code
    _enable_import_sandbox()
    try:
        spec.loader.exec_module(module)
    finally:
        # Keep sandbox enabled - will be disabled after evaluation
        pass

    return module


def _reset_torch_state():
    """Reset torch global state after miner code execution.

    This prevents miners from leaving malicious state that affects
    subsequent evaluations or verification.
    """
    # Reset CUDA state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Reset deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Ensure gradients are enabled
    torch.set_grad_enabled(True)

    # Reset default dtype (miners might change this)
    torch.set_default_dtype(torch.float32)

    logger.debug("Torch state reset complete")


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace or local cache.

    SECURITY: Validators use unpredictable seeds so miners can't pre-compute
    which samples will be used for evaluation.

    When running with --network none, uses pre-cached dataset from Docker image.
    The validator seed is used to shuffle the cached data unpredictably.

    Args:
        dataset_name: HuggingFace dataset name
        model_name: Model name for tokenizer
        num_samples: Number of samples to load
        sequence_length: Sequence length
        split: Dataset split
        validator_seed: Seed string from validator (for unpredictable sampling)

    Returns:
        Tensor of shape [num_samples, sequence_length]
    """
    import json
    import random

    from transformers import AutoTokenizer

    # Determine seed: validators use unpredictable seed
    if validator_seed:
        seed_hash = hashlib.sha256(validator_seed.encode()).hexdigest()
        actual_seed = int(seed_hash[:8], 16)
        logger.info(f"Validator mode: seed={actual_seed} (from {validator_seed})")
    else:
        actual_seed = 42
        logger.info(f"Test mode: fixed seed={actual_seed}")

    logger.info(f"Loading dataset: {dataset_name} (samples={num_samples})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for cached dataset (enables --network none operation)
    cached_path = os.getenv("CACHED_DATASET_PATH", "/home/appuser/.cache/templar/dataset.json")

    if Path(cached_path).exists():
        # Load from cache and shuffle with validator seed
        logger.info(f"Using cached dataset: {cached_path}")
        with open(cached_path) as f:
            all_samples = json.load(f)

        # Shuffle with validator seed for unpredictability
        rng = random.Random(actual_seed)
        rng.shuffle(all_samples)

        tokens_list = []
        for text in all_samples[:num_samples]:
            encoded = tokenizer(
                text,
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens_list.append(encoded["input_ids"].squeeze(0))

        if not tokens_list:
            raise ValueError("No samples in cached dataset")

        data = torch.stack(tokens_list)
        logger.info(f"Loaded cached data: shape={data.shape}, seed={actual_seed}")
        return data

    # Fallback: Load from HuggingFace (requires network)
    logger.info("No cache found, loading from HuggingFace...")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=actual_seed, buffer_size=10000)

    tokens_list = []
    dataset_iter = iter(dataset)

    for _ in range(num_samples):
        try:
            sample = next(dataset_iter)
            text = sample.get("text", sample.get("content", ""))

            encoded = tokenizer(
                text,
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens_list.append(encoded["input_ids"].squeeze(0))

        except StopIteration:
            break

    if not tokens_list:
        raise ValueError(f"No samples loaded from {dataset_name}")

    data = torch.stack(tokens_list)
    logger.info(f"Loaded data: shape={data.shape}, seed={actual_seed}")
    return data


def _set_deterministic(seed: int) -> None:
    """Set deterministic mode for reproducibility."""
    if not DETERMINISTIC_MODE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_code_structure(code: str) -> tuple[bool, str | None]:
    """Validate that train.py has correct structure."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                return False, f"inner_steps has {len(args.args)} args, expected at least 5"
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps"

    return True, None


def _validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    """Validate that inner_steps returned correct type."""
    if isinstance(result, InnerStepsResult):
        return True, None, result

    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return (
            True,
            None,
            InnerStepsResult(
                final_logits=result.final_logits,
                total_tokens=result.total_tokens,
                final_loss=result.final_loss,
            ),
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_path: str, use_random_init: bool = False):
    """Load model from HuggingFace.

    Args:
        model_path: HuggingFace model name/path
        use_random_init: If True, initialize with random weights (anti-cheat)
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if use_random_init:
        # Random init - miners can't cheat by freezing pretrained layers
        logger.info(f"Loading model config from {model_path} with RANDOM initialization")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        logger.info(f"Loading pretrained model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _count_model_params(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def _calculate_mfu(
    total_tokens: int,
    wall_time: float,
    model_params: int,
    gpu_peak_tflops: float = 312.0,
) -> float:
    """Calculate Model FLOPs Utilization (MFU).

    MFU = actual_flops / theoretical_peak_flops

    For transformer training (forward + backward):
    - Forward: 2 * params * tokens (multiply-accumulate)
    - Backward: 4 * params * tokens (gradients are 2x forward)
    - Total: 6 * params * tokens

    Args:
        total_tokens: Total tokens processed
        wall_time: Wall clock time in seconds
        model_params: Number of model parameters
        gpu_peak_tflops: GPU theoretical peak TFLOPS (A100 80GB = 312 TFLOPS bfloat16)

    Returns:
        MFU as a percentage (0-100)
    """
    if wall_time <= 0:
        return 0.0

    # FLOPs for forward + backward pass
    flops_per_token = 6 * model_params
    total_flops = flops_per_token * total_tokens

    # Actual TFLOPS achieved
    actual_tflops = total_flops / wall_time / 1e12

    # MFU as percentage
    mfu = (actual_tflops / gpu_peak_tflops) * 100

    return min(mfu, 100.0)  # Cap at 100%


def _capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass.

    Returns:
        GradientInfo with norm and flattened gradient vector
    """
    grad_list = []
    total_norm_sq = 0.0
    layers_with_grad = 0
    layers_without_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            grad_flat = param.grad.detach().float().view(-1)
            grad_list.append(grad_flat)
            total_norm_sq += grad_flat.pow(2).sum().item()
            # Check if gradient is actually non-zero (not just allocated)
            if grad_flat.abs().sum().item() > 1e-10:
                layers_with_grad += 1
            else:
                layers_without_grad += 1
        else:
            layers_without_grad += 1

    grad_norm = total_norm_sq**0.5

    # Log layer gradient coverage
    total_layers = layers_with_grad + layers_without_grad
    if total_layers > 0:
        coverage = layers_with_grad / total_layers
        logger.info(f"Gradient coverage: {layers_with_grad}/{total_layers} layers ({coverage:.1%})")
        if layers_without_grad > 0:
            logger.warning(f"WARNING: {layers_without_grad} layers have zero/no gradients!")

    # Concatenate all gradients into single vector for cosine similarity
    if grad_list:
        grad_vector = torch.cat(grad_list)
    else:
        grad_vector = None

    return GradientInfo(
        grad_norm=grad_norm,
        grad_vector=grad_vector,
        layers_with_grad=layers_with_grad,
        total_layers=total_layers,
    )


def _verify_trainable_params(
    model: torch.nn.Module,
    min_trainable_ratio: float = 0.9,
) -> tuple[bool, str | None, dict]:
    """Check that minimum % of params are trainable (prevents layer freezing)."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

    details = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "min_required": min_trainable_ratio,
    }

    if trainable_ratio < min_trainable_ratio:
        error = (
            f"Insufficient trainable params: {trainable_ratio:.1%} "
            f"({trainable_params:,}/{total_params:,}) - minimum {min_trainable_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Trainable params check ({trainable_ratio:.1%} >= {min_trainable_ratio:.0%})"
    )
    return True, None, details


def _verify_params_changed(
    model: torch.nn.Module,
    initial_state: dict,
    min_changed_ratio: float = 0.5,
) -> tuple[bool, str | None, dict]:
    """Verify that minimum % of params actually changed during training."""
    total_params = 0
    changed_params = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            initial = initial_state[name].to(param.device)
            diff = (param.data - initial).abs().sum().item()
            num_params = param.numel()
            total_params += num_params
            if diff > 1e-6:  # Threshold for "changed"
                changed_params += num_params

    changed_ratio = changed_params / total_params if total_params > 0 else 0.0

    details = {
        "total_params": total_params,
        "changed_params": changed_params,
        "changed_ratio": changed_ratio,
        "min_required": min_changed_ratio,
    }

    if changed_ratio < min_changed_ratio:
        error = (
            f"Insufficient parameter updates: {changed_ratio:.1%} "
            f"({changed_params:,}/{total_params:,}) - minimum {min_changed_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Parameter changes check ({changed_ratio:.1%} >= {min_changed_ratio:.0%})"
    )
    return True, None, details


def _get_cached_model(model_path: str, use_random_init: bool = False):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    cached_random_init = _CACHE.get("use_random_init")

    # Cache hit only if path AND init mode match
    if cached is not None and cached_path == model_path and cached_random_init == use_random_init:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
        return cached

    model = _load_model(model_path, use_random_init=use_random_init)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
    _CACHE["use_random_init"] = use_random_init
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _create_data_iterator(
    data: torch.Tensor, batch_size: int, sequence_length: int
) -> Iterator[torch.Tensor]:
    """Create infinite data iterator."""
    if data.size(1) < sequence_length:
        raise ValueError(f"Data sequence length {data.size(1)} < required {sequence_length}")

    data = data[:, :sequence_length]
    num_samples = data.size(0)

    def _iter():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > num_samples:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    return _iter()


def _create_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create standard AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )


def _run_reference(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
    capture_final_gradients: bool = True,
) -> tuple[InnerStepsResult, GradientInfo | None]:
    """Run reference implementation for comparison.

    Returns:
        Tuple of (InnerStepsResult, GradientInfo) where GradientInfo
        contains the gradients from the final step (before optimizer.step)
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    final_gradients = None

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()

        # Capture gradients on final step (before optimizer.step clears them)
        if capture_final_gradients and step == num_steps - 1:
            final_gradients = _capture_gradients(model)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    result = InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
    return result, final_gradients


def _verify_gradients(
    reference_grad: GradientInfo | None,
    candidate_grad: GradientInfo | None,
    cosine_min: float = 0.8,
    norm_ratio_min: float = 0.5,
    norm_ratio_max: float = 2.0,
) -> tuple[bool, str | None, dict]:
    """Verify candidate gradients match reference using cosine similarity and norm ratio.

    This is more robust than logits comparison because:
    1. Gradients directly reflect what the model learned
    2. Cheaters who freeze layers will have different gradient patterns
    3. Cosine similarity catches direction changes
    4. Norm ratio catches magnitude cheating

    Args:
        reference_grad: Reference implementation gradients
        candidate_grad: Miner's implementation gradients
        cosine_min: Minimum cosine similarity required
        norm_ratio_min: Minimum ratio of candidate/reference gradient norm
        norm_ratio_max: Maximum ratio of candidate/reference gradient norm

    Returns:
        Tuple of (success, error_message, details)
    """
    details = {
        "cosine_min": cosine_min,
        "norm_ratio_min": norm_ratio_min,
        "norm_ratio_max": norm_ratio_max,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Gradient-based verification")
    logger.info("=" * 60)

    if reference_grad is None or candidate_grad is None:
        error = "Missing gradient information for verification"
        details["checks_failed"].append({"check": "gradient_availability", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    details["reference_grad_norm"] = reference_grad.grad_norm
    details["candidate_grad_norm"] = candidate_grad.grad_norm
    details["candidate_layers_with_grad"] = candidate_grad.layers_with_grad
    details["candidate_total_layers"] = candidate_grad.total_layers

    # Check 0: All layers must have gradients (catches "restore weights" attack)
    if candidate_grad.total_layers > 0:
        grad_coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
        details["gradient_coverage"] = grad_coverage
        logger.info(f"[CHECK 0/3] Gradient coverage: {grad_coverage:.1%}")
        logger.info(
            f"   Layers with gradients: {candidate_grad.layers_with_grad}/{candidate_grad.total_layers}"
        )

        if grad_coverage < 1.0:
            error = (
                f"Not all layers have gradients: {candidate_grad.layers_with_grad}/{candidate_grad.total_layers} "
                f"({grad_coverage:.1%}) - possible layer freezing detected"
            )
            details["checks_failed"].append({"check": "gradient_coverage", "error": error})
            details["error_code"] = "gradient_coverage_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_coverage")
        logger.info("[PASSED] All layers have gradients")

    # Check 1: Gradient norm ratio
    if reference_grad.grad_norm > 0:
        norm_ratio = candidate_grad.grad_norm / reference_grad.grad_norm
        details["norm_ratio"] = norm_ratio
        logger.info(f"[CHECK 1/3] Gradient norm ratio: {norm_ratio:.4f}")
        logger.info(f"   Reference norm: {reference_grad.grad_norm:.6f}")
        logger.info(f"   Candidate norm: {candidate_grad.grad_norm:.6f}")
        logger.info(f"   Allowed range: [{norm_ratio_min}, {norm_ratio_max}]")

        if norm_ratio < norm_ratio_min or norm_ratio > norm_ratio_max:
            error = (
                f"Gradient norm ratio {norm_ratio:.4f} outside allowed range "
                f"[{norm_ratio_min}, {norm_ratio_max}]"
            )
            details["checks_failed"].append({"check": "gradient_norm_ratio", "error": error})
            details["error_code"] = "gradient_norm_ratio_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_norm_ratio")
        logger.info("[PASSED] Gradient norm ratio in acceptable range")
    else:
        logger.warning("Reference gradient norm is zero, skipping norm ratio check")

    # Check 2: Cosine similarity
    ref_vec = reference_grad.grad_vector
    cand_vec = candidate_grad.grad_vector

    if ref_vec is not None and cand_vec is not None:
        # Ensure same device
        if ref_vec.device != cand_vec.device:
            cand_vec = cand_vec.to(ref_vec.device)

        # Ensure same size (might differ if model structure changed)
        if ref_vec.shape != cand_vec.shape:
            error = f"Gradient vector shape mismatch: ref={ref_vec.shape}, cand={cand_vec.shape}"
            details["checks_failed"].append({"check": "gradient_shape", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details

        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(ref_vec.unsqueeze(0), cand_vec.unsqueeze(0)).item()
        details["cosine_similarity"] = cosine_sim

        logger.info(f"[CHECK 2/3] Gradient cosine similarity: {cosine_sim:.4f}")
        logger.info(f"   Minimum required: {cosine_min}")

        if cosine_sim < cosine_min:
            error = f"Gradient cosine similarity {cosine_sim:.4f} below minimum {cosine_min}"
            details["checks_failed"].append({"check": "gradient_cosine", "error": error})
            details["error_code"] = "gradient_cosine_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_cosine_similarity")
        logger.info("[PASSED] Gradient cosine similarity acceptable")
    else:
        logger.warning("Gradient vectors unavailable, skipping cosine similarity check")

    logger.info("=" * 60)
    logger.info("VERIFICATION: GRADIENT CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
    expected_tokens: int,
    reference_grad: GradientInfo | None = None,
    candidate_grad: GradientInfo | None = None,
    max_loss_difference: float = 0.5,
    gradient_cosine_min: float = 0.8,
    gradient_norm_ratio_min: float = 0.5,
    gradient_norm_ratio_max: float = 2.0,
) -> tuple[bool, str | None, dict]:
    """Verify candidate outputs match reference.

    Verification checks:
    1. Token count matches expected
    2. Loss is valid and similar to reference (small difference)
    3. Gradient-based verification (replaces logits comparison)

    Args:
        reference: Reference implementation results
        candidate: Miner's implementation results
        expected_tokens: Expected token count
        reference_grad: Reference gradients for comparison
        candidate_grad: Candidate gradients for comparison
        max_loss_difference: Maximum allowed |candidate_loss - reference_loss|
        gradient_cosine_min: Minimum gradient cosine similarity
        gradient_norm_ratio_min: Min gradient norm ratio
        gradient_norm_ratio_max: Max gradient norm ratio

    Returns:
        Tuple of (success, error_message, verification_details)
    """
    details = {
        "expected_tokens": expected_tokens,
        "candidate_tokens": candidate.total_tokens,
        "candidate_loss": candidate.final_loss,
        "reference_loss": reference.final_loss if reference else None,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Starting output verification")
    logger.info("=" * 60)

    # 1. Verify token count matches expected
    logger.info(
        f"[CHECK 1/3] Token count: expected={expected_tokens}, got={candidate.total_tokens}"
    )
    if candidate.total_tokens != expected_tokens:
        error = f"Token count mismatch: expected {expected_tokens}, got {candidate.total_tokens}"
        details["checks_failed"].append({"check": "token_count", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("token_count")
    logger.info("[PASSED] Token count matches")

    # 2. Verify loss is reasonable and similar to reference
    logger.info(f"[CHECK 2/3] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        error = "Loss is NaN"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if abs(candidate.final_loss) > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Compare losses - check absolute difference
    if reference is not None and reference.final_loss > 0:
        loss_difference = abs(candidate.final_loss - reference.final_loss)
        details["loss_difference"] = loss_difference
        logger.info(
            f"   Reference loss: {reference.final_loss:.4f}, "
            f"Candidate loss: {candidate.final_loss:.4f}"
        )
        logger.info(
            f"   Loss difference: {loss_difference:.4f} (max allowed: {max_loss_difference})"
        )

        if loss_difference > max_loss_difference:
            error = (
                f"Loss difference too large: candidate={candidate.final_loss:.4f}, "
                f"reference={reference.final_loss:.4f}, diff={loss_difference:.4f} > {max_loss_difference}"
            )
            details["checks_failed"].append({"check": "loss_comparison", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details
    details["checks_passed"].append("loss_validity")
    logger.info("[PASSED] Loss is valid and similar to reference")

    # 3. Gradient-based verification (replaces logits comparison)
    logger.info("[CHECK 3/3] Gradient verification")
    grad_ok, grad_error, grad_details = _verify_gradients(
        reference_grad,
        candidate_grad,
        cosine_min=gradient_cosine_min,
        norm_ratio_min=gradient_norm_ratio_min,
        norm_ratio_max=gradient_norm_ratio_max,
    )
    details["gradient_verification"] = grad_details

    if not grad_ok:
        details["checks_failed"].append({"check": "gradient_verification", "error": grad_error})
        return False, grad_error, details
    details["checks_passed"].append("gradient_verification")

    logger.info("=" * 60)
    logger.info("VERIFICATION: ALL CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


class Actor:
    """Templar MFU Evaluation Actor for Affinetes (Validator-Owned).

    This Actor is owned by the validator, not the miner.
    Code is passed directly from the validator (downloaded from URL).
    Uses MFU (Model FLOPs Utilization) instead of TPS as the metric.
    """

    async def evaluate(
        self,
        task_id: int = 0,
        seed: str = "0:0:0",
        model_url: str = "",
        data_url: str = "",
        steps: int = 5,
        batch_size: int = 8,
        timeout: int = 600,
        sequence_length: int | None = None,
        data_samples: int = 10000,
        code: str = "",  # Miner's code passed directly
        # Verification settings
        max_loss_difference: float = 0.5,
        use_random_init: bool = True,
        min_trainable_params_ratio: float = 1.0,
        min_params_changed_ratio: float = 0.5,
        # Gradient verification (replaces logits)
        gradient_cosine_min: float = 0.8,
        gradient_norm_ratio_min: float = 0.5,
        gradient_norm_ratio_max: float = 2.0,
        # MFU calculation
        gpu_peak_tflops: float = 312.0,
        model_params_override: int | None = None,
    ) -> dict:
        """
        Run MFU evaluation on miner's code.

        Args:
            task_id: Evaluation run identifier
            seed: Deterministic seed string (format: "block:uid:run_idx")
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            timeout: Maximum seconds
            sequence_length: Sequence length
            data_samples: Number of data samples
            code: Miner's train.py code (passed directly from validator)
            max_loss_difference: Max allowed |candidate_loss - reference_loss|
            use_random_init: Use random weights (anti-cheat)
            min_trainable_params_ratio: Min % params that must be trainable
            min_params_changed_ratio: Min % params that must change
            gradient_cosine_min: Min gradient cosine similarity
            gradient_norm_ratio_min: Min gradient norm ratio
            gradient_norm_ratio_max: Max gradient norm ratio
            gpu_peak_tflops: GPU peak TFLOPS for MFU calculation
            model_params_override: Override model param count (None = auto-detect)

        Returns:
            Dict with: mfu, tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate code
        if not code:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "No code provided",
                "seed": seed,
            }

        # Validate model/data
        if not model_url or not data_url:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Missing model_url or data_url",
                "seed": seed,
            }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        seed_value = abs(hash(seed)) % (2**32)
        _set_deterministic(seed_value)

        # Validate code structure
        code_ok, code_error = _validate_code_structure(code)
        if not code_ok:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "seed": seed,
                "code": code,
            }

        # Write code to temp file for loading as module
        train_path = CACHE_DIR / "miner_train.py"
        try:
            train_path.write_text(code)
        except Exception as e:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"Failed to write train.py: {e}",
                "seed": seed,
            }

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Timeout before evaluation",
                "seed": seed,
            }

        try:
            # Load miner's module
            miner_module = _load_miner_module(train_path)

            if not hasattr(miner_module, "inner_steps"):
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "train.py missing inner_steps function",
                    "seed": seed,
                    "code": code,
                }

            # Load model and data (with random init if enabled for anti-cheat)
            model = _get_cached_model(model_url, use_random_init=use_random_init)
            model_params = model_params_override or _count_model_params(model)
            logger.info(f"Model loaded: {model_params:,} parameters, random_init={use_random_init}")

            data = _load_hf_dataset(
                dataset_name=data_url,
                model_name=model_url,
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Unpredictable sampling
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH

            # Run reference implementation (captures gradients for verification)
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)

            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            reference, reference_grad = _run_reference(
                model, data_iter_ref, optimizer_ref, steps, device, capture_final_gradients=True
            )

            # Reset model for miner's code
            model.load_state_dict(initial_state)

            # =================================================================
            # EARLY TERMINATION - Run randomized warmup steps to catch errors
            # =================================================================
            # This catches shape mismatches, missing attributes, etc. before
            # running the full evaluation (saves GPU time on broken code)
            # SECURITY: Randomize warmup steps to prevent miners from detecting
            # and behaving differently during warmup vs evaluation
            warmup_steps = random.randint(1, 3)
            data_iter_warmup = _create_data_iterator(data, batch_size, seq_len)
            optimizer_warmup = _create_optimizer(model)

            logger.info(f"Running {warmup_steps} warmup step(s) to check for basic errors...")
            try:
                warmup_result = miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_warmup,
                    optimizer=optimizer_warmup,
                    num_steps=warmup_steps,  # Randomized to prevent detection
                    device=device,
                )

                # Quick validation of warmup result
                if warmup_result is None:
                    return {
                        "task_id": task_id,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps returned None (warmup check)",
                        "seed": seed,
                        "code": code,
                    }

                # Check required attributes exist
                if not hasattr(warmup_result, "final_logits"):
                    return {
                        "task_id": task_id,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps result missing 'final_logits' attribute",
                        "seed": seed,
                        "code": code,
                    }

                # Check logits shape (should be 3D: batch, seq, vocab)
                logits = warmup_result.final_logits
                if logits is not None and len(logits.shape) != 3:
                    return {
                        "task_id": task_id,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": f"Logits shape mismatch: expected 3D (batch, seq, vocab), got {logits.shape}",
                        "seed": seed,
                        "code": code,
                    }

                logger.info("Warmup passed - proceeding with verification checks")

                # Check trainable params AFTER warmup (miner code may modify requires_grad)
                trainable_ok, trainable_error, trainable_details = _verify_trainable_params(
                    model, min_trainable_params_ratio
                )
                if not trainable_ok:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": trainable_error,
                        "error_code": "insufficient_trainable_params",
                        "seed": seed,
                        "code": code,
                        "diagnostics": {"trainable_params": trainable_details},
                    }

            except Exception as e:
                # Early termination - don't waste time on broken code
                error_msg = f"Early termination (warmup failed): {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": error_msg,
                    "seed": seed,
                    "code": code,
                }

            # Reset model state after warmup
            model.load_state_dict(initial_state)

            # =================================================================
            # FULL EVALUATION - Run actual timed evaluation with gradient capture
            # =================================================================
            data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
            optimizer_miner = _create_optimizer(model)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # We need to capture gradients on the final step
            # Run all but last step normally
            start = time.perf_counter()

            if steps > 1:
                # Run steps-1 steps without gradient capture
                miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_miner,
                    optimizer=optimizer_miner,
                    num_steps=steps - 1,
                    device=device,
                )

            # Run final step and capture gradients
            # We run the last step manually to capture gradients before optimizer.step()
            batch = next(data_iter_miner)
            batch = batch.to(device, dtype=torch.long)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                input_ids = batch[:, :-1]
                labels = batch[:, 1:]
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()

            # Capture candidate gradients BEFORE optimizer step
            candidate_grad = _capture_gradients(model)

            optimizer_miner.step()
            optimizer_miner.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            wall_time = time.perf_counter() - start

            # Create result from final step
            total_tokens = batch_size * seq_len * steps
            parsed = InnerStepsResult(
                final_logits=logits.detach().float(),
                total_tokens=total_tokens,
                final_loss=float(loss.item()),
            )

            # =================================================================
            # PARAMS CHANGED CHECK - Verify miner actually trained the model
            # =================================================================
            # SECURITY: This catches the "freeze layers then restore requires_grad" attack.
            # Even if a miner restores requires_grad=True after their inner_steps,
            # the frozen layers wouldn't have changed during training, so this check fails.
            params_ok, params_error, params_details = _verify_params_changed(
                model, initial_state, min_params_changed_ratio
            )
            if not params_ok:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": params_error,
                    "error_code": "insufficient_params_changed",
                    "seed": seed,
                    "code": code,
                    "diagnostics": {"params_changed": params_details},
                }

            # Calculate expected tokens
            expected_tokens = batch_size * seq_len * steps

            # Verify outputs using gradient-based verification
            verified, verify_error, verify_details = _verify_outputs(
                reference,
                parsed,
                expected_tokens,
                reference_grad=reference_grad,
                candidate_grad=candidate_grad,
                max_loss_difference=max_loss_difference,
                gradient_cosine_min=gradient_cosine_min,
                gradient_norm_ratio_min=gradient_norm_ratio_min,
                gradient_norm_ratio_max=gradient_norm_ratio_max,
            )

            # Calculate MFU and TPS
            total_tokens_int = int(parsed.total_tokens)
            tps = float(total_tokens_int) / max(wall_time, 1e-6)
            mfu = _calculate_mfu(total_tokens_int, wall_time, model_params, gpu_peak_tflops)

            # Diagnostics
            diagnostics = {
                "verification": verify_details,
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
                "expected_tokens": expected_tokens,
                "actual_tokens": parsed.total_tokens,
                "model_params": model_params,
                "gpu_peak_tflops": gpu_peak_tflops,
                "trainable_params": trainable_details,
                "params_changed": params_details,
            }

            # Extract error_code from verification details if present
            error_code = None
            if not verified and verify_details:
                # Check gradient_verification sub-details
                grad_details = verify_details.get("gradient_verification", {})
                error_code = grad_details.get("error_code")
                # Check other verification failures
                if not error_code:
                    for check in verify_details.get("checks_failed", []):
                        if check.get("check") == "token_count":
                            error_code = "token_count_mismatch"
                        elif check.get("check") == "loss_comparison":
                            error_code = "loss_mismatch"

            return {
                "task_id": task_id,
                "mfu": mfu if verified else 0.0,
                "tps": tps if verified else 0.0,
                "total_tokens": total_tokens_int if verified else 0,
                "wall_time_seconds": wall_time,
                "success": verified,
                "error": verify_error,
                "error_code": error_code,
                "seed": seed,
                "diagnostics": diagnostics,
                "code": code,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "seed": seed,
                "code": code,
            }

        finally:
            # SECURITY: Disable import sandbox
            _disable_import_sandbox()

            # SECURITY: Reset torch state to prevent cross-evaluation contamination
            _reset_torch_state()

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# FastAPI HTTP Server (for Basilica custom Docker deployment)
# =============================================================================

app = FastAPI(title="Templar MFU Evaluation", version="2.0.0")

# Global actor instance (reused for efficiency)
_actor: Actor | None = None


def get_actor() -> Actor:
    """Get or create singleton Actor instance."""
    global _actor
    if _actor is None:
        _actor = Actor()
    return _actor


class EvaluateRequest(BaseModel):
    """Request body for /evaluate endpoint."""

    task_id: int = 0
    seed: str = "0:0:0"
    model_url: str
    data_url: str
    steps: int = 5
    batch_size: int = 8
    timeout: int = 600
    sequence_length: int | None = None
    data_samples: int = 10000
    code: str  # Miner's train.py code
    # Verification settings
    max_loss_difference: float = 0.5
    use_random_init: bool = True
    min_trainable_params_ratio: float = 1.0
    min_params_changed_ratio: float = 0.5
    # Gradient verification
    gradient_cosine_min: float = 0.8
    gradient_norm_ratio_min: float = 0.5
    gradient_norm_ratio_max: float = 2.0
    # MFU calculation
    gpu_peak_tflops: float = 312.0


class EvaluateResponse(BaseModel):
    """Response body from /evaluate endpoint."""

    task_id: int
    mfu: float  # Model FLOPs Utilization (primary metric)
    tps: float  # Tokens per second (secondary metric)
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None
    seed: str
    diagnostics: dict = {}


@app.get("/health")
async def health():
    """Health check endpoint for Basilica."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate miner's train.py code and return MFU score.

    This endpoint is called by the validator via Basilica.
    """
    actor = get_actor()

    result = await actor.evaluate(
        task_id=request.task_id,
        seed=request.seed,
        model_url=request.model_url,
        data_url=request.data_url,
        steps=request.steps,
        batch_size=request.batch_size,
        timeout=request.timeout,
        sequence_length=request.sequence_length,
        data_samples=request.data_samples,
        code=request.code,
        max_loss_difference=request.max_loss_difference,
        use_random_init=request.use_random_init,
        min_trainable_params_ratio=request.min_trainable_params_ratio,
        min_params_changed_ratio=request.min_params_changed_ratio,
        gradient_cosine_min=request.gradient_cosine_min,
        gradient_norm_ratio_min=request.gradient_norm_ratio_min,
        gradient_norm_ratio_max=request.gradient_norm_ratio_max,
        gpu_peak_tflops=request.gpu_peak_tflops,
    )

    return EvaluateResponse(
        task_id=result.get("task_id", request.task_id),
        mfu=result.get("mfu", 0.0),
        tps=result.get("tps", 0.0),
        total_tokens=result.get("total_tokens", 0),
        wall_time_seconds=result.get("wall_time_seconds", 0.0),
        success=result.get("success", False),
        error=result.get("error"),
        seed=result.get("seed", request.seed),
        diagnostics=result.get("diagnostics", {}),
    )


# Entry point when running directly (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
