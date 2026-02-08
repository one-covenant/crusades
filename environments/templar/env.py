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
4. This Actor runs benchmark and returns MFU
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

# Named constants for magic numbers
GPU_MEMORY_PRESSURE_THRESHOLD = 0.80  # Fraction of GPU memory usage that triggers cache eviction
GRADIENT_ZERO_THRESHOLD = 1e-10  # Threshold below which a gradient is considered zero
ELEMENT_CHANGE_THRESHOLD = 1e-6  # Threshold for counting individual parameter element changes


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Captured gradient information for verification.

    MEMORY OPTIMIZATION: grad_vector is a list of per-layer tensors stored on CPU,
    rather than a single concatenated tensor. This avoids allocating ~6-12GB for
    large models. Cosine similarity is computed incrementally layer-by-layer.
    """

    grad_norm: float  # L2 norm of all gradients
    grad_vector: list | None = None  # List of per-layer gradient tensors (on CPU)
    layers_with_grad: int = 0  # Layers with non-zero gradients
    total_layers: int = 0  # Total trainable layers


# Global cache for model (data is NOT cached for validators)
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
}


def _load_miner_module(train_path: Path):
    """Dynamically load miner's train.py as a module.

    Args:
        train_path: Path to train.py

    Returns:
        Loaded module with inner_steps function

    Security:
        - Docker provides isolation (--network=none, restricted filesystem)
        - No import sandboxing (incompatible with ML libraries)
    """
    spec = importlib.util.spec_from_file_location("miner_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["miner_train"] = module
    spec.loader.exec_module(module)
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

    # Clear torch.compile / dynamo caches (major memory leak source)
    # Miners using torch.compile leave compiled graphs + triton kernels in GPU memory
    try:
        import torch._dynamo

        torch._dynamo.reset()
        logger.debug("torch._dynamo cache cleared")
    except (ImportError, AttributeError):
        pass

    # Clear torch.compile inductor caches
    try:
        import torch._inductor.codecache

        if hasattr(torch._inductor.codecache, "PyCodeCache"):
            torch._inductor.codecache.PyCodeCache.cache_clear()
    except (ImportError, AttributeError):
        pass

    # Remove miner module from sys.modules to release all its global state
    # (closures, leaked tensor references, global variables, etc.)
    miner_module = sys.modules.pop("miner_train", None)
    if miner_module is not None:
        # Delete all attributes to break reference cycles
        for attr in list(vars(miner_module).keys()):
            try:
                delattr(miner_module, attr)
            except Exception:
                pass
        del miner_module

    logger.debug("Torch state reset complete")


def _check_gpu_memory_pressure() -> bool:
    """Check if GPU memory is under pressure and evict cache if needed.

    Returns True if cache was evicted (caller should reload model).
    """
    if not torch.cuda.is_available():
        return False

    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_fraction = 1.0 - (free_mem / total_mem)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)

        logger.info(
            f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({used_fraction:.0%} used)"
        )

        # If more than 80% of GPU memory is used, evict the model cache
        # to prevent OOM on the next evaluation
        if used_fraction > GPU_MEMORY_PRESSURE_THRESHOLD:
            logger.warning(
                f"GPU memory pressure detected ({used_fraction:.0%} used, "
                f"{free_gb:.1f}GB free). Evicting model cache."
            )
            _evict_model_cache()
            return True

    except Exception as e:
        logger.debug(f"Could not check GPU memory: {e}")

    return False


def _evict_model_cache():
    """Evict cached model and state to free GPU+CPU memory."""
    if _CACHE.get("model") is not None:
        # Move model to CPU first, then delete
        try:
            _CACHE["model"].cpu()
        except Exception:
            pass
        _CACHE["model"] = None
        _CACHE["model_path"] = None
        _CACHE["use_random_init"] = None

    if _CACHE.get("initial_state") is not None:
        _CACHE["initial_state"] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024**3)
        logger.info(f"After cache eviction: {free_gb:.1f}GB free")
    except Exception:
        pass


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


def _validate_code_structure(code: str) -> tuple[bool, str | None, str | None]:
    """Validate that train.py has correct structure and no forbidden patterns.

    Checks:
    1. Code parses without syntax errors
    2. inner_steps function exists with correct signature
    3. No forbidden patterns that violate miner rules (optimizer bypass, memory
       manipulation, gradient checkpointing disable, module manipulation)

    Returns:
        (ok, error_message, error_code) - error_code is a structured code for
        reliable error handling (maps to EvaluationErrorCode).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}", "syntax_error"

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                return (
                    False,
                    f"inner_steps has {len(args.args)} args, expected at least 5",
                    "missing_inner_steps",
                )
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps", "missing_inner_steps"

    # Scan for forbidden patterns
    forbidden_error = _scan_forbidden_patterns(tree)
    if forbidden_error is not None:
        return False, forbidden_error, "forbidden_code_pattern"

    return True, None, None


# =========================================================================
# AST scanner constants — forbidden names, modules, and attributes.
# Defined at module level so they are created once and shared across calls.
# =========================================================================

# Function/builtin names that are always forbidden
_BANNED_CALL_NAMES = frozenset(
    {
        "eval",  # Dynamic code execution (complete AST bypass)
        "exec",  # Dynamic code execution (complete AST bypass)
        "compile",  # Code compilation (used with eval/exec)
        "__import__",  # Dynamic import (bypasses all module-level checks)
        "globals",  # Access to global namespace (smuggles __builtins__)
        "breakpoint",  # Debugger access
        "autocast",  # Precision manipulation (from torch.cuda.amp import autocast)
    }
)

# Attribute names on 'sys' that are always forbidden
_BANNED_SYS_ATTRS = frozenset(
    {
        "modules",  # Module manipulation
        "_getframe",  # Stack walking (reads validator locals including reference gradients)
        "_current_frames",  # Stack walking
        "_current_exceptions",  # Stack walking
    }
)

# Attribute names on 'gc' that are always forbidden
_BANNED_GC_ATTRS = frozenset(
    {
        "get_objects",  # Walk all process objects
        "get_referents",  # Walk object references (finds real optimizer)
        "get_referrers",  # Walk object referrers
    }
)

# Module names that are always forbidden to import or reference
_BANNED_MODULES = frozenset(
    {
        "ctypes",  # Memory manipulation
        "inspect",  # Introspection (getattr_static, stack frames, etc.)
        "importlib",  # Dynamic module loading
        "code",  # Interactive interpreter
        "codeop",  # Compile code with possibility of incomplete input
        "dis",  # Bytecode disassembler (introspection)
        "types",  # Type manipulation (FunctionType, CodeType)
    }
)

# Names (variables/builtins) that are forbidden to READ at all
_BANNED_NAMES = frozenset(
    {
        "__builtins__",  # Access to eval/exec/compile via __builtins__['eval']
    }
)

# Attribute names that are always forbidden on ANY object (read or call)
_BANNED_ATTRS_ANY_OBJECT = frozenset(
    {
        # Gradient checkpointing
        "gradient_checkpointing_disable",  # Disabling gradient checkpointing
        # Model hook manipulation (intercept/modify forward/backward pass)
        "register_forward_hook",  # Intercept model forward outputs
        "register_forward_pre_hook",  # Intercept model forward inputs
        "register_backward_hook",  # Intercept backward gradients
        "register_full_backward_hook",  # Full backward gradient hook
        "remove_hook_from_module",  # Remove existing hooks (e.g. grad ckpt hooks)
        # Attention backend manipulation (force faster but non-deterministic kernels)
        "enable_flash_sdp",  # Toggle Flash SDP attention
        "enable_math_sdp",  # Toggle Math SDP attention
        "enable_mem_efficient_sdp",  # Toggle memory-efficient SDP attention
        # Global precision manipulation
        "set_float32_matmul_precision",  # Change matmul precision globally
    }
)

# Attribute names on 'torch' that are forbidden (torch.compile, torch.autocast)
_BANNED_TORCH_ATTRS = frozenset(
    {
        "compile",  # JIT compilation (non-deterministic with max-autotune, cache leaks)
        "autocast",  # Precision manipulation (changes gradient computation)
    }
)


def _scan_forbidden_patterns(tree: ast.AST) -> str | None:
    """Scan AST for forbidden code patterns that violate miner rules.

    This scanner works in two layers:
    1. BLANKET BANS: Dangerous builtins/modules that have no legitimate use in
       a training loop (eval, exec, gc, sys._getframe, ctypes, etc.)
    2. TARGETED CHECKS: Specific patterns on optimizer/model objects

    Returns an error message if a forbidden pattern is found, None otherwise.
    """
    for node in ast.walk(tree):
        # =================================================================
        # Check for banned name references (variables/builtins)
        # =================================================================
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return (
                f"Forbidden: {node.id} (line {node.lineno}). Accessing '{node.id}' is not allowed."
            )

        # =================================================================
        # Check function/method calls
        # =================================================================
        if isinstance(node, ast.Call):
            func = node.func

            # --- Banned builtins: eval(), exec(), compile(), __import__(), globals() ---
            if isinstance(func, ast.Name) and func.id in _BANNED_CALL_NAMES:
                return (
                    f"Forbidden: {func.id}() (line {node.lineno}). "
                    f"Dynamic code execution is not allowed."
                )

            # --- gc.get_objects / gc.get_referents / gc.get_referrers ---
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _BANNED_GC_ATTRS
                and isinstance(func.value, ast.Name)
                and func.value.id == "gc"
            ):
                return (
                    f"Forbidden: gc.{func.attr}() (line {node.lineno}). "
                    f"Process memory introspection is not allowed."
                )

            # --- __import__('gc').get_objects() pattern ---
            # Catches: __import__('gc').get_objects() etc.
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _BANNED_GC_ATTRS
                and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "__import__"
            ):
                return (
                    f"Forbidden: __import__ + {func.attr}() (line {node.lineno}). "
                    f"Dynamic import for process introspection is not allowed."
                )

            # --- object.__getattribute__(optimizer, ...) ---
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("__getattribute__", "__setattr__", "__delattr__")
                and isinstance(func.value, ast.Name)
                and func.value.id == "object"
            ):
                return (
                    f"Forbidden: object.{func.attr}() (line {node.lineno}). "
                    f"Direct object protocol manipulation is not allowed."
                )

            # --- torch.compile() / torch.autocast() ---
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _BANNED_TORCH_ATTRS
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            ):
                return (
                    f"Forbidden: torch.{func.attr}() (line {node.lineno}). "
                    f"Miners MUST NOT use torch.{func.attr}()."
                )

        # =================================================================
        # Check attribute accesses
        # =================================================================
        if isinstance(node, ast.Attribute):
            # --- sys.modules / sys._getframe / sys._current_frames ---
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr in _BANNED_SYS_ATTRS
            ):
                return (
                    f"Forbidden: sys.{node.attr} (line {node.lineno}). "
                    f"System introspection is not allowed."
                )

            # --- Banned attributes on any object ---
            if node.attr in _BANNED_ATTRS_ANY_OBJECT:
                return (
                    f"Forbidden: .{node.attr} (line {node.lineno}). "
                    f"This operation is not allowed in miner code."
                )

            # --- gradient_checkpointing flag set to False ---
            # Catches: model.gradient_checkpointing = False (or any obj)
            if node.attr == "gradient_checkpointing":
                # Check if this is a store (assignment target)
                # ast.Attribute as store context means it's being assigned to
                if isinstance(node.ctx, ast.Store):
                    return (
                        f"Forbidden: modifying gradient_checkpointing flag (line {node.lineno}). "
                        f"Miners MUST NOT modify gradient checkpointing state."
                    )

        # =================================================================
        # Check imports for banned modules
        # =================================================================
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BANNED_MODULES:
                    return (
                        f"Forbidden: import {alias.name} (line {node.lineno}). "
                        f"Module '{alias.name}' is not allowed."
                    )

        if isinstance(node, ast.ImportFrom):
            if node.module in _BANNED_MODULES:
                return (
                    f"Forbidden: from {node.module} import ... (line {node.lineno}). "
                    f"Module '{node.module}' is not allowed."
                )

    return None


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

    Note:
        Docker evaluation runs with --network=none for security.
        Models must be pre-cached in the Docker image. If cache is missing,
        rebuild the image: docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if use_random_init:
        # Random init - miners can't cheat by freezing pretrained layers
        logger.info(f"Loading model config from {model_path} with RANDOM initialization")

        # Use local cache only - Docker runs with --network=none for security
        # If this fails, the Docker image needs to be rebuilt with the model cached
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Simple model loading - Qwen2.5-3B (~6GB) fits easily on A100 80GB
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = model.to(device)
        logger.info(f"Model loaded on {device}")
    else:
        logger.info(f"Loading pretrained model from {model_path}")
        # Use local cache only - Docker runs with --network=none for security
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )

    model.train()

    # Ensure ALL parameters have requires_grad=True for training
    for param in model.parameters():
        param.requires_grad = True

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
    # Guard against division by zero / invalid inputs
    if wall_time <= 0 or gpu_peak_tflops <= 0:
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

    MEMORY OPTIMIZATION: Instead of concatenating all gradients into a single
    large tensor (which can be ~6-12GB for 3B models), we store per-layer
    gradient vectors on CPU. Cosine similarity is computed incrementally
    in _verify_gradients() to avoid memory pressure.

    Returns:
        GradientInfo with norm and per-layer gradient vectors (on CPU)
    """
    grad_vectors_cpu = []  # Store per-layer on CPU to save GPU memory
    total_norm_sq = 0.0
    layers_with_grad = 0
    layers_without_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            # Move to CPU immediately to free GPU memory
            grad_flat = param.grad.detach().cpu().float().view(-1)
            total_norm_sq += grad_flat.pow(2).sum().item()
            # Check if gradient is actually non-zero (not just allocated)
            if grad_flat.abs().sum().item() > GRADIENT_ZERO_THRESHOLD:
                layers_with_grad += 1
                grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(grad_flat)  # Still store for shape matching
        else:
            layers_without_grad += 1
            grad_vectors_cpu.append(None)

    grad_norm = total_norm_sq**0.5

    # Log layer gradient coverage
    total_layers = layers_with_grad + layers_without_grad
    if total_layers > 0:
        coverage = layers_with_grad / total_layers
        logger.info(f"Gradient coverage: {layers_with_grad}/{total_layers} layers ({coverage:.1%})")
        if layers_without_grad > 0:
            logger.warning(f"WARNING: {layers_without_grad} layers have zero/no gradients!")

    return GradientInfo(
        grad_norm=grad_norm,
        grad_vector=grad_vectors_cpu,  # Now a list of per-layer tensors on CPU
        layers_with_grad=layers_with_grad,
        total_layers=total_layers,
    )


def _verify_trainable_params(
    model: torch.nn.Module,
    min_trainable_ratio: float = 1.0,
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
    element_threshold: float = ELEMENT_CHANGE_THRESHOLD,
) -> tuple[bool, str | None, dict]:
    """Verify that minimum % of individual parameter elements changed during training.

    SECURITY: Counts individual elements, not tensors. This prevents a bypass where
    miners make tiny modifications to many tensors (each barely above threshold) to
    pass the check without meaningful training.

    Args:
        model: Model after training
        initial_state: Model state before training
        min_changed_ratio: Minimum fraction of elements that must change
        element_threshold: Minimum absolute change for an element to count as "changed"
    """
    total_elements = 0
    changed_elements = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            initial = initial_state[name].to(param.device)
            # Count individual elements that changed (not whole tensors)
            element_diffs = (param.data - initial).abs()
            changed_mask = element_diffs > element_threshold

            total_elements += param.numel()
            changed_elements += changed_mask.sum().item()

    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0

    details = {
        "total_elements": total_elements,
        "changed_elements": int(changed_elements),
        "changed_ratio": changed_ratio,
        "min_required": min_changed_ratio,
        "element_threshold": element_threshold,
    }

    if changed_ratio < min_changed_ratio:
        error = (
            f"Insufficient parameter updates: {changed_ratio:.1%} "
            f"({int(changed_elements):,}/{total_elements:,} elements) - minimum {min_changed_ratio:.0%} required"
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


class GradientCapturingOptimizer:
    """Optimizer wrapper that captures gradients before each step.

    SECURITY: This wrapper intercepts optimizer.step() to capture gradients
    from within the miner's inner_steps execution. This ensures we verify
    gradients produced by miner code, not validator code.

    Protection layers:
    1. __getattribute__: blocks reading hidden attrs + __dict__/vars()
    2. __setattr__: blocks writing protected attrs
    3. __delattr__: blocks deleting any protected/hidden attrs
    4. __class__ property: blocks class replacement attacks
    5. Method binding: step/zero_grad are bound at init, can't be replaced

    PERFORMANCE: Tracks overhead time separately and excludes it from wall_time
    for fair MFU/TPS calculation. MFU formula (6 * params * tokens) only accounts
    for forward + backward FLOPs, so we exclude:
    - Gradient capture (validator overhead)

    """

    # Attributes that miner code must not be able to replace
    _PROTECTED_ATTRS = frozenset(
        {
            "optimizer",
            "model",
            "captured_gradients",
            "step_count",
            "gradient_capture_time",
            "_initialized",
            # Method/class protections
            "step",
            "zero_grad",
            "__class__",
            "__dict__",
        }
    )

    # Attributes that miner code must not be able to READ
    # (prevents wrapper bypass via optimizer.optimizer, optimizer.model,
    #  optimizer.__dict__, vars(optimizer), etc.)
    _HIDDEN_ATTRS = frozenset({"optimizer", "model", "__dict__"})

    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module):
        # Use object.__setattr__ during init to bypass our protection
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "captured_gradients", None)
        object.__setattr__(self, "step_count", 0)
        object.__setattr__(self, "gradient_capture_time", 0.0)
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name):
        """Block miner code from reading hidden internal attributes.

        Blocks: optimizer, model, __dict__ (and vars() which uses __dict__)
        """
        # Use CLASS reference to prevent miner from clearing _HIDDEN_ATTRS
        if name in GradientCapturingOptimizer._HIDDEN_ATTRS:
            raise AttributeError(
                f"Cannot access internal attribute '{name}' on optimizer wrapper. "
                f"Use optimizer.step() and optimizer.zero_grad() directly."
            )
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        """Prevent miner code from replacing security-critical attributes."""
        # Use CLASS reference (not self) to prevent miner from clearing the set
        # via optimizer._PROTECTED_ATTRS = frozenset()
        if (
            getattr(self, "_initialized", False)
            and name in GradientCapturingOptimizer._PROTECTED_ATTRS
        ):
            raise AttributeError(f"Cannot modify protected attribute '{name}' on optimizer wrapper")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        """Prevent miner code from deleting any attributes."""
        raise AttributeError(f"Cannot delete attribute '{name}' on optimizer wrapper")

    def step(self, *args, **kwargs):
        """Capture gradients BEFORE optimizer.step() clears them."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Use object.__getattribute__ to bypass __getattribute__ protection
        model = object.__getattribute__(self, "model")
        opt = object.__getattribute__(self, "optimizer")

        # Now time ONLY the gradient capture (GPU->CPU copy for verification)
        # This is validator overhead and should be excluded from wall_time
        capture_start = time.perf_counter()
        # Use object.__setattr__ to update protected attrs internally
        object.__setattr__(self, "captured_gradients", _capture_gradients(model))
        cur_capture_time = object.__getattribute__(self, "gradient_capture_time")
        object.__setattr__(
            self,
            "gradient_capture_time",
            cur_capture_time + (time.perf_counter() - capture_start),
        )

        # Actual optimizer step (part of miner's training)
        cur_step_count = object.__getattribute__(self, "step_count")
        object.__setattr__(self, "step_count", cur_step_count + 1)
        return opt.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        return opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        """Forward param_groups access to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        return opt.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Forward param_groups setter to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        opt.param_groups = value

    def state_dict(self):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        return opt.state_dict()

    def load_state_dict(self, state_dict):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        return opt.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        """Forward to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        return opt.add_param_group(param_group)

    @property
    def state(self):
        """Forward state access to underlying optimizer."""
        opt = object.__getattribute__(self, "optimizer")
        return opt.state

    def __getattr__(self, name):
        """Forward any other attribute access to underlying optimizer.

        SECURITY: Also block hidden attrs here since __getattr__ is called
        as a fallback when __getattribute__ raises AttributeError.
        """
        if name in GradientCapturingOptimizer._HIDDEN_ATTRS:
            raise AttributeError(
                f"Cannot access internal attribute '{name}' on optimizer wrapper. "
                f"Use optimizer.step() and optimizer.zero_grad() directly."
            )
        opt = object.__getattribute__(self, "optimizer")
        return getattr(opt, name)


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

        # No autocast - model is already in bfloat16.
        # Using autocast here would change loss computation precision
        # and create gradient mismatches with miner code that doesn't use autocast.
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
    norm_ratio_max: float = 1.02,  # Relative error threshold: 1.02 = 2% max error
) -> tuple[bool, str | None, dict]:
    """Verify candidate gradients match reference using relative error |g - g_truth| / |g_truth|.

    This is a direct comparison that catches any deviation including truncation,
    layer freezing, and step skipping. The threshold should be near numerical
    precision (bfloat16 has ~0.8% relative error for accumulated ops).

    Args:
        reference_grad: Reference implementation gradients
        candidate_grad: Miner's implementation gradients
        norm_ratio_max: Maximum allowed relative error |g - g_truth| / |g_truth|
            Repurposed: norm_ratio_max is used as the relative error threshold.
            Production default: 0.01 (1% relative error)

    Returns:
        Tuple of (success, error_message, details)
    """
    # Repurpose norm_ratio_max as the relative error threshold
    # This keeps backward compat with hparams without adding a new field
    # norm_ratio_max=1.1 means 10% relative error allowed (old default)
    # norm_ratio_max=1.01 means 1% relative error allowed (tight)
    relative_error_threshold = norm_ratio_max - 1.0  # e.g., 1.01 -> 0.01

    details = {
        "relative_error_threshold": relative_error_threshold,
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
        logger.info(f"[CHECK 0/2] Gradient coverage: {grad_coverage:.1%}")
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

    # Check 1: Relative gradient error |g - g_truth| / |g_truth|
    # This single check replaces both norm ratio and cosine similarity.
    # It catches ALL deviations: truncation, layer freezing, step skipping, etc.
    ref_vecs = reference_grad.grad_vector
    cand_vecs = candidate_grad.grad_vector

    if ref_vecs is not None and cand_vecs is not None and len(ref_vecs) > 0:
        if len(ref_vecs) != len(cand_vecs):
            error = f"Gradient layer count mismatch: ref={len(ref_vecs)}, cand={len(cand_vecs)}"
            details["checks_failed"].append({"check": "gradient_shape", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details

        # Compute |g - g_truth|^2 and |g_truth|^2 incrementally (layer-by-layer)
        diff_norm_sq = 0.0
        ref_norm_sq = 0.0
        total_elements = 0

        for ref_layer, cand_layer in zip(ref_vecs, cand_vecs):
            if ref_layer is None or cand_layer is None:
                continue

            if ref_layer.shape != cand_layer.shape:
                error = f"Gradient shape mismatch at layer: ref={ref_layer.shape}, cand={cand_layer.shape}"
                details["checks_failed"].append({"check": "gradient_shape", "error": error})
                logger.error(f"[FAILED] {error}")
                return False, error, details

            diff = cand_layer - ref_layer
            diff_norm_sq += (diff * diff).sum().item()
            ref_norm_sq += (ref_layer * ref_layer).sum().item()
            total_elements += ref_layer.numel()

        ref_norm = ref_norm_sq**0.5
        diff_norm = diff_norm_sq**0.5

        if ref_norm > 0:
            relative_error = diff_norm / ref_norm
        else:
            relative_error = 0.0 if diff_norm == 0 else float("inf")

        details["relative_error"] = relative_error
        details["diff_norm"] = diff_norm
        details["ref_norm"] = ref_norm
        details["gradient_elements"] = total_elements

        logger.info(f"[CHECK 1/2] Gradient relative error: {relative_error:.6f}")
        logger.info(f"   |g - g_truth|: {diff_norm:.6f}")
        logger.info(f"   |g_truth|: {ref_norm:.6f}")
        logger.info(f"   Max allowed: {relative_error_threshold:.6f}")
        logger.info(f"   Total gradient elements: {total_elements:,}")

        if relative_error > relative_error_threshold:
            error = (
                f"Gradient relative error {relative_error:.6f} exceeds threshold "
                f"{relative_error_threshold:.6f} (|g - g_truth| / |g_truth|)"
            )
            details["checks_failed"].append({"check": "gradient_relative_error", "error": error})
            details["error_code"] = "gradient_norm_ratio_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_relative_error")
        logger.info("[PASSED] Gradient relative error within threshold")
    else:
        logger.warning("Gradient vectors unavailable, skipping relative error check")

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
    gradient_norm_ratio_max: float = 1.02,
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
        gradient_norm_ratio_max: Max gradient norm ratio (relative error threshold)

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
    if candidate.final_loss <= 0:
        error = f"Loss must be positive (cross-entropy > 0): got {candidate.final_loss:.4f}"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if candidate.final_loss > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Compare losses - check absolute difference
    if reference is None:
        error = "No reference result available for loss comparison"
        details["checks_failed"].append({"check": "loss_comparison", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    loss_difference = abs(candidate.final_loss - reference.final_loss)
    details["loss_difference"] = loss_difference
    details["reference_loss"] = reference.final_loss
    logger.info(
        f"   Reference loss: {reference.final_loss:.4f}, Candidate loss: {candidate.final_loss:.4f}"
    )
    logger.info(f"   Loss difference: {loss_difference:.4f} (max allowed: {max_loss_difference})")

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
        gradient_norm_ratio_max: float = 1.02,
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
            gradient_norm_ratio_max: Max gradient norm ratio (relative error threshold)
            gpu_peak_tflops: GPU peak TFLOPS for MFU calculation
            model_params_override: Override model param count (None = auto-detect)

        Returns:
            Dict with: mfu, tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate code
        if not code:
            return {
                "task_id": task_id,
                "mfu": 0.0,
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
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Missing model_url or data_url",
                "seed": seed,
            }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Pre-evaluation memory check: evict cache if GPU is under pressure
        # This prevents OOM from accumulated leaks across evaluations
        _check_gpu_memory_pressure()

        deadline = time.monotonic() + timeout
        seed_value = abs(hash(seed)) % (2**32)
        _set_deterministic(seed_value)

        # Validate code structure and scan for forbidden patterns
        code_ok, code_error, code_error_code = _validate_code_structure(code)
        if not code_ok:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "error_code": code_error_code,
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
                "mfu": 0.0,
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
                "mfu": 0.0,
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
                    "mfu": 0.0,
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

            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            # =================================================================
            # WARMUP FIRST - Catch broken code BEFORE wasting GPU on reference
            # =================================================================
            # Run cheap warmup (2 steps) to detect basic errors early.
            # If this fails, we skip the expensive reference run entirely.
            # SECURITY: Use GradientCapturingOptimizer for warmup too, so
            # miners cannot detect warmup vs evaluation by checking optimizer type.
            warmup_steps = 2
            data_iter_warmup = _create_data_iterator(data, batch_size, seq_len)
            warmup_base_opt = _create_optimizer(model)
            optimizer_warmup = GradientCapturingOptimizer(warmup_base_opt, model)

            logger.info(f"Running {warmup_steps} warmup step(s) to check for basic errors...")
            try:
                warmup_result = miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_warmup,
                    optimizer=optimizer_warmup,
                    num_steps=warmup_steps,
                    device=device,
                )

                # Quick validation of warmup result
                if warmup_result is None:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
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
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps result missing 'final_logits' attribute",
                        "seed": seed,
                        "code": code,
                    }

                # Check logits are present and valid shape
                logits = warmup_result.final_logits
                if logits is None:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "final_logits is None - must return actual logits for verification",
                        "error_code": "missing_logits",
                        "seed": seed,
                        "code": code,
                    }
                if len(logits.shape) != 3:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": f"Logits shape mismatch: expected 3D (batch, seq, vocab), got {logits.shape}",
                        "error_code": "invalid_logits_shape",
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

            # Reset model state after warmup and restore gradient checkpointing
            model.load_state_dict(initial_state)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            # =================================================================
            # REFERENCE RUN - Only after warmup passes (saves GPU on failures)
            # =================================================================
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)

            reference, reference_grad = _run_reference(
                model, data_iter_ref, optimizer_ref, steps, device, capture_final_gradients=True
            )

            # Reset model for miner's full evaluation
            model.load_state_dict(initial_state)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            # =================================================================
            # FULL EVALUATION - Run actual timed evaluation with gradient capture
            # =================================================================
            # SECURITY: Use GradientCapturingOptimizer to capture gradients from
            # WITHIN the miner's inner_steps execution. This ensures we verify
            # gradients produced by miner code, not validator code.
            #
            # Previous approach (running final step manually) was flawed because
            # when steps==1, miner's inner_steps was never called during evaluation.
            data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
            base_optimizer = _create_optimizer(model)
            optimizer_miner = GradientCapturingOptimizer(base_optimizer, model)

            # SECURITY: Snapshot torch global state BEFORE miner code runs.
            # Compared after miner returns to detect dynamic manipulation that
            # the AST scanner can't catch (e.g. via string concatenation tricks).
            _pre_state = {}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                    _pre_state["flash_sdp"] = torch.backends.cuda.flash_sdp_enabled()
                if hasattr(torch.backends.cuda, "math_sdp_enabled"):
                    _pre_state["math_sdp"] = torch.backends.cuda.math_sdp_enabled()
                if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
                    _pre_state["mem_efficient_sdp"] = (
                        torch.backends.cuda.mem_efficient_sdp_enabled()
                    )
            if hasattr(torch, "get_float32_matmul_precision"):
                _pre_state["matmul_precision"] = torch.get_float32_matmul_precision()

            start = time.perf_counter()

            # Run ALL steps through miner's inner_steps
            miner_result = miner_module.inner_steps(
                model=model,
                data_iterator=data_iter_miner,
                optimizer=optimizer_miner,
                num_steps=steps,
                device=device,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_time = time.perf_counter() - start

            # Exclude gradient capture time from wall_time for fair MFU calculation
            # Gradient capture is validator overhead (copies gradients GPU->CPU for verification)
            # optimizer.step() and zero_grad() ARE part of miner's training, included in wall_time
            gradient_capture_time = optimizer_miner.gradient_capture_time
            wall_time = total_time - gradient_capture_time
            logger.info(
                f"Timing: total={total_time:.2f}s, grad_capture={gradient_capture_time:.2f}s, "
                f"wall_time={wall_time:.2f}s (used for MFU/TPS)"
            )

            # SECURITY: Verify torch state hasn't been tampered with
            # Miners may change these for speed (non-deterministic = ~5-10% faster,
            # disable checkpointing = ~30% faster backward). We check at runtime
            # because AST scans can't catch all dynamic manipulation methods.
            if torch.backends.cudnn.deterministic is False:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": (
                        "torch.backends.cudnn.deterministic was set to False during evaluation. "
                        "Miners MUST NOT modify deterministic settings."
                    ),
                    "error_code": "forbidden_code_pattern",
                    "seed": seed,
                    "code": code,
                }

            if torch.backends.cudnn.benchmark is True:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": (
                        "torch.backends.cudnn.benchmark was set to True during evaluation. "
                        "Miners MUST NOT enable benchmark mode."
                    ),
                    "error_code": "forbidden_code_pattern",
                    "seed": seed,
                    "code": code,
                }

            if hasattr(model, "is_gradient_checkpointing") and not model.is_gradient_checkpointing:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": (
                        "Gradient checkpointing was disabled during evaluation. "
                        "Miners MUST NOT disable gradient checkpointing."
                    ),
                    "error_code": "forbidden_code_pattern",
                    "seed": seed,
                    "code": code,
                }

            # SECURITY: Verify SDP attention backend state wasn't changed.
            # Switching attention kernels (flash/math/mem_efficient) can give
            # speed gains with subtly different numerics.
            _sdp_checks = [
                ("flash_sdp", "flash_sdp_enabled", "enable_flash_sdp"),
                ("math_sdp", "math_sdp_enabled", "enable_math_sdp"),
                ("mem_efficient_sdp", "mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
            ]
            for state_key, check_fn_name, setter_name in _sdp_checks:
                if state_key in _pre_state and hasattr(torch.backends.cuda, check_fn_name):
                    current_val = getattr(torch.backends.cuda, check_fn_name)()
                    if current_val != _pre_state[state_key]:
                        return {
                            "task_id": task_id,
                            "mfu": 0.0,
                            "tps": 0.0,
                            "total_tokens": 0,
                            "wall_time_seconds": wall_time,
                            "success": False,
                            "error": (
                                f"torch.backends.cuda.{setter_name}() was called during evaluation "
                                f"(changed from {_pre_state[state_key]} to {current_val}). "
                                f"Miners MUST NOT modify attention backend settings."
                            ),
                            "error_code": "forbidden_code_pattern",
                            "seed": seed,
                            "code": code,
                        }

            # SECURITY: Verify matmul precision wasn't changed.
            if "matmul_precision" in _pre_state and hasattr(torch, "get_float32_matmul_precision"):
                current_precision = torch.get_float32_matmul_precision()
                if current_precision != _pre_state["matmul_precision"]:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": wall_time,
                        "success": False,
                        "error": (
                            f"torch.set_float32_matmul_precision() was called during evaluation "
                            f"(changed from '{_pre_state['matmul_precision']}' to '{current_precision}'). "
                            f"Miners MUST NOT modify matmul precision."
                        ),
                        "error_code": "forbidden_code_pattern",
                        "seed": seed,
                        "code": code,
                    }

            # Verify miner called optimizer.step() on every step (not bypassing wrapper)
            miner_step_count = optimizer_miner.step_count
            if miner_step_count != steps:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": (
                        f"optimizer.step() called {miner_step_count} times, expected {steps}. "
                        f"Miners MUST call the provided optimizer.step() on every training step."
                    ),
                    "error_code": "step_count_mismatch",
                    "seed": seed,
                    "code": code,
                }

            # Get gradients captured from the final step (inside miner's code)
            candidate_grad = optimizer_miner.captured_gradients

            if candidate_grad is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "No gradients captured - miner may not be calling optimizer.step()",
                    "error_code": "no_gradients_captured",
                    "seed": seed,
                    "code": code,
                }

            # Validate miner's returned result
            ok, error, parsed = _validate_return_type(miner_result)
            if not ok or parsed is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": error,
                    "seed": seed,
                    "code": code,
                }

            # =================================================================
            # SEQUENCE LENGTH CHECK - Detect truncation cheating
            # =================================================================
            # SECURITY: Miners may truncate sequences to process fewer tokens
            # while reporting full token count. Check logits sequence dimension.
            # Expected: seq_len - 1 (causal LM uses batch[:, :-1] as input)
            expected_seq_len = seq_len - 1

            # SECURITY: Require final_logits to be present (prevent None bypass)
            if parsed.final_logits is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "final_logits is None - must return actual logits for verification",
                    "error_code": "missing_logits",
                    "seed": seed,
                    "code": code,
                }

            # SECURITY: Verify logits is 3D tensor (batch, seq, vocab)
            if len(parsed.final_logits.shape) != 3:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Logits shape invalid: expected 3D (batch, seq, vocab), got shape {parsed.final_logits.shape}",
                    "error_code": "invalid_logits_shape",
                    "seed": seed,
                    "code": code,
                }

            logits_seq_len = parsed.final_logits.shape[1]

            # SECURITY: Verify sequence length - miner can't truncate sequences
            if logits_seq_len != expected_seq_len:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": f"Sequence length mismatch (possible truncation): expected {expected_seq_len}, got {logits_seq_len}",
                    "error_code": "sequence_truncation",
                    "seed": seed,
                    "code": code,
                }
            logger.info(f"[PASSED] Sequence length check: {logits_seq_len} == {expected_seq_len}")

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
                gradient_norm_ratio_max=gradient_norm_ratio_max,
            )

            # This ensures miner can't inflate MFU by reporting higher token counts
            total_tokens_int = expected_tokens  # Validator knows exact expected count
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
            # SECURITY: Reset torch state to prevent cross-evaluation contamination
            # (also clears dynamo caches and removes miner module from sys.modules)
            _reset_torch_state()

            # Restore gradient checkpointing on the cached model if miner disabled it
            # (defense-in-depth: AST scan blocks this, but belt-and-suspenders)
            cached_model = _CACHE.get("model")
            if cached_model is not None and hasattr(cached_model, "gradient_checkpointing_enable"):
                try:
                    cached_model.gradient_checkpointing_enable()
                except Exception:
                    pass

            # Aggressive memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check GPU memory pressure and evict model cache if needed
            # This prevents OOM on the next evaluation due to accumulated leaks
            _check_gpu_memory_pressure()


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
    gradient_norm_ratio_max: float = 1.02
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
    error_code: str | None = None
    seed: str
    diagnostics: dict = {}


@app.get("/health")
async def health():
    """Health check endpoint for Basilica with GPU memory diagnostics."""
    info = {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            info["gpu_memory_free_gb"] = round(free_mem / (1024**3), 2)
            info["gpu_memory_total_gb"] = round(total_mem / (1024**3), 2)
            info["gpu_memory_used_pct"] = round((1.0 - free_mem / total_mem) * 100, 1)
            info["model_cached"] = _CACHE.get("model") is not None
        except Exception:
            pass
    return info


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
        error_code=result.get("error_code"),
        seed=result.get("seed", request.seed),
        diagnostics=result.get("diagnostics", {}),
    )


# Entry point when running directly (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
