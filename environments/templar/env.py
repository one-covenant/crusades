"""
Templar TPS Evaluation Environment for Affinetes

This Actor class is packaged with the miner's train.py into a Docker image.
Validators call env.evaluate() to benchmark the miner's submission.

Flow:
1. Miner packages train.py + env.py + Dockerfile into Docker image
2. Validator loads image via affinetes: env = af.load_env(image=...)
3. Validator calls: result = await env.evaluate(seed=..., model_url=..., data_url=...)
4. This Actor runs the benchmark and returns TPS

Based on Chi's templar_env_bootstrap.py pattern.
"""

import ast
import gc
import hashlib
import os
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

# Add /app to path to import miner's train.py
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Miner's submission - MUST define inner_steps function
try:
    import train
except ImportError:
    train = None

# Configuration from environment variables
OUTPUT_VECTOR_TOLERANCE = float(os.getenv("OUTPUT_VECTOR_TOLERANCE", "0.02"))
VERIFY_LOSS = os.getenv("VERIFY_LOSS", "0") == "1"
LOSS_TOLERANCE = float(os.getenv("LOSS_TOLERANCE", "1e-3"))
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "1") == "1"
EVAL_SEQUENCE_LENGTH = int(os.getenv("EVAL_SEQUENCE_LENGTH", "1024"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "8"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/templar_eval"))


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


# Global cache to avoid reloading model/data
_CACHE = {
    "model": None,
    "model_path": None,
    "data": None,
    "data_path": None,
    "initial_state": None,
}


def _download(url: str, dst: Path) -> None:
    """Download file from URL to destination."""
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    seed: int | None = None,
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace.
    
    SECURITY: Validators use unpredictable seeds so miners can't pre-compute
    which samples will be used for evaluation.
    
    Args:
        dataset_name: HuggingFace dataset (e.g., "HuggingFaceFW/fineweb")
        model_name: Model name for tokenizer (e.g., "Qwen/Qwen2.5-7B")
        num_samples: Number of samples to load
        sequence_length: Sequence length for tokenization
        split: Dataset split
        seed: Fixed seed for miner local testing (use 42 for reproducibility)
        validator_seed: Seed string from validator (format: "block:uid:run")
                       If provided, overrides seed with unpredictable value
        
    Returns:
        Tensor of shape [num_samples, sequence_length] with token IDs
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import hashlib
    
    # Determine seed: validators use unpredictable seed, miners use fixed seed
    if validator_seed:
        # Hash the validator seed to get unpredictable but deterministic value
        seed_hash = hashlib.sha256(validator_seed.encode()).hexdigest()
        actual_seed = int(seed_hash[:8], 16)  # Use first 8 hex chars
        print(f"Validator mode: seed={actual_seed} (from {validator_seed})")
    else:
        actual_seed = seed or 42  # Default fixed seed for miner testing
        print(f"Miner mode: fixed seed={actual_seed}")
    
    print(f"Loading dataset: {dataset_name} (split={split}, samples={num_samples})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset with streaming for efficiency
    print(f"Loading dataset: {dataset_name} (streaming)...")
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )
    
    # Shuffle with buffer for efficiency (key: buffer_size parameter!)
    dataset = dataset.shuffle(seed=actual_seed, buffer_size=10000)
    
    # Sample and tokenize
    tokens_list = []
    dataset_iter = iter(dataset)
    
    for _ in range(num_samples):
        try:
            sample = next(dataset_iter)
            text = sample.get("text", sample.get("content", ""))
            
            # Tokenize
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
    print(f"Loaded data: shape={data.shape}, dtype={data.dtype}, seed={actual_seed}")
    return data


def _maybe_extract(archive_path: Path, extract_dir: Path) -> Path:
    """Extract archive if needed, return path to extracted content."""
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        return archive_path

    children = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_dir


def _set_deterministic(seed: int) -> None:
    """Set deterministic mode for reproducibility."""
    if not DETERMINISTIC_MODE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_code_structure(code_path: Path) -> tuple[bool, str | None]:
    """Validate that train.py has correct structure."""
    try:
        code = code_path.read_text()
    except Exception as exc:
        return False, f"Failed to read train.py: {exc}"

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
                return False, (
                    f"inner_steps has {len(args.args)} args, expected at least 5"
                )
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps"

    return True, None


def _validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    """Validate that inner_steps returned correct type."""
    if isinstance(result, InnerStepsResult):
        return True, None, result

    # Duck-typing: accept any object with required attributes
    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return True, None, InnerStepsResult(
            final_logits=result.final_logits,
            total_tokens=result.total_tokens,
            final_loss=result.final_loss,
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_path: Path | str):
    """Load model from path or HuggingFace model ID."""
    from transformers import AutoModelForCausalLM
    
    # Convert Path to string for from_pretrained
    model_path_str = str(model_path) if isinstance(model_path, Path) else model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch native attention
    )
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _load_data(
    data_path: Path | str,
    model_name: str = "Qwen/Qwen2.5-7B",
    num_samples: int = 10000,
    sequence_length: int = 1024,
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load data from file or HuggingFace dataset.
    
    Supports:
    - .pt files (pre-tokenized tensors)
    - HuggingFace dataset names (e.g., "HuggingFaceFW/fineweb")
    
    For HuggingFace datasets:
    - Miners (validator_seed=None): Use fixed seed for reproducible local testing
    - Validators (validator_seed set): Use unpredictable seed based on block/task
    """
    data_str = str(data_path)
    
    # Check if it's a HuggingFace dataset
    if "/" in data_str and not data_str.startswith("http") and not data_str.endswith(".pt"):
        # Looks like a HuggingFace dataset name
        return _load_hf_dataset(
            dataset_name=data_str,
            model_name=model_name,
            num_samples=num_samples,
            sequence_length=sequence_length,
            validator_seed=validator_seed,  # Unpredictable if set by validator
        )
    
    # Load pre-tokenized .pt file
    return torch.load(data_path, weights_only=True)


def _get_cached_model(model_path: Path | str):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    model_path_str = str(model_path) if isinstance(model_path, Path) else model_path
    
    if cached is not None and cached_path == model_path_str:
        # Reset to initial state before each evaluation
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
        return cached
    
    model = _load_model(model_path)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path_str
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _get_cached_data(
    data_path: Path | str,
    model_name: str = "Qwen/Qwen2.5-7B",
    num_samples: int = 10000,
    sequence_length: int = 1024,
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Get data from cache or load it.
    
    NOTE: For validators, we DON'T cache HuggingFace data because each
    evaluation should use different random samples (unpredictable).
    """
    data_path_str = str(data_path) if isinstance(data_path, Path) else data_path
    is_hf_dataset = "/" in data_path_str and not data_path_str.startswith("http") and not data_path_str.endswith(".pt")
    
    # For validators with HuggingFace data, always reload (different samples each time)
    if is_hf_dataset and validator_seed:
        # Don't cache - validators get fresh random samples each evaluation
        return _load_data(
            data_path,
            model_name=model_name,
            num_samples=num_samples,
            sequence_length=sequence_length,
            validator_seed=validator_seed,
        )
    
    # For miners or .pt files, use cache
    cached = _CACHE.get("data")
    cached_path = _CACHE.get("data_path")
    
    if cached is not None and cached_path == data_path_str:
        return cached
    
    data = _load_data(
        data_path,
        model_name=model_name,
        num_samples=num_samples,
        sequence_length=sequence_length,
        validator_seed=validator_seed,
    )
    _CACHE["data"] = data
    _CACHE["data_path"] = data_path_str
    return data


def _create_data_iterator(data: torch.Tensor, batch_size: int, sequence_length: int) -> Iterator[torch.Tensor]:
    """Create infinite data iterator."""
    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Unsupported data format: {type(data)}")

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
) -> InnerStepsResult:
    """Run reference implementation for comparison.
    
    Must match the expected behavior of miner's inner_steps exactly.
    Uses same settings as baseline train.py to ensure deterministic comparison.
    """
    # Match miner's deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for _ in range(num_steps):
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
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # Match miner

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
) -> tuple[bool, str | None]:
    """Verify candidate outputs match reference within tolerance."""
    # Check token count (exact match required)
    if reference.total_tokens != candidate.total_tokens:
        return (
            False,
            f"Token count mismatch: expected {reference.total_tokens}, got {candidate.total_tokens}",
        )

    # Check logits (within tolerance)
    ref_logits = reference.final_logits
    cand_logits = candidate.final_logits
    if isinstance(cand_logits, str):
        cand_logits = torch.load(cand_logits, weights_only=True)

    ref_logits = ref_logits.to(cand_logits.device)
    diff = (ref_logits - cand_logits).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    mean_abs = ref_logits.abs().mean().item()
    aggregate = mean_diff / mean_abs if mean_abs > 0 else mean_diff

    if aggregate > OUTPUT_VECTOR_TOLERANCE:
        return (
            False,
            (
                "Output logits mismatch: "
                f"mean_diff={mean_diff:.6f} max_diff={max_diff:.6f} "
                f"aggregate={aggregate:.6f} tol={OUTPUT_VECTOR_TOLERANCE}"
            ),
        )

    # Optional: verify loss
    if VERIFY_LOSS:
        loss_diff = abs(reference.final_loss - candidate.final_loss)
        if loss_diff > LOSS_TOLERANCE:
            return (
                False,
                f"Loss mismatch: expected {reference.final_loss:.6f}, "
                f"got {candidate.final_loss:.6f}, tol={LOSS_TOLERANCE}",
            )

    return True, None


class Actor:
    """Templar TPS Evaluation Actor for Affinetes.
    
    This class is called by validators to benchmark a miner's train.py submission.
    The miner's train.py is already in this Docker image at /app/train.py.
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
    ) -> dict:
        """
        Run TPS evaluation on the miner's train.py.
        
        Args:
            task_id: Evaluation run identifier
            seed: Deterministic seed string (format: "block:uid:run_idx")
            model_url: URL to download benchmark model
            data_url: URL to download benchmark data
            steps: Number of training steps to run
            batch_size: Batch size for evaluation
            timeout: Maximum seconds for evaluation
            sequence_length: Sequence length (defaults to EVAL_SEQUENCE_LENGTH)
            data_samples: Number of data samples to load (from hparams)
            
        Returns:
            Dict with: tps, total_tokens, wall_time_seconds, success, error, diagnostics
        """
        # Check train.py exists
        if train is None:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "train.py not found in Docker image",
                "seed": seed,
            }

        # Validate model/data URLs
        if not model_url or not data_url:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "missing model_url or data_url",
                "seed": seed,
            }

        # Validate code structure
        train_path = Path("/app/train.py")
        code_ok, code_error = _validate_code_structure(train_path)
        if not code_ok:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "seed": seed,
            }

        # Setup
        deadline = time.monotonic() + timeout
        seed_value = abs(hash(seed)) % (2**32)
        _set_deterministic(seed_value)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = CACHE_DIR / "model"
        data_path = CACHE_DIR / "data.pt"

        # Handle model loading
        # If model_url looks like a HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B"), use it directly
        is_hf_model = "/" in model_url and not model_url.startswith("http")
        
        if is_hf_model:
            # Use HuggingFace model ID directly
            model_dir = model_url  # This will be passed to from_pretrained
        elif not model_dir.exists():
            if time.monotonic() > deadline:
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "timeout before model download",
                    "seed": seed,
                }
            
            try:
                archive_path = CACHE_DIR / "model_download"
                _download(model_url, archive_path)
                if archive_path.is_file():
                    extracted = _maybe_extract(archive_path, model_dir)
                    model_dir = extracted if extracted.is_dir() else model_dir
            except Exception as e:
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": f"model download failed: {e}",
                    "seed": seed,
                }

        # Handle data loading - HuggingFace dataset or file download
        is_hf_dataset = "/" in data_url and not data_url.startswith("http")
        
        if is_hf_dataset:
            # Load from HuggingFace dataset (e.g., "HuggingFaceFW/fineweb")
            data_path = data_url  # Will be handled by _get_cached_data
        elif not data_path.exists():
            if time.monotonic() > deadline:
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "timeout before data download",
                    "seed": seed,
                }
            
            try:
                _download(data_url, data_path)
            except Exception as e:
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": f"data download failed: {e}",
                    "seed": seed,
                }

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "timeout before evaluation",
                "seed": seed,
            }

        try:
            # Load model and data
            # For HuggingFace datasets, pass the seed for unpredictable sampling
            model = _get_cached_model(model_dir)
            data = _get_cached_data(
                data_path,
                model_name=model_url if is_hf_model else "Qwen/Qwen2.5-7B",
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Validators pass unpredictable seed
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH

            # Run reference implementation
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)

            # Reset model to initial state
            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            reference = _run_reference(model, data_iter_ref, optimizer_ref, steps, device)

            # Reset model for miner's code
            model.load_state_dict(initial_state)

            # Run miner's code
            data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
            optimizer_miner = _create_optimizer(model)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            miner_result = train.inner_steps(
                model=model,
                data_iterator=data_iter_miner,
                optimizer=optimizer_miner,
                num_steps=steps,
                device=device,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            wall_time = time.perf_counter() - start

            # Validate return type
            ok, error, parsed = _validate_return_type(miner_result)
            if not ok or parsed is None:
                return {
                    "task_id": task_id,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": error,
                    "seed": seed,
                }

            # Verify outputs match reference
            verified, verify_error = _verify_outputs(reference, parsed)
            
            # Build diagnostics
            diagnostics = {
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
            }
            
            if reference.final_logits is not None and parsed.final_logits is not None:
                cand_logits = parsed.final_logits
                if isinstance(cand_logits, str):
                    cand_logits = torch.load(cand_logits, weights_only=True)
                ref_logits = reference.final_logits.to(cand_logits.device)
                diff = (ref_logits - cand_logits).abs()
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                mean_abs = ref_logits.abs().mean().item()
                aggregate = mean_diff / mean_abs if mean_abs > 0 else mean_diff
                diagnostics.update({
                    "logits_mean_diff": mean_diff,
                    "logits_max_diff": max_diff,
                    "logits_aggregate_diff": aggregate,
                })

            total_tokens = int(parsed.total_tokens)
            tps = float(total_tokens) / max(wall_time, 1e-6)

            return {
                "task_id": task_id,
                "tps": tps if verified else 0.0,
                "total_tokens": total_tokens if verified else 0,
                "wall_time_seconds": wall_time,
                "success": verified,
                "error": verify_error,
                "seed": seed,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            import traceback
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "seed": seed,
            }
        
        finally:
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


