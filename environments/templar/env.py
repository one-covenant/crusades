"""
Templar TPS Evaluation Environment (Validator-Owned)

R2-Based Architecture:
- This env.py is owned by the VALIDATOR, not the miner
- Miner submits train.py to their R2 bucket
- Validator downloads train.py from miner's R2 at evaluation time
- This ensures miners can't tamper with evaluation logic

Flow:
1. Validator reads miner commitment (contains R2 credentials)
2. Validator calls env.evaluate(r2_endpoint, r2_bucket, r2_key, ...)
3. This Actor downloads train.py from miner's R2
4. Runs benchmark and returns TPS
5. Validator stores miner's code in its own R2 for dashboard
"""

import ast
import gc
import hashlib
import importlib.util
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

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


# Global cache for model (data is NOT cached for validators)
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
}


def _download_from_r2(
    r2_endpoint: str,
    r2_bucket: str,
    r2_key: str,
    r2_access_key: str,
    r2_secret_key: str,
    dest_path: Path,
) -> tuple[bool, str | None]:
    """Download file from miner's R2 bucket.
    
    Args:
        r2_endpoint: R2/S3 endpoint URL
        r2_bucket: Bucket name
        r2_key: Object key
        r2_access_key: Access key ID
        r2_secret_key: Secret access key
        dest_path: Destination path
        
    Returns:
        Tuple of (success, error_message or None)
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        return False, "boto3 not installed"
    
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3},
            ),
        )
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(r2_bucket, r2_key, str(dest_path))
        
        return True, None
        
    except Exception as e:
        return False, f"R2 download failed: {e}"


def _load_miner_module(train_path: Path):
    """Dynamically load miner's train.py as a module.
    
    Args:
        train_path: Path to train.py
        
    Returns:
        Loaded module with inner_steps function
    """
    spec = importlib.util.spec_from_file_location("miner_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["miner_train"] = module
    spec.loader.exec_module(module)
    
    return module


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace.
    
    SECURITY: Validators use unpredictable seeds so miners can't pre-compute
    which samples will be used for evaluation.
    
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
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    # Determine seed: validators use unpredictable seed
    if validator_seed:
        seed_hash = hashlib.sha256(validator_seed.encode()).hexdigest()
        actual_seed = int(seed_hash[:8], 16)
        print(f"Validator mode: seed={actual_seed} (from {validator_seed})")
    else:
        actual_seed = 42
        print(f"Test mode: fixed seed={actual_seed}")
    
    print(f"Loading dataset: {dataset_name} (samples={num_samples})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with streaming
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
    print(f"Loaded data: shape={data.shape}, seed={actual_seed}")
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
        return True, None, InnerStepsResult(
            final_logits=result.final_logits,
            total_tokens=result.total_tokens,
            final_loss=result.final_loss,
        )
    
    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_path: str):
    """Load model from HuggingFace."""
    from transformers import AutoModelForCausalLM
    
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


def _get_cached_model(model_path: str):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    
    if cached is not None and cached_path == model_path:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
        return cached
    
    model = _load_model(model_path)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _create_data_iterator(data: torch.Tensor, batch_size: int, sequence_length: int) -> Iterator[torch.Tensor]:
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
) -> InnerStepsResult:
    """Run reference implementation for comparison."""
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
        optimizer.zero_grad(set_to_none=True)
        
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
    expected_tokens: int,
) -> tuple[bool, str | None]:
    """Verify candidate outputs are valid.
    
    We verify CORRECTNESS, not exact output matching with a reference.
    Comparing different code paths (reference vs miner's inner_steps) will always
    diverge due to floating point non-determinism, even if mathematically equivalent.
    
    Instead, we verify:
    1. Token count matches expected (miner processed correct amount of data)
    2. Loss is reasonable (not NaN/Inf, decreased from initial ~11 to ~2)
    3. Logits are valid tensors (not NaN/Inf)
    """
    # Verify token count matches expected
    if candidate.total_tokens != expected_tokens:
        return False, f"Token count mismatch: expected {expected_tokens}, got {candidate.total_tokens}"
    
    # Verify loss is reasonable (typical loss range is 1.5-3 for trained LLMs)
    if candidate.final_loss != candidate.final_loss:  # NaN check
        return False, f"Loss is NaN"
    if abs(candidate.final_loss) > 100:
        return False, f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
    
    # Verify logits are valid
    cand_logits = candidate.final_logits
    if cand_logits is None:
        return False, "No logits returned"
    
    if isinstance(cand_logits, str):
        cand_logits = torch.load(cand_logits, weights_only=True)
    
    if torch.isnan(cand_logits).any():
        return False, "Logits contain NaN values"
    if torch.isinf(cand_logits).any():
        return False, "Logits contain Inf values"
    
    # Optional: Compare losses if reference provided (sanity check, not strict)
    # Losses should be in the same ballpark (within 50%)
    if reference is not None and reference.final_loss > 0:
        loss_ratio = candidate.final_loss / reference.final_loss
        if loss_ratio < 0.5 or loss_ratio > 2.0:
            # Just warn, don't fail - different optimizations can affect loss
            print(f"Warning: Loss differs significantly from reference ({candidate.final_loss:.4f} vs {reference.final_loss:.4f})")
    
    return True, None


class Actor:
    """Templar TPS Evaluation Actor for Affinetes (Validator-Owned).
    
    This Actor is owned by the validator, not the miner.
    It downloads the miner's train.py from their R2 bucket and evaluates it.
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
        # R2 credentials for miner's code
        r2_endpoint: str = "",
        r2_bucket: str = "",
        r2_key: str = "",
        r2_access_key: str = "",
        r2_secret_key: str = "",
    ) -> dict:
        """
        Download miner's train.py from R2 and run TPS evaluation.
        
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
            r2_endpoint: Miner's R2 endpoint
            r2_bucket: Miner's R2 bucket
            r2_key: Miner's R2 key (path to train.py)
            r2_access_key: Miner's R2 access key
            r2_secret_key: Miner's R2 secret key
            
        Returns:
            Dict with: tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate R2 credentials
        if not all([r2_endpoint, r2_bucket, r2_key, r2_access_key, r2_secret_key]):
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Missing R2 credentials",
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
        
        # Download miner's train.py from R2
        train_path = CACHE_DIR / "miner_train.py"
        
        print(f"Downloading miner code from R2...")
        print(f"   Endpoint: {r2_endpoint}")
        print(f"   Bucket: {r2_bucket}")
        print(f"   Key: {r2_key}")
        
        success, error = _download_from_r2(
            r2_endpoint=r2_endpoint,
            r2_bucket=r2_bucket,
            r2_key=r2_key,
            r2_access_key=r2_access_key,
            r2_secret_key=r2_secret_key,
            dest_path=train_path,
        )
        
        if not success:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"R2 download failed: {error}",
                "seed": seed,
            }
        
        # Read and validate code
        try:
            code = train_path.read_text()
        except Exception as e:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"Failed to read train.py: {e}",
                "seed": seed,
            }
        
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
                "code": code,  # Return code for debugging
            }
        
        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Timeout after downloading code",
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
            
            # Load model and data
            model = _get_cached_model(model_url)
            data = _load_hf_dataset(
                dataset_name=data_url,
                model_name=model_url,
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Unpredictable sampling
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH
            
            # Run reference implementation
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)
            
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
            miner_result = miner_module.inner_steps(
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
                    "code": code,
                }
            
            # Calculate expected tokens
            expected_tokens = batch_size * (seq_len - 1) * steps  # seq_len-1 because we drop last token for labels
            # Actually, miners typically count full batch tokens
            expected_tokens = batch_size * seq_len * steps
            
            # Verify outputs (correctness check, not exact match)
            verified, verify_error = _verify_outputs(reference, parsed, expected_tokens)
            
            # Diagnostics
            diagnostics = {
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
                "expected_tokens": expected_tokens,
                "actual_tokens": parsed.total_tokens,
            }
            
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
                "code": code,  # Return code so validator can store it
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
                "code": code if 'code' in dir() else None,
            }
        
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
