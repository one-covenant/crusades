#!/usr/bin/env python3
"""
Sandbox runner script - executes inside Docker container.

This script:
1. Loads the benchmark config
2. Imports and validates the miner's train.py
3. Runs the miner's inner_steps function
4. Captures output logits and loss for verification
5. Writes results to /output/result.json
"""

import importlib.util
import json
import sys
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InnerStepsResult:
    """Result from miner's inner_steps function."""

    final_logits: "torch.Tensor"  # noqa: F821
    total_tokens: int
    final_loss: float


def load_module(path: Path, name: str = "train"):
    """Dynamically load a Python module from path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_result(
    output_dir: Path,
    total_tokens: int,
    num_steps: int,
    success: bool,
    error: str | None = None,
    final_loss: float | None = None,
    final_logits_path: str | None = None,
):
    """Write result to output file."""
    result = {
        "total_tokens": total_tokens,
        "num_steps": num_steps,
        "success": success,
        "error": error,
        "final_loss": final_loss,
        "final_logits_path": final_logits_path,
    }
    output_file = output_dir / "result.json"
    output_file.write_text(json.dumps(result))


def create_data_iterator(
    data_path: str,
    batch_size: int,
    sequence_length: int,
) -> Iterator["torch.Tensor"]:  # noqa: F821
    """Create a data iterator from the benchmark data.

    Args:
        data_path: Path to pre-tokenized data file.
        batch_size: Batch size for training.
        sequence_length: Sequence length for training.

    Yields:
        Batches of input tensors.
    """
    import torch

    data = torch.load(data_path, weights_only=True)

    if isinstance(data, torch.Tensor):
        # Data is a single tensor
        num_samples = data.size(0)
        if data.size(1) < sequence_length:
            raise ValueError(f"Data sequence length {data.size(1)} < required {sequence_length}")
        data = data[:, :sequence_length]

        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > num_samples:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")


def main():
    output_dir = Path("/output")
    sandbox_dir = Path("/sandbox")

    try:
        # Load config
        config_path = sandbox_dir / "config.json"
        if not config_path.exists():
            write_result(output_dir, 0, 0, False, "Config file not found")
            return 1

        config = json.loads(config_path.read_text())
        model_path = config["model_path"]
        data_path = config["data_path"]
        sequence_length = config["sequence_length"]
        batch_size = config["batch_size"]
        num_steps = config["num_steps"]
        random_seed = config["random_seed"]

        # Load miner's training code
        train_path = sandbox_dir / "train.py"
        if not train_path.exists():
            write_result(output_dir, 0, 0, False, "train.py not found")
            return 1

        train_module = load_module(train_path)

        # Validate required function exists
        if not hasattr(train_module, "inner_steps"):
            write_result(
                output_dir,
                0,
                0,
                False,
                "Missing required function: inner_steps\n"
                "Your train.py must define:\n"
                "  def inner_steps(model, data_iterator, optimizer, num_steps, device) -> InnerStepsResult",
            )
            return 1

        # Set up deterministic mode
        import torch

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        else:
            write_result(output_dir, 0, 0, False, f"Unknown model format at {model_path}")
            return 1

        model = model.to(device)
        model.train()
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Load initial model state if provided (for verification)
        initial_state_path = sandbox_dir / "initial_state.pt"
        if initial_state_path.exists():
            print("Loading initial model state for verification...")
            initial_state = torch.load(initial_state_path, map_location=device, weights_only=True)
            model.load_state_dict(initial_state)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        # Create data iterator
        print(f"Loading data from {data_path}...")
        data_iterator = create_data_iterator(data_path, batch_size, sequence_length)

        # Run miner's inner_steps
        print(f"Running inner_steps for {num_steps} steps...")
        result = train_module.inner_steps(
            model=model,
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_steps=num_steps,
            device=device,
        )

        # Validate result
        if not isinstance(result, InnerStepsResult):
            # Try to convert if miner returned a compatible object
            if (
                hasattr(result, "final_logits")
                and hasattr(result, "total_tokens")
                and hasattr(result, "final_loss")
            ):
                result = InnerStepsResult(
                    final_logits=result.final_logits,
                    total_tokens=result.total_tokens,
                    final_loss=result.final_loss,
                )
            else:
                write_result(
                    output_dir,
                    0,
                    0,
                    False,
                    f"Invalid return type from inner_steps: {type(result)}\n"
                    f"Expected InnerStepsResult with final_logits, total_tokens, final_loss",
                )
                return 1

        # Sync GPU before saving
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Save logits for verification
        logits_path = output_dir / "final_logits.pt"
        torch.save(result.final_logits.cpu(), logits_path)

        print(f"Completed: {result.total_tokens:,} tokens, loss={result.final_loss:.4f}")

        write_result(
            output_dir,
            total_tokens=result.total_tokens,
            num_steps=num_steps,
            success=True,
            final_loss=result.final_loss,
            final_logits_path=str(logits_path),
        )
        return 0

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        write_result(output_dir, 0, 0, False, error_msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
