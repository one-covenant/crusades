#!/usr/bin/env python3
"""
HTTP server wrapper for the sandbox runner.

Exposes the sandbox runner functionality via HTTP endpoints:
- GET /health - Health check with GPU info
- POST /evaluate - Submit training code for evaluation

The server runs training code directly on the GPU and returns metrics.
"""

import importlib.util
import json
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Templar Tournament Sandbox",
    description="GPU sandbox for evaluating training code efficiency",
    version="1.0.0",
)


@dataclass
class InnerStepsResult:
    """Result from miner's inner_steps function."""
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


class EvaluateRequest(BaseModel):
    """Request to evaluate training code."""
    code: str
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    num_steps: int = 10
    batch_size: int = 4
    sequence_length: int = 512
    random_seed: int = 42


class EvaluateResponse(BaseModel):
    """Response from evaluation."""
    success: bool
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    final_loss: float | None = None
    error: str | None = None


def load_module_from_string(code: str, name: str = "train") -> Any:
    """Load a Python module from code string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location(name, temp_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load module from code")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        Path(temp_path).unlink(missing_ok=True)


def create_data_iterator(
    batch_size: int,
    sequence_length: int,
    vocab_size: int = 32000,
) -> Iterator[torch.Tensor]:
    """Create synthetic data iterator for training."""
    while True:
        yield torch.randint(0, vocab_size, (batch_size, sequence_length))


@app.get("/health")
async def health():
    """Health check with GPU information."""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None
    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        gpu_memory = f"{props.total_memory / 1024**3:.1f}GB"

    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """Evaluate training code and return metrics."""
    start_time = time.perf_counter()

    try:
        train_module = load_module_from_string(request.code)

        if not hasattr(train_module, "inner_steps"):
            return EvaluateResponse(
                success=False,
                error="Missing required function: inner_steps. "
                      "Code must define: def inner_steps(model, data_iterator, optimizer, num_steps, device) -> InnerStepsResult",
            )

        torch.manual_seed(request.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            request.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        data_iterator = create_data_iterator(
            request.batch_size,
            request.sequence_length,
        )

        result = train_module.inner_steps(
            model=model,
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_steps=request.num_steps,
            device=device,
        )

        if not hasattr(result, "total_tokens") or not hasattr(result, "final_loss"):
            return EvaluateResponse(
                success=False,
                error="Invalid return from inner_steps. Expected object with total_tokens and final_loss.",
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_time = time.perf_counter() - start_time
        tps = result.total_tokens / wall_time if wall_time > 0 else 0.0

        return EvaluateResponse(
            success=True,
            tokens_per_second=tps,
            total_tokens=result.total_tokens,
            wall_time_seconds=wall_time,
            final_loss=float(result.final_loss),
        )

    except Exception as e:
        wall_time = time.perf_counter() - start_time
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return EvaluateResponse(
            success=False,
            wall_time_seconds=wall_time,
            error=error_msg,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
