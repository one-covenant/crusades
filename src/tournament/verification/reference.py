"""Reference executor for running canonical inner_steps implementation."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..schemas import BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class ReferenceResult:
    """Result from reference execution."""

    final_logits: torch.Tensor  # Logits from last forward pass
    total_tokens: int  # Total tokens processed
    final_loss: float  # Loss from last step
    initial_state: dict[str, torch.Tensor]  # Model state before training


class ReferenceExecutor:
    """Runs reference inner_steps for verification.

    This executor runs a canonical training loop that miners' code must match.
    It uses deterministic settings to ensure reproducibility.
    """

    def __init__(
        self,
        model_path: str,
        data_path: str,
        config: BenchmarkConfig,
        device: torch.device | None = None,
    ):
        """Initialize reference executor.

        Args:
            model_path: Path to the benchmark model checkpoint.
            data_path: Path to the benchmark dataset.
            config: Benchmark configuration (batch size, seq len, num steps).
            device: Device to run on (defaults to CPU to save GPU memory).
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.config = config
        # Run reference on CPU to save GPU memory for sandbox evaluation
        self.device = device or torch.device("cpu")
        
        # Cache for loaded model (load once, reuse for all evaluations)
        self._model_cache: nn.Module | None = None
        
        logger.info(f"Reference executor will use device: {self.device}")

    def _set_deterministic_mode(self, seed: int) -> None:
        """Set deterministic mode for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_model(self) -> nn.Module:
        """Load the official 7B benchmark model.
        
        Uses caching: loads once, then reuses for all evaluations.
        This saves time and memory while maintaining fairness.

        Returns:
            Loaded model on the target device in train mode.
        """
        # Return cached model if available
        if self._model_cache is not None:
            logger.info(f"Using cached reference model ({sum(p.numel() for p in self._model_cache.parameters()):,} parameters)")
            return self._model_cache
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                f"Run: uv run python scripts/setup_validator.py"
            )

        try:
            from transformers import AutoModelForCausalLM
            
            # Load model from HuggingFace format on CPU (saves GPU for sandbox)
            logger.info(f"Loading model from {self.model_path} on {self.device} (first time)")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float32 if self.device.type == "cpu" else torch.bfloat16,
                device_map="cpu" if self.device.type == "cpu" else "auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            model.train()
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded and cached: {num_params:,} parameters")
            
            # Cache for future evaluations
            self._model_cache = model
            
            return model
            
        except ImportError:
            raise ImportError("transformers library required. Run: uv sync")
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for reference execution.

        Uses AdamW with standard hyperparameters.
        """
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

    def _create_data_iterator(self) -> Iterator[torch.Tensor]:
        """Create data iterator for reference execution.

        Returns:
            Iterator yielding batches of input tensors.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        # Load pre-tokenized data
        # This is a placeholder - actual implementation depends on data format
        data = torch.load(self.data_path, weights_only=True)

        if isinstance(data, torch.Tensor):
            # Data is a single tensor, split into batches
            num_samples = data.size(0)
            batch_size = self.config.batch_size
            seq_len = self.config.sequence_length

            # Ensure data has correct sequence length
            if data.size(1) < seq_len:
                raise ValueError(f"Data sequence length {data.size(1)} < required {seq_len}")

            data = data[:, :seq_len]

            # Create iterator
            def _iter():
                idx = 0
                while True:
                    end_idx = idx + batch_size
                    if end_idx > num_samples:
                        # Wrap around
                        idx = 0
                        end_idx = batch_size
                    batch = data[idx:end_idx]
                    yield batch
                    idx = end_idx

            return _iter()

        elif isinstance(data, list):
            # Data is a list of samples
            batch_size = self.config.batch_size
            seq_len = self.config.sequence_length

            def _iter():
                idx = 0
                while True:
                    batch = []
                    for _ in range(batch_size):
                        sample = data[idx % len(data)]
                        if len(sample) >= seq_len:
                            batch.append(sample[:seq_len])
                        else:
                            # Pad if needed
                            padded = sample + [0] * (seq_len - len(sample))
                            batch.append(padded)
                        idx += 1
                    yield torch.tensor(batch, dtype=torch.long)

            return _iter()

        else:
            raise ValueError(f"Unknown data format: {type(data)}")

    def execute(self, seed: int = 42) -> ReferenceResult:
        """Execute reference training and capture expected outputs.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            ReferenceResult with expected logits, tokens, loss, and model state.
        """
        logger.info(f"Running reference execution with seed={seed}")

        # Set deterministic mode
        self._set_deterministic_mode(seed)

        # Load model
        logger.info("Loading model...")
        model = self._load_model()

        # Capture initial state
        initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        logger.info(f"Captured initial state with {len(initial_state)} parameters")

        # Create optimizer
        optimizer = self._create_optimizer(model)

        # Create data iterator
        data_iter = self._create_data_iterator()

        # Run reference inner_steps
        total_tokens = 0
        final_logits = None
        final_loss = None

        logger.info(f"Running {self.config.num_steps} training steps...")

        for step in range(self.config.num_steps):
            batch = next(data_iter)
            batch = batch.to(self.device, dtype=torch.long)

            # Forward pass with autocast for bf16
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                # Standard causal LM forward: predict next token
                input_ids = batch[:, :-1]
                labels = batch[:, 1:]

                outputs = model(input_ids)
                # Handle HuggingFace models that return objects
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Cross entropy loss
                # Use reshape instead of view to handle non-contiguous tensors
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            # Backward + optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Track metrics
            total_tokens += batch.numel()
            final_logits = logits.detach().float()  # Convert back to fp32 for comparison
            final_loss = loss.item()

            if (step + 1) % 10 == 0:
                logger.info(f"  Step {step + 1}/{self.config.num_steps}, loss={final_loss:.4f}")

        # Sync GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"Reference execution complete: {total_tokens} tokens, loss={final_loss:.4f}")

        return ReferenceResult(
            final_logits=final_logits,
            total_tokens=total_tokens,
            final_loss=final_loss,
            initial_state=initial_state,
        )

    def save_reference_outputs(
        self,
        result: ReferenceResult,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Save reference outputs to files for sandbox comparison.

        Args:
            result: Reference execution result.
            output_dir: Directory to save outputs.

        Returns:
            Dictionary mapping output names to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save logits
        logits_path = output_dir / "reference_logits.pt"
        torch.save(result.final_logits, logits_path)
        paths["logits"] = logits_path

        # Save initial state
        state_path = output_dir / "initial_state.pt"
        torch.save(result.initial_state, state_path)
        paths["initial_state"] = state_path

        # Save metadata
        metadata = {
            "total_tokens": result.total_tokens,
            "final_loss": result.final_loss,
        }
        metadata_path = output_dir / "reference_metadata.pt"
        torch.save(metadata, metadata_path)
        paths["metadata"] = metadata_path

        logger.info(f"Saved reference outputs to {output_dir}")
        return paths
