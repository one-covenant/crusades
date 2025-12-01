"""Verbose verification error types with actionable feedback for miners."""

import torch


class VerificationError(Exception):
    """Base verification error with verbose messaging."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class LogitsMismatchError(VerificationError):
    """Output logits don't match expected values."""

    def __init__(self, expected: torch.Tensor, actual: torch.Tensor, tolerance: float = 1e-3):
        diff = (expected - actual).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Find location of max difference for debugging
        max_idx = diff.argmax().item()
        flat_expected = expected.flatten()
        flat_actual = actual.flatten()

        message = (
            f"VERIFICATION FAILED: Output logits don't match expected values.\n"
            f"\n"
            f"  Statistics:\n"
            f"    Max difference:  {max_diff:.6f}\n"
            f"    Mean difference: {mean_diff:.6f}\n"
            f"    Tolerance:       {tolerance}\n"
            f"\n"
            f"  Shapes:\n"
            f"    Expected: {tuple(expected.shape)}\n"
            f"    Actual:   {tuple(actual.shape)}\n"
            f"\n"
            f"  Sample values at max difference (index {max_idx}):\n"
            f"    Expected: {flat_expected[max_idx].item():.6f}\n"
            f"    Actual:   {flat_actual[max_idx].item():.6f}\n"
            f"\n"
            f"  Possible causes:\n"
            f"    - Using different precision (fp16 vs bf16 vs fp32)\n"
            f"    - Non-deterministic operations (set torch.backends.cudnn.deterministic = True)\n"
            f"    - Incorrect forward pass implementation\n"
            f"    - Modified model architecture\n"
            f"    - Gradient checkpointing with different segment boundaries\n"
            f"\n"
            f"  Suggested fixes:\n"
            f"    1. Ensure you're using bf16 autocast: torch.autocast('cuda', dtype=torch.bfloat16)\n"
            f"    2. Set deterministic mode at the start of your inner_steps function\n"
            f"    3. Verify your forward pass matches the reference implementation"
        )
        super().__init__(message)


class TokenCountMismatchError(VerificationError):
    """Token count doesn't match expected."""

    def __init__(self, expected: int, actual: int):
        diff = actual - expected
        diff_pct = (diff / expected * 100) if expected > 0 else float("inf")

        message = (
            f"VERIFICATION FAILED: Token count mismatch.\n"
            f"\n"
            f"  Token counts:\n"
            f"    Expected: {expected:,}\n"
            f"    Actual:   {actual:,}\n"
            f"    Difference: {diff:+,} ({diff_pct:+.2f}%)\n"
            f"\n"
            f"  Possible causes:\n"
            f"    - Skipping batches in the training loop\n"
            f"    - Incorrect batch size calculation\n"
            f"    - Early termination before completing all steps\n"
            f"    - Processing fewer steps than requested\n"
            f"    - Incorrect token counting (should be batch.numel())\n"
            f"\n"
            f"  Suggested fixes:\n"
            f"    1. Ensure you process all num_steps iterations\n"
            f"    2. Count tokens as: total_tokens += batch.numel()\n"
            f"    3. Don't skip any batches from the data iterator"
        )
        super().__init__(message)


class LossMismatchError(VerificationError):
    """Loss value doesn't match expected."""

    def __init__(self, expected: float, actual: float, tolerance: float = 1e-3):
        diff = abs(actual - expected)
        diff_pct = (diff / abs(expected) * 100) if expected != 0 else float("inf")

        message = (
            f"VERIFICATION FAILED: Loss value mismatch.\n"
            f"\n"
            f"  Loss values:\n"
            f"    Expected: {expected:.6f}\n"
            f"    Actual:   {actual:.6f}\n"
            f"    Difference: {diff:.6f} ({diff_pct:.2f}%)\n"
            f"    Tolerance: {tolerance}\n"
            f"\n"
            f"  Possible causes:\n"
            f"    - Modified loss function\n"
            f"    - Incorrect gradient accumulation\n"
            f"    - Precision issues (fp16/bf16 differences)\n"
            f"    - Label preparation differs from reference\n"
            f"    - Different token masking strategy\n"
            f"\n"
            f"  Suggested fixes:\n"
            f"    1. Use standard cross_entropy_loss for next-token prediction\n"
            f"    2. Ensure labels are shifted correctly: labels = input_ids[:, 1:]\n"
            f"    3. Verify loss is not scaled (no gradient accumulation division)"
        )
        super().__init__(message)


class SandboxExecutionError(VerificationError):
    """Sandbox execution failed."""

    def __init__(
        self,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
        error: str | None = None,
    ):
        # Truncate outputs if too long
        max_output_len = 2000
        stdout_truncated = (
            stdout[:max_output_len] + "..." if len(stdout) > max_output_len else stdout
        )
        stderr_truncated = (
            stderr[:max_output_len] + "..." if len(stderr) > max_output_len else stderr
        )

        message = f"VERIFICATION FAILED: Sandbox execution error.\n\n  Exit code: {exit_code}\n"

        if error:
            message += f"  Error: {error}\n"

        if stderr_truncated:
            message += f"\n  Stderr:\n{_indent(stderr_truncated, 4)}\n"

        if stdout_truncated:
            message += f"\n  Stdout:\n{_indent(stdout_truncated, 4)}\n"

        message += (
            "\n"
            "  Possible causes:\n"
            "    - Syntax error in train.py\n"
            "    - Missing inner_steps function\n"
            "    - Runtime exception during execution\n"
            "    - Out of memory error\n"
            "    - Import error (missing dependencies)\n"
            "\n"
            "  Suggested fixes:\n"
            "    1. Test your code locally before submitting\n"
            "    2. Ensure inner_steps function is defined\n"
            "    3. Check for CUDA out of memory errors"
        )
        super().__init__(message)


class MissingFunctionError(VerificationError):
    """Required function is missing from submission."""

    def __init__(self, function_name: str, available_functions: list[str] | None = None):
        available_str = ", ".join(available_functions) if available_functions else "none found"

        message = (
            f"VERIFICATION FAILED: Missing required function '{function_name}'.\n"
            f"\n"
            f"  Available functions: {available_str}\n"
            f"\n"
            f"  Required signature:\n"
            f"    def inner_steps(\n"
            f"        model: torch.nn.Module,\n"
            f"        data_iterator: Iterator[torch.Tensor],\n"
            f"        optimizer: torch.optim.Optimizer,\n"
            f"        num_steps: int,\n"
            f"        device: torch.device,\n"
            f"    ) -> InnerStepsResult:\n"
            f"        ...\n"
            f"\n"
            f"  Suggested fixes:\n"
            f"    1. Define the inner_steps function in your train.py\n"
            f"    2. Ensure the function name is exactly 'inner_steps'\n"
            f"    3. Check for typos in the function name"
        )
        super().__init__(message)


class InvalidReturnTypeError(VerificationError):
    """Return type from inner_steps is invalid."""

    def __init__(self, expected_type: str, actual_type: str, actual_value: str | None = None):
        message = (
            f"VERIFICATION FAILED: Invalid return type from inner_steps.\n"
            f"\n"
            f"  Expected: {expected_type}\n"
            f"  Actual:   {actual_type}\n"
        )

        if actual_value:
            message += (
                f"  Value:    {actual_value[:200]}...\n"
                if len(actual_value) > 200
                else f"  Value:    {actual_value}\n"
            )

        message += (
            "\n"
            "  Required return type:\n"
            "    @dataclass\n"
            "    class InnerStepsResult:\n"
            "        final_logits: torch.Tensor\n"
            "        total_tokens: int\n"
            "        final_loss: float\n"
            "\n"
            "  Suggested fixes:\n"
            "    1. Import InnerStepsResult from the schemas module\n"
            "    2. Return InnerStepsResult with all required fields\n"
            "    3. Ensure final_logits is a torch.Tensor"
        )
        super().__init__(message)


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text by the specified number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))
