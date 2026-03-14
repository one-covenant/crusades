"""Multi-provider LLM client for generating train.py improvements.

Supports Chutes (default), OpenRouter, and Anthropic as LLM backends.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger("arbos.llm")

PROMPT_FILE = Path(__file__).parent / "PROMPT.md"


def _load_system_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text().strip()


@dataclass
class LLMResponse:
    reasoning: str
    code: str


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models (DeepSeek-R1 etc.)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_think_summary(raw_text: str, max_chars: int = 500) -> str:
    """Extract the last paragraph from <think> tags as a reasoning summary."""
    match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if not match:
        return ""
    thinking = match.group(1).strip()
    paragraphs = [p.strip() for p in thinking.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    summary = paragraphs[-1]
    return summary[:max_chars]


def _parse_response(raw_text: str) -> LLMResponse:
    """Extract reasoning and code from LLM response."""
    think_summary = _extract_think_summary(raw_text)
    text = _strip_thinking_tags(raw_text)
    reasoning = ""
    code = ""

    reasoning_match = re.search(r"REASONING:\s*\n?(.*?)(?=\nCODE:|\n```)", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    if not reasoning:
        reasoning = think_summary

    # Find all python code blocks; pick the one after CODE: marker, or the largest
    all_blocks = list(re.finditer(r"```python\s*\n(.*?)```", text, re.DOTALL))
    if all_blocks:
        code_section = re.search(r"CODE:\s*\n(.*)", text, re.DOTALL)
        if code_section:
            # Find the first code block that starts within or after the CODE: section
            code_start = code_section.start()
            for block in all_blocks:
                if block.start() >= code_start:
                    code = block.group(1).strip()
                    break
        if not code:
            # Fall back to the largest block (most likely the full file)
            code = max((b.group(1).strip() for b in all_blocks), key=len)
    elif "def inner_steps" in text:
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith(("import ", "from ", "def ", "class ", "@")):
                in_code = True
            if in_code:
                code_lines.append(line)
        code = "\n".join(code_lines).strip()

    if not code:
        raise ValueError("Could not extract code from LLM response")

    return LLMResponse(reasoning=reasoning or "No reasoning provided", code=code)


class LLMClient:
    """Multi-provider LLM client."""

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "chutes")
        if self.provider not in ("chutes", "openrouter", "anthropic"):
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Must be one of: chutes, openrouter, anthropic"
            )
        self.timeout = int(os.environ.get("LLM_TIMEOUT", "600"))

        if self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self.base_url = "https://api.anthropic.com/v1"
            self.model = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
        elif self.provider == "openrouter":
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")
        else:
            self.api_key = os.environ.get("CHUTES_API_KEY", "")
            self.base_url = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
            self.model = os.environ.get(
                "LLM_MODEL",
                "deepseek-ai/DeepSeek-R1-0528-TEE",
            )

        if not self.api_key:
            key_name = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }.get(self.provider, "CHUTES_API_KEY")
            raise ValueError(f"{key_name} not set. Set it in .env or environment.")

        logger.info(f"LLM provider: {self.provider} ({self.model})")

    def _call_anthropic(self, messages: list[dict]) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 8192,
            "system": _load_system_prompt(),
            "messages": messages,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/messages", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    def _call_openai_compat(self, messages: list[dict]) -> str:
        """Call OpenAI-compatible API (Chutes, OpenRouter)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": _load_system_prompt()},
                *messages,
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _call(self, messages: list[dict]) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai_compat(messages)

    def suggest_improvement(
        self,
        code: str,
        mfu: float,
        history: str,
        hparams: dict,
    ) -> LLMResponse:
        """Ask the LLM to suggest one improvement to the train.py code."""
        num_gpus = (hparams.get("docker") or {}).get("num_gpus", 2)
        user_msg = f"""## Current best train.py (MFU: {mfu:.2f}%)

```python
{code}
```

## Evaluation environment
- Model: {hparams["benchmark_model_name"]}
- Batch size: {hparams["benchmark_batch_size"]}
- Sequence length: {hparams["benchmark_sequence_length"]}
- Steps: {hparams["eval_steps"]}
- GPUs: {num_gpus} x A100 80GB SXM (NVLink connected)
- Peak FLOPs: 312 TFLOPS per GPU (bf16)

## Key constraints
- With only {num_gpus} GPUs, FULL_SHARD adds more communication than it saves. SHARD_GRAD_OP is usually better.
- The evaluator measures wall-clock time for all {hparams["eval_steps"]} steps. First-step compilation overhead hurts MFU.
- Loss must stay within 0.3 of reference — aggressive lr/precision changes can cause loss_mismatch failures.
- Code MUST return final_state (full state dict) for verification.

## Previous attempts — CRITICAL, read carefully
{history}

## Your task
Beat the current best MFU of {mfu:.2f}%. Suggest ONE focused, NOVEL improvement.

Rules:
1. Do NOT repeat any strategy from the history above — each must be genuinely different.
2. Study the current code carefully. Only change what is necessary for your optimization.
3. Keep working code patterns intact (e.g. FSDP wrapping, state dict gathering).
4. Prefer small, targeted changes over full rewrites.
5. Return the complete updated train.py file."""

        messages = [{"role": "user", "content": user_msg}]

        start = time.time()
        logger.info(f"Requesting improvement from {self.provider} ({self.model})...")

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                text = self._call(messages)
                elapsed = time.time() - start
                logger.info(f"LLM responded in {elapsed:.1f}s")
                return _parse_response(text)
            except httpx.ReadTimeout:
                logger.warning(
                    f"LLM read timeout (attempt {attempt}/{max_retries}, timeout={self.timeout}s)"
                )
                if attempt == max_retries:
                    raise
                time.sleep(10)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                detail = ""
                try:
                    body = e.response.json()
                    detail = body.get("detail", {})
                    if isinstance(detail, dict):
                        detail = detail.get("message", str(detail))
                except Exception:
                    detail = e.response.text[:200]

                if status == 402:
                    raise RuntimeError(
                        f"LLM API payment required (HTTP 402): {detail}. "
                        f"Your {self.provider} account has no balance. "
                        f"Add funds or switch LLM_PROVIDER in arbos/.env"
                    ) from e

                if status == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' not found on {self.provider} (HTTP 404). "
                        f"Check available models or set LLM_MODEL in arbos/.env"
                    ) from e

                logger.warning(
                    f"LLM API error (attempt {attempt}/{max_retries}): HTTP {status} — {detail}"
                )
                if status not in (429, 500, 502, 503, 504):
                    raise
                if attempt == max_retries:
                    raise
                wait = 30 * attempt if status == 429 else 2**attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            except ValueError as e:
                logger.warning(
                    f"Failed to parse LLM response (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt == max_retries:
                    raise
                time.sleep(2)
