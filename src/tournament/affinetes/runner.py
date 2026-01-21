"""Affinetes runner for evaluating miner Docker images.

Supports two execution modes:
1. Local Docker mode (for testing without Basilica)
2. Basilica mode (production - remote GPU execution)
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluating a miner's submission."""
    
    success: bool
    tps: float = 0.0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    seed: str = ""
    task_id: int = 0
    diagnostics: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """Create from dictionary response."""
        return cls(
            success=data.get("success", False),
            tps=float(data.get("tps", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            wall_time_seconds=float(data.get("wall_time_seconds", 0.0)),
            error=data.get("error"),
            seed=str(data.get("seed", "")),
            task_id=int(data.get("task_id", 0)),
            diagnostics=data.get("diagnostics", {}),
        )
    
    @classmethod
    def failure(cls, error: str, task_id: int = 0) -> "EvaluationResult":
        """Create a failure result."""
        return cls(success=False, error=error, task_id=task_id)


class AffinetesRunner:
    """Runs evaluations via affinetes (Docker or Basilica).
    
    Example:
        runner = AffinetesRunner(mode="docker")
        result = await runner.evaluate(
            image="templar-submission:v1",
            seed="12345",
            model_url="https://...",
            data_url="https://...",
        )
        if result.success:
            print(f"TPS: {result.tps}")
    """
    
    def __init__(
        self,
        mode: Literal["docker", "basilica"] = "docker",
        basilica_endpoint: str | None = None,
        basilica_api_key: str | None = None,
        docker_gpu: bool = True,
        timeout: int = 600,
        model_url: str | None = None,
        data_url: str | None = None,
    ):
        """Initialize the runner.
        
        Args:
            mode: Execution mode ("docker" for local, "basilica" for remote)
            basilica_endpoint: Basilica API endpoint (required for basilica mode)
            basilica_api_key: Basilica API key (required for basilica mode)
            docker_gpu: Enable GPU in local Docker mode
            timeout: Evaluation timeout in seconds
            model_url: Default model URL for evaluations
            data_url: Default data URL for evaluations
        """
        self.mode = mode
        self.basilica_endpoint = basilica_endpoint or os.getenv("BASILICA_ENDPOINT")
        self.basilica_api_key = basilica_api_key or os.getenv("BASILICA_API_KEY")
        self.docker_gpu = docker_gpu
        self.timeout = timeout
        self.default_model_url = model_url
        self.default_data_url = data_url
        
        if mode == "basilica" and not self.basilica_endpoint:
            logger.warning("Basilica mode selected but no endpoint configured")
    
    async def evaluate(
        self,
        image: str,
        seed: str | int = 0,
        model_url: str | None = None,
        data_url: str | None = None,
        steps: int = 5,
        batch_size: int = 8,
        sequence_length: int = 1024,
        data_samples: int = 10000,
        task_id: int = 0,
    ) -> EvaluationResult:
        """Evaluate a miner's Docker image.
        
        Args:
            image: Docker image name (e.g., "templar-submission:v1")
            seed: Random seed for evaluation
            model_url: URL to benchmark model (or HuggingFace model ID)
            data_url: URL to benchmark data (or HuggingFace dataset name)
            steps: Number of training steps
            batch_size: Batch size
            sequence_length: Sequence length for evaluation
            data_samples: Number of data samples to load
            task_id: Evaluation task identifier
            
        Returns:
            EvaluationResult with TPS if successful
        """
        model_url = model_url or self.default_model_url
        data_url = data_url or self.default_data_url
        
        if not model_url or not data_url:
            return EvaluationResult.failure(
                "model_url and data_url are required",
                task_id=task_id,
            )
        
        if self.mode == "docker":
            return await self._evaluate_docker(
                image=image,
                seed=str(seed),
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                task_id=task_id,
            )
        elif self.mode == "basilica":
            return await self._evaluate_basilica(
                image=image,
                seed=str(seed),
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                task_id=task_id,
            )
        else:
            return EvaluationResult.failure(
                f"Unknown mode: {self.mode}",
                task_id=task_id,
            )
    
    async def _evaluate_docker(
        self,
        image: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation locally using Docker.
        
        This creates a container from the miner's image and calls
        the Actor.evaluate() method via a Python script.
        """
        logger.info(f"Running Docker evaluation: {image}")
        
        # Check if image exists
        check_cmd = ["docker", "image", "inspect", image]
        check_result = subprocess.run(check_cmd, capture_output=True)
        
        if check_result.returncode != 0:
            return EvaluationResult.failure(
                f"Docker image not found: {image}",
                task_id=task_id,
            )
        
        # Create evaluation script to run inside container
        eval_script = f'''
import asyncio
import json
import sys
sys.path.insert(0, '/app')

from env import Actor

async def main():
    actor = Actor()
    result = await actor.evaluate(
        task_id={task_id},
        seed="{seed}",
        model_url="{model_url}",
        data_url="{data_url}",
        steps={steps},
        batch_size={batch_size},
        sequence_length={sequence_length},
        data_samples={data_samples},
        timeout={self.timeout},
    )
    print("EVAL_RESULT:" + json.dumps(result))

asyncio.run(main())
'''
        
        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(eval_script)
            script_path = f.name
        
        try:
            # Build Docker run command
            docker_cmd = [
                "docker", "run",
                "--rm",
                "-v", f"{script_path}:/tmp/eval_script.py:ro",
            ]
            
            # Add GPU if available and requested
            if self.docker_gpu:
                docker_cmd.extend(["--gpus", "all"])
            
            # Add memory limits
            docker_cmd.extend([
                "--memory", "32g",
                "--shm-size", "8g",
            ])
            
            # Set timeout
            docker_cmd.extend([
                "--stop-timeout", str(self.timeout),
            ])
            
            # Image and command
            docker_cmd.extend([
                image,
                "python", "/tmp/eval_script.py",
            ])
            
            logger.info(f"Running: {' '.join(docker_cmd[:10])}...")
            
            # Run with timeout
            start_time = time.perf_counter()
            
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout + 30,  # Extra buffer for container startup
                )
            except asyncio.TimeoutError:
                process.kill()
                return EvaluationResult.failure(
                    f"Evaluation timed out after {self.timeout}s",
                    task_id=task_id,
                )
            
            wall_time = time.perf_counter() - start_time
            
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""
            
            if process.returncode != 0:
                error_msg = stderr_text[:500] if stderr_text else f"Exit code: {process.returncode}"
                return EvaluationResult.failure(
                    f"Container failed: {error_msg}",
                    task_id=task_id,
                )
            
            # Parse result from output
            for line in stdout_text.split('\n'):
                if line.startswith("EVAL_RESULT:"):
                    result_json = line[len("EVAL_RESULT:"):]
                    try:
                        result_data = json.loads(result_json)
                        return EvaluationResult.from_dict(result_data)
                    except json.JSONDecodeError as e:
                        return EvaluationResult.failure(
                            f"Invalid result JSON: {e}",
                            task_id=task_id,
                        )
            
            # No result found in output
            return EvaluationResult.failure(
                f"No evaluation result in output. stdout: {stdout_text[:200]}",
                task_id=task_id,
            )
            
        finally:
            # Clean up temp script
            try:
                os.unlink(script_path)
            except Exception:
                pass
    
    async def _evaluate_basilica(
        self,
        image: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation remotely via Basilica.
        
        This uses the affinetes library to call Basilica's API.
        """
        logger.info(f"Running Basilica evaluation: {image}")
        
        if not self.basilica_endpoint:
            return EvaluationResult.failure(
                "Basilica endpoint not configured",
                task_id=task_id,
            )
        
        try:
            # Try to import affinetes
            import affinetes as af
            
            # Load environment from Docker image
            env = af.load_env(
                image=image,
                endpoint=self.basilica_endpoint,
                api_key=self.basilica_api_key,
            )
            
            # Call evaluate method
            result = await env.evaluate(
                task_id=task_id,
                seed=seed,
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                timeout=self.timeout,
            )
            
            return EvaluationResult.from_dict(result)
            
        except ImportError:
            logger.warning("affinetes not installed, using HTTP fallback")
            return await self._evaluate_basilica_http(
                image=image,
                seed=seed,
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                task_id=task_id,
            )
        except Exception as e:
            return EvaluationResult.failure(
                f"Basilica error: {e}",
                task_id=task_id,
            )
    
    async def _evaluate_basilica_http(
        self,
        image: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Fallback: Call Basilica directly via HTTP."""
        try:
            import httpx
        except ImportError:
            return EvaluationResult.failure(
                "httpx not installed for Basilica HTTP fallback",
                task_id=task_id,
            )
        
        payload = {
            "image": image,
            "method": "evaluate",
            "params": {
                "task_id": task_id,
                "seed": seed,
                "model_url": model_url,
                "data_url": data_url,
                "steps": steps,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "data_samples": data_samples,
                "timeout": self.timeout,
            },
        }
        
        headers = {}
        if self.basilica_api_key:
            headers["Authorization"] = f"Bearer {self.basilica_api_key}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout + 60) as client:
                response = await client.post(
                    f"{self.basilica_endpoint}/v1/evaluate",
                    json=payload,
                    headers=headers,
                )
                
                if response.status_code != 200:
                    return EvaluationResult.failure(
                        f"Basilica API error: {response.status_code} - {response.text[:200]}",
                        task_id=task_id,
                    )
                
                result_data = response.json()
                return EvaluationResult.from_dict(result_data)
                
        except Exception as e:
            return EvaluationResult.failure(
                f"Basilica HTTP error: {e}",
                task_id=task_id,
            )
    
    async def check_image_exists(self, image: str) -> bool:
        """Check if a Docker image exists (locally or in registry).
        
        Args:
            image: Docker image name
            
        Returns:
            True if image exists
        """
        if self.mode == "docker":
            # Check local Docker
            cmd = ["docker", "image", "inspect", image]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        else:
            # For Basilica, assume image exists (Basilica will pull it)
            return True
    
    async def pull_image(self, image: str) -> bool:
        """Pull Docker image from registry.
        
        Args:
            image: Docker image name
            
        Returns:
            True if pull succeeded
        """
        if self.mode != "docker":
            return True  # Basilica handles pulling
        
        logger.info(f"Pulling image: {image}")
        cmd = ["docker", "pull", image]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to pull {image}: {result.stderr.decode()}")
            return False
        
        return True


def create_runner(
    mode: str = "docker",
    **kwargs,
) -> AffinetesRunner:
    """Factory function to create an AffinetesRunner.
    
    Args:
        mode: "docker" or "basilica"
        **kwargs: Additional arguments for AffinetesRunner
        
    Returns:
        Configured AffinetesRunner
    """
    # Load defaults from environment
    if mode == "basilica":
        kwargs.setdefault("basilica_endpoint", os.getenv("BASILICA_ENDPOINT"))
        kwargs.setdefault("basilica_api_key", os.getenv("BASILICA_API_KEY"))
    
    return AffinetesRunner(mode=mode, **kwargs)

