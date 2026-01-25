"""Affinetes runner for evaluating miner submissions.

URL-Based Architecture:
- Validator downloads train.py from miner's committed URL
- Code is passed directly to the evaluation environment
- No R2 credentials needed - just HTTP access to the URL

Execution modes:
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
from typing import Literal

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
    code: str | None = None  # Miner's code for storage
    
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
            code=data.get("code"),
        )
    
    @classmethod
    def failure(cls, error: str, task_id: int = 0) -> "EvaluationResult":
        """Create a failure result."""
        return cls(success=False, error=error, task_id=task_id)


class AffinetesRunner:
    """Runs evaluations via affinetes (Docker or Basilica).
    
    URL-Based Architecture:
    - Validator downloads code from miner's committed URL
    - Code is passed directly to the evaluation container
    - No R2 credentials or downloads inside container
    
    Example:
        runner = AffinetesRunner(mode="docker")
        result = await runner.evaluate(
            code="def inner_steps(...): ...",
            seed="12345",
        )
        if result.success:
            print(f"TPS: {result.tps}")
    """
    
    # Validator's standard evaluation image
    VALIDATOR_IMAGE = os.getenv("VALIDATOR_EVAL_IMAGE", "templar-eval:latest")
    
    def __init__(
        self,
        mode: Literal["docker", "basilica"] = "docker",
        basilica_endpoint: str | None = None,
        basilica_api_key: str | None = None,
        docker_gpu_devices: str = "all",
        docker_memory_limit: str = "32g",
        docker_shm_size: str = "8g",
        timeout: int = 600,
        model_url: str | None = None,
        data_url: str | None = None,
        output_tolerance: float = 0.02,
        validator_image: str | None = None,
    ):
        """Initialize the runner.
        
        Args:
            mode: Execution mode ("docker" for local, "basilica" for remote)
            basilica_endpoint: Basilica API endpoint
            basilica_api_key: Basilica API key
            docker_gpu_devices: GPU devices for Docker ("all", "0", "0,1", "none")
            docker_memory_limit: Docker memory limit (e.g., "32g")
            docker_shm_size: Shared memory size for Docker (e.g., "8g")
            timeout: Evaluation timeout in seconds
            model_url: Default model URL (HuggingFace model ID)
            data_url: Default data URL (HuggingFace dataset)
            output_tolerance: Verification tolerance (0.02 = 2%)
            validator_image: Validator's Docker image name
        """
        self.mode = mode
        self.basilica_endpoint = basilica_endpoint or os.getenv("BASILICA_ENDPOINT")
        self.basilica_api_key = basilica_api_key or os.getenv("BASILICA_API_KEY")
        self.docker_gpu_devices = docker_gpu_devices
        self.docker_memory_limit = docker_memory_limit
        self.docker_shm_size = docker_shm_size
        self.timeout = timeout
        self.default_model_url = model_url
        self.default_data_url = data_url
        self.output_tolerance = output_tolerance
        self.validator_image = validator_image or self.VALIDATOR_IMAGE
        
        if mode == "basilica" and not self.basilica_endpoint:
            logger.warning("Basilica mode selected but no endpoint configured")
    
    async def evaluate(
        self,
        code: str,
        seed: str | int = 0,
        model_url: str | None = None,
        data_url: str | None = None,
        steps: int = 5,
        batch_size: int = 8,
        sequence_length: int = 1024,
        data_samples: int = 10000,
        task_id: int = 0,
    ) -> EvaluationResult:
        """Evaluate a miner's train.py code.
        
        Args:
            code: Miner's train.py code (already downloaded from URL)
            seed: Random seed for evaluation
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            sequence_length: Sequence length
            data_samples: Number of data samples
            task_id: Evaluation task identifier
            
        Returns:
            EvaluationResult with TPS score
        """
        model_url = model_url or self.default_model_url
        data_url = data_url or self.default_data_url
        
        if not model_url or not data_url:
            return EvaluationResult.failure(
                "model_url and data_url are required",
                task_id=task_id,
            )
        
        if not code or "def inner_steps" not in code:
            return EvaluationResult.failure(
                "Invalid code: must contain 'def inner_steps' function",
                task_id=task_id,
            )
        
        if self.mode == "docker":
            return await self._evaluate_docker(
                code=code,
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
                code=code,
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
        code: str,
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
        
        Code is mounted directly into the container - no downloads needed.
        """
        logger.info(f"Running Docker evaluation")
        logger.info(f"   Code size: {len(code)} bytes")
        
        # Check if validator image exists
        check_cmd = ["docker", "image", "inspect", self.validator_image]
        check_result = subprocess.run(check_cmd, capture_output=True)
        
        if check_result.returncode != 0:
            return EvaluationResult.failure(
                f"Validator image not found: {self.validator_image}. "
                f"Build it first: cd environments/templar && docker build -t {self.validator_image} .",
                task_id=task_id,
            )
        
        # Write miner's code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            prefix='train_',
        ) as f:
            f.write(code)
            train_path = f.name
        
        # Create evaluation script that reads code from mounted file
        eval_script = f'''
import asyncio
import json
import sys
sys.path.insert(0, '/app')

from env import Actor

async def main():
    # Read miner's code
    with open('/tmp/miner_train.py') as f:
        code = f.read()
    
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
        code=code,
    )
    print("EVAL_RESULT:" + json.dumps(result))

asyncio.run(main())
'''
        
        # Write eval script to temp file
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
                "-v", f"{train_path}:/tmp/miner_train.py:ro",
            ]
            
            # Add GPU if configured
            if self.docker_gpu_devices and self.docker_gpu_devices.lower() != "none":
                if self.docker_gpu_devices.lower() == "all":
                    docker_cmd.extend(["--gpus", "all"])
                else:
                    # Specific devices like "0" or "0,1"
                    docker_cmd.extend(["--gpus", f'"device={self.docker_gpu_devices}"'])
            
            # Memory limits (from hparams.json docker config)
            docker_cmd.extend([
                "--memory", self.docker_memory_limit,
                "--shm-size", self.docker_shm_size,
            ])
            
            # Environment variables
            docker_cmd.extend([
                "-e", f"OUTPUT_VECTOR_TOLERANCE={self.output_tolerance}",
            ])
            
            # Timeout
            docker_cmd.extend([
                "--stop-timeout", str(self.timeout),
            ])
            
            # Image and command
            docker_cmd.extend([
                self.validator_image,
                "python", "/tmp/eval_script.py",
            ])
            
            logger.info(f"Running evaluation in {self.validator_image}...")
            
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
                    timeout=self.timeout + 60,
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
            
            # Parse result
            for line in stdout_text.split('\n'):
                if line.startswith("EVAL_RESULT:"):
                    result_json = line[len("EVAL_RESULT:"):]
                    try:
                        result_data = json.loads(result_json)
                        result = EvaluationResult.from_dict(result_data)
                        result.code = code  # Include code in result
                        return result
                    except json.JSONDecodeError as e:
                        return EvaluationResult.failure(
                            f"Invalid result JSON: {e}",
                            task_id=task_id,
                        )
            
            return EvaluationResult.failure(
                f"No evaluation result in output. stdout: {stdout_text[:200]}",
                task_id=task_id,
            )
            
        finally:
            try:
                os.unlink(script_path)
                os.unlink(train_path)
            except Exception:
                pass
    
    async def _evaluate_basilica(
        self,
        code: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation remotely via Basilica."""
        logger.info(f"Running Basilica evaluation")
        
        try:
            from affinetes import env as af_env
            
            env_vars = {
                "OUTPUT_VECTOR_TOLERANCE": str(self.output_tolerance),
            }
            if self.basilica_api_key:
                env_vars["BASILICA_API_KEY"] = self.basilica_api_key
            
            # Load validator's environment
            env = af_env.load_env(
                mode="basilica",
                image=self.validator_image,
                cpu_limit=os.getenv("BASILICA_CPU_LIMIT", "2000m"),
                mem_limit=os.getenv("BASILICA_MEM_LIMIT", "32Gi"),
                env_vars=env_vars,
            )
            
            try:
                result = await asyncio.wait_for(
                    env.evaluate(
                        task_id=task_id,
                        seed=seed,
                        model_url=model_url,
                        data_url=data_url,
                        steps=steps,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        data_samples=data_samples,
                        timeout=self.timeout,
                        code=code,
                    ),
                    timeout=self.timeout + 60,
                )
                result_obj = EvaluationResult.from_dict(result)
                result_obj.code = code
                return result_obj
            finally:
                try:
                    await env.cleanup()
                except Exception as cleanup_err:
                    logger.warning(f"Cleanup failed: {cleanup_err}")
            
        except ImportError:
            logger.warning("affinetes not installed, using HTTP fallback")
            return await self._evaluate_basilica_http(
                code=code,
                seed=seed,
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                task_id=task_id,
            )
        except asyncio.TimeoutError:
            return EvaluationResult.failure(
                f"Basilica timeout after {self.timeout}s",
                task_id=task_id,
            )
        except Exception as e:
            return EvaluationResult.failure(
                f"Basilica error: {e}",
                task_id=task_id,
            )
    
    async def _evaluate_basilica_http(
        self,
        code: str,
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
            "image": self.validator_image,
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
                "code": code,
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
                result = EvaluationResult.from_dict(result_data)
                result.code = code
                return result
                
        except Exception as e:
            return EvaluationResult.failure(
                f"Basilica HTTP error: {e}",
                task_id=task_id,
            )
    
    async def build_validator_image(self, env_path: Path | None = None) -> bool:
        """Build the validator's evaluation Docker image.
        
        Args:
            env_path: Path to environments/templar directory
            
        Returns:
            True if build succeeded
        """
        if env_path is None:
            candidates = [
                Path(__file__).parent.parent.parent.parent / "environments" / "templar",
                Path.cwd() / "environments" / "templar",
            ]
            for candidate in candidates:
                if candidate.exists() and (candidate / "Dockerfile").exists():
                    env_path = candidate
                    break
        
        if env_path is None or not env_path.exists():
            logger.error("Could not find environments/templar directory")
            return False
        
        logger.info(f"Building validator image: {self.validator_image}")
        logger.info(f"   From: {env_path}")
        
        cmd = [
            "docker", "build",
            "-t", self.validator_image,
            str(env_path),
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            logger.error(f"Failed to build validator image")
            return False
        
        logger.info(f"Successfully built: {self.validator_image}")
        return True


def create_runner(
    mode: str = "docker",
    **kwargs,
) -> AffinetesRunner:
    """Factory function to create an AffinetesRunner.
    
    Args:
        mode: "docker" or "basilica"
        **kwargs: Additional arguments
        
    Returns:
        Configured AffinetesRunner
    """
    if mode == "basilica":
        kwargs.setdefault("basilica_endpoint", os.getenv("BASILICA_ENDPOINT"))
        kwargs.setdefault("basilica_api_key", os.getenv("BASILICA_API_KEY"))
    
    return AffinetesRunner(mode=mode, **kwargs)
