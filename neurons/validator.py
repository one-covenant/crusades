"""Tournament validator - evaluates miner submissions and sets weights.

Chi/Affinetes Architecture:
1. Reads Docker image commitments from blockchain
2. Evaluates via affinetes (Docker locally or Basilica remotely)
3. Sets weights based on TPS scores
"""

import argparse
import asyncio
import gc
import logging
import os
import statistics
import time
from typing import Literal

import bittensor as bt
import torch

from tournament.chain.commitments import CommitmentReader, MinerCommitment
from tournament.chain.weights import WeightSetter
from tournament.config import get_config, get_hparams
from tournament.core.protocols import SubmissionStatus
from tournament.affinetes import AffinetesRunner, EvaluationResult
from tournament.storage.database import Database, get_database
from tournament.storage.models import EvaluationModel, SubmissionModel

from .base_node import BaseNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Tournament validator node (Chi/Affinetes Architecture).

    Responsibilities:
    1. Read miner Docker image commitments from blockchain
    2. Evaluate submissions via affinetes (Docker or Basilica)
    3. Calculate and set weights (winner-takes-all)
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        skip_blockchain_check: bool = False,
        affinetes_mode: Literal["docker", "basilica"] = "docker",
    ):
        super().__init__(wallet=wallet)

        self.skip_blockchain_check = skip_blockchain_check
        self.affinetes_mode = affinetes_mode

        # Components (initialized in start)
        self.db: Database | None = None
        self.weight_setter: WeightSetter | None = None
        self.commitment_reader: CommitmentReader | None = None
        self.affinetes_runner: AffinetesRunner | None = None
        
        # State
        self.last_processed_block: int = 0
        self.evaluated_images: set[str] = set()

        # Timing
        self.last_weight_set_time: float = 0
        self.last_sync_time: float = 0

    async def initialize(self) -> None:
        """Initialize validator components."""
        config = get_config()
        hparams = get_hparams()

        logger.info("Initializing validator (Chi/Affinetes mode)")
        
        # Database
        self.db = await get_database()
        
        # Weight setter
        self.weight_setter = WeightSetter(
            chain=self.chain,
            database=self.db,
        )
        
        # Commitment reader for blockchain
        self.commitment_reader = CommitmentReader(
            subtensor=self.chain.subtensor,
            netuid=hparams.netuid,
            network=config.subtensor_network,
        )
        self.commitment_reader.sync()
        
        # Get model/data from hparams - fallback to env variables
        model_url = getattr(hparams, 'benchmark_model_name', None) or os.getenv('BENCHMARK_MODEL_URL', '')
        data_url = getattr(hparams, 'benchmark_dataset_name', None) or os.getenv('BENCHMARK_DATA_URL', '')
        
        if not model_url:
            logger.warning("benchmark_model_name not set in hparams.json")
        
        if not data_url:
            logger.warning("benchmark_dataset_name not set in hparams.json")
        
        # Affinetes runner
        basilica_endpoint = os.getenv('BASILICA_ENDPOINT')
        basilica_api_key = os.getenv('BASILICA_API_KEY')
        
        # Get verification tolerance from hparams
        verification = getattr(hparams, 'verification', None)
        output_tolerance = 0.02  # default 2%
        if verification:
            output_tolerance = getattr(verification, 'output_vector_tolerance', 0.02)
        
        self.affinetes_runner = AffinetesRunner(
            mode=self.affinetes_mode,
            basilica_endpoint=basilica_endpoint,
            basilica_api_key=basilica_api_key,
            model_url=model_url,
            data_url=data_url,
            timeout=getattr(hparams, 'eval_timeout', 600),
            output_tolerance=output_tolerance,
        )
        
        self.last_processed_block = 0
        
        logger.info(f"   Affinetes mode: {self.affinetes_mode}")
        logger.info(f"   Model URL: {model_url or 'NOT SET'}")
        logger.info(f"   Data URL: {data_url or 'NOT SET'}")
        if self.affinetes_mode == "basilica":
            logger.info(f"   Basilica endpoint: {basilica_endpoint or 'NOT SET'}")

    async def start(self) -> None:
        """Start the validator."""
        await self.initialize()
        await super().start()

    async def run_step(self) -> None:
        """Run one iteration of the validator loop."""
        logger.info("Starting validation loop iteration...")
        
        # 1. Read blockchain commitments
        logger.info("Step 1: Reading blockchain commitments...")
        await self.process_blockchain_commitments()

        # 2. Evaluate via affinetes
        logger.info("Step 2: Evaluating via affinetes...")
        await self.evaluate_submissions()
        
        # 3. Set weights
        logger.info("Step 3: Checking weight setting...")
        await self.maybe_set_weights()

        # 4. Sync metagraph
        logger.info("Step 4: Checking metagraph sync...")
        await self.maybe_sync()
        
        # Memory cleanup
        self._cleanup_memory()

        logger.info("Loop iteration complete. Sleeping 10s...")
        await asyncio.sleep(10)
    
    async def process_blockchain_commitments(self) -> None:
        """Read and process new commitments from blockchain."""
        try:
            new_commitments = self.commitment_reader.get_new_commitments_since(
                self.last_processed_block
            )
            
            current_block = self.commitment_reader.get_current_block()
            
            if new_commitments:
                logger.info(f"Found {len(new_commitments)} new commitments")
                
                for commitment in new_commitments:
                    if commitment.image_name in self.evaluated_images:
                        logger.debug(f"Skipping already evaluated: {commitment.image_name}")
                        continue
                    
                    await self._create_submission_from_commitment(commitment)
            
            self.last_processed_block = current_block
            
        except Exception as e:
            logger.error(f"Error processing commitments: {e}")
    
    async def _create_submission_from_commitment(
        self,
        commitment: MinerCommitment,
    ) -> None:
        """Create a submission record from a blockchain commitment.
        
        Includes rate limiting: miners must wait min_blocks_between_commits
        between submissions to prevent spam.
        """
        hparams = get_hparams()
        submission_id = f"chi_{commitment.commit_block}_{commitment.uid}"
        
        existing = await self.db.get_submission(submission_id)
        if existing:
            return
        
        # Rate limiting: Check if miner submitted too recently
        min_blocks = getattr(hparams, 'min_blocks_between_commits', 100)
        last_submission = await self.db.get_latest_submission_by_hotkey(commitment.hotkey)
        
        if last_submission:
            # Extract commit_block from submission_id (format: chi_<block>_<uid>)
            try:
                last_block = int(last_submission.submission_id.split('_')[1])
                blocks_since = commitment.commit_block - last_block
                
                if blocks_since < min_blocks:
                    logger.warning(
                        f"Rate limit: {commitment.hotkey[:16]}... submitted too soon "
                        f"({blocks_since} blocks, min={min_blocks}). Skipping."
                    )
                    return
            except (IndexError, ValueError):
                pass  # Can't parse, allow submission
        
        submission = SubmissionModel(
            submission_id=submission_id,
            miner_hotkey=commitment.hotkey,
            miner_uid=commitment.uid,
            code_hash=commitment.image_hash,
            bucket_path=commitment.image_name,  # Docker image stored here
            status=SubmissionStatus.EVALUATING,
            payment_verified=True,
        )
        
        await self.db.save_submission(submission)
        logger.info(f"Created submission: {submission_id}")
        logger.info(f"   Image: {commitment.image_name}")
        logger.info(f"   UID: {commitment.uid}")
    
    async def evaluate_submissions(self) -> None:
        """Evaluate submissions using affinetes."""
        hparams = get_hparams()
        evaluating = await self.db.get_evaluating_submissions()
        num_runs = getattr(hparams, 'evaluation_runs', 5)

        for submission in evaluating:
            image = submission.bucket_path
            
            # Validate it's a Docker image
            if not image:
                continue
            
            existing_evals = await self.db.get_evaluations(submission.submission_id)
            my_evals = [e for e in existing_evals if e.evaluator_hotkey == self.hotkey]

            if len(my_evals) >= num_runs:
                continue

            runs_remaining = num_runs - len(my_evals)
            logger.info(f"Evaluating {submission.submission_id}")
            logger.info(f"   Image: {image}")
            logger.info(f"   Runs: {len(my_evals)+1}/{num_runs}")

            # Check if image exists
            image_exists = await self.affinetes_runner.check_image_exists(image)
            if not image_exists:
                logger.warning(f"Image not found: {image}, attempting pull...")
                pulled = await self.affinetes_runner.pull_image(image)
                if not pulled:
                    evaluation = EvaluationModel(
                        submission_id=submission.submission_id,
                        evaluator_hotkey=self.hotkey,
                        tokens_per_second=0.0,
                        total_tokens=0,
                        wall_time_seconds=0.0,
                        success=False,
                        error=f"Docker image not found: {image}",
                    )
                    await self.db.save_evaluation(evaluation)
                    continue

            # Run evaluations
            for run_idx in range(runs_remaining):
                current_run = len(my_evals) + run_idx + 1
                seed = f"{submission.miner_uid}:{current_run}:{int(time.time())}"
                
                logger.info(f"Evaluation run {current_run}/{num_runs} (seed: {seed})")
                
                result = await self.affinetes_runner.evaluate(
                    image=image,
                    seed=seed,
                    steps=getattr(hparams, 'eval_steps', 5),
                    batch_size=getattr(hparams, 'benchmark_batch_size', 8),
                    sequence_length=getattr(hparams, 'benchmark_sequence_length', 1024),
                    data_samples=getattr(hparams, 'benchmark_data_samples', 10000),
                    task_id=current_run,
                )

                if result.success:
                    logger.info(f"Run {current_run} PASSED: {result.tps:,.2f} TPS")
                else:
                    logger.warning(f"Run {current_run} FAILED: {result.error}")

                evaluation = EvaluationModel(
                    submission_id=submission.submission_id,
                    evaluator_hotkey=self.hotkey,
                    tokens_per_second=result.tps,
                    total_tokens=result.total_tokens,
                    wall_time_seconds=result.wall_time_seconds,
                    success=result.success,
                    error=result.error,
                )
                await self.db.save_evaluation(evaluation)
                self._cleanup_memory()

            # Mark image as evaluated
            self.evaluated_images.add(image)
            
            # Finalize submission
            await self._finalize_submission(submission.submission_id, num_runs)
    
    async def _finalize_submission(self, submission_id: str, num_runs: int) -> None:
        """Calculate final score and update submission status."""
        hparams = get_hparams()
        num_evals = await self.db.count_evaluations(submission_id)
        required_evals = getattr(hparams, 'num_evals_per_submission', 1) * num_runs
        
        if num_evals >= required_evals:
            all_evals = await self.db.get_evaluations(submission_id)
            successful_evals = [e for e in all_evals if e.success]

            if successful_evals:
                tps_scores = [e.tokens_per_second for e in successful_evals]
                median_tps = statistics.median(tps_scores)
                
                logger.info(
                    f"Final score for {submission_id}:\n"
                    f"   Successful runs: {len(tps_scores)}\n"
                    f"   Scores: {[f'{s:.1f}' for s in sorted(tps_scores)]}\n"
                    f"   Median TPS: {median_tps:,.2f}"
                )
                
                await self.db.update_submission_score(submission_id, median_tps)
            else:
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message="All evaluations failed",
                )
                logger.warning(f"Submission {submission_id} failed: no successful evaluations")
    
    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if not hasattr(self, '_loop_count'):
            self._loop_count = 0
        
        self._loop_count += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if self._loop_count % 10 == 0:
            logger.info(f"Memory cleanup (iteration {self._loop_count})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    async def maybe_sync(self) -> None:
        """Sync metagraph periodically."""
        now = time.time()
        if now - self.last_sync_time >= 300:
            await self.sync()
            if self.commitment_reader:
                self.commitment_reader.sync()
            self.last_sync_time = now

    async def maybe_set_weights(self) -> None:
        """Set weights if enough time has passed."""
        hparams = get_hparams()
        now = time.time()

        if now - self.last_weight_set_time < hparams.set_weights_interval_seconds:
            return

        logger.info("Setting weights...")
        success, message = await self.weight_setter.set_weights()

        if success:
            self.last_weight_set_time = now
            logger.info(f"Weights set: {message}")
        else:
            logger.warning(f"Failed to set weights: {message}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await super().cleanup()
        if self.db:
            await self.db.close()


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    parser = argparse.ArgumentParser(description="Tournament Validator (Chi/Affinetes)")
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Wallet name",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey",
    )
    parser.add_argument(
        "--skip-blockchain-check",
        action="store_true",
        help="Skip blockchain registration check",
    )
    parser.add_argument(
        "--affinetes-mode",
        type=str,
        choices=["docker", "basilica"],
        default="docker",
        help="Execution mode: docker (local) or basilica (remote GPU)",
    )

    args = parser.parse_args()

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    validator = Validator(
        wallet=wallet,
        skip_blockchain_check=args.skip_blockchain_check,
        affinetes_mode=args.affinetes_mode,
    )

    logger.info(f"Starting validator")
    logger.info(f"   Affinetes mode: {args.affinetes_mode}")
    
    asyncio.run(validator.start())


if __name__ == "__main__":
    main()
