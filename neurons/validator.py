"""Tournament validator - evaluates miner submissions and sets weights.

R2-Based Architecture:
1. Reads R2 commitments from blockchain (miner's R2 credentials)
2. Downloads miner's train.py from their R2 at evaluation time
3. Evaluates via affinetes (Docker locally or Basilica remotely)
4. Stores miner's code in validator's R2 for dashboard
5. Sets weights based on TPS scores
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
from tournament.affinetes.runner import R2Info
from tournament.storage.database import Database, get_database
from tournament.storage.models import EvaluationModel, SubmissionModel

from .base_node import BaseNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Tournament validator node (R2-Based Architecture).

    Responsibilities:
    1. Read miner R2 commitments from blockchain
    2. Download and evaluate miner's train.py via affinetes
    3. Store miner's code in validator's R2 for dashboard
    4. Calculate and set weights (winner-takes-all)
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        skip_blockchain_check: bool = False,
        affinetes_mode: Literal["docker", "basilica"] = "docker",
    ):
        super().__init__(wallet=wallet, skip_blockchain_check=skip_blockchain_check)
        self.affinetes_mode = affinetes_mode

        # Components (initialized in start)
        self.db: Database | None = None
        self.weight_setter: WeightSetter | None = None
        self.commitment_reader: CommitmentReader | None = None
        self.affinetes_runner: AffinetesRunner | None = None
        
        # Validator's R2 for storing code (optional)
        self.validator_r2_client = None
        
        # State
        self.last_processed_block: int = 0
        self.evaluated_hashes: set[str] = set()  # code_hash, not image

        # Timing
        self.last_weight_set_time: float = 0
        self.last_sync_time: float = 0

    async def initialize(self) -> None:
        """Initialize validator components."""
        config = get_config()
        hparams = get_hparams()

        logger.info("Initializing validator (R2-Based Architecture)")
        
        # Database
        self.db = await get_database()
        
        # Weight setter and commitment reader
        if self.chain is not None:
            self.weight_setter = WeightSetter(
                chain=self.chain,
                database=self.db,
            )
            
            self.commitment_reader = CommitmentReader(
                subtensor=self.chain.subtensor,
                netuid=hparams.netuid,
                network=config.subtensor_network,
            )
            self.commitment_reader.sync()
        else:
            logger.warning("Running without blockchain (test mode)")
            logger.warning("Using local commitments only")
            
            self.commitment_reader = CommitmentReader(
                subtensor=None,
                netuid=hparams.netuid,
                network="local",
                local_mode=True,
            )
        
        # Get model/data from hparams
        model_url = getattr(hparams, 'benchmark_model_name', None) or os.getenv('BENCHMARK_MODEL_URL', '')
        data_url = getattr(hparams, 'benchmark_dataset_name', None) or os.getenv('BENCHMARK_DATA_URL', '')
        
        if not model_url:
            logger.warning("benchmark_model_name not set in hparams.json")
        
        if not data_url:
            logger.warning("benchmark_dataset_name not set in hparams.json")
        
        # Affinetes runner (uses validator's standard image)
        basilica_endpoint = os.getenv('BASILICA_ENDPOINT')
        basilica_api_key = os.getenv('BASILICA_API_KEY')
        
        # Verification tolerance
        verification = getattr(hparams, 'verification', None)
        output_tolerance = 0.02
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
        
        # Initialize validator's R2 for code storage (optional, from .env)
        self._init_validator_r2()
        
        self.last_processed_block = 0
        
        logger.info(f"   Affinetes mode: {self.affinetes_mode}")
        logger.info(f"   Model URL: {model_url or 'NOT SET'}")
        logger.info(f"   Data URL: {data_url or 'NOT SET'}")
        if self.affinetes_mode == "basilica":
            logger.info(f"   Basilica endpoint: {basilica_endpoint or 'NOT SET'}")
    
    def _init_validator_r2(self) -> None:
        """Initialize validator's R2 client for storing evaluated code.
        
        All R2 config comes from .env:
        - VALIDATOR_R2_* (production: validator's own bucket)
        - TOURNAMENT_R2_* (testing: shared bucket fallback)
        
        Note: Miner R2 credentials come from commitments, not environment.
        """
        # Primary: Validator's own R2
        endpoint = os.getenv('VALIDATOR_R2_ENDPOINT', '')
        bucket = os.getenv('VALIDATOR_R2_BUCKET', '')
        access_key = os.getenv('VALIDATOR_R2_ACCESS_KEY', '')
        secret_key = os.getenv('VALIDATOR_R2_SECRET_KEY', '')
        
        # Fallback: Shared tournament R2 (for testing)
        if not all([endpoint, bucket, access_key, secret_key]):
            account_id = os.getenv('TOURNAMENT_R2_ACCOUNT_ID', '')
            bucket = bucket or os.getenv('TOURNAMENT_R2_BUCKET_NAME', '')
            access_key = access_key or os.getenv('TOURNAMENT_R2_ACCESS_KEY_ID', '')
            secret_key = secret_key or os.getenv('TOURNAMENT_R2_SECRET_ACCESS_KEY', '')
            
            if account_id and not endpoint:
                endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
        
        if not all([endpoint, bucket, access_key, secret_key]):
            logger.info("R2 credentials incomplete - using commitments only")
            return
        
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            
            self.validator_r2_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=BotoConfig(signature_version="s3v4"),
            )
            self.validator_r2_bucket = bucket
            
            # Store R2 config for downloading miner submissions
            self.r2_config = {
                "endpoint": endpoint,
                "bucket": bucket,
                "access_key": access_key,
                "secret_key": secret_key,
            }
            
            logger.info(f"R2 configured: {bucket}")
        except ImportError:
            logger.warning("boto3 not installed - R2 disabled")
        except Exception as e:
            logger.warning(f"Failed to init R2: {e}")

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
        """Read and process new R2 commitments from blockchain."""
        try:
            new_commitments = self.commitment_reader.get_new_commitments_since(
                self.last_processed_block
            )
            
            current_block = self.commitment_reader.get_current_block()
            
            if new_commitments:
                logger.info(f"Found {len(new_commitments)} new commitments")
                
                for commitment in new_commitments:
                    # Skip if no R2 credentials
                    if not commitment.has_r2_credentials():
                        logger.debug(f"Skipping commitment without R2 credentials: UID {commitment.uid}")
                        continue
                    
                    # Skip if already evaluated this code
                    if commitment.code_hash in self.evaluated_hashes:
                        logger.debug(f"Skipping already evaluated: {commitment.code_hash[:16]}...")
                        continue
                    
                    await self._create_submission_from_commitment(commitment)
            
            self.last_processed_block = current_block
            
        except Exception as e:
            logger.error(f"Error processing commitments: {e}")
    
    async def _create_submission_from_commitment(
        self,
        commitment: MinerCommitment,
    ) -> None:
        """Create a submission record from a blockchain commitment."""
        hparams = get_hparams()
        submission_id = f"r2_{commitment.commit_block}_{commitment.uid}"
        
        existing = await self.db.get_submission(submission_id)
        if existing:
            return
        
        # Rate limiting
        min_blocks = getattr(hparams, 'min_blocks_between_commits', 100)
        last_submission = await self.db.get_latest_submission_by_hotkey(commitment.hotkey)
        
        if last_submission:
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
                pass
        
        # Store R2 info as JSON in bucket_path
        import json
        r2_info = {
            "endpoint": commitment.r2_credentials.endpoint,
            "bucket": commitment.r2_credentials.bucket,
            "key": commitment.r2_credentials.key,
            "access_key": commitment.r2_credentials.access_key,
            "secret_key": commitment.r2_credentials.secret_key,
        }
        
        submission = SubmissionModel(
            submission_id=submission_id,
            miner_hotkey=commitment.hotkey,
            miner_uid=commitment.uid,
            code_hash=commitment.code_hash,
            bucket_path=json.dumps(r2_info),  # Store R2 info as JSON
            status=SubmissionStatus.EVALUATING,
            payment_verified=True,
        )
        
        await self.db.save_submission(submission)
        logger.info(f"Created submission: {submission_id}")
        logger.info(f"   R2 bucket: {commitment.r2_credentials.bucket}")
        logger.info(f"   R2 key: {commitment.r2_credentials.key}")
        logger.info(f"   UID: {commitment.uid}")
    
    async def evaluate_submissions(self) -> None:
        """Evaluate submissions by downloading from miner's R2.
        
        Uses R2 credentials from the miner's commitment (stored in bucket_path).
        Each miner provides their own R2 credentials in their commitment.
        """
        hparams = get_hparams()
        evaluating = await self.db.get_evaluating_submissions()
        num_runs = getattr(hparams, 'evaluation_runs', 5)

        for submission in evaluating:
            # Parse R2 info from bucket_path (miner's R2 credentials)
            import json
            try:
                r2_data = json.loads(submission.bucket_path)
                
                # Use miner's R2 credentials from commitment
                r2_info = R2Info(
                    endpoint=r2_data.get("r2_endpoint", r2_data.get("endpoint", "")),
                    bucket=r2_data.get("r2_bucket", r2_data.get("bucket", "")),
                    key=r2_data.get("r2_key", r2_data.get("key", "")),
                    access_key=r2_data.get("r2_access_key", r2_data.get("access_key", "")),
                    secret_key=r2_data.get("r2_secret_key", r2_data.get("secret_key", "")),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Invalid R2 info for {submission.submission_id}: {e}")
                continue
            
            if not r2_info.is_valid():
                logger.warning(f"Invalid R2 credentials for {submission.submission_id}")
                continue
            
            existing_evals = await self.db.get_evaluations(submission.submission_id)
            my_evals = [e for e in existing_evals if e.evaluator_hotkey == self.hotkey]

            if len(my_evals) >= num_runs:
                continue

            runs_remaining = num_runs - len(my_evals)
            logger.info(f"Evaluating {submission.submission_id}")
            logger.info(f"   R2: {r2_info.bucket}/{r2_info.key}")
            logger.info(f"   Runs: {len(my_evals)+1}/{num_runs}")

            # Run evaluations
            miner_code = None  # Will store code from first successful run
            
            for run_idx in range(runs_remaining):
                current_run = len(my_evals) + run_idx + 1
                seed = f"{submission.miner_uid}:{current_run}:{int(time.time())}"
                
                logger.info(f"Evaluation run {current_run}/{num_runs} (seed: {seed})")
                
                result = await self.affinetes_runner.evaluate(
                    r2_info=r2_info,
                    seed=seed,
                    steps=getattr(hparams, 'eval_steps', 5),
                    batch_size=getattr(hparams, 'benchmark_batch_size', 8),
                    sequence_length=getattr(hparams, 'benchmark_sequence_length', 1024),
                    data_samples=getattr(hparams, 'benchmark_data_samples', 10000),
                    task_id=current_run,
                )

                if result.success:
                    logger.info(f"Run {current_run} PASSED: {result.tps:,.2f} TPS")
                    # Store code from first successful run
                    if miner_code is None and result.code:
                        miner_code = result.code
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

            # Mark code hash as evaluated
            self.evaluated_hashes.add(submission.code_hash)
            
            # Store miner's code in database and validator's R2
            if miner_code:
                await self._store_miner_code(submission, miner_code)
            
            # Finalize submission
            await self._finalize_submission(submission.submission_id, num_runs)
    
    async def _store_miner_code(self, submission: SubmissionModel, code: str) -> None:
        """Store miner's code in database and optionally in validator's R2."""
        # Store in database
        await self.db.update_submission_code(submission.submission_id, code)
        logger.info(f"Stored code for {submission.submission_id} in database")
        
        # Optionally store in validator's R2
        if self.validator_r2_client:
            try:
                key = f"submissions/{submission.submission_id}/train.py"
                self.validator_r2_client.put_object(
                    Bucket=self.validator_r2_bucket,
                    Key=key,
                    Body=code.encode(),
                    ContentType="text/x-python",
                )
                logger.info(f"Stored code for {submission.submission_id} in validator R2")
            except Exception as e:
                logger.warning(f"Failed to store code in validator R2: {e}")
    
    async def _finalize_submission(self, submission_id: str, num_runs: int) -> None:
        """Calculate final score and update submission status."""
        num_evals = await self.db.count_evaluations(submission_id)
        required_evals = num_runs
        
        logger.info(f"Finalizing {submission_id}: {num_evals}/{required_evals} evaluations")
        
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
        if self.weight_setter is None:
            logger.debug("Skipping weight setting (test mode)")
            return
            
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
    
    parser = argparse.ArgumentParser(description="Tournament Validator (R2-Based)")
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
