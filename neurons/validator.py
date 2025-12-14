"""Tournament validator - evaluates miner submissions and sets weights."""

import argparse
import asyncio
import logging
import time

import bittensor as bt

from tournament.chain.weights import WeightSetter
from tournament.config import get_config, get_hparams
from tournament.core.protocols import SubmissionStatus
from tournament.payment.verifier import PaymentVerifier
from tournament.pipeline.validator import CodeValidator
from tournament.sandbox.manager import SandboxManager
from tournament.schemas import BenchmarkConfig
from tournament.storage.database import Database, get_database
from tournament.storage.models import EvaluationModel
from tournament.verification import (
    ReferenceExecutor,
    SandboxVerifier,
    VerificationConfig,
)

from .base_node import BaseNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Tournament validator node.

    Responsibilities:
    1. Validate pending submissions (syntax, required functions)
    2. Evaluate validated submissions in sandbox
    3. Calculate and set weights (winner-takes-all)
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        burn_hotkey: str | None = None,
        burn_enabled: bool = False,
    ):
        super().__init__(wallet=wallet)

        self.burn_hotkey = burn_hotkey
        self.burn_enabled = burn_enabled

        # Components (initialized in start)
        self.db: Database | None = None
        self.sandbox: SandboxManager | None = None
        self.verifier: SandboxVerifier | None = None
        self.code_validator: CodeValidator | None = None
        self.weight_setter: WeightSetter | None = None
        self.payment_verifier: PaymentVerifier | None = None

        # Timing
        self.last_weight_set_time: float = 0
        self.last_sync_time: float = 0

    async def initialize(self) -> None:
        """Initialize components."""
        config = get_config()
        hparams = get_hparams()

        # Database
        self.db = await get_database()

        # Sandbox
        self.sandbox = SandboxManager(
            benchmark_model_path=config.benchmark_model_path,
            benchmark_data_path=config.benchmark_data_path,
        )
        await self.sandbox.initialize()

        # Verification configuration
        verification_config = VerificationConfig.from_dict(
            hparams.verification if hasattr(hparams, "verification") else {}
        )

        # Reference executor for verification
        benchmark_config = BenchmarkConfig(
            model_path=config.benchmark_model_path,
            data_path=config.benchmark_data_path,
            sequence_length=hparams.benchmark_sequence_length,
            batch_size=hparams.benchmark_batch_size,
            num_steps=hparams.eval_steps,
        )
        reference_executor = ReferenceExecutor(
            model_path=config.benchmark_model_path,
            data_path=config.benchmark_data_path,
            config=benchmark_config,
        )

        # Verifier (combines sandbox + reference for verification)
        self.verifier = SandboxVerifier(
            sandbox_manager=self.sandbox,
            reference_executor=reference_executor,
            config=verification_config,
        )

        # Code validator
        self.code_validator = CodeValidator()

        # Weight setter
        self.weight_setter = WeightSetter(
            chain=self.chain,
            database=self.db,
            burn_hotkey=self.burn_hotkey,
            burn_enabled=self.burn_enabled,
        )

        # Payment verifier (anti-spam)
        self.payment_verifier = PaymentVerifier(
            recipient_address=self.wallet.hotkey.ss58_address,
            subtensor=chain.subtensor,
        )

    async def start(self) -> None:
        """Start the validator."""
        await self.initialize()
        await super().start()

    async def run_step(self) -> None:
        """Run one iteration of the validator loop."""
        # 1. Process pending submissions (validation)
        await self.process_pending_submissions()

        # 2. Evaluate submissions ready for evaluation
        await self.evaluate_submissions()

        # 3. Set weights periodically
        await self.maybe_set_weights()

        # 4. Sync metagraph periodically (every 5 minutes)
        await self.maybe_sync()

        # Sleep before next iteration
        await asyncio.sleep(10)

    async def maybe_sync(self) -> None:
        """Sync metagraph periodically."""
        now = time.time()
        if now - self.last_sync_time >= 300:  # Every 5 minutes
            await self.sync()
            self.last_sync_time = now

    async def process_pending_submissions(self) -> None:
        """Validate pending submissions (including payment verification)."""
        pending = await self.db.get_pending_submissions()

        for submission in pending:
            logger.info(f"Validating submission {submission.submission_id}")

            # Update status to validating
            await self.db.update_submission_status(
                submission.submission_id,
                SubmissionStatus.VALIDATING,
            )

            # Step 1: Verify payment (if payment info provided)
            if submission.payment_block_hash and not submission.payment_verified:
                logger.info(f"Verifying payment for submission {submission.submission_id}")
                
                payment_valid, payment_error = await self.payment_verifier.verify_payment(
                    block_hash=submission.payment_block_hash,
                    extrinsic_index=submission.payment_extrinsic_index,
                    miner_coldkey=submission.miner_hotkey,  # Assuming hotkey used for payment
                    expected_amount_rao=submission.payment_amount_rao,
                )

                if not payment_valid:
                    await self.db.update_submission_status(
                        submission.submission_id,
                        SubmissionStatus.FAILED_VALIDATION,
                        error_message=f"Payment verification failed: {payment_error}",
                    )
                    logger.warning(
                        f"Submission {submission.submission_id} failed payment verification: {payment_error}"
                    )
                    continue

                # Mark payment as verified
                submission.payment_verified = True
                logger.info(f"Payment verified for submission {submission.submission_id}")

            # Step 2: Download code from R2
            # TODO: Implement R2 download
            # For now, assume code is available locally
            # code = await download_from_r2(submission.bucket_path)

            # Step 3: Validate code
            # result = self.code_validator.validate(code)
            # For now, just mark as evaluating
            result_valid = True

            if result_valid:
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.EVALUATING,
                )
                logger.info(f"Submission {submission.submission_id} passed validation")
            else:
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.FAILED_VALIDATION,
                    error_message="Validation failed",  # result.errors
                )
                logger.warning(f"Submission {submission.submission_id} failed validation")

    async def evaluate_submissions(self) -> None:
        """Evaluate submissions that are ready with verification."""
        hparams = get_hparams()
        evaluating = await self.db.get_evaluating_submissions()

        for submission in evaluating:
            # Check if we've already evaluated this submission
            existing_evals = await self.db.get_evaluations(submission.submission_id)
            my_evals = [e for e in existing_evals if e.evaluator_hotkey == self.hotkey]

            if len(my_evals) > 0:
                # Already evaluated by this validator
                continue

            logger.info(f"Evaluating submission {submission.submission_id}")

            # TODO: Download code from R2
            # code_path = await download_from_r2(submission.bucket_path)
            # For now, use a placeholder path
            code_path = f"/tmp/submissions/{submission.submission_id}/train.py"

            # Run verification and benchmarking
            result = await self.verifier.verify_and_benchmark(code_path)

            if result.success:
                logger.info(
                    f"Verification PASSED for {submission.submission_id}\n"
                    f"  TPS: {result.tokens_per_second:,.2f}\n"
                    f"  Tokens: {result.total_tokens:,}\n"
                    f"  Time: {result.wall_time_seconds:.2f}s"
                )
            else:
                logger.warning(
                    f"Verification FAILED for {submission.submission_id}\n"
                    f"  Error type: {result.error_type}\n"
                    f"  Message: {result.error_message}"
                )

            # Save evaluation result
            evaluation = EvaluationModel(
                submission_id=submission.submission_id,
                evaluator_hotkey=self.hotkey,
                tokens_per_second=result.tokens_per_second,
                total_tokens=result.total_tokens,
                wall_time_seconds=result.wall_time_seconds,
                success=result.success,
                error=result.error_message,
            )
            await self.db.save_evaluation(evaluation)

            # Check if submission has enough evaluations
            num_evals = await self.db.count_evaluations(submission.submission_id)
            if num_evals >= hparams.num_evals_per_submission:
                # Calculate final score (average TPS of successful evaluations)
                all_evals = await self.db.get_evaluations(submission.submission_id)
                successful_evals = [e for e in all_evals if e.success]

                if successful_evals:
                    avg_tps = sum(e.tokens_per_second for e in successful_evals) / len(
                        successful_evals
                    )
                    await self.db.update_submission_score(submission.submission_id, avg_tps)
                    logger.info(
                        f"Submission {submission.submission_id} finished with score {avg_tps:,.2f} TPS"
                    )
                else:
                    # All evaluations failed verification
                    await self.db.update_submission_status(
                        submission.submission_id,
                        SubmissionStatus.FAILED_VALIDATION,
                        error_message="All evaluations failed verification",
                    )
                    logger.warning(
                        f"Submission {submission.submission_id} failed: no successful evaluations"
                    )

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
        if self.sandbox:
            await self.sandbox.cleanup()
        if self.db:
            await self.db.close()


def main():
    parser = argparse.ArgumentParser(description="Tournament Validator")
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
        "--burn-hotkey",
        type=str,
        default=None,
        help="Hotkey to burn emissions to when no winner",
    )
    parser.add_argument(
        "--burn-enabled",
        action="store_true",
        help="Enable burn mode (all emissions to burn hotkey)",
    )

    args = parser.parse_args()

    # Initialize wallet
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    # Create and run validator
    validator = Validator(
        wallet=wallet,
        burn_hotkey=args.burn_hotkey,
        burn_enabled=args.burn_enabled,
    )

    asyncio.run(validator.start())


if __name__ == "__main__":
    main()
