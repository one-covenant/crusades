"""
End-to-end test for miner submission and validator evaluation flow.

This test simulates the full lifecycle:
1. Miner creates and submits training code
2. Code is uploaded to storage (R2 or local fallback)
3. Validator downloads the code
4. Validator evaluates the code in sandbox
5. Results are stored in database
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tournament.core.protocols import SubmissionStatus
from tournament.storage.database import Database
from tournament.storage.models import SubmissionModel
from tournament.storage.r2 import R2Storage


# Sample miner code for testing
SAMPLE_MINER_CODE = """
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from collections.abc import Iterator


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    \"\"\"Simple training loop for testing.\"\"\"

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
"""


@pytest.fixture
async def test_db():
    """Create a test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def temp_code_file():
    """Create a temporary code file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(SAMPLE_MINER_CODE)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def r2_storage():
    """Create R2 storage instance (will use local fallback)."""
    return R2Storage()


class TestEndToEndSubmission:
    """Test the complete submission and evaluation flow."""

    @pytest.mark.asyncio
    async def test_upload_download_code(self, r2_storage, temp_code_file):
        """Test uploading and downloading code."""
        bucket_path = "test/submissions/uid_123/hash_abc/train.py"

        # Upload
        upload_success = await r2_storage.upload_code(temp_code_file, bucket_path)
        assert upload_success, "Code upload should succeed"

        # Download
        download_path = Path("/tmp/test_download_train.py")
        download_success = await r2_storage.download_code(bucket_path, download_path)
        assert download_success, "Code download should succeed"
        assert download_path.exists(), "Downloaded file should exist"

        # Verify content matches
        original_content = temp_code_file.read_text()
        downloaded_content = download_path.read_text()
        assert (
            original_content == downloaded_content
        ), "Downloaded content should match original"

        # Cleanup
        if download_path.exists():
            download_path.unlink()
        await r2_storage.delete_code(bucket_path)

    @pytest.mark.asyncio
    async def test_submission_workflow(self, test_db, r2_storage, temp_code_file):
        """Test the complete submission workflow."""
        import hashlib

        # Step 1: Simulate miner creating submission
        code_content = temp_code_file.read_text()
        code_hash = hashlib.sha256(code_content.encode()).hexdigest()
        miner_hotkey = "5TestMinerHotkey123"
        miner_uid = 42
        bucket_path = f"submissions/{miner_uid}/{code_hash}/train.py"

        # Step 2: Upload code
        upload_success = await r2_storage.upload_code(temp_code_file, bucket_path)
        assert upload_success, "Code upload should succeed"

        # Step 3: Create submission in database
        submission = SubmissionModel(
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            code_hash=code_hash,
            bucket_path=bucket_path,
            payment_verified=True,  # Skip payment verification for testing
        )
        await test_db.save_submission(submission)

        # Step 4: Update status to evaluating (simulating validator)
        await test_db.update_submission_status(
            submission.submission_id, SubmissionStatus.EVALUATING
        )

        # Step 5: Validator downloads code
        download_path = Path(f"/tmp/submissions/{submission.submission_id}/train.py")
        download_path.parent.mkdir(parents=True, exist_ok=True)

        download_success = await r2_storage.download_code(bucket_path, download_path)
        assert download_success, "Code download should succeed"
        assert download_path.exists(), "Downloaded file should exist"

        # Step 6: Verify downloaded code is correct
        downloaded_content = download_path.read_text()
        assert "def inner_steps" in downloaded_content, "Code should contain inner_steps"
        assert (
            "InnerStepsResult" in downloaded_content
        ), "Code should contain InnerStepsResult"

        # Step 7: Verify submission is in correct state
        retrieved_submission = await test_db.get_submission(submission.submission_id)
        assert retrieved_submission is not None
        assert retrieved_submission.status == SubmissionStatus.EVALUATING
        assert retrieved_submission.code_hash == code_hash

        # Cleanup
        if download_path.exists():
            download_path.unlink()
        await r2_storage.delete_code(bucket_path)

    @pytest.mark.asyncio
    async def test_multiple_submissions_same_miner(
        self, test_db, r2_storage, temp_code_file
    ):
        """Test handling multiple submissions from the same miner."""
        import hashlib

        miner_hotkey = "5TestMinerHotkey456"
        miner_uid = 99
        submissions = []

        # Create 3 submissions
        for i in range(3):
            # Modify code slightly for different hashes
            code_content = temp_code_file.read_text() + f"\n# Version {i}\n"
            code_hash = hashlib.sha256(code_content.encode()).hexdigest()[:16]
            bucket_path = f"submissions/{miner_uid}/{code_hash}/train.py"

            # Create temporary file with modified content
            temp_file = Path(f"/tmp/test_code_{i}.py")
            temp_file.write_text(code_content)

            # Upload
            upload_success = await r2_storage.upload_code(temp_file, bucket_path)
            assert upload_success, f"Upload {i} should succeed"

            # Create submission
            submission = SubmissionModel(
                miner_hotkey=miner_hotkey,
                miner_uid=miner_uid,
                code_hash=code_hash,
                bucket_path=bucket_path,
                payment_verified=True,
            )
            await test_db.save_submission(submission)
            submissions.append((submission, bucket_path))

            # Cleanup temp file
            temp_file.unlink()

        # Verify all submissions are in database
        for submission, _ in submissions:
            retrieved = await test_db.get_submission(submission.submission_id)
            assert retrieved is not None
            assert retrieved.miner_hotkey == miner_hotkey
            assert retrieved.miner_uid == miner_uid

        # Cleanup storage
        for _, bucket_path in submissions:
            await r2_storage.delete_code(bucket_path)

    @pytest.mark.asyncio
    async def test_download_nonexistent_code(self, r2_storage):
        """Test handling of downloading non-existent code."""
        bucket_path = "nonexistent/path/to/code.py"
        download_path = Path("/tmp/should_not_exist.py")

        download_success = await r2_storage.download_code(bucket_path, download_path)
        assert not download_success, "Download of non-existent code should fail"
        assert not download_path.exists(), "File should not be created"

    @pytest.mark.asyncio
    async def test_storage_fallback_when_r2_not_configured(self, temp_code_file):
        """Test that local storage fallback works when R2 is not configured."""
        # Create storage without credentials
        storage = R2Storage(
            account_id="", bucket_name="", access_key_id="", secret_access_key=""
        )

        assert not storage.is_configured(), "Storage should not be configured"

        bucket_path = "test/fallback/train.py"

        # Should still work with local fallback
        upload_success = await storage.upload_code(temp_code_file, bucket_path)
        assert upload_success, "Local fallback upload should succeed"

        download_path = Path("/tmp/test_fallback.py")
        download_success = await storage.download_code(bucket_path, download_path)
        assert download_success, "Local fallback download should succeed"

        # Verify content
        assert download_path.exists()
        original = temp_code_file.read_text()
        downloaded = download_path.read_text()
        assert original == downloaded

        # Cleanup
        download_path.unlink()
        await storage.delete_code(bucket_path)


class TestSubmissionIntegration:
    """Integration tests for submission components."""

    @pytest.mark.asyncio
    async def test_validator_evaluates_submission(self, test_db, r2_storage, temp_code_file):
        """Test validator workflow for evaluating a submission."""
        import hashlib

        from tournament.storage.models import EvaluationModel

        # Create submission
        code_content = temp_code_file.read_text()
        code_hash = hashlib.sha256(code_content.encode()).hexdigest()
        bucket_path = f"submissions/100/{code_hash}/train.py"

        await r2_storage.upload_code(temp_code_file, bucket_path)

        submission = SubmissionModel(
            miner_hotkey="5ValidatorTestMiner",
            miner_uid=100,
            code_hash=code_hash,
            bucket_path=bucket_path,
            payment_verified=True,
        )
        await test_db.save_submission(submission)
        await test_db.update_submission_status(
            submission.submission_id, SubmissionStatus.EVALUATING
        )

        # Simulate validator downloading and evaluating
        download_path = Path(f"/tmp/eval/{submission.submission_id}/train.py")
        download_path.parent.mkdir(parents=True, exist_ok=True)

        download_success = await r2_storage.download_code(bucket_path, download_path)
        assert download_success

        # Simulate evaluation result
        evaluation = EvaluationModel(
            submission_id=submission.submission_id,
            evaluator_hotkey="5TestValidatorHotkey",
            tokens_per_second=15000.5,
            total_tokens=819200,
            wall_time_seconds=54.67,
            success=True,
        )
        await test_db.save_evaluation(evaluation)

        # Verify evaluation was saved
        evals = await test_db.get_evaluations(submission.submission_id)
        assert len(evals) == 1
        assert evals[0].success is True
        assert evals[0].tokens_per_second == 15000.5

        # Cleanup
        download_path.unlink()
        await r2_storage.delete_code(bucket_path)

    @pytest.mark.asyncio
    async def test_failed_evaluation_handling(self, test_db, r2_storage):
        """Test handling of failed code download during evaluation."""
        from tournament.storage.models import EvaluationModel

        # Create submission with non-existent code path
        submission = SubmissionModel(
            miner_hotkey="5TestMiner",
            miner_uid=50,
            code_hash="nonexistent_hash",
            bucket_path="nonexistent/path/code.py",
            payment_verified=True,
        )
        await test_db.save_submission(submission)
        await test_db.update_submission_status(
            submission.submission_id, SubmissionStatus.EVALUATING
        )

        # Try to download (should fail)
        download_path = Path(f"/tmp/eval/{submission.submission_id}/train.py")
        download_success = await r2_storage.download_code(
            submission.bucket_path, download_path
        )
        assert not download_success

        # Save failed evaluation
        evaluation = EvaluationModel(
            submission_id=submission.submission_id,
            evaluator_hotkey="5TestValidator",
            tokens_per_second=0.0,
            total_tokens=0,
            wall_time_seconds=0.0,
            success=False,
            error="Failed to download code from storage",
        )
        await test_db.save_evaluation(evaluation)

        # Verify failed evaluation
        evals = await test_db.get_evaluations(submission.submission_id)
        assert len(evals) == 1
        assert evals[0].success is False
        assert "Failed to download" in evals[0].error

