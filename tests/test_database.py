"""
Unit tests for Database Operations

Tests CRUD operations for submissions and evaluations.
"""

from datetime import datetime
from uuid import uuid4

import pytest

from tournament.core.protocols import SubmissionStatus
from tournament.storage.database import Database
from tournament.storage.models import EvaluationModel, SubmissionModel


@pytest.fixture
async def test_db():
    """Create a test database with in-memory SQLite."""
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def sample_submission():
    """Create a sample submission for testing."""
    return SubmissionModel(
        miner_hotkey="test_hotkey_123",
        miner_uid=42,
        code_hash="abc123def456",
        bucket_path="submissions/test/abc123.py",
    )


@pytest.fixture
def sample_evaluation(sample_submission):
    """Create a sample evaluation for testing."""
    return EvaluationModel(
        submission_id=sample_submission.submission_id,
        evaluator_hotkey="validator_hotkey_1",
        tokens_per_second=10000.0,
        total_tokens=100000,
        wall_time_seconds=10.0,
        success=True,
    )


class TestSubmissionOperations:
    """Tests for submission CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_submission(self, test_db, sample_submission):
        """Can save a new submission to database."""
        await test_db.save_submission(sample_submission)

        # Retrieve it
        retrieved = await test_db.get_submission(sample_submission.submission_id)

        assert retrieved is not None
        assert retrieved.miner_hotkey == sample_submission.miner_hotkey
        assert retrieved.miner_uid == sample_submission.miner_uid

    @pytest.mark.asyncio
    async def test_get_nonexistent_submission_returns_none(self, test_db):
        """Getting non-existent submission returns None."""
        result = await test_db.get_submission("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_submission_status(self, test_db, sample_submission):
        """Can update submission status."""
        await test_db.save_submission(sample_submission)

        # Update status
        await test_db.update_submission_status(
            sample_submission.submission_id,
            SubmissionStatus.EVALUATING,
        )

        # Verify update
        updated = await test_db.get_submission(sample_submission.submission_id)
        assert updated.status == SubmissionStatus.EVALUATING

    @pytest.mark.asyncio
    async def test_update_submission_status_with_error(self, test_db, sample_submission):
        """Can update submission status with error message."""
        await test_db.save_submission(sample_submission)

        # Update with error
        error_msg = "Validation failed: missing function"
        await test_db.update_submission_status(
            sample_submission.submission_id,
            SubmissionStatus.FAILED_VALIDATION,
            error_message=error_msg,
        )

        # Verify update
        updated = await test_db.get_submission(sample_submission.submission_id)
        assert updated.status == SubmissionStatus.FAILED_VALIDATION
        assert updated.error_message == error_msg

    @pytest.mark.asyncio
    async def test_update_submission_score(self, test_db, sample_submission):
        """Can update submission final score."""
        await test_db.save_submission(sample_submission)

        # Update score
        final_score = 12345.67
        await test_db.update_submission_score(
            sample_submission.submission_id,
            final_score,
        )

        # Verify update
        updated = await test_db.get_submission(sample_submission.submission_id)
        assert updated.final_score == final_score
        assert updated.status == SubmissionStatus.FINISHED

    @pytest.mark.asyncio
    async def test_get_pending_submissions(self, test_db):
        """Can retrieve all pending submissions."""
        # Create multiple submissions with different statuses
        sub1 = SubmissionModel(
            miner_hotkey="hotkey1",
            miner_uid=1,
            code_hash="hash1",
            bucket_path="path1",
        )
        sub2 = SubmissionModel(
            miner_hotkey="hotkey2",
            miner_uid=2,
            code_hash="hash2",
            bucket_path="path2",
        )
        sub3 = SubmissionModel(
            miner_hotkey="hotkey3",
            miner_uid=3,
            code_hash="hash3",
            bucket_path="path3",
        )

        await test_db.save_submission(sub1)
        await test_db.save_submission(sub2)
        await test_db.save_submission(sub3)

        # Update one to evaluating
        await test_db.update_submission_status(sub2.submission_id, SubmissionStatus.EVALUATING)

        # Get pending
        pending = await test_db.get_pending_submissions()

        assert len(pending) == 2
        pending_ids = {s.submission_id for s in pending}
        assert sub1.submission_id in pending_ids
        assert sub3.submission_id in pending_ids

    @pytest.mark.asyncio
    async def test_get_evaluating_submissions(self, test_db):
        """Can retrieve all evaluating submissions."""
        # Create submissions
        sub1 = SubmissionModel(
            miner_hotkey="hotkey1",
            miner_uid=1,
            code_hash="hash1",
            bucket_path="path1",
        )
        sub2 = SubmissionModel(
            miner_hotkey="hotkey2",
            miner_uid=2,
            code_hash="hash2",
            bucket_path="path2",
        )

        await test_db.save_submission(sub1)
        await test_db.save_submission(sub2)

        # Update one to evaluating
        await test_db.update_submission_status(sub1.submission_id, SubmissionStatus.EVALUATING)

        # Get evaluating
        evaluating = await test_db.get_evaluating_submissions()

        assert len(evaluating) == 1
        assert evaluating[0].submission_id == sub1.submission_id


class TestEvaluationOperations:
    """Tests for evaluation CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_evaluation(self, test_db, sample_submission):
        """Can save evaluation result."""
        # Save submission first
        await test_db.save_submission(sample_submission)

        # Create evaluation for the saved submission
        evaluation = EvaluationModel(
            submission_id=sample_submission.submission_id,
            evaluator_hotkey="validator_hotkey_1",
            tokens_per_second=10000.0,
            total_tokens=100000,
            wall_time_seconds=10.0,
            success=True,
        )
        await test_db.save_evaluation(evaluation)

        # Retrieve evaluations
        evals = await test_db.get_evaluations(sample_submission.submission_id)

        assert len(evals) == 1
        assert evals[0].evaluator_hotkey == evaluation.evaluator_hotkey

    @pytest.mark.asyncio
    async def test_get_evaluations_for_submission(self, test_db, sample_submission):
        """Can retrieve all evaluations for a submission."""
        await test_db.save_submission(sample_submission)

        # Add multiple evaluations
        eval1 = EvaluationModel(
            submission_id=sample_submission.submission_id,
            evaluator_hotkey="val1",
            tokens_per_second=1000.0,
            total_tokens=10000,
            wall_time_seconds=10.0,
            success=True,
        )
        eval2 = EvaluationModel(
            submission_id=sample_submission.submission_id,
            evaluator_hotkey="val2",
            tokens_per_second=1200.0,
            total_tokens=10000,
            wall_time_seconds=8.33,
            success=True,
        )

        await test_db.save_evaluation(eval1)
        await test_db.save_evaluation(eval2)

        # Get evaluations
        evals = await test_db.get_evaluations(sample_submission.submission_id)

        assert len(evals) == 2

    @pytest.mark.asyncio
    async def test_count_evaluations(self, test_db, sample_submission):
        """Can count successful evaluations."""
        await test_db.save_submission(sample_submission)

        # Add evaluations (mix of success/failure)
        eval1 = EvaluationModel(
            submission_id=sample_submission.submission_id,
            evaluator_hotkey="val1",
            tokens_per_second=1000.0,
            total_tokens=10000,
            wall_time_seconds=10.0,
            success=True,
        )
        eval2 = EvaluationModel(
            submission_id=sample_submission.submission_id,
            evaluator_hotkey="val2",
            tokens_per_second=0.0,
            total_tokens=0,
            wall_time_seconds=0.0,
            success=False,
            error="Verification failed",
        )
        eval3 = EvaluationModel(
            submission_id=sample_submission.submission_id,
            evaluator_hotkey="val3",
            tokens_per_second=1100.0,
            total_tokens=10000,
            wall_time_seconds=9.09,
            success=True,
        )

        await test_db.save_evaluation(eval1)
        await test_db.save_evaluation(eval2)
        await test_db.save_evaluation(eval3)

        # Count should only include successful
        count = await test_db.count_evaluations(sample_submission.submission_id)
        assert count == 2


class TestLeaderboardOperations:
    """Tests for leaderboard queries."""

    @pytest.mark.asyncio
    async def test_get_top_submission(self, test_db):
        """Can get the top-scoring finished submission."""
        # Create submissions with different scores
        sub1 = SubmissionModel(
            miner_hotkey="hotkey1",
            miner_uid=1,
            code_hash="hash1",
            bucket_path="path1",
        )
        sub2 = SubmissionModel(
            miner_hotkey="hotkey2",
            miner_uid=2,
            code_hash="hash2",
            bucket_path="path2",
        )
        sub3 = SubmissionModel(
            miner_hotkey="hotkey3",
            miner_uid=3,
            code_hash="hash3",
            bucket_path="path3",
        )

        await test_db.save_submission(sub1)
        await test_db.save_submission(sub2)
        await test_db.save_submission(sub3)

        # Set scores
        await test_db.update_submission_score(sub1.submission_id, 1000.0)
        await test_db.update_submission_score(sub2.submission_id, 2000.0)  # Highest
        await test_db.update_submission_score(sub3.submission_id, 1500.0)

        # Get top
        top = await test_db.get_top_submission()

        assert top is not None
        assert top.submission_id == sub2.submission_id
        assert top.final_score == 2000.0

    @pytest.mark.asyncio
    async def test_get_top_submission_no_finished(self, test_db, sample_submission):
        """Returns None when no finished submissions."""
        await test_db.save_submission(sample_submission)

        top = await test_db.get_top_submission()
        assert top is None

    @pytest.mark.asyncio
    async def test_get_leaderboard(self, test_db):
        """Can get leaderboard sorted by score."""
        # Create multiple finished submissions
        submissions = []
        scores = [1000.0, 3000.0, 2000.0, 500.0]

        for i, score in enumerate(scores):
            sub = SubmissionModel(
                miner_hotkey=f"hotkey{i}",
                miner_uid=i,
                code_hash=f"hash{i}",
                bucket_path=f"path{i}",
            )
            await test_db.save_submission(sub)
            await test_db.update_submission_score(sub.submission_id, score)
            submissions.append(sub)

        # Get leaderboard
        leaderboard = await test_db.get_leaderboard(limit=10)

        assert len(leaderboard) == 4
        # Should be sorted by score descending
        assert leaderboard[0].final_score == 3000.0
        assert leaderboard[1].final_score == 2000.0
        assert leaderboard[2].final_score == 1000.0
        assert leaderboard[3].final_score == 500.0

    @pytest.mark.asyncio
    async def test_get_leaderboard_with_limit(self, test_db):
        """Leaderboard respects limit parameter."""
        # Create 5 submissions
        for i in range(5):
            sub = SubmissionModel(
                miner_hotkey=f"hotkey{i}",
                miner_uid=i,
                code_hash=f"hash{i}",
                bucket_path=f"path{i}",
            )
            await test_db.save_submission(sub)
            await test_db.update_submission_score(sub.submission_id, float(i * 1000))

        # Get top 3
        leaderboard = await test_db.get_leaderboard(limit=3)

        assert len(leaderboard) == 3

