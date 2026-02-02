"""Database abstraction layer."""

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..config import get_hparams
from ..core.protocols import SubmissionStatus
from .models import (
    AdaptiveThresholdModel,
    Base,
    EvaluationModel,
    SubmissionModel,
    ValidatorStateModel,
)


class Database:
    """Async database interface."""

    def __init__(self, url: str | None = None):
        if url is None:
            url = get_hparams().storage.database_url
        self.engine = create_async_engine(url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()

    # Submission operations

    async def save_submission(self, submission: SubmissionModel) -> None:
        """Save a new submission."""
        async with self.session_factory() as session:
            session.add(submission)
            await session.commit()

    async def get_submission(self, submission_id: str) -> SubmissionModel | None:
        """Get submission by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            return result.scalar_one_or_none()

    async def update_submission_status(
        self,
        submission_id: str,
        status: SubmissionStatus,
        error_message: str | None = None,
    ) -> None:
        """Update submission status."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.status = status
                if error_message:
                    submission.error_message = error_message
                await session.commit()

    async def update_submission_score(self, submission_id: str, final_score: float) -> None:
        """Update submission final score."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.final_score = final_score
                submission.status = SubmissionStatus.FINISHED
                await session.commit()

    async def update_submission_code(self, submission_id: str, code: str) -> None:
        """Store miner's code content after evaluation.

        This is called by the validator after downloading and evaluating
        the miner's code, so it can be displayed on the dashboard.

        Args:
            submission_id: Submission ID
            code: Miner's train.py code content
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).where(SubmissionModel.submission_id == submission_id)
            )
            submission = result.scalar_one_or_none()
            if submission:
                submission.code_content = code
                await session.commit()

    async def get_submission_code(self, submission_id: str) -> str | None:
        """Get miner's code content for a submission.

        Args:
            submission_id: Submission ID

        Returns:
            Code content or None if not stored
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel.code_content).where(
                    SubmissionModel.submission_id == submission_id
                )
            )
            return result.scalar_one_or_none()

    async def get_all_submissions(self) -> list[SubmissionModel]:
        """Get all submissions."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel).order_by(desc(SubmissionModel.created_at))
            )
            return list(result.scalars().all())

    async def get_pending_submissions(self, spec_version: int | None = None) -> list[SubmissionModel]:
        """Get submissions pending validation.

        Args:
            spec_version: If provided, only return submissions matching this version
        """
        async with self.session_factory() as session:
            query = select(SubmissionModel).where(SubmissionModel.status == SubmissionStatus.PENDING)
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(query.order_by(SubmissionModel.created_at))
            return list(result.scalars().all())

    async def get_evaluating_submissions(self, spec_version: int | None = None) -> list[SubmissionModel]:
        """Get submissions currently being evaluated.

        Args:
            spec_version: If provided, only return submissions matching this version
        """
        async with self.session_factory() as session:
            query = select(SubmissionModel).where(SubmissionModel.status == SubmissionStatus.EVALUATING)
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(query.order_by(SubmissionModel.created_at))
            return list(result.scalars().all())

    async def get_latest_submission_by_hotkey(self, hotkey: str) -> SubmissionModel | None:
        """Get the most recent submission from a miner.

        Used for rate limiting - checking when miner last submitted.

        Args:
            hotkey: Miner's hotkey

        Returns:
            Most recent submission or None
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(SubmissionModel)
                .where(SubmissionModel.miner_hotkey == hotkey)
                .order_by(desc(SubmissionModel.created_at))
                .limit(1)
            )
            return result.scalar_one_or_none()

    # Evaluation operations

    async def save_evaluation(self, evaluation: EvaluationModel) -> None:
        """Save a new evaluation result."""
        async with self.session_factory() as session:
            session.add(evaluation)
            await session.commit()

    async def get_evaluations(self, submission_id: str) -> list[EvaluationModel]:
        """Get all evaluations for a submission."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(EvaluationModel)
                .where(EvaluationModel.submission_id == submission_id)
                .order_by(EvaluationModel.created_at)
            )
            return list(result.scalars().all())

    async def count_evaluations(self, submission_id: str) -> int:
        """Count ALL evaluations for a submission (including failed)."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(EvaluationModel)
                .where(
                    EvaluationModel.submission_id == submission_id,
                )
            )
            return result.scalar() or 0

    # Leaderboard operations

    async def get_top_submission(self, spec_version: int | None = None) -> SubmissionModel | None:
        """Get the top-scoring finished submission (raw, no threshold).

        Args:
            spec_version: If provided, only consider submissions from this version
        """
        async with self.session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(
                query.order_by(desc(SubmissionModel.final_score)).limit(1)
            )
            return result.scalar_one_or_none()

    async def get_leaderboard_winner(
        self, threshold: float = 0.01, spec_version: int | None = None
    ) -> SubmissionModel | None:
        """Get the rank 1 submission from leaderboard with threshold.

        A new submission only beats an incumbent if it's more than `threshold`
        better. This gives stability to leaders.

        Args:
            threshold: Minimum improvement ratio to beat incumbent
            spec_version: If provided, only consider submissions from this version

        Returns:
            The submission at rank 1, or None if no finished submissions.
        """
        async with self.session_factory() as session:
            # Get all finished submissions ordered by created_at (oldest first)
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(query.order_by(SubmissionModel.created_at.asc()))
            submissions = list(result.scalars().all())

            if not submissions:
                return None

            # Build leaderboard with threshold logic
            leaderboard: list[SubmissionModel] = []

            for submission in submissions:
                score = submission.final_score or 0.0

                # Find insertion position
                insert_pos = len(leaderboard)
                for i, existing in enumerate(leaderboard):
                    existing_score = existing.final_score or 0.0
                    threshold_score = existing_score * (1 + threshold)
                    if score > threshold_score:
                        insert_pos = i
                        break

                leaderboard.insert(insert_pos, submission)

            return leaderboard[0] if leaderboard else None

    async def get_leaderboard(
        self, limit: int = 100, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get top submissions by score.

        Args:
            limit: Maximum number of submissions to return
            spec_version: If provided, only show submissions from this version
        """
        async with self.session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(
                query.order_by(desc(SubmissionModel.final_score)).limit(limit)
            )
            return list(result.scalars().all())

    async def get_top_submissions(
        self, limit: int = 5, spec_version: int | None = None
    ) -> list[SubmissionModel]:
        """Get top N submissions by score for similarity checking.

        Args:
            limit: Number of top submissions to return (default 5)
            spec_version: If provided, only consider submissions from this version

        Returns:
            List of top submissions sorted by score (highest first)
        """
        async with self.session_factory() as session:
            query = select(SubmissionModel).where(
                SubmissionModel.status == SubmissionStatus.FINISHED,
                SubmissionModel.final_score.isnot(None),
            )
            if spec_version is not None:
                query = query.where(SubmissionModel.spec_version == spec_version)
            result = await session.execute(
                query.order_by(desc(SubmissionModel.final_score)).limit(limit)
            )
            return list(result.scalars().all())

    # Validator state operations

    async def get_validator_state(self, key: str) -> str | None:
        """Get a validator state value."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ValidatorStateModel.value).where(ValidatorStateModel.key == key)
            )
            return result.scalar_one_or_none()

    async def set_validator_state(self, key: str, value: str) -> None:
        """Set a validator state value (upsert)."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(ValidatorStateModel).where(ValidatorStateModel.key == key)
            )
            state = result.scalar_one_or_none()
            if state:
                state.value = value
            else:
                session.add(ValidatorStateModel(key=key, value=value))
            await session.commit()

    # Adaptive threshold operations

    async def get_adaptive_threshold(
        self,
        current_block: int,
        base_threshold: float = 0.01,
        decay_rate: float = 0.95,
        decay_interval_blocks: int = 100,
    ) -> float:
        """Get the current adaptive threshold with decay applied.

        The threshold decays towards base_threshold over time.
        threshold = base + (peak - base) * decay^(blocks_since_update / interval)

        Args:
            current_block: Current block number
            base_threshold: Minimum threshold (default 1%)
            decay_rate: Decay multiplier per interval (default 0.95)
            decay_interval_blocks: Blocks between decay steps

        Returns:
            Current threshold value
        """
        async with self.session_factory() as session:
            result = await session.execute(select(AdaptiveThresholdModel).where(AdaptiveThresholdModel.id == 1))
            state = result.scalar_one_or_none()

            if state is None:
                return base_threshold

            # Calculate decay based on blocks elapsed
            blocks_elapsed = max(0, current_block - state.last_update_block)
            decay_steps = blocks_elapsed / decay_interval_blocks
            decay_factor = decay_rate ** decay_steps

            # Decay from current threshold towards base
            decayed = base_threshold + (state.current_threshold - base_threshold) * decay_factor
            return max(base_threshold, decayed)

    async def update_adaptive_threshold(
        self,
        new_score: float,
        old_score: float,
        current_block: int,
        base_threshold: float = 0.01,
        max_threshold: float = 0.50,
        improvement_multiplier: float = 2.0,
    ) -> float:
        """Update threshold when a new leader is established.

        Threshold increases based on the improvement magnitude:
        - Small improvement (1-5%): threshold stays low
        - Big improvement (50%+): threshold becomes high

        Args:
            new_score: New leader's score
            old_score: Previous leader's score
            current_block: Current block number
            base_threshold: Minimum threshold
            max_threshold: Maximum threshold cap
            improvement_multiplier: Threshold = improvement * multiplier

        Returns:
            New threshold value
        """
        async with self.session_factory() as session:
            result = await session.execute(select(AdaptiveThresholdModel).where(AdaptiveThresholdModel.id == 1))
            state = result.scalar_one_or_none()

            # Calculate improvement ratio
            if old_score > 0:
                improvement = (new_score - old_score) / old_score
            else:
                improvement = 1.0  # First submission, use max

            # New threshold based on improvement
            new_threshold = min(max_threshold, max(base_threshold, improvement * improvement_multiplier))

            if state is None:
                # Create new state
                state = AdaptiveThresholdModel(
                    id=1,
                    current_threshold=new_threshold,
                    last_improvement=improvement,
                    last_update_block=current_block,
                )
                session.add(state)
            else:
                # Update existing state
                state.current_threshold = new_threshold
                state.last_improvement = improvement
                state.last_update_block = current_block

            await session.commit()
            return new_threshold


# Global instance
_database: Database | None = None


async def get_database() -> Database:
    """Get or create global database instance."""
    global _database
    if _database is None:
        _database = Database()
        await _database.initialize()
    return _database
