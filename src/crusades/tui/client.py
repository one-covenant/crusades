"""API client for crusades endpoints."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from crusades.core.protocols import SubmissionStatus


@dataclass
class CrusadesData:
    """Container for all crusades data."""

    overview: dict[str, Any]
    validator: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    recent: list[dict[str, Any]]
    queue: dict[str, Any]
    history: list[dict[str, Any]]


@dataclass
class SubmissionDetail:
    """Container for submission detail data."""

    submission: dict[str, Any]
    evaluations: list[dict[str, Any]]
    code: str | None


class MockClient:
    """Mock client that returns demo data."""

    def __init__(self):
        from crusades.tui.mock_data import (
            MOCK_CODE,
            MOCK_EVALUATIONS,
            MOCK_HISTORY,
            MOCK_LEADERBOARD,
            MOCK_OVERVIEW,
            MOCK_QUEUE,
            MOCK_RECENT,
            MOCK_SUBMISSIONS,
            MOCK_VALIDATOR,
            get_default_evaluations,
            get_default_submission,
        )

        self._overview = MOCK_OVERVIEW
        self._validator = MOCK_VALIDATOR
        self._queue = MOCK_QUEUE
        self._leaderboard = MOCK_LEADERBOARD
        self._recent = MOCK_RECENT
        self._history = MOCK_HISTORY
        self._submissions = MOCK_SUBMISSIONS
        self._evaluations = MOCK_EVALUATIONS
        self._code = MOCK_CODE
        self._get_default_submission = get_default_submission
        self._get_default_evaluations = get_default_evaluations

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_overview(self) -> dict[str, Any]:
        return self._overview

    def get_validator_status(self) -> dict[str, Any]:
        return self._validator

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._leaderboard[:limit]

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        return self._recent

    def get_queue_stats(self) -> dict[str, Any]:
        return self._queue

    def get_history(self) -> list[dict[str, Any]]:
        return self._history

    def fetch_all(self) -> CrusadesData:
        return CrusadesData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        return self._submissions.get(submission_id, self._get_default_submission(submission_id))

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        return self._evaluations.get(submission_id, self._get_default_evaluations(submission_id))

    def get_submission_code(self, submission_id: str) -> str | None:
        return self._code

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


class DatabaseClient:
    """Client that reads directly from validator's SQLite database."""

    def __init__(self, db_path: str = "crusades.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute query and return list of dicts."""
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def _query_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute query and return single dict or None."""
        cursor = self._conn.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_overview(self) -> dict[str, Any]:
        """Get dashboard overview stats."""
        now = datetime.now()
        # Use space separator to match SQLite datetime format
        day_ago = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        # Total submissions
        total = self._query_one("SELECT COUNT(*) as count FROM submissions")
        total_count = total["count"] if total else 0

        # Submissions in last 24h
        recent = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE created_at > ?", (day_ago,)
        )
        recent_count = recent["count"] if recent else 0

        # Top score (current best)
        top = self._query_one(
            "SELECT final_score FROM submissions WHERE status = 'finished' "
            "ORDER BY final_score DESC LIMIT 1"
        )
        top_score = top["final_score"] if top and top["final_score"] else 0.0

        # Top score from 24 hours ago (best score that existed then)
        top_24h_ago = self._query_one(
            "SELECT MAX(final_score) as score FROM submissions "
            "WHERE status = 'finished' AND created_at <= ?",
            (day_ago,),
        )
        score_24h_ago = top_24h_ago["score"] if top_24h_ago and top_24h_ago["score"] else 0.0

        # Calculate improvement percentage
        if score_24h_ago > 0:
            # Have baseline from 24h ago
            improvement = ((top_score - score_24h_ago) / score_24h_ago) * 100
        elif top_score > 0:
            # No 24h baseline, use earliest score as reference
            first_score = self._query_one(
                "SELECT final_score FROM submissions WHERE status = 'finished' "
                "AND final_score > 0 ORDER BY created_at ASC LIMIT 1"
            )
            baseline = (
                first_score["final_score"] if first_score and first_score["final_score"] else 0.0
            )
            if baseline > 0 and baseline != top_score:
                improvement = ((top_score - baseline) / baseline) * 100
            else:
                improvement = 0.0
        else:
            improvement = 0.0

        # Active miners (unique hotkeys in last 24h)
        miners = self._query_one(
            "SELECT COUNT(DISTINCT miner_hotkey) as count FROM submissions WHERE created_at > ?",
            (day_ago,),
        )
        active_miners = miners["count"] if miners else 0

        return {
            "submissions_24h": recent_count,
            "current_top_score": top_score,
            "score_improvement_24h": round(improvement, 2),
            "total_submissions": total_count,
            "active_miners": active_miners,
        }

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 0:
            return "N/A"

        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def get_validator_status(self) -> dict[str, Any]:
        """Get validator status."""
        now = datetime.now()
        # Use space separator to match SQLite datetime format
        hour_ago = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        # Evaluations in last hour
        evals = self._query_one(
            "SELECT COUNT(*) as count FROM evaluations WHERE created_at > ?", (hour_ago,)
        )
        eval_count = evals["count"] if evals else 0

        # Current evaluation (most recent evaluating submission)
        current = self._query_one(
            "SELECT submission_id FROM submissions WHERE status = 'evaluating' "
            "ORDER BY created_at DESC LIMIT 1"
        )

        # Queue stats
        queue = self.get_queue_stats()

        # Success rate
        finished = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE status = 'finished'"
        )
        failed = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE status LIKE 'failed%'"
        )
        finished_count = finished["count"] if finished else 0
        failed_count = failed["count"] if failed else 0
        total = finished_count + failed_count
        success_rate = (finished_count / total * 100) if total > 0 else 0

        # Calculate uptime from first submission
        first_submission = self._query_one(
            "SELECT MIN(created_at) as start_time FROM submissions"
        )
        if first_submission and first_submission["start_time"]:
            try:
                # Parse the timestamp (handles both ISO and SQLite formats)
                start_str = first_submission["start_time"]
                # Replace space with T for fromisoformat compatibility
                start_str = start_str.replace(" ", "T")
                start_time = datetime.fromisoformat(start_str)
                uptime_seconds = (now - start_time).total_seconds()
                uptime = self._format_duration(uptime_seconds)
            except (ValueError, TypeError):
                uptime = "N/A"
        else:
            uptime = "N/A"

        return {
            "status": "running" if current else "idle",
            "evaluations_completed_1h": eval_count,
            "current_evaluation": current["submission_id"] if current else None,
            "uptime": uptime,
            "queued_count": queue["queued_count"],
            "running_count": queue["running_count"],
            "finished_count": queue["finished_count"],
            "failed_count": queue["failed_count"],
            "success_rate": f"{success_rate:.1f}%",
        }

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get leaderboard entries with 1% threshold for position changes.

        A new submission only gets placed ABOVE an existing entry if it beats
        that entry by more than 1%. This gives incumbents stability - they keep
        their position unless significantly beaten.

        Submissions are processed in order of creation (oldest first), and each
        new submission finds its position by comparing against existing entries.

        Ranks are strictly sequential: 1, 2, 3, 4...
        """
        # Get all finished submissions ordered by created_at (oldest first)
        # This ensures earlier submissions have "incumbent advantage"
        rows = self._query(
            """SELECT s.submission_id, s.miner_hotkey, s.miner_uid, s.final_score,
                      s.created_at, COUNT(e.evaluation_id) as eval_count
               FROM submissions s
               LEFT JOIN evaluations e ON s.submission_id = e.submission_id
               WHERE s.status = ? AND s.final_score IS NOT NULL
               GROUP BY s.submission_id
               ORDER BY s.created_at ASC""",
            (SubmissionStatus.FINISHED,),
        )

        # 1% threshold - must beat incumbent by more than this to take their spot
        rank_threshold = 0.01

        # Build leaderboard incrementally (oldest submissions first)
        leaderboard: list[dict[str, Any]] = []

        for row in rows:
            score = row["final_score"] or 0.0
            entry = {
                "rank": 0,  # Will be assigned later
                "submission_id": row["submission_id"],
                "miner_hotkey": row["miner_hotkey"],
                "miner_uid": row["miner_uid"],
                "final_score": score,
                "num_evaluations": row["eval_count"],
                "created_at": row["created_at"],
            }

            # Find insertion position (scan from top)
            # New entry only goes above existing if it beats them by >1%
            insert_pos = len(leaderboard)  # Default: add at bottom
            for i, existing in enumerate(leaderboard):
                threshold_score = existing["final_score"] * (1 + rank_threshold)
                if score > threshold_score:
                    # Beats this incumbent by >1%, insert here
                    insert_pos = i
                    break

            leaderboard.insert(insert_pos, entry)

        # Assign sequential ranks
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard[:limit]

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        """Get recent submissions (all final statuses)."""
        rows = self._query(
            "SELECT submission_id, miner_hotkey, miner_uid, status, final_score, "
            "created_at, error_message FROM submissions "
            "WHERE status IN ('finished', 'failed_validation', 'failed_evaluation', "
            "'failed_copy', 'error') "
            "ORDER BY created_at DESC LIMIT 20"
        )

        recent = []
        for row in rows:
            recent.append(
                {
                    "submission_id": row["submission_id"],
                    "miner_hotkey": row["miner_hotkey"],
                    "miner_uid": row["miner_uid"],
                    "status": row["status"],
                    "final_score": row["final_score"],  # Match app.py expectation
                    "created_at": row["created_at"],
                    "error": row["error_message"],
                }
            )
        return recent

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        # Pending + validating = queued
        queued = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE status IN ('pending', 'validating')"
        )

        # Running = evaluating
        running = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE status = 'evaluating'"
        )

        # Finished
        finished = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE status = 'finished'"
        )

        # All failures
        failed = self._query_one(
            "SELECT COUNT(*) as count FROM submissions WHERE status IN (?, ?, ?)",
            (
                SubmissionStatus.FAILED_VALIDATION,
                SubmissionStatus.FAILED_EVALUATION,
                SubmissionStatus.ERROR,
            ),
        )

        # Average score
        avg = self._query_one(
            "SELECT AVG(final_score) as avg FROM submissions "
            "WHERE status = 'finished' AND final_score IS NOT NULL"
        )

        return {
            "queued_count": queued["count"] if queued else 0,
            "running_count": running["count"] if running else 0,
            "finished_count": finished["count"] if finished else 0,
            "failed_count": failed["count"] if failed else 0,
            "avg_wait_time_seconds": 0.0,
            "avg_score": avg["avg"] if avg and avg["avg"] else 0.0,
            "pending_count": queued["count"] if queued else 0,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Get TPS history for chart - shows top TPS progression over time."""
        # Get ALL finished submissions ordered by time, tracking best score
        rows = self._query(
            "SELECT submission_id, final_score as tps, created_at "
            "FROM submissions "
            "WHERE status = 'finished' AND final_score IS NOT NULL "
            "ORDER BY created_at ASC"
        )

        history = []
        running_best = 0.0

        for row in rows:
            tps = row["tps"] or 0.0
            # Track the running best TPS
            if tps > running_best:
                running_best = tps

            history.append(
                {
                    "submission_id": row["submission_id"],
                    "tps": running_best,  # Show best TPS so far
                    "timestamp": row["created_at"],
                }
            )
        return history

    def fetch_all(self) -> CrusadesData:
        """Fetch all crusades data."""
        return CrusadesData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        """Get submission details."""
        row = self._query_one("SELECT * FROM submissions WHERE submission_id = ?", (submission_id,))
        return row if row else {}

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        """Get evaluations for a submission."""
        return self._query(
            "SELECT * FROM evaluations WHERE submission_id = ? ORDER BY created_at",
            (submission_id,),
        )

    def get_submission_code(self, submission_id: str) -> str | None:
        """Get code content from database (stored after evaluation).

        Code is stored in code_content column after validator evaluates.
        """
        row = self._query_one(
            "SELECT code_content, code_hash FROM submissions WHERE submission_id = ?",
            (submission_id,),
        )
        if not row:
            return None

        code = row.get("code_content")
        if code:
            return code

        # Fallback: code not yet stored (still evaluating)
        return f"# Code not yet available\n# Submission may still be evaluating\n# code_hash: {row.get('code_hash', 'N/A')}"

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        """Fetch all details for a submission."""
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


class CrusadesClient:
    """Client for fetching crusades data from API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get(self, endpoint: str) -> dict[str, Any] | list[Any]:
        """Make a GET request to the API."""
        try:
            response = self._client.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return {}

    def get_overview(self) -> dict[str, Any]:
        """Get dashboard overview stats."""
        data = self._get("/api/stats/overview")
        if not data:
            return {
                "submissions_24h": 0,
                "current_top_score": 0.0,
                "score_improvement_24h": 0.0,
                "total_submissions": 0,
                "active_miners": 0,
            }
        return data

    def get_validator_status(self) -> dict[str, Any]:
        """Get validator status."""
        data = self._get("/api/stats/validator")
        if not data:
            return {
                "status": "unknown",
                "evaluations_completed_1h": 0,
                "current_evaluation": None,
                "uptime": "N/A",
            }
        return data

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get leaderboard entries."""
        data = self._get(f"/leaderboard?limit={limit}")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        """Get recent submissions."""
        data = self._get("/api/stats/recent")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        data = self._get("/api/stats/queue")
        if not data:
            return {
                "queued_count": 0,
                "running_count": 0,
                "finished_count": 0,
                "failed_count": 0,
                "avg_wait_time_seconds": 0.0,
                "avg_score": 0.0,
                "pending_count": 0,
            }
        return data

    def get_history(self) -> list[dict[str, Any]]:
        """Get TPS history."""
        data = self._get("/api/stats/history")
        if not data or not isinstance(data, list):
            return []
        return data

    def fetch_all(self) -> CrusadesData:
        """Fetch all crusades data."""
        return CrusadesData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        """Get submission details."""
        data = self._get(f"/api/submissions/{submission_id}")
        if not data:
            return {}
        return data

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        """Get evaluations for a submission."""
        data = self._get(f"/api/submissions/{submission_id}/evaluations")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_submission_code(self, submission_id: str) -> str | None:
        """Get code for a submission."""
        data = self._get(f"/api/submissions/{submission_id}/code")
        if not data or not isinstance(data, dict):
            return None
        return data.get("code")

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        """Fetch all details for a submission."""
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


def format_time_ago(timestamp_str: str | None) -> str:
    """Format a timestamp as 'X ago' string."""
    if not timestamp_str:
        return "N/A"
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(timestamp_str)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt

        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes}m ago"
        return "just now"
    except (ValueError, TypeError):
        return "N/A"
