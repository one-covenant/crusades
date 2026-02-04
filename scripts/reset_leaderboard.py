#!/usr/bin/env python3
"""Reset leaderboard state after removing invalid/cheating submissions.

This script safely removes submissions and resets the adaptive threshold
and winner tracking state so the system can recalculate from legitimate data.

Usage:
    # List current state (safe, read-only)
    uv run scripts/reset_leaderboard.py --status

    # Remove a specific submission and reset threshold
    uv run scripts/reset_leaderboard.py --remove <submission_id>

    # Remove multiple submissions
    uv run scripts/reset_leaderboard.py --remove id1 --remove id2

    # Only reset threshold (keep submissions)
    uv run scripts/reset_leaderboard.py --reset-threshold

    # Full reset: clear all submissions, threshold, and winner state
    uv run scripts/reset_leaderboard.py --full-reset
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def get_db_path(args_db: str) -> str:
    """Find the database file."""
    db_path = args_db
    if not Path(db_path).exists():
        project_root = Path(__file__).parent.parent
        db_path = project_root / "crusades.db"
        if not db_path.exists():
            print(f"Database not found: {args_db}")
            sys.exit(1)
        db_path = str(db_path)
    return db_path


def show_status(db_path: str):
    """Show current leaderboard state."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()

        print("=" * 70)
        print("CURRENT LEADERBOARD STATE")
        print("=" * 70)

        # Adaptive threshold
        print("\n[Adaptive Threshold]")
        cur.execute("SELECT * FROM adaptive_threshold")
        row = cur.fetchone()
        if row:
            print(f"  Current threshold: {row['current_threshold']:.4f} ({row['current_threshold']*100:.2f}%)")
            print(f"  Last improvement:  {row['last_improvement']:.4f} ({row['last_improvement']*100:.2f}%)")
            print(f"  Last update block: {row['last_update_block']}")
            print(f"  Updated at:        {row['updated_at']}")
        else:
            print("  (empty - using base threshold)")

        # Previous winner
        print("\n[Previous Winner State]")
        cur.execute("SELECT * FROM validator_state WHERE key LIKE 'previous_winner%'")
        rows = cur.fetchall()
        if rows:
            for r in rows:
                print(f"  {r['key']}: {r['value']}")
        else:
            print("  (no winner tracked)")

        # Top submissions
        print("\n[Top 10 Submissions by MFU]")
        cur.execute("""
            SELECT submission_id, miner_uid, final_score, status, created_at
            FROM submissions
            WHERE final_score IS NOT NULL
            ORDER BY final_score DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        if rows:
            for i, r in enumerate(rows, 1):
                status_icon = "✓" if r["status"] == "finished" else "✗"
                print(f"  #{i}: {r['submission_id']:<30} UID={r['miner_uid']:<3} "
                      f"MFU={r['final_score']:.2f}% [{status_icon}]")
        else:
            print("  (no submissions)")

        print()

    finally:
        conn.close()


def remove_submission(db_path: str, submission_id: str, dry_run: bool = False) -> bool:
    """Remove a specific submission."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()

        # Check if submission exists
        cur.execute(
            "SELECT submission_id, miner_uid, final_score FROM submissions WHERE submission_id = ?",
            (submission_id,)
        )
        row = cur.fetchone()

        if not row:
            print(f"  Submission '{submission_id}' not found.")
            return False

        print(f"  Found: {row['submission_id']} (UID={row['miner_uid']}, MFU={row['final_score']}%)")

        if dry_run:
            print("  [DRY RUN] Would delete this submission")
            return True

        # Delete evaluations first (foreign key)
        cur.execute("DELETE FROM evaluations WHERE submission_id = ?", (submission_id,))
        eval_count = cur.rowcount

        # Delete submission
        cur.execute("DELETE FROM submissions WHERE submission_id = ?", (submission_id,))

        conn.commit()
        print(f"  Deleted submission and {eval_count} evaluation(s)")
        return True

    finally:
        conn.close()


def reset_threshold(db_path: str, dry_run: bool = False):
    """Reset adaptive threshold to base."""
    conn = sqlite3.connect(db_path)

    try:
        cur = conn.cursor()

        if dry_run:
            print("  [DRY RUN] Would clear adaptive_threshold table")
            print("  [DRY RUN] Would clear previous_winner_id and previous_winner_score")
            return

        # Clear adaptive threshold
        cur.execute("DELETE FROM adaptive_threshold")
        threshold_deleted = cur.rowcount

        # Clear previous winner tracking
        cur.execute("DELETE FROM validator_state WHERE key IN ('previous_winner_id', 'previous_winner_score')")
        state_deleted = cur.rowcount

        conn.commit()
        print(f"  Cleared {threshold_deleted} threshold record(s)")
        print(f"  Cleared {state_deleted} winner state record(s)")
        print("  System will use base_threshold (1%) until new leader emerges")

    finally:
        conn.close()


def full_reset(db_path: str, dry_run: bool = False):
    """Full reset: clear all submissions, evaluations, threshold, and state."""
    conn = sqlite3.connect(db_path)

    try:
        cur = conn.cursor()

        if dry_run:
            cur.execute("SELECT COUNT(*) FROM submissions")
            sub_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM evaluations")
            eval_count = cur.fetchone()[0]
            print(f"  [DRY RUN] Would delete {sub_count} submission(s)")
            print(f"  [DRY RUN] Would delete {eval_count} evaluation(s)")
            print("  [DRY RUN] Would clear adaptive_threshold")
            print("  [DRY RUN] Would clear previous_winner state")
            return

        # Delete all evaluations
        cur.execute("DELETE FROM evaluations")
        eval_count = cur.rowcount

        # Delete all submissions
        cur.execute("DELETE FROM submissions")
        sub_count = cur.rowcount

        # Clear adaptive threshold
        cur.execute("DELETE FROM adaptive_threshold")

        # Clear previous winner tracking
        cur.execute("DELETE FROM validator_state WHERE key IN ('previous_winner_id', 'previous_winner_score')")

        conn.commit()
        print(f"  Deleted {sub_count} submission(s)")
        print(f"  Deleted {eval_count} evaluation(s)")
        print("  Cleared adaptive threshold")
        print("  Cleared previous winner state")
        print("  Leaderboard is now empty")

    finally:
        conn.close()


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"\n{message} [y/N]: ").strip().lower()
    return response in ("y", "yes")


def main():
    parser = argparse.ArgumentParser(
        description="Reset leaderboard state after removing invalid submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current state
  uv run scripts/reset_leaderboard.py --status

  # Remove a cheater's submission and reset threshold
  uv run scripts/reset_leaderboard.py --remove v2_commit_87982_1

  # Preview what would be deleted (dry run)
  uv run scripts/reset_leaderboard.py --remove v2_commit_87982_1 --dry-run

  # Only reset threshold/winner state (keep all submissions)
  uv run scripts/reset_leaderboard.py --reset-threshold

  # Full reset (delete everything)
  uv run scripts/reset_leaderboard.py --full-reset
        """,
    )
    parser.add_argument("--status", action="store_true", help="Show current leaderboard state")
    parser.add_argument("--remove", action="append", metavar="ID", help="Remove submission(s) by ID")
    parser.add_argument("--reset-threshold", action="store_true", help="Reset adaptive threshold only")
    parser.add_argument("--full-reset", action="store_true", help="Delete ALL data (submissions, threshold, state)")
    parser.add_argument("--db", default="crusades.db", help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    # Default to --status if no action specified
    if not any([args.status, args.remove, args.reset_threshold, args.full_reset]):
        args.status = True

    db_path = get_db_path(args.db)

    # Always show status first
    if args.status or args.remove or args.reset_threshold or args.full_reset:
        show_status(db_path)

    # Handle --remove
    if args.remove:
        print("\n[Removing Submissions]")
        for submission_id in args.remove:
            remove_submission(db_path, submission_id, dry_run=args.dry_run)

        if not args.dry_run:
            print("\n[Resetting Threshold & Winner State]")
            reset_threshold(db_path)

        if not args.dry_run:
            print("\n[Updated State]")
            show_status(db_path)

    # Handle --reset-threshold only
    elif args.reset_threshold:
        if not args.yes and not args.dry_run:
            if not confirm_action("Reset adaptive threshold and winner state?"):
                print("Aborted.")
                return

        print("\n[Resetting Threshold & Winner State]")
        reset_threshold(db_path, dry_run=args.dry_run)

    # Handle --full-reset
    elif args.full_reset:
        if not args.yes and not args.dry_run:
            if not confirm_action("⚠️  DELETE ALL DATA? This cannot be undone!"):
                print("Aborted.")
                return

        print("\n[Full Reset]")
        full_reset(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
