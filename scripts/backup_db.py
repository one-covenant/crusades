#!/usr/bin/env python3
"""Periodic backup of crusades.db to Cloudflare R2."""

import logging
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "crusades.db"
ENV_FILE = PROJECT_DIR / ".env"
MAX_BACKUPS = 48

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("backup")


def load_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def get_r2_client(env: dict):
    return boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT"],
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def rotate_r2(client, bucket: str, prefix: str, keep: int) -> None:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = resp.get("Contents", [])
    objects.sort(key=lambda o: o["LastModified"], reverse=True)
    for old in objects[keep:]:
        client.delete_object(Bucket=bucket, Key=old["Key"])
        log.info("Deleted R2: %s", old["Key"])


def main() -> int:
    if not DB_PATH.exists():
        log.error("DB not found: %s", DB_PATH)
        return 1

    if not ENV_FILE.exists():
        log.error(".env not found: %s", ENV_FILE)
        return 1

    env = load_env(ENV_FILE)
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"):
        if key not in env:
            log.error("Missing %s in .env", key)
            return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    backup_name = f"crusades_{timestamp}.db"

    # 1. Safe SQLite snapshot to a temp file (safe even while validator is writing)
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            tmp_path = tmp.name
        conn_src = sqlite3.connect(str(DB_PATH))
        conn_dst = sqlite3.connect(tmp_path)
        try:
            conn_src.backup(conn_dst)
        finally:
            conn_dst.close()
            conn_src.close()
        size = Path(tmp_path).stat().st_size
        log.info("Snapshot: %d bytes", size)
    except Exception as e:
        log.error("SQLite snapshot failed: %s", e)
        return 1

    # 2. Upload to R2
    try:
        client = get_r2_client(env)
        client.upload_file(tmp_path, env["R2_BUCKET"], f"backups/{backup_name}")
        log.info("Uploaded to R2: backups/%s", backup_name)
    except Exception as e:
        log.error("R2 upload failed: %s", e)
        return 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 3. Rotate old backups on R2
    try:
        rotate_r2(client, env["R2_BUCKET"], "backups/", MAX_BACKUPS)
    except Exception as e:
        log.warning("Rotation error (non-fatal): %s", e)

    log.info("Backup complete: %s", backup_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
