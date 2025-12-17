"""Submission API endpoints.

SECURITY MODEL:
- Miners submit code via API (POST /api/submissions)
- Validator stores code in PRIVATE R2 bucket
- Miners receive only submission_id (no storage details)
- Miners can check status (GET /api/submissions/{id})
- Storage credentials never exposed to miners

This prevents miners from:
- Accessing other miners' code
- Manipulating storage
- Seeing internal evaluation details
"""

import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from tournament.schemas import EvaluationResponse, SubmissionResponse
from tournament.storage.database import Database, get_database
from tournament.storage.models import SubmissionModel
from tournament.storage.r2 import get_r2_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/submissions", tags=["submissions"])


class SubmissionRequest(BaseModel):
    """Request model for creating a submission."""
    
    code: str
    code_hash: str
    miner_hotkey: str
    miner_uid: int
    payment_block_hash: str | None = None
    payment_extrinsic_index: int | None = None
    payment_amount_rao: int | None = None


@router.post("", status_code=200)
async def create_submission(
    request: SubmissionRequest,
    db: Database = Depends(get_database),
) -> dict:
    """Create a new submission (called by miner).
    
    SECURITY: This endpoint receives code from miners and stores it privately.
    Miners never get direct access to storage.
    """
    logger.info(f"Received submission from miner {request.miner_hotkey} (UID: {request.miner_uid})")
    
    # Verify code hash
    actual_hash = hashlib.sha256(request.code.encode()).hexdigest()
    if actual_hash != request.code_hash:
        logger.warning(f"Code hash mismatch for miner {request.miner_hotkey}")
        raise HTTPException(status_code=400, detail="Code hash verification failed")
    
    # Store code in PRIVATE storage (validator only)
    bucket_path = f"submissions/{request.miner_uid}/{request.code_hash}/train.py"
    r2_storage = get_r2_storage()
    
    # Save code to temporary file first
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(request.code)
        temp_path = Path(f.name)
    
    try:
        # Upload to private storage
        upload_success = await r2_storage.upload_code(temp_path, bucket_path)
        
        if not upload_success:
            logger.error(f"Failed to upload code for miner {request.miner_hotkey}")
            raise HTTPException(status_code=500, detail="Failed to store code")
        
        logger.info(f"Code stored privately at: {bucket_path}")
        
        # Create submission record
        submission = SubmissionModel(
            miner_hotkey=request.miner_hotkey,
            miner_uid=request.miner_uid,
            code_hash=request.code_hash,
            bucket_path=bucket_path,
            payment_block_hash=request.payment_block_hash,
            payment_extrinsic_index=request.payment_extrinsic_index,
            payment_amount_rao=request.payment_amount_rao,
            payment_verified=False,  # Validator will verify
        )
        await db.save_submission(submission)
        
        logger.info(f"Submission created: {submission.submission_id}")
        
        return {
            "submission_id": submission.submission_id,
            "status": "pending",
            "message": "Submission received and queued for evaluation"
        }
        
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@router.get("/{submission_id}", response_model=SubmissionResponse)
async def get_submission(
    submission_id: str,
    db: Database = Depends(get_database),
) -> SubmissionResponse:
    """Get submission status and details."""
    submission = await db.get_submission(submission_id)

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    return SubmissionResponse.model_validate(submission)


@router.get("/{submission_id}/evaluations", response_model=list[EvaluationResponse])
async def get_submission_evaluations(
    submission_id: str,
    db: Database = Depends(get_database),
) -> list[EvaluationResponse]:
    """Get all evaluations for a submission."""
    submission = await db.get_submission(submission_id)

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    evaluations = await db.get_evaluations(submission_id)

    return [EvaluationResponse.model_validate(e) for e in evaluations]
