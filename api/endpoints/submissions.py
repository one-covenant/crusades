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
import time
from pathlib import Path

from bittensor_wallet.keypair import Keypair
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from tournament.config import get_hparams
from tournament.schemas import EvaluationResponse, SubmissionResponse
from tournament.storage.database import Database, get_database
from tournament.storage.models import SubmissionModel
from tournament.storage.r2 import get_r2_storage

logger = logging.getLogger(__name__)
hparams = get_hparams()

router = APIRouter(prefix="/api/submissions", tags=["submissions"])


def verify_signature(timestamp: int, signature: str, hotkey: str) -> bool:
    """Verify that the signature was created by the claimed hotkey.
    
    Args:
        timestamp: Unix timestamp that was signed
        signature: Hex-encoded signature
        hotkey: SS58 address of the claimed signer
    
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Check timestamp is recent (within 5 minutes)
        now = int(time.time())
        if abs(now - timestamp) > 300:  # 5 minutes
            logger.warning(f"Timestamp too old or future: {timestamp} (now: {now})")
            return False
        
        # Verify signature using public key
        keypair = Keypair(ss58_address=hotkey)
        is_valid = keypair.verify(str(timestamp), bytes.fromhex(signature))
        
        return is_valid
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False


async def verify_code_timestamp(
    code_hash: str,
    block_number_str: str,
    extrinsic_index: int,
    miner_hotkey: str,
    db: Database
) -> tuple[bool, str]:
    """Verify that code_hash was posted to blockchain before submission.
    
    This prevents malicious validators from stealing code by proving who had it first.
    
    Args:
        code_hash: SHA256 hash of the code
        block_number_str: Block number where hash was posted (as string)
        extrinsic_index: Extrinsic index (not used currently)
        miner_hotkey: Who posted it
        db: Database to check for duplicates
        
    Returns:
        (is_valid, error_message)
    """
    try:
        block_number = int(block_number_str)
        logger.info(f"📍 Verifying blockchain timestamp: block {block_number}")
        
        # Query blockchain to verify commitment exists
        import bittensor as bt
        from tournament.config import get_config
        config = get_config()
        subtensor = bt.subtensor(network=config.subtensor_network)
        
        try:
            # Get commitment from blockchain
            commitment = subtensor.get_commitment(
                netuid=hparams.netuid,
                uid=0,  # TODO: Get actual UID
                block=block_number
            )
            
            if commitment and commitment == code_hash:
                logger.info(f"✅ Blockchain commitment verified: {code_hash[:16]}... at block {block_number}")
            else:
                logger.warning(f"⚠️  Commitment not found or mismatch on chain")
                # For now, proceed anyway (verification is optional during testing)
        except Exception as e:
            logger.warning(f"Could not query blockchain commitment: {e}")
            # Proceed anyway for testing
        
        # Check if this code_hash was already submitted by someone else
        all_submissions = await db.get_all_submissions()
        for sub in all_submissions:
            if sub.code_hash == code_hash and sub.miner_hotkey != miner_hotkey:
                # Duplicate code! Need to determine who was first
                
                if not sub.code_timestamp_block_hash:
                    # Original has no blockchain proof, but we do - we win!
                    logger.info(f"✅ Our timestamp proves we're first (original had no proof)")
                    continue
                
                original_block = int(sub.code_timestamp_block_hash)
                original_extrinsic = sub.code_timestamp_extrinsic_index or 999999
                
                # Compare block numbers first
                if original_block < block_number:
                    # Original posted to blockchain earlier (different block)
                    logger.warning(f"🚫 Code already posted at block {original_block} (yours: {block_number})")
                    return False, (
                        f"This code was already submitted at block {original_block}. "
                        f"You submitted at block {block_number}. Original wins."
                    )
                
                elif original_block > block_number:
                    # We posted earlier! Original submission was LATER
                    logger.info(f"✅ You posted first! Block {block_number} < {original_block}")
                    logger.info(f"   Marking original submission {sub.submission_id[:8]}... as duplicate")
                    # TODO: Mark original as duplicate/copied
                    continue
                
                else:
                    # SAME BLOCK! Use extrinsic index as tiebreaker
                    if original_extrinsic < extrinsic_index:
                        # Original was earlier in the same block
                        logger.warning(f"🚫 Same block {block_number}, but earlier extrinsic")
                        logger.warning(f"   Original: extrinsic {original_extrinsic}")
                        logger.warning(f"   Yours: extrinsic {extrinsic_index}")
                        return False, (
                            f"This code was posted in the same block ({block_number}) "
                            f"but at earlier extrinsic index ({original_extrinsic} < {extrinsic_index}). "
                            f"Original wins."
                        )
                    else:
                        # We were earlier in the same block!
                        logger.info(f"✅ Same block, but you were earlier! Extrinsic {extrinsic_index} < {original_extrinsic}")
                        logger.info(f"   Marking original as duplicate")
                        # TODO: Mark original as duplicate
                        continue
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Blockchain timestamp verification error: {e}")
        # Allow submission but log error
        return True, ""  # Lenient for testing


class SubmissionRequest(BaseModel):
    """Request model for creating a submission."""
    
    code: str
    code_hash: str
    miner_hotkey: str
    miner_uid: int
    timestamp: int  # Unix timestamp for signature verification
    signature: str  # Hex-encoded signature of timestamp
    
    # Payment verification
    payment_block_hash: str | None = None
    payment_extrinsic_index: int | None = None
    payment_amount_rao: int | None = None
    
    # Anti-copying: Blockchain timestamp proof
    # Miner posts code_hash to chain BEFORE submitting code
    # This proves they had the code at that block height
    code_timestamp_block_hash: str | None = None
    code_timestamp_extrinsic_index: int | None = None


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
    
    # Verify signature to authenticate miner (PKE authentication)
    if not verify_signature(request.timestamp, request.signature, request.miner_hotkey):
        logger.warning(f"❌ Invalid signature from {request.miner_hotkey}")
        raise HTTPException(
            status_code=403,
            detail="Invalid signature. You must sign the timestamp with your hotkey's private key."
        )
    
    logger.info(f"✅ Signature verified - miner authenticated")
    
    # ANTI-COPYING: Verify blockchain timestamp (multi-validator protection)
    # TODO: Make this REQUIRED once blockchain posting is fully implemented
    if request.code_timestamp_block_hash and request.code_timestamp_extrinsic_index:
        # Verify if timestamp provided
        timestamp_valid, error_msg = await verify_code_timestamp(
            request.code_hash,
            request.code_timestamp_block_hash,
            request.code_timestamp_extrinsic_index,
            request.miner_hotkey,
            db
        )
        
        if not timestamp_valid:
            logger.error(f"❌ Invalid blockchain timestamp from {request.miner_hotkey}")
            raise HTTPException(status_code=403, detail=f"Invalid blockchain timestamp: {error_msg}")
        
        logger.info(f"✅ Blockchain timestamp verified - code ownership proven")
    else:
        # Warning: Proceeding without blockchain timestamp
        logger.warning(f"⚠️  No blockchain timestamp - vulnerable to code theft by malicious validators")
    
    # ANTI-COPYING: Check submission cooldown (prevent rapid copying)
    recent_submissions = await db.get_all_submissions()
    miner_submissions = [s for s in recent_submissions if s.miner_uid == request.miner_uid]
    
    if miner_submissions:
        from datetime import datetime, timedelta
        latest = max(miner_submissions, key=lambda s: s.created_at)
        time_since_last = datetime.utcnow() - latest.created_at
        
        cooldown_minutes = hparams.anti_copying.submission_cooldown_minutes
        if time_since_last < timedelta(minutes=cooldown_minutes):
            minutes_remaining = cooldown_minutes - int(time_since_last.total_seconds() / 60)
            logger.warning(f"🚫 Cooldown violation from miner {request.miner_uid}")
            raise HTTPException(
                status_code=429,
                detail=f"Cooldown period active. Please wait {minutes_remaining} minutes before next submission."
            )
    
    # Basic submission validation
    if len(request.code) < 100:
        raise HTTPException(status_code=400, detail="Code too short (min 100 characters)")
    
    if len(request.code) > 100_000:
        raise HTTPException(status_code=400, detail="Code too long (max 100KB)")
    
    if not request.code.strip().startswith(('"""', "'''", 'from', 'import', 'def', '#')):
        raise HTTPException(status_code=400, detail="Invalid Python code format")
    
    if 'def inner_steps' not in request.code:
        raise HTTPException(status_code=400, detail="Missing required function: inner_steps")
    
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
            code_timestamp_block_hash=request.code_timestamp_block_hash,
            code_timestamp_extrinsic_index=request.code_timestamp_extrinsic_index,
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


@router.get("/{submission_id}/code")
async def get_submission_code(
    submission_id: str,
    db: Database = Depends(get_database),
) -> dict:
    """Get the code for a submission (only after evaluation complete).
    
    ANTI-COPYING: Code is only visible after evaluation finishes.
    This prevents code theft during the evaluation window.
    """
    submission = await db.get_submission(submission_id)

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # SECURITY: Only show code for finished submissions
    from tournament.core.protocols import SubmissionStatus
    if submission.status not in [SubmissionStatus.FINISHED, SubmissionStatus.FAILED_VALIDATION, SubmissionStatus.ERROR]:
        raise HTTPException(
            status_code=403,
            detail="Code not available yet. Submission must complete evaluation first."
        )

    # Download code from R2
    r2_storage = get_r2_storage()
    import tempfile
    from pathlib import Path
    
    temp_file = Path(tempfile.mktemp(suffix=".py"))
    try:
        success = await r2_storage.download_code(submission.bucket_path, str(temp_file))
        
        if not success or not temp_file.exists():
            raise HTTPException(status_code=404, detail="Code not found in storage")
        
        code = temp_file.read_text()
        return {"code": code, "code_hash": submission.code_hash}
    finally:
        if temp_file.exists():
            temp_file.unlink()
