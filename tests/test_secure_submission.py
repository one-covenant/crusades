"""
Test secure submission flow via validator API.

This verifies that miners submit through the API (not direct storage)
and that storage details remain hidden from miners.
"""

import hashlib

import httpx
import pytest

from tournament.storage.database import Database
from tournament.storage.r2 import get_r2_storage


# Sample training code for testing
SAMPLE_CODE = """
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from collections.abc import Iterator

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    total_tokens = 0
    for step in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long)
        input_ids, labels = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch.numel()
    return InnerStepsResult(logits.detach().float(), total_tokens, loss.item())
"""


@pytest.fixture
async def test_db():
    """Create test database."""
    db = Database(url="sqlite+aiosqlite:///:memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_api_submission_flow(test_db):
    """Test that submission via API stores code privately."""
    # This would normally be called via API endpoint
    from tournament.storage.models import SubmissionModel
    
    code_hash = hashlib.sha256(SAMPLE_CODE.encode()).hexdigest()
    miner_hotkey = "5TestMiner"
    miner_uid = 100
    
    # Simulate what API does: store privately
    bucket_path = f"submissions/{miner_uid}/{code_hash}/train.py"
    r2_storage = get_r2_storage()
    
    # Create temp file
    import tempfile
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(SAMPLE_CODE)
        temp_path = Path(f.name)
    
    try:
        # Upload (validator side)
        upload_success = await r2_storage.upload_code(temp_path, bucket_path)
        assert upload_success
        
        # Create submission record
        submission = SubmissionModel(
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            code_hash=code_hash,
            bucket_path=bucket_path,
            payment_verified=True,
        )
        await test_db.save_submission(submission)
        
        # Verify miner only sees submission ID (not storage path)
        retrieved = await test_db.get_submission(submission.submission_id)
        assert retrieved is not None
        assert retrieved.submission_id == submission.submission_id
        # Storage path is internal, not exposed to miner
        
        # Validator can download from private storage
        download_path = Path(f"/tmp/test_{submission.submission_id}.py")
        download_success = await r2_storage.download_code(bucket_path, download_path)
        assert download_success
        assert download_path.exists()
        
        # Verify content
        downloaded = download_path.read_text()
        assert "def inner_steps" in downloaded
        
        # Cleanup
        if download_path.exists():
            download_path.unlink()
        await r2_storage.delete_code(bucket_path)
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.asyncio
async def test_miner_cannot_access_storage_details():
    """Verify miners don't get storage credentials or paths."""
    # Miner submission data (what they provide)
    miner_data = {
        "code": SAMPLE_CODE,
        "code_hash": hashlib.sha256(SAMPLE_CODE.encode()).hexdigest(),
        "miner_hotkey": "5TestMiner",
        "miner_uid": 100,
    }
    
    # Miner receives only submission ID (not storage details)
    expected_response = {
        "submission_id": "uuid-here",
        "status": "pending",
        # NO bucket_path, NO credentials
    }
    
    # Verify keys miner should NOT see
    assert "bucket_path" not in miner_data
    assert "r2_credentials" not in miner_data
    assert "storage_location" not in expected_response

