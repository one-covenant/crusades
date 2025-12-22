# Tournament Sandbox

GPU sandbox for evaluating training code efficiency via HTTP API.

## Quickstart

```bash
# 1. Build and push image
./build.sh --push

# 2. Deploy to Basilica
export BASILICA_API_TOKEN="your-token"
uv run deploy.py

# 3. Test the deployment
./test_evaluate.sh https://YOUR-DEPLOYMENT.deployments.basilica.ai
```

## API Endpoints

### Health Check
```bash
curl https://YOUR-URL/health
```

### Evaluate Code
```bash
curl -X POST https://YOUR-URL/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "...",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "num_steps": 10,
    "batch_size": 4,
    "sequence_length": 512
  }'
```

## Code Requirements

Your code must define an `inner_steps` function:

```python
from dataclasses import dataclass
import torch

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float

def inner_steps(model, data_iterator, optimizer, num_steps, device) -> InnerStepsResult:
    total_tokens = 0
    for step in range(num_steps):
        batch = next(data_iterator).to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch.numel()
    return InnerStepsResult(
        final_logits=outputs.logits,
        total_tokens=total_tokens,
        final_loss=float(loss)
    )
```

## Files

- `server.py` - FastAPI HTTP server
- `runner.py` - Docker-based sandbox runner
- `deploy.py` - Basilica deployment script
- `build.sh` - Docker build/push script
- `test_evaluate.sh` - Test script
