#!/bin/bash
set -euo pipefail

# Test the tournament sandbox evaluate endpoint
#
# Usage:
#   ./test_evaluate.sh <URL>
#
# Example:
#   ./test_evaluate.sh https://bd85329e-3485-4045-b171-5917cf0d8c42.deployments.basilica.ai

if [ $# -lt 1 ]; then
    echo "Usage: $0 <URL>"
    echo "Example: $0 https://my-sandbox.deployments.basilica.ai"
    exit 1
fi

URL="$1"

echo "Testing sandbox at: ${URL}"
echo ""

echo "=== Health Check ==="
curl -s "${URL}/health" 2>&1
echo ""
echo ""

echo "=== Evaluate Test ==="
curl -s -X POST "${URL}/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from dataclasses import dataclass\nimport torch\n\n@dataclass\nclass InnerStepsResult:\n    final_logits: torch.Tensor\n    total_tokens: int\n    final_loss: float\n\ndef inner_steps(model, data_iterator, optimizer, num_steps, device):\n    total_tokens = 0\n    loss = None\n    logits = None\n    for step in range(num_steps):\n        batch = next(data_iterator).to(device)\n        outputs = model(batch, labels=batch)\n        loss = outputs.loss\n        logits = outputs.logits\n        loss.backward()\n        optimizer.step()\n        optimizer.zero_grad()\n        total_tokens += batch.numel()\n    return InnerStepsResult(final_logits=logits, total_tokens=total_tokens, final_loss=float(loss))\n",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "num_steps": 3,
    "batch_size": 2,
    "sequence_length": 128,
    "random_seed": 42
  }' 2>&1

echo ""
echo ""
echo "=== Done ==="
