# Templar TPS Evaluation Environment (VALIDATOR-OWNED)
#
# Usage (for validators):
#   docker build -t templar-eval:latest .
#
# Deploy to Basilica:
#   docker push ghcr.io/one-covenant/templar-eval:latest
#   python -c "from basilica import BasilicaClient; c = BasilicaClient(); d = c.deploy(name='templar-eval', image='ghcr.io/one-covenant/templar-eval:latest', port=8000, ttl_seconds=3600)"

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system dependencies (as root, before switching user)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Copy and install dependencies (owned by appuser)
COPY --chown=appuser:appuser requirements.txt /app/

# Switch to non-root user
USER appuser

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy validator's evaluation logic
# This Actor.evaluate() receives miner's code directly from validator
COPY --chown=appuser:appuser env.py /app/

# NOTE: train.py is NOT included here!
# It's passed directly to the container at evaluation time
# This ensures miners can't tamper with the evaluation logic

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OUTPUT_VECTOR_TOLERANCE=0.02
ENV DETERMINISTIC_MODE=1
ENV EVAL_SEQUENCE_LENGTH=1024
ENV EVAL_BATCH_SIZE=8
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Expose HTTP port for Basilica
EXPOSE 8000

# Default command: Run FastAPI server for Basilica mode
# Docker mode overrides this by passing: python /tmp/eval_script.py
CMD ["python", "-m", "uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000"]
