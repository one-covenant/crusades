# Templar TPS Evaluation Environment (VALIDATOR-OWNED)
#
# URL-Based Architecture:
# - This is the VALIDATOR's evaluation image
# - Miner's train.py is downloaded from their committed URL at runtime
# - Miners do NOT build their own Docker images
#
# Security:
# - Runs as non-root user (appuser) for container security
# - Root user is disabled per Basilica requirements
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

# Note: HTTP server is auto-injected by affinetes framework
# The Actor class in env.py will be exposed as an HTTP endpoint
