# Templar TPS Evaluation Environment (VALIDATOR-OWNED)
#
# R2-Based Architecture:
# - This is the VALIDATOR's evaluation image
# - Miner's train.py is downloaded from their R2 at runtime
# - Miners do NOT build their own Docker images
#
# Usage (for validators):
#   docker build -t templar-eval:latest .

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy validator's evaluation logic
# This Actor.evaluate() downloads miner's code from R2 at runtime
COPY env.py /app/

# NOTE: train.py is NOT included here!
# It's downloaded from miner's R2 bucket at evaluation time
# This ensures miners can't tamper with the evaluation logic

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OUTPUT_VECTOR_TOLERANCE=0.02
ENV DETERMINISTIC_MODE=1
ENV EVAL_SEQUENCE_LENGTH=1024
ENV EVAL_BATCH_SIZE=8

# Note: HTTP server is auto-injected by affinetes framework
# The Actor class in env.py will be exposed as an HTTP endpoint
