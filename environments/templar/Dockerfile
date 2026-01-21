# Templar TPS Evaluation Environment
# This Dockerfile packages a miner's train.py for evaluation via Affinetes
#
# Usage:
#   1. Copy your train.py into this folder
#   2. Build: docker build -t your-registry/templar-submission:v1 .
#   3. Push: docker push your-registry/templar-submission:v1
#   4. Commit image URL to blockchain

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

# Copy environment files (env.py for Actor class)
COPY env.py /app/

# Copy miner's submission (train.py)
# This MUST define: inner_steps(model, data_iterator, optimizer, num_steps, device)
COPY train.py /app/

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OUTPUT_VECTOR_TOLERANCE=0.02
ENV DETERMINISTIC_MODE=1
ENV EVAL_SEQUENCE_LENGTH=1024
ENV EVAL_BATCH_SIZE=8

# Note: HTTP server is auto-injected by affinetes framework
# The Actor class in env.py will be exposed as an HTTP endpoint


