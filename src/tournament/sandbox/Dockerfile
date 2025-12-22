FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

# Configure uv for production
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3.12

# Create non-root user for security
RUN useradd -m -s /bin/bash -u 2000 runner

# Set up working directory
WORKDIR /home/runner

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /home/runner/.venv && \
    uv pip install --python=/home/runner/.venv/bin/python \
    torch \
    transformers \
    datasets \
    accelerate \
    fastapi \
    uvicorn \
    pydantic

# Set PATH to use venv
ENV PATH="/home/runner/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/runner/.venv"

# Copy scripts
COPY runner.py /home/runner/runner.py
COPY server.py /home/runner/server.py
RUN chown -R runner:runner /home/runner

# Switch to non-root user
USER runner

# Expose HTTP port
EXPOSE 8000

# Entry point - HTTP server mode
ENTRYPOINT ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
