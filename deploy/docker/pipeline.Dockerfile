# Base image: Uses a minimal Python 3.10 Debian-based image.
FROM python:3.10-slim

# Sets the working directory inside the container. All relative paths below are now rooted here.
WORKDIR /app

# Environment variables:
# - PYTHONUNBUFFERED=1: Forces stdout/stderr to be unbuffered (good for logging).
# - PYTHONPATH=/app: Ensures Python can find project modules.
# - VENV_PATH: Defines path for virtual environment.
# - PATH update: Adds virtual env's bin to PATH so installed packages are available.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VENV_PATH=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies needed for building some Python packages (e.g., numpy, pandas).
# Clean up package cache afterward to keep image size small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' — a fast dependency and virtual environment manager for Python.
RUN pip install --no-cache-dir "uv>=0.1.24"

# Copy project dependency configuration into container
COPY pyproject.toml ./

# Create and activate a virtual environment, then install project dependencies from pyproject.toml
RUN uv venv $VENV_PATH && \
    uv pip install -r pyproject.toml

# Copy project source code and data into the container.
COPY src/ ./src/
COPY data/ ./data/

# Define default container behavior: run the full ML pipeline script.
CMD ["python", "src/run_pipeline.py"]
