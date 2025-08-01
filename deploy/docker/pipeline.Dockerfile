# Base image: Uses a minimal Python 3.10 Debian-based image.
FROM python:3.10-slim

# Sets the working directory inside the container. All relative paths below are now rooted here
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VENV_PATH=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies and cleans up to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installs uv with latest support for --system and virtual environments
RUN pip install --no-cache-dir "uv>=0.1.24"

# Copy pyproject.toml
COPY pyproject.toml ./

# Create and activate virtual environment using uv
RUN uv venv $VENV_PATH && \
    uv pip install -r pyproject.toml

# Copies the source code
COPY src/ ./src/
COPY data/ ./data/

# Runs your ML pipeline script when the container starts
CMD ["python", "src/run_pipeline.py"]
