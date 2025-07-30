# Use minimal Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install only necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pin version for reproducibility)
RUN pip install uv==0.7.19

# Copy dependency manifests first for cache efficiency
COPY pyproject.toml ./
COPY uv.lock ./

RUN uv sync --frozen

# Copy application source code
COPY src/ ./src/
COPY data/ ./data/

# Set execute permissions on scripts (only if needed)
RUN chmod +x src/*.py

# Default entrypoint to run the ML pipeline
CMD ["python", "src/run_pipeline.py"]
