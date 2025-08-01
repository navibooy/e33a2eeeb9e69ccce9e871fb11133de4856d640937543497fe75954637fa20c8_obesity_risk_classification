# Start from the official Airflow image
FROM apache/airflow:3.0.3-python3.10

# Set working directory
WORKDIR /opt/airflow

# Install system packages required by data science tools
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python dependency manager)
RUN pip install --no-cache-dir "uv>=0.1.24"

# Copy only dependency manifest
COPY pyproject.toml .

# Install dependencies into system environment
RUN uv pip install -r pyproject.toml --system

USER airflow
